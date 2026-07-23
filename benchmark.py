import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"
os.environ["VLLM_CPU_INT4_W4A8"] = "1"

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.tokenizers import get_tokenizer

INPUT_LEN = 512
OUTPUT_LEN = 64
BATCH_SIZES = [32, 64, 96]
WARMUP_ITERS = 5
MEASURE_ITERS = 15
MODEL_PATH = "/home/kasw/文档/GitHub/GPU-Project/vllm/qwen2.5-1.5B-int4"
DTYPE = "bfloat16"
ENFORCE_EAGER = False
DATASET_PATH = "datasets/sharegpt_workload.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    batch_sizes = [args.batch_size] if args.batch_size else BATCH_SIZES

    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=OUTPUT_LEN,
    )

    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = Path("bench_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "bench_results.csv"
    results = []
    config_captured = None

    for bs in batch_sizes:
        print(f"\n{'='*50}")
        print(f"Batch size: {bs}")

        try:
            llm = LLM(
                model=MODEL_PATH,
                dtype=DTYPE,
                quantization="awq",
                max_num_batched_tokens=10240,
                max_model_len=4096,
                gpu_memory_utilization=0.8,
                enforce_eager=ENFORCE_EAGER,
                enable_prefix_caching=True,
                disable_log_stats=True,
            )

            if config_captured is None and not args.no_csv:
                vc = llm.llm_engine.vllm_config
                config_captured = {
                    "model": MODEL_PATH,
                    "dtype": DTYPE,
                    "batch_size": str(batch_sizes).replace(" ", ""),
                    "input_len": INPUT_LEN,
                    "output_len": OUTPUT_LEN,
                    "warmup_iterations": WARMUP_ITERS,
                    "measurement_iterations": MEASURE_ITERS,
                    "enable_prefix_caching": vc.cache_config.enable_prefix_caching,
                    "enforce_eager": ENFORCE_EAGER,
                    "tensor_parallel_size": vc.parallel_config.tensor_parallel_size,
                    "pipeline_parallel_size": vc.parallel_config.pipeline_parallel_size,
                    "kv_cache_space_gb": os.environ.get("VLLM_CPU_KVCACHE_SPACE", ""),
                    "omp_threads": os.environ.get(
                        "OMP_NUM_THREADS", str(os.cpu_count())
                    ),
                    "cpu_binding": os.environ.get(
                        "VLLM_CPU_OMP_THREADS_BIND", "auto"
                    ),
                    "seed": vc.model_config.seed,
                }

            tokenizer = get_tokenizer(MODEL_PATH)
            with open(DATASET_PATH) as f:
                workload = json.load(f)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

            padded_tokens = []
            for i in range(bs):
                tokens = tokenizer.encode(workload[i % len(workload)]["prompt"])
                if len(tokens) >= INPUT_LEN:
                    tokens = tokens[:INPUT_LEN]
                else:
                    tokens += [pad_id] * (INPUT_LEN - len(tokens))
                padded_tokens.append(tokens)
            prompts = [TokensPrompt(prompt_token_ids=t) for t in padded_tokens]

            for _ in range(WARMUP_ITERS):
                llm.generate(
                    prompts, sampling_params=sampling_params, use_tqdm=False
                )

            latencies = []
            for _ in range(MEASURE_ITERS):
                t0 = time.perf_counter()
                llm.generate(
                    prompts, sampling_params=sampling_params, use_tqdm=False
                )
                latencies.append(time.perf_counter() - t0)

            latencies = np.array(latencies)
            total_tokens = bs * (INPUT_LEN + OUTPUT_LEN)

            r = {
                "batch_size": bs,
                "avg_latency": float(np.mean(latencies)),
                "min_latency": float(np.min(latencies)),
                "max_latency": float(np.max(latencies)),
                "p50_latency": float(np.percentile(latencies, 50)),
                "p99_latency": float(np.percentile(latencies, 99)),
                "tok_per_s": total_tokens / float(np.mean(latencies)),
                "req_per_s": bs / float(np.mean(latencies)),
            }
            results.append(r)

            print(f"  avg_latency: {r['avg_latency']:.4f}s")
            print(f"  p50_latency: {r['p50_latency']:.4f}s")
            print(f"  p99_latency: {r['p99_latency']:.4f}s")
            print(f"  tok/s:       {r['tok_per_s']:.1f}")
            print(f"  req/s:       {r['req_per_s']:.1f}")

            del llm

        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print(f"\n{'='*80}")
    header = (
        f"{'batch':>6} | {'avg_lat(s)':>10} | {'P50(s)':>10} | {'P99(s)':>10}"
        f" | {'tok/s':>12} | {'req/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['batch_size']:>6} | {r['avg_latency']:>10.4f} | "
            f"{r['p50_latency']:>10.4f} | {r['p99_latency']:>10.4f} | "
            f"{r['tok_per_s']:>12.1f} | {r['req_per_s']:>10.1f}"
        )

    if args.no_csv:
        print("\n[CSV output disabled]")
        return

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")
    else:
        print("\nNo results to save")

    if config_captured:
        config_path = run_dir / "bench_configurations.csv"
        with open(config_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(config_captured.keys())
            writer.writerow(config_captured.values())
        print(f"Configurations saved to {config_path}")


if __name__ == "__main__":
    main()
