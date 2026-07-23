import ctypes
import os
import struct

os.environ["VLLM_CPU_KVCACHE_SPACE"] = "2"

import csv
import time
from pathlib import Path

import numpy as np

# ── PMU via perf_event_open ────────────────────────────────────────────────

libc = ctypes.CDLL("libc.so.6", use_errno=True)

PERF_TYPE_HARDWARE = 0
PERF_TYPE_HW_CACHE = 3
PERF_TYPE_RAW = 4

PERF_COUNT_HW_CPU_CYCLES = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_BRANCH_INSTRUCTIONS = 4
PERF_COUNT_HW_BRANCH_MISSES = 5

PERF_EVENT_IOC_RESET = 0x2402
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401

_PMU_REGULAR = [
    ("cycles",         PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES),
    ("instructions",   PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS),
    ("branches",       PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS),
    ("branch_misses",  PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES),
    ("l1_dcache_loads",       PERF_TYPE_HW_CACHE, 0x0),
    ("l1_dcache_load_misses", PERF_TYPE_HW_CACHE, 0x10000),
    ("llc_loads",             PERF_TYPE_HW_CACHE, 0x2),
    ("llc_load_misses",       PERF_TYPE_HW_CACHE, 0x10002),
    ("dtlb_loads",            PERF_TYPE_HW_CACHE, 0x3),
    ("dtlb_load_misses",      PERF_TYPE_HW_CACHE, 0x10003),
]

_TOPDOWN_LEADER = ("td_slots", 0x0400)
_TOPDOWN_METRICS = [
    ("td_fe_bound",    0x82),
    ("td_be_bound",    0x83),
    ("td_bad_spec",    0x81),
    ("td_retiring",    0x80),
]

_PERF_FORMAT_GROUP = 1 << 3

_ATTR_SIZE = 136  # struct perf_event_attr size for Linux 6.8


class CpuPmu:
    """Lightweight perf_event_open wrapper – own process, not system-wide."""

    def __init__(self, pid=0, cpu=-1):
        self._fds = {}
        self._slots_fd = None
        self._td_names = [m[0] for m in _TOPDOWN_METRICS]

        attr_tmpl = bytearray(_ATTR_SIZE)
        struct.pack_into("I", attr_tmpl, 4, _ATTR_SIZE)
        struct.pack_into("Q", attr_tmpl, 40, 1)

        # Regular counters (no group)
        for name, etype, econfig in _PMU_REGULAR:
            attr = bytearray(attr_tmpl)
            struct.pack_into("I", attr, 0, etype)
            struct.pack_into("Q", attr, 8, econfig)
            buf = ctypes.create_string_buffer(bytes(attr))
            fd = libc.syscall(ctypes.c_long(298), buf,
                              ctypes.c_int(pid), ctypes.c_int(cpu),
                              ctypes.c_int(-1), ctypes.c_ulong(0))
            if fd < 0:
                print(f"  [PMU skip] {name}: {os.strerror(ctypes.get_errno())}")
                continue
            self._fds[name] = fd

        # Top-Down slots group leader (with PERF_FORMAT_GROUP)
        name, config = _TOPDOWN_LEADER
        attr = bytearray(attr_tmpl)
        struct.pack_into("I", attr, 0, PERF_TYPE_RAW)
        struct.pack_into("Q", attr, 8, config)
        struct.pack_into("Q", attr, 32, _PERF_FORMAT_GROUP)
        buf = ctypes.create_string_buffer(bytes(attr))
        fd = libc.syscall(ctypes.c_long(298), buf,
                          ctypes.c_int(pid), ctypes.c_int(cpu),
                          ctypes.c_int(-1), ctypes.c_ulong(0))
        if fd < 0:
            print(f"  [PMU skip] {name}: {os.strerror(ctypes.get_errno())}")
        else:
            self._slots_fd = fd

        # Top-Down metric events (group members)
        if self._slots_fd is not None:
            for name, config in _TOPDOWN_METRICS:
                attr = bytearray(attr_tmpl)
                struct.pack_into("I", attr, 0, PERF_TYPE_RAW)
                struct.pack_into("Q", attr, 8, config)
                buf = ctypes.create_string_buffer(bytes(attr))
                fd = libc.syscall(ctypes.c_long(298), buf,
                                  ctypes.c_int(pid), ctypes.c_int(cpu),
                                  ctypes.c_int(self._slots_fd), ctypes.c_ulong(0))
                if fd < 0:
                    print(f"  [PMU skip] {name}: {os.strerror(ctypes.get_errno())}")
                    continue
                self._fds[name] = fd

        if not self._fds:
            raise RuntimeError("no PMU counters available")

    def reset_and_enable(self):
        for fd in self._fds.values():
            libc.ioctl(ctypes.c_int(fd), ctypes.c_uint(PERF_EVENT_IOC_RESET))
            libc.ioctl(ctypes.c_int(fd), ctypes.c_uint(PERF_EVENT_IOC_ENABLE))

    def disable(self):
        for fd in self._fds.values():
            libc.ioctl(ctypes.c_int(fd), ctypes.c_uint(PERF_EVENT_IOC_DISABLE))

    def read_all(self):
        values = {}
        for name, fd in self._fds.items():
            if name.startswith("td_"):
                continue
            buf = os.read(fd, 8)
            values[name] = struct.unpack("Q", buf)[0]

        if self._slots_fd is not None:
            n = 1 + len(self._td_names)
            buf = os.read(self._slots_fd, n * 8)
            vals = struct.unpack(f"{n}Q", buf)
            values["td_slots"] = vals[0]
            for i, name in enumerate(self._td_names):
                values[name] = vals[i + 1]
            os.close(self._slots_fd)
            self._slots_fd = None

        return values

    def close(self):
        if self._slots_fd is not None:
            os.close(self._slots_fd)
            self._slots_fd = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    from datetime import datetime

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    INPUT_LEN = 32
    OUTPUT_LEN = 64
    BATCH_SIZES = [32, 64, 96]
    WARMUP_ITERS = 5
    MEASURE_ITERS = 15
    MODEL_PATH = "./qwen2.5-1.5B-int4"
    DTYPE = "bfloat16"
    ENFORCE_EAGER = False

    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=OUTPUT_LEN,
    )

    run_dir = Path("bench_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "bench_results.csv"
    results = []
    pmu_results = []
    td_results = []
    config_captured = None

    for bs in BATCH_SIZES:
        print(f"\n{'='*50}")
        print(f"Batch size: {bs}")

        try:
            llm = LLM(
                model=MODEL_PATH,
                dtype=DTYPE,
                gpu_memory_utilization=1.0,
                enforce_eager=ENFORCE_EAGER,
                disable_log_stats=True,
            )

            if config_captured is None:
                vc = llm.llm_engine.vllm_config
                config_captured = {
                    "model": MODEL_PATH,
                    "dtype": DTYPE,
                    "batch_size": str(BATCH_SIZES).replace(" ", ""),
                    "input_len": INPUT_LEN,
                    "output_len": OUTPUT_LEN,
                    "warmup_iterations": WARMUP_ITERS,
                    "measurement_iterations": MEASURE_ITERS,
                    "enable_prefix_caching": vc.cache_config.enable_prefix_caching,
                    "enforce_eager": ENFORCE_EAGER,
                    "tensor_parallel_size": vc.parallel_config.tensor_parallel_size,
                    "pipeline_parallel_size": vc.parallel_config.pipeline_parallel_size,
                    "kv_cache_space_gb": os.environ.get("VLLM_CPU_KVCACHE_SPACE", ""),
                    "omp_threads": os.environ.get("OMP_NUM_THREADS", str(os.cpu_count())),
                    "cpu_binding": os.environ.get("VLLM_CPU_OMP_THREADS_BIND", "auto"),
                    "seed": vc.model_config.seed,
                }

            dummy_token_ids = np.random.randint(10000, size=(bs, INPUT_LEN))
            dummy_prompts = [
                TokensPrompt(prompt_token_ids=batch.tolist())
                for batch in dummy_token_ids
            ]

            for _ in range(WARMUP_ITERS):
                llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)

            pmu = None
            try:
                pmu = CpuPmu()
                pmu.reset_and_enable()
            except Exception as e:
                print(f"  [PMU init failed] {e}")

            latencies = []
            for _ in range(MEASURE_ITERS):
                t0 = time.perf_counter()
                llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
                latencies.append(time.perf_counter() - t0)

            pmu_raw = {}
            if pmu is not None:
                pmu.disable()
                pmu_raw = pmu.read_all()
                pmu.close()

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

            if pmu_raw:
                n = MEASURE_ITERS
                cycles = pmu_raw.get("cycles", 0) / n
                instructions = pmu_raw.get("instructions", 0) / n
                branches = pmu_raw.get("branches", 0)
                bmiss = pmu_raw.get("branch_misses", 0)

                l1_loads = pmu_raw.get("l1_dcache_loads", 0)
                l1_misses = pmu_raw.get("l1_dcache_load_misses", 0)
                llc_loads = pmu_raw.get("llc_loads", 0)
                llc_misses = pmu_raw.get("llc_load_misses", 0)
                dtlb_loads = pmu_raw.get("dtlb_loads", 0)
                dtlb_misses = pmu_raw.get("dtlb_load_misses", 0)

                pmu_r = {
                    "batch_size": bs,
                    "cycles": int(cycles),
                    "instructions": int(instructions),
                    "ipc": instructions / cycles if cycles else 0.0,
                    "branch_miss_pct": (bmiss / branches * 100) if branches else 0.0,
                    "l1_miss_pct": (l1_misses / l1_loads * 100) if l1_loads else 0.0,
                    "llc_miss_pct": (llc_misses / llc_loads * 100) if llc_loads else 0.0,
                    "dtlb_miss_pct": (dtlb_misses / dtlb_loads * 100) if dtlb_loads else 0.0,
                }
                pmu_results.append(pmu_r)

            if pmu_raw and "td_slots" in pmu_raw and pmu_raw["td_slots"]:
                slots = pmu_raw["td_slots"]
                td_r = {
                    "batch_size": bs,
                    "frontend_bound_pct": pmu_raw.get("td_fe_bound", 0) / slots * 100,
                    "backend_bound_pct": pmu_raw.get("td_be_bound", 0) / slots * 100,
                    "bad_speculation_pct": pmu_raw.get("td_bad_spec", 0) / slots * 100,
                    "retiring_pct": pmu_raw.get("td_retiring", 0) / slots * 100,
                }
                td_results.append(td_r)

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
    header = f"{'batch':>6} | {'avg_lat(s)':>10} | {'P50(s)':>10} | {'P99(s)':>10} | {'tok/s':>12} | {'req/s':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['batch_size']:>6} | {r['avg_latency']:>10.4f} | {r['p50_latency']:>10.4f} | "
            f"{r['p99_latency']:>10.4f} | {r['tok_per_s']:>12.1f} | {r['req_per_s']:>10.1f}"
        )

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
    else:
        print("No configuration captured")

    if pmu_results:
        pmu_path = run_dir / "cpu_pmu.csv"
        with open(pmu_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pmu_results[0].keys())
            writer.writeheader()
            writer.writerows(pmu_results)
        print(f"PMU saved to {pmu_path}")

    if td_results:
        td_path = run_dir / "top_down.csv"
        with open(td_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=td_results[0].keys())
            writer.writeheader()
            writer.writerows(td_results)
        print(f"Top-down saved to {td_path}")


if __name__ == "__main__":
    main()
