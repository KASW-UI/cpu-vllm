#!/bin/bash
# run_benchmarks.sh — 5-pass CPU benchmark orchestration
#
# Pass 1: latency baseline (all batch sizes, no perf)
# Pass 2: perf stat PMU (cycles, instructions, branches, branch-misses)
# Pass 3: perf stat Cache/TLB (L1-dcache, LLC, dTLB)
# Pass 4: perf stat Topdown (topdown.slots, -retiring, -bad-spec, -fe-bound, -be-bound)
# Pass 5: perf stat Memory BW (unc_m_cas_count_rd, unc_m_cas_count_wr) — requires sudo
#
# Each pass runs per batch_size independently, producing:
#   bench_results.csv  (from pass 1 only)
#   bench_configurations.csv (from pass 1 only)
#   cpu_pmu.csv        (parsed from pass 2 + 3)
#   top_down.csv       (parsed from pass 4)
#   memory.csv         (parsed from pass 5)

set -o pipefail

RUN_DIR="bench_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

BATCH_SIZES=(32 64 96)

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: $1 not found" >&2
        exit 1
    fi
}

check_cmd python
check_cmd perf

PYTHON="/home/kasw/workspace/cpu-vllm-infra/.venv/bin/python"
echo "Using Python: $PYTHON"

echo "========================================"
echo "  vLLM CPU Benchmark — 5-pass run"
echo "  Output dir: $RUN_DIR"
echo "========================================"

# ── Pass 1: Latency baseline (all batch sizes in one invocation) ──

echo ""
echo "=== Pass 1/5: Latency baseline ==="
"$PYTHON" benchmark.py --output-dir "$RUN_DIR" || {
    echo "FATAL: baseline run failed"
    exit 1
}

# ── Passes 2-4: per-batch perf stat runs ──

for BS in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "=== Pass 2/5: PMU (bs=$BS) ==="
    perf stat \
        -e cycles,instructions,branches,branch-misses \
        -o "$RUN_DIR/pmu_bs${BS}.txt" \
        "$PYTHON" benchmark.py --batch-size "$BS" --output-dir "$RUN_DIR" --no-csv \
        || echo "WARNING: PMU run for bs=$BS failed, skipping"

    echo ""
    echo "=== Pass 3/5: Cache/TLB (bs=$BS) ==="
    perf stat \
        -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses \
        -o "$RUN_DIR/cache_bs${BS}.txt" \
        "$PYTHON" benchmark.py --batch-size "$BS" --output-dir "$RUN_DIR" --no-csv \
        || echo "WARNING: Cache/TLB run for bs=$BS failed, skipping"

    echo ""
    echo "=== Pass 4/5: Topdown (bs=$BS) ==="
    perf stat \
        -e topdown.slots,topdown-retiring,topdown-bad-spec,topdown-fe-bound,topdown-be-bound \
        -o "$RUN_DIR/td_bs${BS}.txt" \
        "$PYTHON" benchmark.py --batch-size "$BS" --output-dir "$RUN_DIR" --no-csv \
        || echo "WARNING: Topdown run for bs=$BS failed, skipping"

    echo ""
    echo "=== Pass 5/5: Memory BW (bs=$BS) ==="
    perf stat \
        -e unc_m_cas_count_rd,unc_m_cas_count_wr \
        -o "$RUN_DIR/mem_bs${BS}.txt" \
        "$PYTHON" benchmark.py --batch-size "$BS" --output-dir "$RUN_DIR" --no-csv \
        || echo "WARNING: Memory BW run for bs=$BS failed, skipping"
done

# ── Parse perf stat output into CSVs ──

echo ""
echo "=== Parsing perf stat output ==="
"$PYTHON" parse_perf.py "$RUN_DIR" || echo "WARNING: parse_perf.py failed"

rm -f "$RUN_DIR"/pmu_bs*.txt "$RUN_DIR"/cache_bs*.txt "$RUN_DIR"/td_bs*.txt "$RUN_DIR"/mem_bs*.txt

echo ""
echo "=== Done ==="
echo "Output directory: $RUN_DIR"
ls -la "$RUN_DIR"/*.csv 2>/dev/null || echo "(no CSV files)"
