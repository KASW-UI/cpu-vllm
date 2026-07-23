import csv
import re
import sys
from pathlib import Path


def normalize_event(name):
    """Strip cpu_core/ prefix; discard cpu_atom/ events entirely."""
    m = re.match(r"^cpu_core/(.+?)/?$", name)
    if m:
        return m.group(1)
    if name.startswith("cpu_atom/"):
        return None
    return name


def parse_perf_stat(filepath):
    """Parse perf stat -o output into {event_name: count} dict (P-core only)."""
    counts = {}
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.lstrip().startswith("#"):
                continue
            if "<not supported>" in line:
                continue
            if "Performance counter stats" in line:
                continue
            if "seconds" in line and (
                "time elapsed" in line
                or "seconds user" in line
                or "seconds sys" in line
            ):
                continue
            m = re.match(r"^\s*([\d,]+)\s+(\S+)", line)
            if m:
                count_str = m.group(1).replace(",", "")
                base = normalize_event(m.group(2))
                if base is not None:
                    counts[base] = int(count_str)
    return counts


def parse_perf_time(filepath):
    """Extract elapsed seconds from perf stat -o output."""
    with open(filepath) as f:
        for line in f:
            m = re.match(r"\s*([\d.]+)\s+seconds time elapsed", line)
            if m:
                return float(m.group(1))
    return 0.0


def safe_pct(numerator, denominator):
    if denominator:
        return numerator / denominator * 100
    return 0.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_perf.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}")
        sys.exit(1)

    batch_sizes = set()
    for f in run_dir.glob("pmu_bs*.txt"):
        m = re.match(r"pmu_bs(\d+)\.txt", f.name)
        if m:
            batch_sizes.add(int(m.group(1)))
    for f in run_dir.glob("mem_bs*.txt"):
        m = re.match(r"mem_bs(\d+)\.txt", f.name)
        if m:
            batch_sizes.add(int(m.group(1)))

    if not batch_sizes:
        print("No pmu_bs*.txt files found")
        sys.exit(1)

    pmu_rows = []
    td_rows = []
    mem_rows = []

    for bs in sorted(batch_sizes):
        pmu_file = run_dir / f"pmu_bs{bs}.txt"
        cache_file = run_dir / f"cache_bs{bs}.txt"
        td_file = run_dir / f"td_bs{bs}.txt"
        mem_file = run_dir / f"mem_bs{bs}.txt"

        pmu_counts = parse_perf_stat(pmu_file) if pmu_file.exists() else {}
        cache_counts = parse_perf_stat(cache_file) if cache_file.exists() else {}
        td_counts = parse_perf_stat(td_file) if td_file.exists() else {}
        mem_counts = parse_perf_stat(mem_file) if mem_file.exists() else {}
        elapsed = parse_perf_time(mem_file) if mem_file.exists() else 0.0

        cycles = pmu_counts.get("cycles", 0)
        instructions = pmu_counts.get("instructions", 0)
        branches = pmu_counts.get("branches", 0)
        bmiss = pmu_counts.get("branch-misses", 0)

        l1_loads = cache_counts.get("L1-dcache-loads", 0)
        l1_misses = cache_counts.get("L1-dcache-load-misses", 0)
        llc_loads = cache_counts.get("LLC-loads", 0)
        llc_misses = cache_counts.get("LLC-load-misses", 0)
        dtlb_loads = cache_counts.get("dTLB-loads", 0)
        dtlb_misses = cache_counts.get("dTLB-load-misses", 0)

        td_slots = td_counts.get("topdown.slots", 0)

        pmu_row = {
            "batch_size": bs,
            "cycles": cycles,
            "instructions": instructions,
            "ipc": instructions / cycles if cycles else 0.0,
            "branch_miss_pct": safe_pct(bmiss, branches),
            "l1_miss_pct": safe_pct(l1_misses, l1_loads),
            "llc_miss_pct": safe_pct(llc_misses, llc_loads),
            "dtlb_miss_pct": safe_pct(dtlb_misses, dtlb_loads),
        }
        pmu_rows.append(pmu_row)

        td_row = {
            "batch_size": bs,
            "frontend_bound_pct": safe_pct(
                td_counts.get("topdown-fe-bound", 0), td_slots
            ),
            "backend_bound_pct": safe_pct(
                td_counts.get("topdown-be-bound", 0), td_slots
            ),
            "bad_speculation_pct": safe_pct(
                td_counts.get("topdown-bad-spec", 0), td_slots
            ),
            "retiring_pct": safe_pct(
                td_counts.get("topdown-retiring", 0), td_slots
            ),
        }
        td_rows.append(td_row)

        rd = mem_counts.get("unc_m_cas_count_rd", 0)
        wr = mem_counts.get("unc_m_cas_count_wr", 0)
        dram_read_gbps = (rd * 64) / elapsed / 1e9 if elapsed else 0.0
        dram_write_gbps = (wr * 64) / elapsed / 1e9 if elapsed else 0.0
        mem_rows.append({
            "batch_size": bs,
            "memory_bandwidth_gbps": dram_read_gbps + dram_write_gbps,
            "dram_read_gbps": dram_read_gbps,
            "dram_write_gbps": dram_write_gbps,
            "numa_node": 0,
            "remote_memory_pct": 0.0,
        })

    if pmu_rows:
        pmu_path = run_dir / "cpu_pmu.csv"
        with open(pmu_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pmu_rows[0].keys())
            writer.writeheader()
            writer.writerows(pmu_rows)
        print(f"cpu_pmu.csv saved to {pmu_path}")

    if td_rows:
        td_path = run_dir / "top_down.csv"
        with open(td_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=td_rows[0].keys())
            writer.writeheader()
            writer.writerows(td_rows)
        print(f"top_down.csv saved to {td_path}")

    if mem_rows:
        mem_path = run_dir / "memory.csv"
        with open(mem_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mem_rows[0].keys())
            writer.writeheader()
            writer.writerows(mem_rows)
        print(f"memory.csv saved to {mem_path}")


if __name__ == "__main__":
    main()
