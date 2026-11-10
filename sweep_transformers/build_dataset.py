#!/usr/bin/env python3
"""
Combines the equal-length traces from one or more sweep output folders into
one wide CSV: one row per trace, label/metadata columns first, then the power
values p0, p1, ..., pN.

Power column p_k is the power (W) at time  -pre + k * grid  (ms) relative to
the start of the inference, so the first columns are the idle period before
the run. Only traces with status "ok" are included.

Usage:
    python build_dataset.py released_out
    python build_dataset.py released_out arch_out --out transformer_dataset.csv
"""

import argparse
import csv
import sys
from pathlib import Path

LABEL_COLUMNS = [
    "trace_id", "model_type", "source", "model", "family", "kind",
    "layers", "hidden", "heads", "params_m", "dtype",
    "tokens", "input_len", "batch", "kv_cache",
    "run_ms", "mean_run_w", "idle_w", "gpu_temp_c", "sm_clock_mhz",
    "input_shape", "weights", "arch_desc",
    "pre_s", "window_s", "grid_ms", "n_points", "folder",
]


def build(folders, out_path):
    """Combine the folders' traces into one wide CSV at out_path. Returns the number of traces."""
    rows, shape = [], None          # shape = (pre_s, window_s, grid_ms, n_points), must match everywhere
    for folder in map(Path, folders):
        manifest = folder / "manifest.csv"
        if not manifest.exists():
            sys.exit(f"{manifest} not found")
        with open(manifest, newline="") as f:
            entries = list(csv.DictReader(f))
        ok = [r for r in entries if r["status"] == "ok"]
        print(f"{folder}: {len(ok)} ok traces ({len(entries) - len(ok)} skipped: out of memory / error)")

        for r in ok:
            this = (float(r["pre_s"]), float(r["window_s"]), float(r["grid_ms"]), int(r["n_points"]))
            if shape is None:
                shape = this
            elif this != shape:
                sys.exit(f"{folder} uses pre {this[0]} s, window {this[1]} s, grid {this[2]} ms "
                         f"({this[3]} points), but earlier folders use pre {shape[0]} s, window "
                         f"{shape[1]} s, grid {shape[2]} ms ({shape[3]} points).\n"
                         f"Traces must be the same length to share one CSV: rerun with the same --window.")
            with open(folder / r["trace_file"], newline="") as f:
                power = [line["power_w"] for line in csv.DictReader(f)]
            if len(power) != shape[3]:
                sys.exit(f"{folder / r['trace_file']} has {len(power)} points, expected {shape[3]}")
            label = {k: r.get(k, "") for k in LABEL_COLUMNS}
            label["folder"] = folder.name
            rows.append((label, power))

    if not rows:
        sys.exit("no traces found")
    n = shape[3]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LABEL_COLUMNS + [f"p{k}" for k in range(n)])
        for label, power in rows:
            w.writerow([label[k] for k in LABEL_COLUMNS] + power)

    pre_ms, grid = shape[0] * 1000, shape[2]
    counts = {}
    for label, _ in rows:
        counts[label["model_type"]] = counts.get(label["model_type"], 0) + 1
    print(f"\nwrote {out_path}: {len(rows)} traces x {n} power points "
          f"(p0 = {-pre_ms:g} ms ... p{n - 1} = {-pre_ms + (n - 1) * grid:g} ms, every {grid:g} ms)")
    print("traces per model_type: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return len(rows)


def main():
    p = argparse.ArgumentParser(description="Combine sweep traces into one wide CSV.")
    p.add_argument("folders", nargs="+", help="sweep output folders (each containing manifest.csv)")
    p.add_argument("--out", default="dataset.csv", help="output CSV file (default: dataset.csv)")
    args = p.parse_args()
    build(args.folders, args.out)


if __name__ == "__main__":
    main()