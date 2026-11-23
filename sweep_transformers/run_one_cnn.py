#!/usr/bin/env python3
"""
Record one CNN several times in a single power trace, with idle gaps between runs.

Default: ResNet-152, batch 64, 288x288 image, fp32, random weights.
Each run = back-to-back forward passes for about 1 s (10 passes of ~99 ms on the 4090).
5 runs, 1 s idle between runs, 0.5 s idle before the first and 1.5 s after the last.

    python run_one_cnn.py
    python run_one_cnn.py --runs 5 --idle 1.0 --run-ms 1000
    python run_one_cnn.py --model vgg19_bn --batch 32 --image-size 224 --out vgg19bn_repeat5
    python run_one_cnn.py --passes-list 2 4 6 8 10      # run 1 = 2 passes, run 2 = 4 passes, ...

Output folder (default resnet152_b64_288_x5/):
    power_raw.csv   every NVML sample: time (s, 0 = first run start), watts
    runs.csv        start/end time and duration of each run
    trace.png       power vs. time, each run shaded gray
Must sit in the same folder as sweep.py (it reuses sweep.py's PowerSampler).
"""

import argparse
import csv
import time
from pathlib import Path

import torch

import sweep as S


def parse_args():
    p = argparse.ArgumentParser(description="Record one CNN several times with idle gaps between runs.")
    p.add_argument("--model", default="resnet152", help="any torchvision model function name")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--image-size", type=int, default=288, help="square image size")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--pretrained", action="store_true", help="ImageNet weights instead of random (downloads)")
    p.add_argument("--runs", type=int, default=5, help="how many runs")
    p.add_argument("--idle", type=float, default=1.0, help="idle seconds between runs")
    p.add_argument("--run-ms", type=float, default=1000,
                   help="length of one run; passes = whole number closest to run-ms / one pass")
    p.add_argument("--passes", type=int, default=0, help="fixed passes per run (overrides --run-ms)")
    p.add_argument("--passes-list", type=int, nargs="+", default=None,
                   help="passes for each run in order, e.g. 2 4 6 8 10 (overrides --runs, --passes, --run-ms)")
    p.add_argument("--pre", type=float, default=0.5, help="idle seconds recorded before the first run")
    p.add_argument("--post", type=float, default=1.5, help="idle seconds recorded after the last run")
    p.add_argument("--cooldown", type=float, default=2.0, help="seconds to wait after warm-up before recording")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default="", help="output folder (default: <model>_b<batch>_<size>_x<runs>)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.passes_list:
        args.runs = len(args.passes_list)
    import torchvision.models as M

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found")
    device = torch.device(f"cuda:{args.gpu}")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    tag = ("_p" + "-".join(map(str, args.passes_list))) if args.passes_list else f"_x{args.runs}"
    out = Path(args.out or f"{args.model}_b{args.batch}_{args.image_size}{tag}")
    out.mkdir(parents=True, exist_ok=True)

    # ---- build the model and input (same as the CNN sweep) ----
    model = getattr(M, args.model)(weights="DEFAULT" if args.pretrained else None)
    model = model.to(device=device, dtype=dtype).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn((args.batch, 3, args.image_size, args.image_size), generator=g).to(device=device, dtype=dtype)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # ---- warm up, then time one pass to decide how many passes make one run ----
    with torch.inference_mode():
        model(x)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(3):
            model(x)
        torch.cuda.synchronize(device)
    pass_s = (time.perf_counter() - t0) / 3
    if args.passes_list:
        plan = list(args.passes_list)
    else:
        plan = [args.passes or max(1, int(round(args.run_ms / 1000 / pass_s)))] * args.runs
    varied = len(set(plan)) > 1
    passes_txt = ", ".join(map(str, plan)) if varied else str(plan[0])
    print(f"{args.model}: {params_m:.1f}M params, batch {args.batch}, {args.image_size}x{args.image_size}, "
          f"{args.dtype}")
    print(f"one pass {pass_s * 1000:.1f} ms -> passes per run: {passes_txt} "
          f"(~{', '.join(f'{n * pass_s * 1000:.0f}' for n in sorted(set(plan)))} ms); "
          f"{args.runs} runs, {args.idle:g} s idle between")

    def one_run(passes):
        with torch.inference_mode():
            for _ in range(passes):
                model(x)
        torch.cuda.synchronize(device)

    # ---- record ----
    sampler = S.PowerSampler(args.gpu)
    static = sampler.static_info()
    time.sleep(args.cooldown)                    # let the warm-up power settle
    sampler.start()
    rec_start = time.time()
    time.sleep(args.pre)
    runs = []
    for k, n in enumerate(plan):
        start = time.time()
        one_run(n)
        end = time.time()
        runs.append((k + 1, start, end, n))
        print(f"  run {k + 1}: {n} passes, {(end - start) * 1000:.0f} ms")
        if k < args.runs - 1:
            time.sleep(args.idle)
    time.sleep(args.post)
    rec_end = time.time()
    time.sleep(0.3)                              # let the last samples reach the buffer
    times, watts = sampler.window(rec_start, rec_end)
    sampler.stop()

    t_ref = runs[0][1]                           # time 0 = start of the first run
    rel = [t - t_ref for t in times]

    # ---- save ----
    with open(out / "power_raw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "watts"])
        w.writerows((f"{t:.4f}", f"{p:.2f}") for t, p in zip(rel, watts))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "start_s", "end_s", "duration_ms", "passes", "mean_w"])
        for k, s, e, n in runs:
            inside = [p for t, p in zip(times, watts) if s <= t <= e]
            mean_w = sum(inside) / len(inside) if inside else float("nan")
            w.writerow([k, f"{s - t_ref:.4f}", f"{e - t_ref:.4f}", f"{(e - s) * 1000:.1f}", n, f"{mean_w:.1f}"])
            print(f"  run {k}: mean {mean_w:.0f} W over {len(inside)} samples")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    limit = float(static.get("power_limit_w") or 450)
    ymax = max(limit * 1.15, 1.05 * max(watts) if watts else 0)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (k, s, e, n) in enumerate(runs):
        ax.axvspan(s - t_ref, e - t_ref, color="0.88", label="run (inference)" if i == 0 else None)
        if varied:
            ax.text((s + e) / 2 - t_ref, ymax * 0.97, f"{n}", ha="center", va="top", fontsize=9, color="0.3")
    ax.step(rel, watts, where="post", lw=1.2, color="tab:blue", label="power (NVML samples)")
    ax.set_xlim(-args.pre, rel[-1] if rel else 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Time since first run start (s)")
    ax.set_ylabel("Power (W)")
    ax.set_title(f"{args.model} ({params_m:.1f}M params) | batch {args.batch} | {args.image_size}x{args.image_size} | "
                 f"{args.dtype} | {'trained' if args.pretrained else 'random'} weights\n"
                 f"{args.runs} runs of {passes_txt} back-to-back passes ({pass_s * 1000:.1f} ms each"
                 f"{'; number above each run = passes' if varied else ''}), "
                 f"{args.idle:g} s idle between runs | {static.get('gpu', '')}", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "trace.png", dpi=150)
    print(f"saved {out}/power_raw.csv, runs.csv, trace.png")


if __name__ == "__main__":
    main()