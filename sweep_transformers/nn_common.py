#!/usr/bin/env python3
"""
Shared code for sweep_mlp.py, sweep_bnn.py and sweep_cnn.py: building the run plan,
recording one forward pass per trace, and saving traces/manifest/plots in the same
format as sweep.py. You don't run this file directly.
"""

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

import sweep as S          # sweep.py in the same folder: power recorder, resampling, plotting

FIELDS = S.MANIFEST_FIELDS + ["input_shape", "weights", "arch_desc"]


# =========================================================================== BNN repo helpers

def import_bnn_repo(path):
    repo = Path(path).resolve()
    if not (repo / "models" / "binary_modules.py").exists():
        sys.exit(f"BNN code not found at {repo} (expected models/binary_modules.py).\n"
                 f"Pass the folder of efficient-binary-neural-networks with --bnn-repo.")
    sys.path.insert(0, str(repo))
    import models.binary_modules as bm
    import models.lenet as lenet
    import models.vgg as vgg
    return repo, bm, lenet, vgg


def bnn_args(repo, net, binary=True, **override):
    """The args object the repo's model classes expect, from its config files."""
    cfg = {}
    name = ("binary_" if binary else "ref_") + ("mnist" if net == "LeNet5" else "cifar10")
    try:
        import yaml
        with open(repo / "config" / f"{name}.yml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    defaults = dict(dataset="mnist" if net == "LeNet5" else "cifar10", width_infl=1, width_scale=1,
                    tern_acts="0" * (3 if net == "LeNet5" else 8),
                    tern_weights="0" * (4 if net == "LeNet5" else 9),
                    l1_fp=False, ln_fp=False, no_biases=True, no_batchnorm=False,
                    gradient_estimator="bnn", swish_beta=10, layer_order="CPBA",
                    apply_xnor_net=False, first_weights_fullprecision=False,
                    last_weights_fullprecision=False, binarize_input=False,
                    binarize_input_threshold=0.5)
    defaults.update({k: v for k, v in cfg.items() if k in defaults})
    defaults.update(override)
    defaults["tern_acts"], defaults["tern_weights"] = str(defaults["tern_acts"]), str(defaults["tern_weights"])
    defaults["cuda"] = torch.cuda.is_available()
    return SimpleNamespace(**defaults)


# =========================================================================== networks and plan

class Spec:
    """One network: how to build it, its input, and its labels."""

    def __init__(self, mtype, family, model_id, build, input_kind, desc, layers="", width="",
                 weights="random", fixed_size=None, input_dim=None):
        self.mtype, self.family, self.id, self.build = mtype, family, model_id, build
        self.input_kind = input_kind            # "vector" (MLP/BNN-MLP) or "image"
        self.desc, self.layers, self.width, self.weights = desc, layers, width, weights
        self.fixed_size = fixed_size            # (C, H, W) for networks that only accept one size
        self.input_dim = input_dim


def workloads_for(spec, args):
    """(input_shape, batch) pairs. Vector nets: feature vector; fixed-size CNNs: their size only."""
    if spec.input_kind == "vector":
        shapes = [(spec.input_dim,)]
    elif spec.fixed_size:
        shapes = [spec.fixed_size]
    else:
        shapes = [(3, s, s) for s in args.image_size]   # only image networks reach here
    return [(sh, b) for sh in shapes for b in args.batch]


def shape_tag(shape):
    return "x".join(str(s) for s in shape)


def trace_id(spec, wl, rep):
    return f"{spec.id}_in{shape_tag(wl[0])}_b{wl[1]}_r{rep}"


def build_plan(specs, args):
    plan = []
    for s in specs:
        wls = workloads_for(s, args)
        if args.pilot:
            light = min(wls, key=lambda w: (w[1], w[0]))
            heavy = max(wls, key=lambda w: (w[1], w[0]))
            wls = sorted({light, heavy})
        plan.append((s, wls))
    if args.pilot:                      # per family: first (smallest) and last (largest) + trained ones
        keep, by_fam = [], {}
        for p in plan:
            by_fam.setdefault((p[0].family, p[0].weights), []).append(p)
        for (fam, weights), ps in by_fam.items():
            keep += ps if weights == "trained" else ([ps[0], ps[-1]] if len(ps) > 1 else ps)
        plan = keep
    return plan


# =========================================================================== running

def make_input(spec, wl, device, dtype):
    shape, batch = wl
    g = torch.Generator().manual_seed(1)
    return torch.randn((batch,) + tuple(shape), generator=g).to(device=device, dtype=dtype)


def load(spec, device, dtype):
    torch.manual_seed(0)
    model = spec.build().to(device=device, dtype=dtype).eval()
    params = round(sum(p.numel() for p in model.parameters()) / 1e6, 4)
    return model, params


def measure_window(plan, args, device):
    """Time the heaviest workload of the largest network of each family."""
    by_fam = {}
    for p in sorted(plan, key=lambda p: p[0].rank):
        if p[0].weights == "random" or p[0].family not in by_fam:
            by_fam[p[0].family] = p             # ends on the largest random-weight network
    slowest = 0.0
    for spec, wls in by_fam.values():
        wl = max(wls, key=lambda w: (w[1], w[0]))
        print(f"timing {spec.id} with input {shape_tag(wl[0])}, batch {wl[1]} ...", flush=True)
        model = None
        try:
            model, _ = load(spec, device, args.torch_dtype)
            x = make_input(spec, wl, device, args.torch_dtype)
            with torch.inference_mode():
                model(x)
                S.sync(device)
                t = time.time()
                model(x)
                S.sync(device)
            run = time.time() - t
            print(f"    {run * 1000:.1f} ms")
            slowest = max(slowest, run)
        except torch.cuda.OutOfMemoryError:
            print("    out of GPU memory (will be skipped in the sweep)")
        finally:
            del model
            S.free(device)
    return float(max(1.0, -(-(slowest * 1.5 + 0.5) * 2 // 1) / 2))


def append(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def common_args(description, default_batch, default_out):
    """Options shared by sweep_mlp.py, sweep_bnn.py and sweep_cnn.py. Each script adds its own."""
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    w = p.add_argument_group("workload and recording")
    w.add_argument("--batch", nargs="+", type=int, default=default_batch, help="batch sizes")
    w.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    w.add_argument("--match", default=None,
                   help="sweep output folder (e.g. released_out) whose window/idle/grid to copy, "
                        "so traces can share one CSV")
    w.add_argument("--window", default="auto", help="seconds after the pass starts ('auto' = measure)")
    w.add_argument("--pre", type=float, default=0.5, help="idle seconds recorded before the pass")
    w.add_argument("--grid-ms", type=float, default=20.0, help="spacing of the resampled trace")
    w.add_argument("--repeats", type=int, default=1, help="traces per (network, workload)")
    w.add_argument("--cooldown", type=float, default=2.0, help="idle seconds after each warm-up")
    w.add_argument("--ymax", type=float, default=None, help="plot y-axis top")
    o = p.add_argument_group("running")
    o.add_argument("--out", default=default_out, help="output folder")
    o.add_argument("--device", default="cuda")
    o.add_argument("--seed", type=int, default=0, help="seed for the shuffled run order")
    o.add_argument("--no-shuffle", action="store_true")
    o.add_argument("--no-plots", action="store_true", help="skip plots (redraw later with --replot)")
    o.add_argument("--pilot", action="store_true",
                   help="smallest + largest network of each family (plus trained ones), lightest + "
                        "heaviest workload; writes to <out>_pilot")
    o.add_argument("--limit", type=int, default=None, help="stop after this many new traces")
    o.add_argument("--dry-run", action="store_true", help="list the runs and estimated time, then exit")
    o.add_argument("--replot", action="store_true", help="redraw all plots from saved traces, then exit")
    return p


def finish_args(args, mtype):
    args.type = mtype
    args.batch = sorted(args.batch)
    if args.pilot:
        args.out += "_pilot"
    args.torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    return args


# =========================================================================== main loop

def run(args, make_specs):
    """The whole sweep. make_specs(args) returns the list of networks to run."""
    out = Path(args.out)
    manifest = out / "manifest.csv"
    previous = S.load_manifest(manifest)

    # Window: from this folder (resume), else from --match, else --window / auto.
    ref = previous or (S.load_manifest(Path(args.match) / "manifest.csv") if args.match else [])
    if args.match and not previous and not ref:
        sys.exit(f"--match {args.match}: no manifest.csv there")
    if ref:
        args.window, args.pre, args.grid_ms = (float(ref[0]["window_s"]), float(ref[0]["pre_s"]),
                                               float(ref[0]["grid_ms"]))
        src = out if previous else Path(args.match)
        print(f"using window {args.window:g} s, idle {args.pre:g} s, grid {args.grid_ms:g} ms from {src}")
    elif args.window == "auto":
        args.window = None
    else:
        args.window = float(args.window)

    if args.replot:
        rows = [r for r in previous if r["status"] == "ok"]
        limits = [float(r["power_limit_w"]) for r in rows if r.get("power_limit_w")]
        ymax = args.ymax or (round(limits[0] * 1.15) if limits else 450)
        groups = {}
        for r in rows:
            groups.setdefault(r["plot_id"], []).append(r)
        (out / "plots").mkdir(parents=True, exist_ok=True)
        for rs in groups.values():
            S.plot_group(rs, [S.read_trace(out, r) for r in rs], out / rs[0]["plot_file"], args, ymax)
        print(f"redrew {len(groups)} plots in {out / 'plots'}")
        return

    specs = make_specs(args)
    for i, s in enumerate(specs):
        s.rank = i                      # lists are written smallest -> largest within a family
    plan = build_plan(specs, args)
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(plan)
        for _, wls in plan:
            rng.shuffle(wls)
    done = {r["trace_id"] for r in previous if r["status"] in ("ok", "oom")}
    todo, n = [], 0
    for spec, wls in plan:
        left = [(wl, rep) for wl in wls for rep in range(1, args.repeats + 1)
                if trace_id(spec, wl, rep) not in done]
        if args.limit is not None:
            left = left[:max(0, args.limit - n)]
        n += len(left)
        if left:
            todo.append((spec, left))
    total = sum(len(w) for _, w in plan) * args.repeats
    print(f"{args.type}: {len(plan)} networks, {total} traces in total; {len(done)} already done; {n} to run.")
    est = n * (args.pre + (args.window or 3.0) + 0.5 + args.cooldown + 0.5) + len(todo) * 5
    print(f"estimated time: ~{est / 3600:.1f} h")
    if args.dry_run:
        for spec, left in todo:
            print(f"  {spec.id:34s} {spec.weights:8s} {len(left):4d} traces")
        return
    if not todo:
        print("nothing to do")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA is not available (pass --device cpu only for testing).")
    for sub in ("traces", "raw", "plots"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    sampler = S.PowerSampler(device.index or 0)
    sampler.start()
    static = sampler.static_info()
    ymax = args.ymax or round((static.get("power_limit_w") or 0) * 1.15) or 450
    print(f"GPU: {static.get('gpu')}  driver {static.get('driver')}  power limit {static.get('power_limit_w')} W")
    if args.window is None:
        args.window = measure_window(plan, args, device)
    n_points = int(round((args.pre + args.window) * 1000 / args.grid_ms))
    print(f"window: {args.window:g} s after the pass starts -> {n_points} points per trace", flush=True)

    t_begin, n_done = time.time(), 0
    try:
        for mi, (spec, left) in enumerate(todo, 1):
            print(f"[network {mi}/{len(todo)}] {spec.id}: {len(left)} traces", flush=True)
            model = None
            base = dict(model_type=spec.mtype, source="sweep" if spec.weights == "random" else "pretrained",
                        model=spec.id, family=spec.family, kind="forward", layers=spec.layers,
                        hidden=spec.width, heads="", dtype=args.dtype, tokens=0, kv_cache="",
                        weights=spec.weights, arch_desc=spec.desc, pre_s=args.pre, window_s=args.window,
                        grid_ms=args.grid_ms, n_points=n_points, gpu=static.get("gpu", ""),
                        driver=static.get("driver", ""), power_limit_w=static.get("power_limit_w", ""))
            try:
                model, params = load(spec, device, args.torch_dtype)
                base["params_m"] = params
            except Exception as e:
                status = "oom" if isinstance(e, torch.cuda.OutOfMemoryError) else "error"
                print(f"    {status} while building: {type(e).__name__}: {str(e)[:150]}")
                for wl, rep in left:
                    append(manifest, dict(base, trace_id=trace_id(spec, wl, rep), status=status,
                                          error=f"{type(e).__name__}: {e}"[:200]))
                S.free(device)
                continue

            last_wl = None
            for wl, rep in left:
                shape, batch = wl
                plot_id = f"{spec.id}_in{shape_tag(shape)}_b{batch}"
                row = dict(base, trace_id=trace_id(spec, wl, rep), plot_id=plot_id, repeat=rep,
                           input_len=shape[-1] if len(shape) > 1 else shape[0], batch=batch,
                           input_shape=shape_tag(shape), plot_file=f"plots/{plot_id}.png")
                try:
                    x = make_input(spec, wl, device, args.torch_dtype)

                    def infer():
                        with torch.inference_mode():
                            model(x)

                    if wl != last_wl:
                        infer()
                        S.sync(device)
                        time.sleep(args.cooldown)
                        last_wl = wl
                    row.update(sampler.info(), started_at=datetime.now().isoformat(timespec="seconds"))
                    time.sleep(args.pre)
                    S.sync(device)
                    t0 = time.time()
                    infer()
                    S.sync(device)
                    run_s = time.time() - t0
                    time.sleep(max(0.0, t0 + args.window - time.time()) + 0.3)

                    start, end = t0 - args.pre, t0 + args.window
                    raw_t, raw_w = sampler.window(start, end)
                    grid = S.to_grid(raw_t, raw_w, start, n_points, args.grid_ms / 1000)
                    in_win = [(t, w) for t, w in zip(raw_t, raw_w) if start <= t <= end]
                    gaps = [b - a for a, b in zip(raw_t, raw_t[1:])]
                    idle = sorted(w for t, w in in_win if t < t0)
                    run = [w for t, w in in_win if t0 <= t <= t0 + run_s]

                    row["trace_file"] = f"traces/{row['trace_id']}.csv"
                    with open(out / row["trace_file"], "w", newline="") as f:
                        wr = csv.writer(f)
                        wr.writerow(["t_ms", "power_w"])
                        for k, v in enumerate(grid):
                            wr.writerow([f"{(-args.pre + k * args.grid_ms / 1000) * 1000:.1f}", f"{v:.3f}"])
                    row["raw_file"] = f"raw/{row['trace_id']}.csv"
                    with open(out / row["raw_file"], "w", newline="") as f:
                        wr = csv.writer(f)
                        wr.writerow(["t_ms", "power_w", "driver_time_us"])
                        for t, v in in_win:
                            wr.writerow([f"{(t - t0) * 1000:.1f}", f"{v:.3f}", int(round(t * 1e6))])

                    row.update(status="ok", run_ms=round(run_s * 1000, 2), truncated=run_s > args.window,
                               raw_samples=len(in_win), max_gap_ms=round(max(gaps) * 1000, 1) if gaps else "",
                               idle_w=round(idle[len(idle) // 2], 2) if idle else "",
                               mean_run_w=round(sum(run) / len(run), 2) if run else "")
                    append(manifest, row)
                    n_done += 1
                    print(f"    in {shape_tag(shape):>9s} b {batch:5d}: {row['run_ms']:9.2f} ms, "
                          f"{len(run)} samples during pass, mean {row['mean_run_w'] or '-'} W", flush=True)
                    if not args.no_plots:
                        rs = [r for r in S.load_manifest(manifest) if r["plot_id"] == plot_id and r["status"] == "ok"]
                        S.plot_group(rs, [S.read_trace(out, r) for r in rs], out / row["plot_file"], args, ymax)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"    in {shape_tag(shape)} b {batch}: out of GPU memory, skipped")
                    append(manifest, dict(row, status="oom", error=str(e)[:200]))
                    last_wl = None
                    S.free(device)

            del model
            S.free(device)
            el = time.time() - t_begin
            print(f"    {n_done}/{n} traces, elapsed {el / 60:.1f} min, about "
                  f"{el / max(n_done, 1) * (n - n_done) / 60:.0f} min left", flush=True)
    except KeyboardInterrupt:
        print("\nstopped by user; run the same command again to continue where it left off")
    finally:
        sampler.stop()

    rows = S.load_manifest(manifest)
    trunc = [r["trace_id"] for r in rows if r.get("truncated") == "True"]
    bad = [r["trace_id"] for r in rows if r["status"] != "ok"]
    print(f"\ndone: {n_done} new traces this session; manifest: {manifest}")
    if trunc:
        print(f"WARNING: {len(trunc)} passes were longer than the window, e.g. {trunc[:3]}")
    if bad:
        print(f"{len(bad)} traces failed (out of memory or error), e.g. {bad[:3]}; see the manifest")