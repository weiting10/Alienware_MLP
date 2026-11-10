#!/usr/bin/env python3
"""
Records GPU power traces of transformers for a classifier dataset.

Two modes, same output format (equal-length traces, manifest.csv, titled plots):

  --mode released   (default) released models loaded by name (T5, GPT-2, Pythia, OPT, BERT)
                    with their trained weights, over a grid of workloads:
                    generated tokens x input length x batch size.
  --mode arch       random-weight T5 / GPT-2 models for every combination of
                    layers x hidden size x attention heads (no download, no training).

For every (model, workload):
  1. load/build the model once, run one warm-up inference (not recorded), cool down
  2. record a fixed time window: PRE seconds of idle, then the inference, then idle
     until WINDOW seconds after the inference started
  3. resample the driver's power samples onto a fixed time grid, so every trace has
     exactly the same number of points
  4. save the trace CSV, a manifest row (model_type = transformer) and a titled plot

Power comes from the driver's sample buffer (nvmlDeviceGetSamples with
NVML_TOTAL_POWER_SAMPLES), the same source as MLPlot_buffer.py.

Quick start:
    python sweep.py --dry-run                 # list the runs and estimate the time
    python sweep.py --pilot                   # downloads every model, runs smallest + largest workload
    python sweep.py                           # full released-model sweep
    python sweep.py --mode arch               # random-architecture sweep
Run "python sweep.py --help" for all options.
"""

import argparse
import bisect
import csv
import itertools
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

DEFAULT_MODELS = [
    "t5-small", "t5-base", "t5-large",
    "gpt2", "gpt2-medium", "gpt2-large",
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "bert-base-uncased", "bert-large-uncased",
]
ENCODER_ONLY = {"bert", "roberta", "distilbert", "albert", "electra", "deberta", "deberta-v2"}

MANIFEST_FIELDS = [
    "trace_id", "plot_id", "model_type", "source", "model", "family", "kind",
    "layers", "hidden", "heads", "params_m", "dtype",
    "repeat", "tokens", "input_len", "batch", "kv_cache",
    "status", "run_ms", "truncated", "pre_s", "window_s", "grid_ms", "n_points",
    "raw_samples", "max_gap_ms", "idle_w", "mean_run_w", "gpu_temp_c", "sm_clock_mhz",
    "gpu", "driver", "power_limit_w", "started_at", "trace_file", "raw_file", "plot_file", "error",
]
KIND_NAMES = {"seq2seq": "encoder-decoder", "causal": "decoder-only", "encoder": "encoder-only"}

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, SHADE = "#0b0b0b", "#52514e", "#e8e7e2"


# =========================================================================== power

class PowerSampler(threading.Thread):
    """Continuously drains the driver's power sample buffer into memory."""

    def __init__(self, gpu_index=0, poll_s=0.05, keep_s=180):
        super().__init__(daemon=True)
        try:
            import pynvml
        except ImportError:
            raise SystemExit("Power capture needs NVML. Install it with: pip install nvidia-ml-py")
        self.N = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.poll_s, self.keep_s = poll_s, keep_s
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.times, self.watts = [], []      # seconds since 1970, watts (sorted by time)
        self.last = 0
        self._read()                         # skip what was already in the buffer
        with self.lock:
            self.times, self.watts = [], []

    def _read(self):
        N = self.N
        try:
            _, batch = N.nvmlDeviceGetSamples(self.h, N.NVML_TOTAL_POWER_SAMPLES, self.last)
        except N.NVMLError_NotFound:
            return
        new = sorted((s.timeStamp, s.sampleValue.uiVal) for s in batch if s.timeStamp > self.last)
        if not new:
            return
        self.last = new[-1][0]
        with self.lock:
            for ts, mw in new:
                self.times.append(ts / 1e6)
                self.watts.append(mw / 1000)
            drop = bisect.bisect_left(self.times, time.time() - self.keep_s)
            if drop:
                del self.times[:drop], self.watts[:drop]

    def run(self):
        while not self.stop_event.is_set():
            self._read()
            self.stop_event.wait(self.poll_s)

    def window(self, start, end):
        """Raw samples with start <= t <= end, plus the last sample before start."""
        self._read()
        with self.lock:
            i = max(bisect.bisect_left(self.times, start) - 1, 0)
            j = bisect.bisect_right(self.times, end)
            return self.times[i:j], self.watts[i:j]

    def info(self):
        N, out = self.N, {}
        for key, fn in (("gpu_temp_c", lambda: N.nvmlDeviceGetTemperature(self.h, N.NVML_TEMPERATURE_GPU)),
                        ("sm_clock_mhz", lambda: N.nvmlDeviceGetClockInfo(self.h, N.NVML_CLOCK_SM))):
            try:
                out[key] = fn()
            except N.NVMLError:
                out[key] = ""
        return out

    def static_info(self):
        N, out = self.N, {}
        for key, fn in (("gpu", lambda: N.nvmlDeviceGetName(self.h)),
                        ("driver", N.nvmlSystemGetDriverVersion),
                        ("power_limit_w", lambda: N.nvmlDeviceGetPowerManagementLimit(self.h) / 1000)):
            try:
                v = fn()
                out[key] = v.decode() if isinstance(v, bytes) else v
            except N.NVMLError:
                out[key] = ""
        return out

    def stop(self):
        self.stop_event.set()
        self.join()
        self.N.nvmlShutdown()


def to_grid(times, watts, start, n_points, step_s):
    """Sample-and-hold resampling: value at each grid time = latest raw sample at or before it."""
    if not times:
        return [float("nan")] * n_points
    out = []
    for k in range(n_points):
        i = bisect.bisect_right(times, start + k * step_s) - 1
        out.append(watts[max(i, 0)])
    return out


# =========================================================================== models

class Spec:
    """One model to run: a released model name, or a random architecture."""

    def __init__(self, source, name, family=None, layers=None, hidden=None, heads=None):
        self.source, self.name = source, name
        self.family, self.layers, self.hidden, self.heads = family, layers, hidden, heads

    @property
    def id(self):
        if self.source == "sweep":
            return f"{self.family}_L{self.layers}_H{self.hidden}_A{self.heads}"
        return self.name.split("/")[-1]

    def size_key(self):                      # rough compute size, for picking the largest
        return (self.layers or 0) * (self.hidden or 0) ** 2


def _cfg_get(cfg, *names):
    d = cfg.to_dict()
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return ""


def load_model(spec, device, dtype):
    """Returns (model, kind, vocab, info dict). kind is seq2seq, causal or encoder."""
    if spec.source == "sweep":
        from transformers import GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration
        L, H, A = spec.layers, spec.hidden, spec.heads
        if spec.family == "t5":
            cfg = T5Config(d_model=H, d_ff=4 * H, num_layers=L, num_decoder_layers=L, num_heads=A,
                           d_kv=H // A, feed_forward_proj="relu", vocab_size=32128,
                           decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
            cls, kind = T5ForConditionalGeneration, "seq2seq"
        else:
            cfg = GPT2Config(n_layer=L, n_embd=H, n_head=A, n_inner=4 * H, n_positions=1024,
                             vocab_size=50257, bos_token_id=50256, eos_token_id=50256, pad_token_id=50256)
            cls, kind = GPT2LMHeadModel, "causal"
        torch.manual_seed(0)
        if device.type == "cuda":
            with torch.device(device):          # create weights directly on the GPU (much faster)
                model = cls(cfg)
        else:
            model = cls(cfg).to(device)
        model = model.to(dtype)
        family = spec.family
    else:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
        cfg = AutoConfig.from_pretrained(spec.name)
        if cfg.is_encoder_decoder:
            cls, kind = AutoModelForSeq2SeqLM, "seq2seq"
        elif cfg.model_type in ENCODER_ONLY:
            cls, kind = AutoModel, "encoder"
        else:
            cls, kind = AutoModelForCausalLM, "causal"
        try:
            model = cls.from_pretrained(spec.name, torch_dtype=dtype)
        except TypeError:
            model = cls.from_pretrained(spec.name, dtype=dtype)
        model = model.to(device)
        family = cfg.model_type
    model.eval()
    info = dict(family=family, kind=kind,
                layers=_cfg_get(cfg, "num_layers", "n_layer", "num_hidden_layers"),
                hidden=_cfg_get(cfg, "d_model", "n_embd", "hidden_size"),
                heads=_cfg_get(cfg, "num_heads", "n_head", "num_attention_heads"),
                params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
    pad = _cfg_get(cfg, "pad_token_id")
    if pad == "":
        pad = _cfg_get(cfg, "eos_token_id")
    info["pad"] = pad if pad != "" else 0
    return model, kind, cfg.vocab_size, info


def make_infer(model, kind, vocab, info, wl, args, device):
    """Returns a function running one inference for workload wl = (tokens, input_len, batch)."""
    tokens, input_len, batch = wl
    g = torch.Generator().manual_seed(1)    # same input token IDs every time
    ids = torch.randint(100, vocab - 100, (batch, input_len), generator=g).to(device)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if kind == "encoder":
        def infer():
            with torch.inference_mode():
                model(**inputs)
    else:
        gen = dict(max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False, num_beams=1,
                   use_cache=not args.no_cache, pad_token_id=info["pad"])

        def infer():
            with torch.inference_mode():
                model.generate(**inputs, **gen)
    return infer


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def free(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


# =========================================================================== plotting

def describe(r):
    """Two-line plot title from a manifest row."""
    if r.get("kind") == "forward":            # MLP / BNN / CNN rows written by sweep_nn.py
        line1 = (f"{r['model']}  ({r['model_type'].upper()}, {r['family']})  |  {r.get('arch_desc') or ''}  |  "
                 f"{float(r['params_m']):.2f}M params  |  {r.get('weights') or 'random'} weights  |  {r['dtype']}")
        work = f"single forward pass, input {r.get('input_shape') or r['input_len']}, batch {r['batch']}"
        return line1, work
    name = r["model"] if r["source"] == "released" else f"{r['family'].upper()} random architecture"
    line1 = (f"{name}  ({r['family']}, {KIND_NAMES.get(r['kind'], r['kind'])})  |  {r['layers']} layers  |  "
             f"hidden {r['hidden']}  |  {r['heads']} heads  |  {float(r['params_m']):.1f}M params  |  {r['dtype']}")
    if r["kind"] == "encoder":
        work = f"single forward pass, input {r['input_len']} tokens, batch {r['batch']}"
    else:
        work = (f"{r['tokens']} generated tokens, input {r['input_len']} tokens, batch {r['batch']}, "
                f"KV cache {'on' if r['kv_cache'] == 'True' else 'off'}")
    return line1, work


def plot_group(rows, traces, path, args, ymax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r0 = rows[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    run_ms = [float(r["run_ms"]) for r in rows]
    ax.axvspan(0, max(run_ms), color=SHADE, lw=0, zorder=0, label="inference")
    for k, (row, (t_ms, w, raw_t, raw_w)) in enumerate(zip(rows, traces)):
        c = COLORS[k % len(COLORS)]
        lab = f"repeat {row['repeat']}" if len(rows) > 1 else "power (resampled)"
        ax.step(t_ms, w, where="post", color=c, lw=1.6, zorder=3, label=lab)
        ax.plot(raw_t, raw_w, "o", color=c, ms=3.5, zorder=4, label="driver samples" if k == 0 else None)
    if r0["idle_w"] != "":
        ax.axhline(float(r0["idle_w"]), color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2,
                   label=f"idle before run ({float(r0['idle_w']):.0f} W)")
    line1, work = describe(r0)
    sub = (f"{work}  |  run {' / '.join(f'{x:.0f}' for x in run_ms)} ms  |  "
           f"GPU {r0['gpu_temp_c']} °C, {r0['sm_clock_mhz']} MHz  |  {r0['n_points']} points")
    if any(r["truncated"] == "True" for r in rows):
        sub += "  |  WARNING: run longer than window"
    ax.set_title(f"{line1}\n{sub}", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Time since inference start (ms)", color=INK)
    ax.set_ylabel("Power (W)", color=INK)
    ax.set_xlim(-float(r0["pre_s"]) * 1000, float(r0["window_s"]) * 1000)
    # Shared scale for comparison, but never cut off a reading: expand if the data goes higher.
    peak = max([v for _, w, _, rw in traces for v in list(w) + list(rw) if v == v] or [0])
    ax.set_ylim(0, max(ymax, peak * 1.05))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=SHADE, lw=0.8)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def read_trace(out_dir, row):
    t, w, rt, rw = [], [], [], []
    with open(out_dir / row["trace_file"], newline="") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_ms"]))
            w.append(float(r["power_w"]))
    with open(out_dir / row["raw_file"], newline="") as f:
        for r in csv.DictReader(f):
            rt.append(float(r["t_ms"]))
            rw.append(float(r["power_w"]))
    return t, w, rt, rw


def plot_id_rows(out, manifest_path, plot_id):
    return [r for r in load_manifest(manifest_path) if r["plot_id"] == plot_id and r["status"] == "ok"]


# =========================================================================== bookkeeping

def load_manifest(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_manifest(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def specs_for(args):
    if args.mode == "released":
        return [Spec("released", m) for m in args.models]
    specs = []
    for family in args.families:
        for L in args.layers:
            for H in args.hidden:
                for A in args.heads:
                    if H % A:
                        print(f"skipping {family} hidden {H} / heads {A}: not divisible")
                        continue
                    specs.append(Spec("sweep", None, family, L, H, A))
    return specs


def workloads_for(spec_kind, args):
    """All (tokens, input_len, batch) combinations; encoder-only models ignore tokens."""
    toks = [0] if spec_kind == "encoder" else args.tokens
    return list(itertools.product(toks, args.input_len, args.batch))


def guess_kind(spec):
    if spec.source == "sweep":
        return "seq2seq" if spec.family == "t5" else "causal"
    n = spec.name.lower()
    if any(e in n for e in ENCODER_ONLY):
        return "encoder"
    return "seq2seq" if "t5" in n or "bart" in n else "causal"


def trace_id(spec, wl, rep):
    tokens, input_len, batch = wl
    return f"{spec.id}_tok{tokens}_in{input_len}_b{batch}_r{rep}"


def build_plan(args):
    specs = specs_for(args)
    plan = []                                 # list of (spec, [workloads])
    for s in specs:
        wls = workloads_for(guess_kind(s), args)
        if args.pilot:
            wls = sorted({min(wls), max(wls)})
        plan.append((s, wls))
    if args.pilot and args.mode == "arch":    # smallest and largest architecture per family
        keep = []
        for fam in args.families:
            fam_specs = [p for p in plan if p[0].family == fam]
            keep.append(min(fam_specs, key=lambda p: (p[0].size_key(), -p[0].heads)))
            keep.append(max(fam_specs, key=lambda p: (p[0].size_key(), p[0].heads)))
        plan = keep
    return plan


def parse_args():
    p = argparse.ArgumentParser(description="Record GPU power traces of transformers.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["released", "arch"], default="released",
                   help="released = released models by name; arch = random architectures")
    m = p.add_argument_group("released mode")
    m.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Hugging Face model names")
    a = p.add_argument_group("arch mode (every combination is run)")
    a.add_argument("--families", nargs="+", choices=["t5", "gpt2"], default=["t5", "gpt2"])
    a.add_argument("--layers", nargs="+", type=int, default=[2, 4, 6, 12, 24])
    a.add_argument("--hidden", nargs="+", type=int, default=[256, 384, 512, 768, 1024])
    a.add_argument("--heads", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    w = p.add_argument_group("workload grid (every combination is run for every model)")
    w.add_argument("--tokens", nargs="+", type=int, default=None,
                   help="generated tokens (default: released 4 16 32 64 128; arch 64)")
    w.add_argument("--input-len", nargs="+", type=int, default=None,
                   help="input length in tokens (default: released 16 32 64 128 256; arch 32)")
    w.add_argument("--batch", nargs="+", type=int, default=None,
                   help="batch size (default: released 1 2 4 8 16; arch 1)")
    w.add_argument("--no-cache", action="store_true", help="turn off the KV cache")
    w.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                   help="weight precision (real deployments often use fp16/bf16)")
    r = p.add_argument_group("recording")
    r.add_argument("--pre", type=float, default=0.5, help="idle seconds recorded before each inference")
    r.add_argument("--window", default="auto",
                   help="seconds recorded after the inference starts. 'auto' times the slowest model + "
                        "workload first and uses 1.5x its run time + 0.5 s, rounded up to 0.5 s. "
                        "Use the same number for every dataset you want to combine.")
    r.add_argument("--grid-ms", type=float, default=20.0,
                   help="spacing of the resampled trace; points per trace = (pre + window) / grid")
    r.add_argument("--repeats", type=int, default=1, help="traces per (model, workload)")
    r.add_argument("--cooldown", type=float, default=2.0, help="idle seconds after each warm-up")
    r.add_argument("--ymax", type=float, default=None,
                   help="y-axis top for every plot (default: 15%% above the GPU's power limit)")
    o = p.add_argument_group("running")
    o.add_argument("--out", default=None, help="output folder (default: released_out or arch_out)")
    o.add_argument("--device", default="cuda", help="cuda, cuda:1, ... (cpu only for testing)")
    o.add_argument("--seed", type=int, default=0, help="seed for the shuffled run order")
    o.add_argument("--no-shuffle", action="store_true", help="run in list order instead of shuffled")
    o.add_argument("--no-plots", action="store_true", help="skip plots (redraw later with --replot)")
    o.add_argument("--pilot", action="store_true",
                   help="every model (arch: smallest + largest) with only the smallest and largest "
                        "workload; writes to <out>_pilot")
    o.add_argument("--limit", type=int, default=None, help="stop after this many new traces")
    o.add_argument("--dry-run", action="store_true", help="list the runs and estimated time, then exit")
    o.add_argument("--replot", action="store_true", help="redraw all plots from saved traces, then exit")
    args = p.parse_args()

    rel = args.mode == "released"
    args.tokens = args.tokens or ([4, 16, 32, 64, 128] if rel else [64])
    args.input_len = args.input_len or ([16, 32, 64, 128, 256] if rel else [32])
    args.batch = args.batch or ([1, 2, 4, 8, 16] if rel else [1])
    args.out = args.out or ("released_out" if rel else "arch_out")
    if args.pilot:
        args.out += "_pilot"
    args.torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    return args


# =========================================================================== main

def measure_window(plan, args, device):
    """Time the heaviest workload of the models most likely to be slowest."""
    if args.mode == "arch":                   # largest architecture of each family
        cands = [max((p for p in plan if p[0].family == f), key=lambda p: p[0].size_key())
                 for f in args.families if any(p[0].family == f for p in plan)]
    else:                                     # every released model (sizes vary a lot)
        cands = plan
    slowest = 0.0
    for spec, wls in cands:
        wl = max(wls, key=lambda w: (w[0] + 1) * w[1] * w[2])
        print(f"timing {spec.id} with tokens {wl[0]}, input {wl[1]}, batch {wl[2]} ...", flush=True)
        model = None
        try:
            model, kind, vocab, info = load_model(spec, device, args.torch_dtype)
            infer = make_infer(model, kind, vocab, info, wl, args, device)
            infer()
            sync(device)
            t = time.time()
            infer()
            sync(device)
            run = time.time() - t
            print(f"    {run * 1000:.0f} ms")
            slowest = max(slowest, run)
        except torch.cuda.OutOfMemoryError:
            print("    out of GPU memory at this workload (it will be skipped in the sweep)")
        finally:
            del model
            free(device)
    return float(max(1.0, -(-(slowest * 1.5 + 0.5) * 2 // 1) / 2))   # round up to 0.5 s


def main():
    args = parse_args()
    out = Path(args.out)
    manifest_path = out / "manifest.csv"
    previous = load_manifest(manifest_path)
    if previous:                              # resuming: keep the same window so traces stay equal length
        old = float(previous[0]["window_s"])
        if args.window != "auto" and float(args.window) != old:
            raise SystemExit(f"{manifest_path} already uses a {old:g} s window, so --window {args.window} "
                             f"can't be applied here (traces in one folder must be the same length).\n"
                             f"Use a new --out folder, or delete {out} to start over.")
        print(f"note: {out} already exists, reusing its {old:g} s window. If you've changed the "
              f"workload (e.g. more tokens), use a new --out folder so the window is measured again.")
        args.window = old
        args.pre = float(previous[0]["pre_s"])
        args.grid_ms = float(previous[0]["grid_ms"])
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
            plot_group(rs, [read_trace(out, r) for r in rs], out / rs[0]["plot_file"], args, ymax)
        print(f"redrew {len(groups)} plots in {out / 'plots'}")
        return

    plan = build_plan(args)
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
    n_total = sum(len(wls) for _, wls in plan) * args.repeats

    win = f"{args.window:g} s" if args.window else "auto"
    print(f"{len(plan)} models, {n_total} traces in total; {len(done)} already done; {n} to run.")
    print(f"window: {args.pre:g} s idle before + {win} after inference start, every {args.grid_ms:g} ms")
    est = len(todo) * (args.cooldown + 15) + n * (args.pre + (args.window or 3.0) + 1.0)
    print(f"estimated time: ~{est / 3600:.1f} h (with a {args.window or 3.0:g} s window; "
          f"model loading time varies)")
    if args.dry_run:
        for spec, left in todo:
            print(f"  {spec.id:28s} {len(left):4d} traces")
        return
    if not todo:
        print("nothing to do")
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available (pass --device cpu only for testing).")
    for sub in ("traces", "raw", "plots"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    sampler = PowerSampler(device.index or 0)
    sampler.start()
    static = sampler.static_info()
    ymax = args.ymax or round((static.get("power_limit_w") or 0) * 1.15) or 450   # headroom above the limit
    print(f"GPU: {static.get('gpu')}  driver {static.get('driver')}  power limit {static.get('power_limit_w')} W")

    if args.window is None:
        args.window = measure_window(plan, args, device)
    n_points = int(round((args.pre + args.window) * 1000 / args.grid_ms))
    print(f"window: {args.window:g} s after inference start -> {n_points} points per trace", flush=True)

    t_begin, n_done = time.time(), 0
    try:
        for mi, (spec, left) in enumerate(todo, 1):
            print(f"[model {mi}/{len(todo)}] {spec.id}: {len(left)} traces", flush=True)
            model = None
            try:
                model, kind, vocab, info = load_model(spec, device, args.torch_dtype)
            except torch.cuda.OutOfMemoryError as e:
                print("    out of GPU memory while loading, skipped")
                for wl, rep in left:
                    append_manifest(manifest_path, dict(trace_id=trace_id(spec, wl, rep), status="oom",
                                                        model=spec.name or spec.id, error=str(e)[:200]))
                free(device)
                continue
            except Exception as e:
                print(f"    ERROR loading: {type(e).__name__}: {e}")
                for wl, rep in left:
                    append_manifest(manifest_path, dict(trace_id=trace_id(spec, wl, rep), status="error",
                                                        model=spec.name or spec.id,
                                                        error=f"{type(e).__name__}: {e}"[:200]))
                continue

            base = dict(model_type="transformer", source=spec.source, model=spec.name or spec.id,
                        family=info["family"], kind=kind, layers=info["layers"], hidden=info["hidden"],
                        heads=info["heads"], params_m=info["params_m"], dtype=args.dtype,
                        kv_cache=(kind != "encoder") and not args.no_cache, pre_s=args.pre,
                        window_s=args.window, grid_ms=args.grid_ms, n_points=n_points,
                        gpu=static.get("gpu", ""), driver=static.get("driver", ""),
                        power_limit_w=static.get("power_limit_w", ""))
            last_wl = None
            for wl, rep in left:
                tokens, input_len, batch = wl
                plot_id = f"{spec.id}_tok{tokens}_in{input_len}_b{batch}"
                row = dict(base, trace_id=trace_id(spec, wl, rep), plot_id=plot_id, repeat=rep,
                           tokens=tokens, input_len=input_len, batch=batch,
                           plot_file=f"plots/{plot_id}.png")
                try:
                    infer = make_infer(model, kind, vocab, info, wl, args, device)
                    if wl != last_wl:                  # warm-up for each new workload
                        infer()
                        sync(device)
                        time.sleep(args.cooldown)
                        last_wl = wl
                    row.update(sampler.info(), started_at=datetime.now().isoformat(timespec="seconds"))
                    time.sleep(args.pre)
                    sync(device)
                    t0 = time.time()
                    infer()
                    sync(device)
                    run_s = time.time() - t0
                    time.sleep(max(0.0, t0 + args.window - time.time()) + 0.3)   # buffer catch-up

                    start, end = t0 - args.pre, t0 + args.window
                    raw_t, raw_w = sampler.window(start, end)
                    grid = to_grid(raw_t, raw_w, start, n_points, args.grid_ms / 1000)
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

                    row.update(status="ok", run_ms=round(run_s * 1000, 1), truncated=run_s > args.window,
                               raw_samples=len(in_win),
                               max_gap_ms=round(max(gaps) * 1000, 1) if gaps else "",
                               idle_w=round(idle[len(idle) // 2], 2) if idle else "",
                               mean_run_w=round(sum(run) / len(run), 2) if run else "")
                    append_manifest(manifest_path, row)
                    n_done += 1
                    flag = "  WARNING: longer than --window" if row["truncated"] else ""
                    print(f"    tok {tokens:4d} in {input_len:4d} b {batch:3d} r{rep}: "
                          f"{row['run_ms']:8.0f} ms, mean {row['mean_run_w'] or '-'} W{flag}", flush=True)
                    if not args.no_plots:
                        rs = plot_id_rows(out, manifest_path, plot_id)
                        plot_group(rs, [read_trace(out, r) for r in rs], out / row["plot_file"], args, ymax)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"    tok {tokens} in {input_len} b {batch}: out of GPU memory, skipped")
                    append_manifest(manifest_path, dict(row, status="oom", error=str(e)[:200]))
                    last_wl = None
                    free(device)

            del model
            free(device)
            elapsed = time.time() - t_begin
            print(f"    {n_done}/{n} traces, elapsed {elapsed / 60:.1f} min, "
                  f"about {elapsed / max(n_done, 1) * (n - n_done) / 60:.0f} min left", flush=True)
    except KeyboardInterrupt:
        print("\nstopped by user; run the same command again to continue where it left off")
    finally:
        sampler.stop()

    rows = load_manifest(manifest_path)
    trunc = [r["trace_id"] for r in rows if r.get("truncated") == "True"]
    bad = [r["trace_id"] for r in rows if r["status"] != "ok"]
    print(f"\ndone: {n_done} new traces this session; manifest: {manifest_path}")
    if trunc:
        print(f"WARNING: {len(trunc)} runs were longer than the window ({args.window} s), e.g. {trunc[:3]}. "
              f"Rerun into a new --out folder with a larger --window.")
    if bad:
        print(f"{len(bad)} traces failed (out of memory or error), e.g. {bad[:3]}; see the manifest")


if __name__ == "__main__":
    main()