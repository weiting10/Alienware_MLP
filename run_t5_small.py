#!/usr/bin/env python3
"""
Run a pretrained T5-small model in a repeatable loop so a separate power
logger (e.g. nvidia-smi in another terminal) can record the GPU power trace.

Every important moment (idle start, each run's start/end, end of experiment)
is printed and saved to a CSV event log with timestamps in the same format
nvidia-smi uses, so the two logs can be lined up afterwards.

Example:
    python run_t5_small.py                       # defaults: 50 runs on GPU
    python run_t5_small.py --runs 200 --batch-size 8 --new-tokens 64
    python run_t5_small.py --mode encoder        # encoder-only passes
"""

import argparse
import csv
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL_NAME = "google-t5/t5-small"

# T5 was trained with task prefixes like "translate English to German:".
PROMPT = (
    "translate English to German: The quick brown fox jumps over the lazy dog "
    "while the researchers measure the power consumption of the graphics card."
)


def now_stamp():
    """Wall-clock time in nvidia-smi's timestamp format, plus Unix epoch."""
    t = time.time()
    return t, datetime.fromtimestamp(t).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


class EventLog:
    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["epoch_s", "timestamp", "event", "run", "detail"])

    def mark(self, event, run="", detail=""):
        epoch, stamp = now_stamp()
        self.writer.writerow([f"{epoch:.6f}", stamp, event, run, detail])
        self.file.flush()
        print(f"[{stamp}] {event} {run} {detail}".rstrip(), flush=True)

    def close(self):
        self.file.close()


def sync(device):
    # GPU work is asynchronous; wait for it so timestamps reflect real work.
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description="Run T5-small for power-trace collection.")
    p.add_argument("--runs", type=int, default=50, help="number of measured runs")
    p.add_argument("--batch-size", type=int, default=1, help="copies of the prompt per run")
    p.add_argument("--new-tokens", type=int, default=32,
                   help="tokens generated per run (forced to exactly this many)")
    p.add_argument("--mode", choices=["generate", "encoder"], default="generate",
                   help="generate = full encoder + decoder loop; encoder = encoder pass only")
    p.add_argument("--warmup", type=int, default=5, help="unmeasured warmup runs")
    p.add_argument("--idle", type=float, default=5.0,
                   help="seconds of idle before and after the runs (baseline power)")
    p.add_argument("--gap", type=float, default=0.5,
                   help="idle seconds between runs so each run is visible in the trace")
    p.add_argument("--device", default="cuda", help="cuda, cuda:1, or cpu")
    p.add_argument("--log", default=None, help="event log CSV path (default: timestamped name)")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Check the GPU/driver, or pass --device cpu.")

    log_path = args.log or f"t5_events_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log = EventLog(log_path)

    # ---- Load model (downloads once, then uses the local Hugging Face cache) ----
    log.mark("load_start", detail=MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()
    log.mark("load_end", detail=f"device={device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    inputs = tokenizer([PROMPT] * args.batch_size, return_tensors="pt").to(device)
    print(f"Input shape (batch, tokens): {tuple(inputs['input_ids'].shape)}")

    # min = max new tokens makes every run do the same amount of work.
    gen_kwargs = dict(
        max_new_tokens=args.new_tokens,
        min_new_tokens=args.new_tokens,
        do_sample=False,
        num_beams=1,
    )

    def one_run():
        with torch.inference_mode():
            if args.mode == "generate":
                return model.generate(**inputs, **gen_kwargs)
            return model.encoder(**inputs).last_hidden_state

    # ---- Warmup: first runs include CUDA init and kernel setup, so skip them ----
    log.mark("warmup_start", detail=f"{args.warmup} runs")
    for _ in range(args.warmup):
        one_run()
    sync(device)
    log.mark("warmup_end")

    # Show one output so you can see the model really works.
    if args.mode == "generate":
        out = one_run()
        sync(device)
        print("Sample output:", tokenizer.decode(out[0], skip_special_tokens=True))

    # ---- Idle baseline ----
    log.mark("idle_start")
    time.sleep(args.idle)
    log.mark("idle_end")

    # ---- Measured runs ----
    log.mark("experiment_start",
             detail=f"mode={args.mode} runs={args.runs} batch={args.batch_size} "
                    f"new_tokens={args.new_tokens}")
    durations = []
    for i in range(1, args.runs + 1):
        sync(device)
        log.mark("run_start", run=i)
        t0 = time.perf_counter()
        one_run()
        sync(device)
        dt = time.perf_counter() - t0
        durations.append(dt)
        log.mark("run_end", run=i, detail=f"{dt * 1000:.2f} ms")
        if args.gap > 0:
            time.sleep(args.gap)
    log.mark("experiment_end")

    # ---- Idle tail ----
    log.mark("idle_start")
    time.sleep(args.idle)
    log.mark("idle_end")
    log.close()

    durations.sort()
    print(f"\nRun time: median {durations[len(durations) // 2] * 1000:.2f} ms, "
          f"min {durations[0] * 1000:.2f} ms, max {durations[-1] * 1000:.2f} ms")
    print(f"Event log saved to {log_path}")


if __name__ == "__main__":
    main()
