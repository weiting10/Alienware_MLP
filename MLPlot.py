# Example command line, sample interval and duration are in units of seconds
# python ~/MLPlot_buffer.py --duration 50 --poll-interval 0.1 --output ~/mlp_gpu_power.csv --plot-output ~/mlp_gpu_power.png --no-show

# Records GPU power from the driver's sample buffer (nvmlDeviceGetSamples),
# which gives a real sample every ~20 ms with the driver's own timestamps.
# Example command line, poll interval and duration are in units of seconds:
# python MLPlot_buffer.py --duration 60 --poll-interval 0.1 --output trace.csv

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pynvml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record GPU power from the NVML sample buffer to a CSV file and plot it."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30,
        help="Recording duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="How often to read new samples from the buffer, in seconds (default: 0.1). "
             "This does not change the sampling rate; the driver decides that.",
    )
    parser.add_argument(
        "--output",
        default="gpu_power.csv",
        help="Output CSV filename (default: gpu_power.csv).",
    )
    parser.add_argument(
        "--plot-output",
        help="Output plot filename (default: CSV filename with a .png extension).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive window.",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than 0")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be greater than 0")
    return args


def read_new_samples(gpu, last_ts):
    """Return [(driver_time_us, watts), ...] for samples newer than last_ts."""
    try:
        _, samples = pynvml.nvmlDeviceGetSamples(gpu, pynvml.NVML_TOTAL_POWER_SAMPLES, last_ts)
    except pynvml.NVMLError_NotFound:
        return []  # no new samples since last_ts
    return [(s.timeStamp, s.sampleValue.uiVal / 1000) for s in samples if s.timeStamp > last_ts]


def plot_power_trace(csv_path, plot_path, show=True):
    time_s = []
    power_w = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_s.append(float(row["time_s"]))
            power_w.append(float(row["power_w"]))

    if not time_s:
        raise ValueError(f"No GPU power samples found in {csv_path}.")

    fig, ax = plt.subplots()
    ax.plot(time_s, power_w, marker=".", markersize=2)
    ax.set_title("GPU Power Consumption Over Time (NVML sample buffer)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    print(f"GPU power plot saved to {plot_path}.")
    if show:
        plt.show()


def main():
    args = parse_args()
    pynvml.nvmlInit()
    count = 0
    try:
        gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Skip samples that were already in the buffer before we started.
        old = read_new_samples(gpu, 0)
        last_ts = old[-1][0] if old else 0
        first_ts = None

        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "power_w", "driver_time_us"])

            start = time.perf_counter()
            while time.perf_counter() - start < args.duration:
                for ts, power in read_new_samples(gpu, last_ts):
                    if first_ts is None:
                        first_ts = ts
                    writer.writerow([(ts - first_ts) / 1e6, power, ts])
                    last_ts = ts
                    count += 1
                time.sleep(args.poll_interval)
    finally:
        pynvml.nvmlShutdown()

    if count == 0:
        print("No power samples came from the buffer. This GPU/driver may not support "
              "NVML_TOTAL_POWER_SAMPLES (e.g. some laptop GPUs).")
        return

    print(f"GPU power trace saved to {args.output} ({count} samples).")
    plot_path = args.plot_output or str(Path(args.output).with_suffix(".png"))
    plot_power_trace(args.output, plot_path, show=not args.no_show)


if __name__ == "__main__":
    main()
