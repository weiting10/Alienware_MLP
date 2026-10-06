# Example command line, sample interval and duration are in units of seconds
# python MLPlot.py --duration 60 --sample-interval 0.1 --output trace.csv

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pynvml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record GPU power readings to a CSV file and plot them."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30,
        help="Recording duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.02,
        help="Time between readings in seconds (default: 0.02).",
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
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be greater than 0")
    return args


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
    ax.plot(time_s, power_w)
    ax.set_title("GPU Power Consumption Over Time")
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
    try:
        gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "power_w"])

            start = time.perf_counter()
            while time.perf_counter() - start < args.duration:
                elapsed = time.perf_counter() - start
                power = pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000
                writer.writerow([elapsed, power])
                time.sleep(args.sample_interval)
    finally:
        pynvml.nvmlShutdown()

    print(f"GPU power trace saved to {args.output}.")
    plot_path = args.plot_output or str(Path(args.output).with_suffix(".png"))
    plot_power_trace(args.output, plot_path, show=not args.no_show)


if __name__ == "__main__":
    main()
