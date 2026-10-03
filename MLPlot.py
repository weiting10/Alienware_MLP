# Example command line, sample interval and duration are in units of seconds
# python MLPlot.py --duration 60 --sample-interval 0.1 --output trace.csv

import argparse
import pynvml
import time
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record GPU power readings to a CSV file."
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
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than 0")
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be greater than 0")
    return args


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


if __name__ == "__main__":
    main()
