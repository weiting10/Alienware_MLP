"""
High-rate capture for ElmorLabs PMD2/PMD2X.

Polls sensor frames as fast as the USB link allows (raw bytes only, no parsing
in the hot loop), then decodes, prints timing stats and writes a wide CSV.

Usage:
    python pmd_capture.py                    # run until Ctrl+C
    python pmd_capture.py --samples 5000     # fixed-length capture
    python pmd_capture.py --depth 2          # experimental: 2 requests in flight
    python pmd_capture.py --port COM5 --out run1.csv

Expects the optimized pmd2.py next to this file's pmd_pycore package, i.e.
importable as pmd_pycore.core.pmd2. If you keep it elsewhere, adjust the import.
"""

import argparse
import csv
import statistics
import time

import serial

from pmd_pycore.core import get_pmdusb_ports
from pmd_pycore.core.pmd2 import (
    PMD2X_PRODUCT_ID,
    capture_pmd2_raw,
    decode_sensor_frame,
    get_pmd2_ports,
    read_pmd2_device,
    translate_pmd2_aggregate_power,
    translate_pmd2_chip_sensors,
    translate_pmd2_sensor_struct,
    translate_pmd2x_hpwr_temperatures,
)


def parse_args():
    p = argparse.ArgumentParser(description="High-rate PMD2/PMD2X capture")
    p.add_argument("--samples", type=int, default=0,
                   help="number of samples to capture; 0 = run until Ctrl+C (default)")
    p.add_argument("--depth", type=int, default=1,
                   help="requests kept in flight (1 = classic; 2 = experimental pipelining)")
    p.add_argument("--timeout", type=float, default=0.05,
                   help="serial read timeout in seconds (only costs time on a failed read)")
    p.add_argument("--port", default=None, help="serial port, e.g. COM5 or /dev/ttyACM0")
    p.add_argument("--out", default="capture.csv", help="output CSV path")
    return p.parse_args()


def choose_port(requested):
    if requested:
        return requested

    print("\nFinding potential PMD2/PMD2X Serial Ports...\n")
    ports = get_pmd2_ports()
    for i, p in enumerate(ports):
        print(f"{i} - {p['port']}")
    print()

    if not ports:
        if get_pmdusb_ports():
            print("Found a PMD-USB, but this capture script only supports PMD2/PMD2X.")
        else:
            print("No PMD2/PMD2X serial ports found.")
        return None

    index = 0
    if len(ports) > 1:
        print("Multiple PMD2/PMD2X serial ports found. Input which port to use:")
        index = int(input())
        if index < 0 or index >= len(ports):
            print("Invalid port index.")
            return None
    return ports[index]["port"]


def open_port(port, timeout):
    ser = serial.Serial(port, 115200, timeout=timeout, write_timeout=1,
                        rtscts=False, dsrdtr=False)
    # Larger driver-side buffers (Windows only; harmless elsewhere).
    try:
        ser.set_buffer_size(rx_size=1 << 16, tx_size=1 << 12)
    except (AttributeError, serial.SerialException, OSError):
        pass
    ser.dtr = True
    time.sleep(0.1)          # let the device's unsolicited welcome push land
    ser.reset_input_buffer()
    return ser


def report_timing(frames, stamps, failures):
    n = len(frames)
    print(f"\nGood frames: {n}   failed reads: {failures}")
    if n < 2:
        return
    span = stamps[-1] - stamps[0]
    dts = sorted(b - a for a, b in zip(stamps, stamps[1:]))
    p99 = dts[min(len(dts) - 1, int(0.99 * len(dts)))]
    dupes = sum(1 for a, b in zip(frames, frames[1:]) if a == b) #Calculates number of identical measurements by comparing neighbors

    print(f"Duration:      {span:.3f} s")
    print(f"Sample rate:   {(n - 1) / span:.1f} Hz")
    print(f"Interval:      median {statistics.median(dts) * 1e3:.2f} ms, "
          f"p99 {p99 * 1e3:.2f} ms, max {dts[-1] * 1e3:.2f} ms")
    print(f"Identical consecutive frames: {dupes} ({100 * dupes / (n - 1):.1f}%)")
    if dupes / (n - 1) > 0.2:
        print("  -> many repeats: the device itself likely refreshes slower than you poll.")


def write_csv(path, frames, stamps, product_id):
    is_pmd2x = product_id == PMD2X_PRODUCT_ID
    first = decode_sensor_frame(frames[0], product_id)
    rails = list(translate_pmd2_sensor_struct(first, product_id).keys())

    header = ["Sample", "Time_s"]
    for r in rails:
        header += [f"{r}_V", f"{r}_A", f"{r}_W"]
    header += ["EPS_W", "PCIE_W", "MB_W", "TOTAL_W", "Vdd_V", "Tchip_C"]
    if is_pmd2x:
        header += ["HPWR_in_C", "HPWR_out_C"]

    t0 = stamps[0]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (buf, t) in enumerate(zip(frames, stamps)):
            s = decode_sensor_frame(buf, product_id)
            row = [i, f"{t - t0:.6f}"]
            for reading in translate_pmd2_sensor_struct(s, product_id).values():
                row += [reading["voltage"], reading["current"], reading["power"]]
            agg = translate_pmd2_aggregate_power(s)
            chip = translate_pmd2_chip_sensors(s)
            row += [agg["eps"], agg["pcie"], agg["mb"], agg["total"], chip["vdd"], chip["tchip"]]
            if is_pmd2x:
                temps = translate_pmd2x_hpwr_temperatures(s)
                row += ["" if temps["ts_in"] is None else temps["ts_in"],
                        "" if temps["ts_out"] is None else temps["ts_out"]]
            writer.writerow(row)


def main():
    args = parse_args()
    port = choose_port(args.port)
    if port is None:
        return
    print(f"Using {port}\n")

    try:
        ser = open_port(port, args.timeout)
    except serial.SerialException as e:
        print(f"Failed to open {port}: {e}")
        return

    try:
        print("Reading vendor data...")
        info = read_pmd2_device(ser)
        if info is None:
            print("Failed to read vendor data from device.")
            return
        print(f"Device Info: {info}\n")
        product_id = info["ProductId"]

        if args.samples > 0:
            print(f"Capturing {args.samples} samples (depth={args.depth})...")
        else:
            print(f"Capturing until Ctrl+C (depth={args.depth})...")

        frames, stamps, failures = capture_pmd2_raw(
            ser,
            product_id,
            max_samples=args.samples or None,
            depth=args.depth,
        )
    finally:
        ser.close()

    if not frames:
        print("No samples were captured.")
        return

    report_timing(frames, stamps, failures)
    print(f"\nWriting {args.out} ...")
    write_csv(args.out, frames, stamps, product_id)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
