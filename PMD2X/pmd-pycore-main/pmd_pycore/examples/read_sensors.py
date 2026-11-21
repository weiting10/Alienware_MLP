"""
Connect to the first supported device found -- trying PMD2/PMD2X first, then
PMD-USB -- and print averaged readings in a loop, refreshing in place.
"""

import statistics
import sys
import time

import serial

from pmd_pycore.core import (
    get_pmd2_ports,
    read_pmd2_device,
    read_pmd2_sensors,
    translate_pmd2_sensor_struct,
    translate_pmd2_aggregate_power,
    translate_pmd2_chip_sensors,
    translate_pmd2x_hpwr_temperatures,
    PMD2X_PRODUCT_ID,
    get_pmdusb_ports,
    read_pmdusb_id,
    read_pmdusb_sensors,
    translate_pmdusb_sensors,
    translate_pmdusb_aggregate_power,
)

# Number of sensor reads averaged into each displayed update.
NUM_SAMPLES = 1

# Target time between displayed updates. On PMD2/PMD2X, read_pmd2_uart()'s 5ms
# inter-command delay dominates per-sample cost, so NUM_SAMPLES * 5ms
# (~100ms) alone is already close to this budget -- actual cycles land
# around 110-125ms there, not exactly 100ms. PMD-USB's 2ms delay leaves more
# headroom. Either way this is a target to approach, not a hard cap.
UPDATE_INTERVAL_S = 0.1

CLEAR_SCREEN = "\x1b[2J\x1b[H"


def _enable_windows_ansi() -> None:
    """Turn on ANSI escape processing for this console, if needed.

    Windows Terminal and PowerShell 7+ enable this by default, but plain
    cmd.exe and older consoles don't -- without it, CLEAR_SCREEN's escape
    bytes print as literal garbage instead of clearing the screen, so the
    display just scrolls forever rather than refreshing in place.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass  # Not a console we can configure (e.g. redirected to a file); harmless either way.


def _average_dicts(dicts, fields):
    return {field: statistics.mean(d[field] for d in dicts) for field in fields}


def _rail_lines(averaged_rails):
    lines = [f'{"Rail":<10}{"Voltage":>10}{"Current":>10}{"Power":>10}']
    for rail, reading in averaged_rails.items():
        lines.append(
            f'{rail:<10}{reading["voltage"]:>10.3f}'
            f'{reading["current"]:>10.3f}{reading["power"]:>10.3f}'
        )
    lines.append("")
    return lines

# --- PMD2X Higher Speed --- Built with for loop

def run_pmd_high_speed(port, target_samples=5000):
    try:
        # Lower serial timeout for faster non-blocking checks
        ser = serial.Serial(port, 115200, timeout=0.01, write_timeout=1, rtscts=False, dsrdtr=False)
    except serial.SerialException as e:
        print(f"Failed to open {port}: {e}")
        return

    try:
        ser.dtr = True
        time.sleep(0.1)
        ser.reset_input_buffer()

        print("Reading vendor data...")
        device_info = read_pmd2_device(ser)
        if device_info is None:
            print("Failed to read vendor data from device.")
            return
        print(f"Device Info: {device_info}\n")
        time.sleep(0.5)

        product_id = device_info["ProductId"]

        print(f"Starting high-speed capture of {target_samples} samples...")
        
        # In-memory buffer to store raw measurements instantly
        sample_buffer = []
        start_time = time.monotonic()

        # HIGH-SPEED CAPTURE LOOP: Zero prints, zero UI delays
        for _ in range(target_samples):
            try:
                sensors = read_pmd2_sensors(ser, product_id)
                if sensors is not None:
                    parsed_rails = translate_pmd2_sensor_struct(sensors, product_id)
                    sample_buffer.append(parsed_rails)
            except serial.SerialException:
                pass  # Skip transient serial hiccups during max-speed polling

        duration = time.monotonic() - start_time
        actual_rate = len(sample_buffer) / duration
        print(f"Finished! Captured {len(sample_buffer)} samples in {duration:.3f} seconds.")
        print(f"Achieved sampling rate: {actual_rate:.2f} Hz\n")

        # POST-PROCESSING: Write to disk after high-speed capture is complete
        print("Saving capture data to 'high_speed_log.csv'...")
        import csv
        with open("high_speed_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Sample_Index", "Rail", "Voltage", "Current", "Power"])
            for idx, sample in enumerate(sample_buffer):
                for rail_name, metrics in sample.items():
                    writer.writerow([idx, rail_name, metrics["voltage"], metrics["current"], metrics["power"]])
        print("Data successfully saved to CSV!")

    finally:
        ser.close()

# --- PMD2X Higher Speed Continuous
def run_pmd_continuous(port):
    try:
        # Low timeout ensures non-blocking checks in the loop
        ser = serial.Serial(port, 115200, timeout=0.01, write_timeout=1, rtscts=False, dsrdtr=False)
    except serial.SerialException as e:
        print(f"Failed to open {port}: {e}")
        return

    try:
        ser.dtr = True
        time.sleep(0.1)
        ser.reset_input_buffer()

        print("Reading vendor data...")
        device_info = read_pmd2_device(ser)
        if device_info is None:
            print("Failed to read vendor data from device.")
            return
        print(f"Device Info: {device_info}\n")
        time.sleep(0.5)

        product_id = device_info["ProductId"]
        is_pmd2x = product_id == PMD2X_PRODUCT_ID

        print("Starting continuous high-speed capture...")
        print("Press **Ctrl + C** at any time to stop capturing and save data.\n")

        sample_buffer = []
        start_time = time.monotonic()

        # CONTINUOUS WHILE LOOP: Runs forever until you press Ctrl + C
        while True:
            try:
                sensors = read_pmd2_sensors(ser, product_id)
                if sensors is not None:
                    parsed_rails = translate_pmd2_sensor_struct(sensors, product_id)
                    sample_buffer.append(parsed_rails)
            except serial.SerialException:
                # Silently catch minor serial blips during rapid polling so the loop doesn't crash
                pass

    except KeyboardInterrupt:
        # This catches when you press Ctrl + C
        print("\n\nCapture stopped by user.")
        
    finally:
        ser.close()
        
        # POST-PROCESSING: Calculate stats and save to CSV after stopping
        if sample_buffer:
            duration = time.monotonic() - start_time
            actual_rate = len(sample_buffer) / duration
            print(f"Captured {len(sample_buffer)} samples in {duration:.3f} seconds.")
            print(f"Average sampling rate achieved: {actual_rate:.2f} Hz\n")

            print("Saving data to 'continuous_sensor_log.csv'...")
            import csv
            with open("continuous_sensor_log.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Sample_Index", "Rail", "Voltage", "Current", "Power"])
                for idx, sample in enumerate(sample_buffer):
                    for rail_name, metrics in sample.items():
                        writer.writerow([idx, rail_name, metrics["voltage"], metrics["current"], metrics["power"]])
            print("Data successfully saved!")
        else:
            print("No samples were captured.")

# --- PMD2 / PMD2X ---

def run_pmd(port):
    try:
        ser = serial.Serial(port, 115200, timeout=1, write_timeout=1, rtscts=False, dsrdtr=False)
    except serial.SerialException as e:
        print(f"Failed to open {port}: {e}")
        return

    try:
        ser.dtr = True
        # The device pushes an unsolicited welcome string on DTR assert, but it
        # can arrive after this point rather than before it. Give it a moment to
        # land, then flush -- read_pmd2_uart() also flushes before each command, so a
        # late push can never corrupt the next real response.
        time.sleep(0.1)
        ser.reset_input_buffer()

        print("Reading vendor data...")
        device_info = read_pmd2_device(ser)
        if device_info is None:
            print("Failed to read vendor data from device.")
            return
        print(f"Result: {device_info}\n")
        time.sleep(1)

        product_id = device_info["ProductId"]
        is_pmd2x = product_id == PMD2X_PRODUCT_ID

        while True:
            cycle_start = time.monotonic()
            rail_samples = []
            aggregate_samples = []
            chip_samples = []
            hpwr_temp_samples = []

            for _ in range(NUM_SAMPLES):
                sensors = None
                while sensors is None:
                    try:
                        sensors = read_pmd2_sensors(ser, product_id)
                    except serial.SerialException:
                        time.sleep(0.1)
                rail_samples.append(translate_pmd2_sensor_struct(sensors, product_id))
                aggregate_samples.append(translate_pmd2_aggregate_power(sensors))
                chip_samples.append(translate_pmd2_chip_sensors(sensors))
                if is_pmd2x:
                    hpwr_temp_samples.append(translate_pmd2x_hpwr_temperatures(sensors))

            rail_names = rail_samples[0].keys()
            averaged_rails = {
                rail: _average_dicts([s[rail] for s in rail_samples], ("voltage", "current", "power"))
                for rail in rail_names
            }
            averaged_aggregates = _average_dicts(aggregate_samples, ("eps", "pcie", "mb", "total"))
            averaged_chip = _average_dicts(chip_samples, ("vdd", "tchip"))

            lines = _rail_lines(averaged_rails)

            lines.append("--- Aggregate power ---")
            lines.append(f'EPS (CPU):       {averaged_aggregates["eps"]:>8.3f} W')
            lines.append(f'PCIe (GPU):      {averaged_aggregates["pcie"]:>8.3f} W')
            lines.append(f'Motherboard:     {averaged_aggregates["mb"]:>8.3f} W')
            lines.append(f'Total:           {averaged_aggregates["total"]:>8.3f} W')
            lines.append("")

            lines.append("--- Device health ---")
            lines.append(f'{"Supply (Vdd):":<17}{averaged_chip["vdd"]:>8.3f} V')
            lines.append(f'{"Chip temp:":<17}{averaged_chip["tchip"]:>8.1f} C')
            if is_pmd2x:
                def fmt_temp(samples, key):
                    values = [s[key] for s in samples if s[key] is not None]
                    return f"{statistics.mean(values):>8.1f} C" if values else f"{'n/c':>8} C"

                lines.append(f'{"12VHPWR in:":<17}{fmt_temp(hpwr_temp_samples, "ts_in")}')
                lines.append(f'{"12VHPWR out:":<17}{fmt_temp(hpwr_temp_samples, "ts_out")}')

            print(CLEAR_SCREEN + "\n".join(lines), flush=True)

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.0, UPDATE_INTERVAL_S - elapsed))
    finally:
        ser.close()


# --- PMD-USB ---

# def run_pmdusb(port):
#     try:
#         ser = serial.Serial(port, 115200, timeout=1, write_timeout=1)
#     except serial.SerialException as e:
#         print(f"Failed to open {port}: {e}")
#         return

#     try:
#         print("Reading device id...")
#         device_info = read_pmdusb_id(ser)
#         if device_info is None:
#             print("Failed to read device id.")
#             return
#         print(f"Result: {device_info}\n")
#         time.sleep(1)

#         while True:
#             cycle_start = time.monotonic()
#             channel_samples = []
#             aggregate_samples = []

#             for _ in range(NUM_SAMPLES): #Number of Samples Averaged into on reading
#                 sensors = None
#                 while sensors is None:
#                     try:
#                         sensors = read_pmdusb_sensors(ser)
#                     except serial.SerialException:
#                         time.sleep(0.1)
#                 channel_samples.append(translate_pmdusb_sensors(sensors))
#                 aggregate_samples.append(translate_pmdusb_aggregate_power(sensors))
#             #end for loop
#             channel_names = channel_samples[0].keys()
#             averaged_channels = {
#                 channel: _average_dicts(
#                     [s[channel] for s in channel_samples], ("voltage", "current", "power")
#                 )
#                 for channel in channel_names
#             }
#             averaged_aggregates = _average_dicts(aggregate_samples, ("gpu", "cpu", "total"))

#             lines = _rail_lines(averaged_channels)
#             lines.append("--- Aggregate power ---")
#             lines.append(f'GPU (PCIE1+2):   {averaged_aggregates["gpu"]:>8.3f} W')
#             lines.append(f'CPU (EPS1+2):    {averaged_aggregates["cpu"]:>8.3f} W')
#             lines.append(f'Total:           {averaged_aggregates["total"]:>8.3f} W')
#             print(CLEAR_SCREEN + "\n".join(lines), flush=True)

#             elapsed = time.monotonic() - cycle_start
#             time.sleep(max(0.0, UPDATE_INTERVAL_S - elapsed))
#     finally:
#         ser.close()


# --- Entry point ---

def main():
    _enable_windows_ansi()

    print("\nFinding potential PMD2/PMD2X Serial Ports...\n")
    pmd_ports = get_pmd2_ports()
    for i, p in enumerate(pmd_ports):
        print(f"{i} - {p['port']}")
    print()

    if pmd_ports:
        port_index = 0
        if len(pmd_ports) > 1:
            print("Multiple PMD2/PMD2X serial ports found. Input which port to use:")
            port_index = int(input())
            if port_index >= len(pmd_ports):
                print("Invalid port index.")
                return

        port = pmd_ports[port_index]["port"]
        print(f"Using port index {port_index} ({port})\n")
        run_pmd_continuous(port)
        return

    print("No PMD2/PMD2X serial ports found. Checking for PMD-USB...\n")
    pmdusb_ports = get_pmdusb_ports()
    if not pmdusb_ports:
        print("No PMD2/PMD2X or PMD-USB serial ports found.")
        return

    port_index = 0
    if len(pmdusb_ports) > 1:
        print("Multiple PMD-USB serial ports found. Input which port to use:")
        port_index = int(input())
        if port_index >= len(pmdusb_ports):
            print("Invalid port index.")
            return

    port = pmdusb_ports[port_index]["port"]
    print(f"Using port index {port_index} ({port})\n")
    #run_pmdusb(port)
    run_pmd_continuous(port)
    return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
