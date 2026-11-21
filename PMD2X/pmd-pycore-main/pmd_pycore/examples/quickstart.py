"""
Minimal example: connect to the first supported device found -- trying
PMD2/PMD2X first, then PMD-USB -- print its identity, and read sensors once.
"""

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import serial

from pmd_pycore.core import (
    get_pmd2_ports,
    read_pmd2_device,
    read_pmd2_uid,
    read_pmd2_sensors,
    translate_pmd2_sensor_struct,
    translate_pmd2_aggregate_power,
    translate_pmd2_chip_sensors,
    translate_pmd2x_hpwr_temperatures,
    PMD2_PRODUCT_ID,
    PMD2X_PRODUCT_ID,
    get_pmdusb_ports,
    read_pmdusb_id,
    read_pmdusb_sensors,
    translate_pmdusb_sensors,
    translate_pmdusb_aggregate_power,
)

PRODUCT_NAMES = {PMD2_PRODUCT_ID: "PMD2", PMD2X_PRODUCT_ID: "PMD2X"}


# --- PMD2 / PMD2X ---

@contextmanager
def connect_pmd(port: str) -> Iterator[serial.Serial]:
    try:
        # rtscts/dsrdtr disabled: the device uses DTR as its "is a host listening"
        # signal, so pyserial's automatic flow-control handling must stay out of the way.
        with serial.Serial(
            port, 115200, timeout=1, write_timeout=1, rtscts=False, dsrdtr=False
        ) as ser:
            ser.dtr = True
            # The device pushes an unsolicited welcome string on DTR assert, but
            # the push can arrive after this point rather than before it. Give it
            # a moment to land, then flush -- read_pmd2_uart() also flushes before each
            # command, so a late push can never corrupt the next real response.
            time.sleep(0.1)
            ser.reset_input_buffer()
            yield ser
    except serial.SerialException as e:
        raise SystemExit(
            f"Failed to open {port}: {e}\n"
            "(is another program -- a monitoring app, another instance of this "
            "script -- already using the port?)"
        )


def print_pmd_device_info(device_info: Dict[str, int], uid: Optional[str]) -> None:
    product_name = PRODUCT_NAMES.get(device_info["ProductId"], "unknown")
    print("--- Device ---")
    print(f"Product:    {product_name}")
    print(f"Vendor ID:  {hex(device_info['VendorId'])}")
    print(f"Product ID: {hex(device_info['ProductId'])}")
    print(f"Firmware:   {hex(device_info['FwVersion'])}")
    print(f"UID:        {uid or 'unavailable'}")
    print()


def print_rail_readings(readings: Dict[str, Dict[str, float]]) -> None:
    print("--- Rails ---")
    print(f'{"Rail":<10}{"Voltage":>9}{"Current":>9}{"Power":>9}')
    for rail, reading in readings.items():
        print(
            f'{rail:<10}{reading["voltage"]:>8.3f}V{reading["current"]:>8.3f}A'
            f'{reading["power"]:>8.3f}W'
        )
    print()


def print_aggregate_power(aggregates: Dict[str, float]) -> None:
    print("--- Aggregate power ---")
    print(f'EPS (CPU):       {aggregates["eps"]:>8.3f} W')
    print(f'PCIe (GPU):      {aggregates["pcie"]:>8.3f} W')
    print(f'Motherboard:     {aggregates["mb"]:>8.3f} W')
    print(f'Total:           {aggregates["total"]:>8.3f} W')
    print()


def print_pmd_device_health(chip: Dict[str, float], hpwr_temps: Optional[Dict[str, Optional[float]]]) -> None:
    def fmt_temp(value: Optional[float]) -> str:
        return f"{value:>8.1f} C" if value is not None else f"{'n/c':>8} C"

    print("--- Device health ---")
    print(f'{"Supply (Vdd):":<17}{chip["vdd"]:>8.3f} V')
    print(f'{"Chip temp:":<17}{chip["tchip"]:>8.1f} C')
    if hpwr_temps is not None:
        print(f'{"12VHPWR in:":<17}{fmt_temp(hpwr_temps["ts_in"])}')
        print(f'{"12VHPWR out:":<17}{fmt_temp(hpwr_temps["ts_out"])}')


def run_pmd(port: str) -> None:
    with connect_pmd(port) as ser:
        device_info = read_pmd2_device(ser)
        if device_info is None:
            raise SystemExit("Failed to read device info -- is this a PMD2/PMD2X device?")

        product_id = device_info["ProductId"]
        if product_id not in PRODUCT_NAMES:
            raise SystemExit(f"Unrecognized ProductId {hex(product_id)} -- not a PMD2/PMD2X device?")

        print_pmd_device_info(device_info, read_pmd2_uid(ser))

        sensors = read_pmd2_sensors(ser, product_id)
        if sensors is None:
            raise SystemExit("Failed to read sensor values.")

        print_rail_readings(translate_pmd2_sensor_struct(sensors, product_id))
        print_aggregate_power(translate_pmd2_aggregate_power(sensors))
        hpwr_temps = translate_pmd2x_hpwr_temperatures(sensors) if product_id == PMD2X_PRODUCT_ID else None
        print_pmd_device_health(translate_pmd2_chip_sensors(sensors), hpwr_temps)


# --- PMD-USB ---

def print_pmdusb_device_info(device_info: Dict[str, int]) -> None:
    print("--- Device ---")
    print("Product:    PMD-USB")
    print(f"Vendor ID:  {hex(device_info['VendorId'])}")
    print(f"Product ID: {hex(device_info['ProductId'])}")
    print(f"Firmware:   {hex(device_info['FwVersion'])}")
    print()


def run_pmdusb(port: str) -> None:
    try:
        ser = serial.Serial(port, 115200, timeout=1, write_timeout=1)
    except serial.SerialException as e:
        raise SystemExit(f"Failed to open {port}: {e}")

    try:
        device_info = read_pmdusb_id(ser)
        if device_info is None:
            raise SystemExit("Failed to read device info -- is this a PMD-USB device?")
        print_pmdusb_device_info(device_info)

        sensors = read_pmdusb_sensors(ser)
        if sensors is None:
            raise SystemExit("Failed to read sensor values.")

        print_rail_readings(translate_pmdusb_sensors(sensors))

        aggregates = translate_pmdusb_aggregate_power(sensors)
        print("--- Aggregate power ---")
        print(f'GPU (PCIE1+2):   {aggregates["gpu"]:>8.3f} W')
        print(f'CPU (EPS1+2):    {aggregates["cpu"]:>8.3f} W')
        print(f'Total:           {aggregates["total"]:>8.3f} W')
    finally:
        ser.close()


# --- Entry point ---

def main() -> None:
    pmd_ports = get_pmd2_ports()
    if pmd_ports:
        if len(pmd_ports) > 1:
            print(f"Found {len(pmd_ports)} PMD2/PMD2X devices, using the first: {pmd_ports[0]['port']}")
        port = pmd_ports[0]["port"]
        print(f"Connecting to {port} (PMD2/PMD2X)...")
        run_pmd(port)
        return

    pmdusb_ports = get_pmdusb_ports()
    if pmdusb_ports:
        if len(pmdusb_ports) > 1:
            print(f"Found {len(pmdusb_ports)} PMD-USB devices, using the first: {pmdusb_ports[0]['port']}")
        port = pmdusb_ports[0]["port"]
        print(f"Connecting to {port} (PMD-USB)...")
        run_pmdusb(port)
        return

    raise SystemExit("No PMD2/PMD2X or PMD-USB device found.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
