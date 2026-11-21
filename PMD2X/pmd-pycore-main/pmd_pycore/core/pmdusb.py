"""
Wire-format structures, serial I/O, and sensor translation for the ElmorLabs PMD-USB.

PMD-USB is a different device family from PMD2/PMD2X: it connects via a
CH340 USB-serial bridge rather than an STM32 CDC-ACM VCP, and uses its own
command numbering and sensor layout (PCIE1/PCIE2/EPS1/EPS2 only -- no ATX24
rails, no 12VHPWR). There is no public firmware source; this was ported
from and verified against:
https://github.com/ElmorLabs/PMDLogger (C# reference, PMD_USB_Device.cs)
https://github.com/ElmorLabs/PMD-USB (official GUI app, FormPMD.cs)
https://github.com/ElmorLabs/pmd-usb-serial-interface-fast
https://github.com/bjorntas/elmorlabs-pmd-usb-serial-interface

Example:
    >>> from pmd_pycore.core import get_pmdusb_ports, read_pmdusb_id, read_pmdusb_sensors
    >>> ports = get_pmdusb_ports()
    >>> if ports:
    >>>     import serial
    >>>     ser = serial.Serial(ports[0]['port'], 115200, timeout=1)
    >>>     info = read_pmdusb_id(ser)
    >>>     sensors = read_pmdusb_sensors(ser)
    >>>     ser.close()
"""

import os
import sys
import time
import logging
import serial
import serial.tools.list_ports
from ctypes import LittleEndianStructure, c_uint8, c_int16, sizeof
from enum import IntEnum
from typing import List, Dict, Optional

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("pmd_pycore.core.pmdusb")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --- Structures ---

PMDUSB_VENDOR_ID = 0xEE
PMDUSB_PRODUCT_ID = 0x0A

PMDUSB_PRODUCT_STRING = "ElmorLabs PMD-USB"

PMDUSB_CHANNEL_NUM = 4


class PMDUSB_CMD(IntEnum):
    """UART_CMD, per PMDLogger's PMD_USB_Device.cs and the official GUI app's FormPMD.cs."""

    CMD_WELCOME = 0
    CMD_READ_ID = 1
    CMD_READ_SENSORS = 2
    CMD_READ_SENSOR_VALUES = 3
    CMD_READ_CONFIG = 4
    CMD_WRITE_CONFIG = 5
    CMD_READ_ADC = 6
    CMD_WRITE_CONFIG_CONT_TX = 7
    CMD_WRITE_CONFIG_UART = 8
    CMD_RESET = 0xF0
    CMD_BOOTLOADER = 0xF1
    CMD_NOP = 0xFF

    def toByte(self) -> bytes:
        return self.value.to_bytes(1, byteorder="little")


class PMDUSBIdStruct(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("VendorId", c_uint8),
        ("ProductId", c_uint8),
        ("FwVersion", c_uint8),
    ]

    def __str__(self):
        return (
            f"VendorId: {hex(self.VendorId)}, "
            f"ProductId: {hex(self.ProductId)}, "
            f"FwVersion: {hex(self.FwVersion)}"
        )


class PMDUSBChannelValues(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("Voltage", c_int16),
        ("Current", c_int16),
    ]


class PMDUSBSensorStruct(LittleEndianStructure):
    """CMD_READ_SENSOR_VALUES reply: 4 channels x {Voltage, Current}, 16 bytes total.

    Voltage is raw*0.01 V, current is raw*0.1 A -- plain signed int16.
    Already calibrated on-device (confirmed against the official GUI app's
    live monitoring loop, which uses this command directly with no
    client-side offset/gain correction); CMD_READ_ADC's raw-ADC path isn't
    needed or implemented here.
    """

    _pack_ = 1
    _fields_ = [
        ("Channels", PMDUSBChannelValues * PMDUSB_CHANNEL_NUM),
    ]


# Channel index -> rail name, per PMDLogger's PMD_USB_Device constructor and
# bjorntas's elmorlabs-pmd-usb-serial-interface main.py.
PMDUSB_CHANNELS = {
    0: "PCIE1",
    1: "PCIE2",
    2: "EPS1",
    3: "EPS2",
}

# Channel indices feeding each aggregate, matching the official PMD-USB GUI
# app's update_task() (FormPMD.cs): PCIE1+PCIE2 -> GPU, EPS1+EPS2 -> CPU.
_GPU_CHANNELS = (0, 1)
_CPU_CHANNELS = (2, 3)


# --- Serial I/O ---

# PMD-USB uses a CH340 USB-serial bridge, not the STM32 VCP that PMD2/PMD2X
# use -- a different VID:PID entirely.
_PMDUSB_VID_PID = "VID:PID=1A86:7523"

_WELCOME_REPLY_LEN = len(PMDUSB_PRODUCT_STRING)


def get_pmdusb_ports() -> List[Dict[str, str]]:
    """Return all candidate PMD-USB COM ports without opening them.

    Returns:
        List[Dict[str, str]]: List of dicts with a 'port' key, e.g. [{'port': 'COM3'}].
    """
    try:
        ports = serial.tools.list_ports.comports()
    except serial.SerialException as e:
        logger.error("Failed to enumerate serial ports: %s", e)
        raise

    pmdusb_ports = []
    for p in ports:
        if _PMDUSB_VID_PID in p.hwid.upper():
            pmdusb_ports.append({"port": p.device})
    return pmdusb_ports


def read_pmdusb_uart(ser: serial.Serial, cmd: PMDUSB_CMD, size: int) -> Optional[bytes]:
    """Send a command and read `size` bytes from the device.

    Returns:
        Optional[bytes]: Raw bytes read, or None if the read was incomplete.
    """
    ser.reset_input_buffer()
    ser.write(cmd.toByte())
    # Measured 0/5000 failures at 0ms across 5 trials against a live
    # PMD-USB (its CH340 bridge responds faster than PMD2/PMD2X's STM32
    # CDC-ACM stack did), but a small margin is kept anyway -- that same
    # "0ms looked fine" read on PMD2/PMD2X didn't hold up under more
    # sustained testing, and a cheap sleep costs far less than a stall on
    # the read timeout if it's ever wrong on different hardware.
    time.sleep(0.002)
    buffer = ser.read(size)
    if len(buffer) != size:
        logger.warning("UART read incomplete: expected %d, got %d bytes", size, len(buffer))
        return None
    return buffer


def read_pmdusb_welcome_string(ser: serial.Serial) -> Optional[str]:
    """Send CMD_WELCOME and return the device's product name.

    Returns:
        Optional[str]: Decoded product name, or None if the read failed.
    """
    buffer = read_pmdusb_uart(ser, PMDUSB_CMD.CMD_WELCOME, _WELCOME_REPLY_LEN)
    if buffer is None:
        return None
    try:
        return buffer.decode("ascii").rstrip("\x00")
    except UnicodeDecodeError:
        logger.warning("Failed to decode welcome string: %r", buffer)
        return None


def read_pmdusb_id(ser: serial.Serial) -> Optional[Dict[str, int]]:
    """Read and validate vendor/product/firmware info from the device.

    Returns:
        Optional[Dict[str, int]]: Dict with 'VendorId', 'ProductId', 'FwVersion',
            or None if the read failed or the vendor/product ID didn't match.
    """
    buffer = read_pmdusb_uart(ser, PMDUSB_CMD.CMD_READ_ID, sizeof(PMDUSBIdStruct))
    if buffer is None:
        return None

    id_struct = PMDUSBIdStruct.from_buffer_copy(buffer)
    if id_struct.VendorId != PMDUSB_VENDOR_ID or id_struct.ProductId != PMDUSB_PRODUCT_ID:
        logger.warning("Unexpected vendor data: %s", id_struct)
        return None

    return {
        "VendorId": id_struct.VendorId,
        "ProductId": id_struct.ProductId,
        "FwVersion": id_struct.FwVersion,
    }


def read_pmdusb_sensors(ser: serial.Serial) -> Optional[PMDUSBSensorStruct]:
    """Read all sensor values from the device.

    Returns:
        Optional[PMDUSBSensorStruct]: Decoded sensor structure, or None if the
            read failed.
    """
    buffer = read_pmdusb_uart(ser, PMDUSB_CMD.CMD_READ_SENSOR_VALUES, sizeof(PMDUSBSensorStruct))
    if buffer is None:
        logger.warning("Failed to read sensors from device")
        return None
    return PMDUSBSensorStruct.from_buffer_copy(buffer)


# --- Sensor translation ---

def translate_pmdusb_sensors(sensors: PMDUSBSensorStruct) -> Dict[str, Dict[str, float]]:
    """Convert a raw PMD-USB sensor struct into a dict of channel name -> reading.

    Returns:
        Dict[str, Dict[str, float]]: Channel name -> {voltage, current, power}.
    """
    result = {}
    for idx, name in PMDUSB_CHANNELS.items():
        channel = sensors.Channels[idx]
        voltage = channel.Voltage * 0.01
        current = channel.Current * 0.1
        result[name] = {"voltage": voltage, "current": current, "power": voltage * current}
    return result


def translate_pmdusb_aggregate_power(sensors: PMDUSBSensorStruct) -> Dict[str, float]:
    """Compute GPU/CPU/Total power from the per-channel readings, in watts.

    Unlike PMD2/PMD2X, the PMD-USB firmware doesn't compute these aggregates
    on-device -- they're derived client-side here, matching the official
    GUI app: gpu = PCIE1 + PCIE2, cpu = EPS1 + EPS2, total = gpu + cpu.
    """
    readings = translate_pmdusb_sensors(sensors)
    channel_names = list(PMDUSB_CHANNELS.values())

    gpu = sum(readings[channel_names[i]]["power"] for i in _GPU_CHANNELS)
    cpu = sum(readings[channel_names[i]]["power"] for i in _CPU_CHANNELS)

    return {"gpu": gpu, "cpu": cpu, "total": gpu + cpu}
