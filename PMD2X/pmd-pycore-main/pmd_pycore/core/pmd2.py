"""
Wire-format structures, serial I/O, and sensor translation for ElmorLabs PMD2/PMD2X.

Example:
    >>> from pmd_pycore.core import get_pmd2_ports, read_pmd2_device, read_pmd2_sensors
    >>> ports = get_pmd2_ports()
    >>> if ports:
    >>>     import serial
    >>>     ser = serial.Serial(ports[0]['port'], 115200, timeout=1,
    >>>                          rtscts=False, dsrdtr=False)
    >>>     info = read_pmd2_device(ser)
    >>>     sensors = read_pmd2_sensors(ser, info['ProductId'])
    >>>     ser.close()
"""

import os
import sys
import time
import logging
import serial
import serial.tools.list_ports
from ctypes import LittleEndianStructure, c_uint8, c_int16, c_int32, c_uint16, sizeof
from enum import IntEnum
from typing import List, Dict, Optional, Union

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("pmd_pycore.core.pmd2")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --- Structures ---

PMD2_VENDOR_ID = 0xEE
PMD2_PRODUCT_ID = 0x15
PMD2X_PRODUCT_ID = 0x17

PMD2_PRODUCT_STRING = "ElmorLabs PMD2"
PMD2X_PRODUCT_STRING = "ElmorLabs PMD2X"

PMD2_SENSOR_POWER_NUM = 10
PMD2X_SENSOR_POWER_NUM = 16
PMD2X_SENSOR_TS_NUM = 2

PMD2X_TS_IN = 0   # 12VHPWR connector temperature sensor, cable side
PMD2X_TS_OUT = 1  # 12VHPWR connector temperature sensor, PSU/device side
PMD2X_TS_NOT_CONNECTED = 0x7FFF


class PMD2_CMD(IntEnum):
    """USB_CMD, per PMD2-FW's pmd2.h (private repo).

    CMD_READ_DEVICE_DATA is declared there but has no case in the device's
    CDC_Receive_FS dispatch (usbd_cdc_if.c) -- sending it is a no-op.
    """

    CMD_WELCOME = 0
    CMD_READ_VENDOR_DATA = 1
    CMD_READ_UID = 2
    CMD_READ_DEVICE_DATA = 3
    CMD_READ_SENSOR_VALUES = 4
    CMD_WRITE_CONT_TX = 5
    CMD_READ_CONFIG = 6
    CMD_WRITE_CONFIG = 7
    CMD_RESET = 0xF0
    CMD_BOOTLOADER = 0xF1
    CMD_MCU_CMD = 0xF2
    CMD_NOP = 0xFF

    def toByte(self) -> bytes:
        return self.value.to_bytes(1, byteorder="little")


class PMD2VendorDataStruct(LittleEndianStructure):
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


class PMD2PowerSensor(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("Voltage", c_int16),
        ("Current", c_int32),
        ("Power", c_int32),
    ]

    def __str__(self):
        return f"Voltage: {self.Voltage}, Current: {self.Current}, Power: {self.Power}"


class PMD2SensorStruct(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("Vdd", c_uint16),
        ("Tchip", c_int16),
        ("PowerReadings", PMD2PowerSensor * PMD2_SENSOR_POWER_NUM),
        ("EpsPower", c_uint16),
        ("PciePower", c_uint16),
        ("MbPower", c_uint16),
        ("TotalPower", c_uint16),
        ("Ocp", c_uint8 * PMD2_SENSOR_POWER_NUM),
    ]


class PMD2XSensorStruct(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("Vdd", c_uint16),
        ("Tchip", c_int16),
        ("Ts", c_int16 * PMD2X_SENSOR_TS_NUM),
        ("PowerReadings", PMD2PowerSensor * PMD2X_SENSOR_POWER_NUM),
        ("EpsPower", c_uint16),
        ("PciePower", c_uint16),
        ("MbPower", c_uint16),
        ("TotalPower", c_uint16),
        ("Ocp", c_uint8 * PMD2X_SENSOR_POWER_NUM),
    ]


# --- Serial I/O ---

# VID:PID shared by PMD2 and PMD2X (checked case-insensitively).
_PMD2_VID_PID = "VID:PID=0483:5740"

# Longest welcome string a device can push on connect.
_WELCOME_PUSH_MAX_LEN = max(len(PMD2_PRODUCT_STRING), len(PMD2X_PRODUCT_STRING)) + 1

# CMD_WELCOME replies with BuildData.ProductName truncated to 15 bytes
# (CDC_Transmit_FS((void*)BuildData.ProductName, 15) in usbd_cdc_if.c) --
# a fixed size independent of the unsolicited DTR-triggered push above.
_WELCOME_REPLY_LEN = 15

_UID_LEN = 12


def get_pmd2_ports() -> List[Dict[str, str]]:
    """Return all candidate PMD2/PMD2X COM ports without opening them.

    Returns:
        List[Dict[str, str]]: List of dicts with a 'port' key, e.g. [{'port': 'COM3'}].
    """
    try:
        ports = serial.tools.list_ports.comports()
    except serial.SerialException as e:
        logger.error("Failed to enumerate serial ports: %s", e)
        raise

    pmd2_ports = []
    for p in ports:
        if _PMD2_VID_PID in p.hwid.upper():
            pmd2_ports.append({"port": p.device})
    return pmd2_ports


def _drain_pmd2_welcome_push(ser: serial.Serial) -> Optional[str]:
    """Read and decode the unsolicited welcome string a device pushes after DTR is asserted."""
    buffer = ser.read(_WELCOME_PUSH_MAX_LEN)
    try:
        return buffer.decode("ascii").rstrip("\x00")
    except UnicodeDecodeError:
        return None


def find_pmd2_devices() -> List[Dict[str, str]]:
    """Scan and return all connected PMD2/PMD2X devices with identity info.

    Returns:
        List[Dict[str, str]]: List of dicts with 'port', 'product_id', 'fw' keys.

    Note:
        This opens and closes a serial connection per candidate port.
    """
    devices = []
    for port_info in get_pmd2_ports():
        port = port_info["port"]
        product_id, fw = "?", "?"
        try:
            with serial.Serial(port, 115200, timeout=1, rtscts=False, dsrdtr=False) as ser:
                ser.dtr = True
                _drain_pmd2_welcome_push(ser)
                device_info = read_pmd2_device(ser)
                if device_info:
                    product_id = device_info["ProductId"]
                    fw = device_info["FwVersion"]
            logger.info("Found device on %s: ProductId=%s, FW=%s", port, product_id, fw)
        except serial.SerialException as e:
            logger.warning("Failed to open serial port %s: %s", port, e)
        except Exception as e:
            logger.warning("Failed to read device on %s: %s", port, e)
        devices.append({"port": port, "product_id": product_id, "fw": fw})
    return devices


def read_pmd2_uart(ser: serial.Serial, cmd: PMD2_CMD, size: int) -> Optional[bytes]:
    """Send a command and read `size` bytes from the device.

    Returns:
        Optional[bytes]: Raw bytes read, or None if the read was incomplete.
    """
    ser.reset_input_buffer()
    ser.write(cmd.toByte())
    # Without a short gap the device occasionally hasn't queued its response
    # yet when we read. 2ms was 0/5000 on a PMD2X but 2-3% failures on a
    # PMD2 (different unit/firmware); 5ms was 0/5000 on both -- cheap
    # insurance against a read that otherwise blocks for the full port
    # timeout before coming back empty.
    time.sleep(0.005)
    buffer = ser.read(size)
    if len(buffer) != size:
        logger.warning("UART read incomplete: expected %d, got %d bytes", size, len(buffer))
        return None
    return buffer


def read_pmd2_device(ser: serial.Serial) -> Optional[Dict[str, int]]:
    """Read and validate vendor/product/firmware info from the device.

    Returns:
        Optional[Dict[str, int]]: Dict with 'VendorId', 'ProductId', 'FwVersion',
            or None if the read failed or the vendor ID didn't match.
    """
    buffer = read_pmd2_uart(ser, PMD2_CMD.CMD_READ_VENDOR_DATA, sizeof(PMD2VendorDataStruct))
    if buffer is None:
        return None

    vendor_struct = PMD2VendorDataStruct.from_buffer_copy(buffer)
    if vendor_struct.VendorId != PMD2_VENDOR_ID or vendor_struct.ProductId not in (
        PMD2_PRODUCT_ID,
        PMD2X_PRODUCT_ID,
    ):
        logger.warning("Unexpected vendor data: %s", vendor_struct)
        return None

    return {
        "VendorId": vendor_struct.VendorId,
        "ProductId": vendor_struct.ProductId,
        "FwVersion": vendor_struct.FwVersion,
    }


def read_pmd2_welcome_string(ser: serial.Serial) -> Optional[str]:
    """Send CMD_WELCOME and return the device's product name.

    Returns:
        Optional[str]: Decoded product name, or None if the read failed.
    """
    buffer = read_pmd2_uart(ser, PMD2_CMD.CMD_WELCOME, _WELCOME_REPLY_LEN)
    if buffer is None:
        return None
    try:
        return buffer.decode("ascii").rstrip("\x00")
    except UnicodeDecodeError:
        logger.warning("Failed to decode welcome string: %r", buffer)
        return None


def read_pmd2_uid(ser: serial.Serial) -> Optional[str]:
    """Read the device's unique ID (the STM32's hardware UID).

    Returns:
        Optional[str]: UID as an uppercase hex string, or None if the read failed.
    """
    buffer = read_pmd2_uart(ser, PMD2_CMD.CMD_READ_UID, _UID_LEN)
    if buffer is None:
        return None
    return buffer.hex().upper()


def read_pmd2_sensors(
    ser: serial.Serial,
    product_id: int,
) -> Optional[Union[PMD2SensorStruct, PMD2XSensorStruct]]:
    """Read all sensor values from the device.

    Args:
        ser (serial.Serial): Opened serial connection to the device.
        product_id (int): PMD2_PRODUCT_ID or PMD2X_PRODUCT_ID, selecting which
            sensor structure to decode the response as.

    Returns:
        Optional[Union[PMD2SensorStruct, PMD2XSensorStruct]]: Decoded sensor
            structure, or None if the read failed.
    """
    struct_type = PMD2XSensorStruct if product_id == PMD2X_PRODUCT_ID else PMD2SensorStruct
    buffer = read_pmd2_uart(ser, PMD2_CMD.CMD_READ_SENSOR_VALUES, sizeof(struct_type))
    if buffer is None:
        logger.warning("Failed to read sensors from device")
        return None
    return struct_type.from_buffer_copy(buffer)


# --- Sensor translation ---

# Channel index -> rail name, for each product. Matches SENSOR_POWER_* in
# the PMD2-FW pmd2.h header (private repo).
# Index 4 ("HPWR"/"HPWR1") is computed by the firmware itself from the six
# wire channels (10-15) -- it isn't something the host needs to average.
PMD2_CHANNELS = {
    0: "12V", 1: "5V", 2: "5VSB", 3: "3.3V",
    4: "HPWR",
    5: "EPS1", 6: "EPS2",
    7: "PCIE1", 8: "PCIE2", 9: "PCIE3",
}

PMD2X_CHANNELS = {
    0: "12V", 1: "5V", 2: "5VSB", 3: "3.3V",
    4: "HPWR",
    5: "EPS1", 6: "EPS2",
    7: "PCIE1", 8: "PCIE2", 9: "PCIE3",
    10: "HPWR_W1", 11: "HPWR_W2", 12: "HPWR_W3",
    13: "HPWR_W4", 14: "HPWR_W5", 15: "HPWR_W6",
}


def _pmd2_reading(voltage: int, current: int, power: int) -> Dict[str, float]:
    return {"voltage": voltage / 1000, "current": current / 1000, "power": power / 1000}


def translate_pmd2_sensor_struct(
    sensors: Union[PMD2SensorStruct, PMD2XSensorStruct],
    product_id: int,
) -> Dict[str, Dict[str, float]]:
    """Convert a raw sensor struct into a dict of rail name -> {voltage, current, power} (SI units).

    Args:
        sensors: Decoded sensor struct from read_pmd2_sensors().
        product_id: PMD2_PRODUCT_ID or PMD2X_PRODUCT_ID.

    Returns:
        Dict[str, Dict[str, float]]: Rail name -> reading.
    """
    channels = PMD2_CHANNELS if product_id == PMD2_PRODUCT_ID else PMD2X_CHANNELS
    readings = sensors.PowerReadings

    return {
        name: _pmd2_reading(readings[idx].Voltage, readings[idx].Current, readings[idx].Power)
        for idx, name in channels.items()
    }


def translate_pmd2_aggregate_power(
    sensors: Union[PMD2SensorStruct, PMD2XSensorStruct],
) -> Dict[str, float]:
    """Read the firmware-computed power aggregates, in watts.

    The device sums these on-board rather than requiring the host to add up
    individual rails:
        - eps: EPS/CPU power (EPS1 + EPS2)
        - pcie: PCIe/GPU power (PCIE1-3, plus HPWR on PMD2X)
        - mb: motherboard power (ATX24 rails: 12V/5V/5VSB/3.3V)
        - total: sum of all of the above

    Unlike PowerReadings[i].Power (milliwatts), these fields are already in
    whole watts -- firmware computes them as div10(div10(div10(sum_mW))),
    i.e. sum_mW / 1000. Confirmed against a live, loaded PMD2X: raw
    MbPower matched the rail-sum-in-watts, not rail-sum-in-milliwatts.
    """
    return {
        "eps": float(sensors.EpsPower),
        "pcie": float(sensors.PciePower),
        "mb": float(sensors.MbPower),
        "total": float(sensors.TotalPower),
    }


def translate_pmd2_chip_sensors(sensors: Union[PMD2SensorStruct, PMD2XSensorStruct]) -> Dict[str, float]:
    """Read the device's own supply voltage and chip temperature.

    Tchip is in tenths of a degree C, per sensor.c's TH_GetTs()/
    __LL_ADC_CALC_TEMPERATURE() usage -- confirmed against a live device
    (raw 320 read back as a plausible 32.0 C, not 320 C).

    Returns:
        Dict[str, float]: 'vdd' in volts, 'tchip' in degrees Celsius.
    """
    return {"vdd": sensors.Vdd / 1000, "tchip": sensors.Tchip / 10}


def translate_pmd2x_hpwr_temperatures(sensors: PMD2XSensorStruct) -> Dict[str, Optional[float]]:
    """Read the 12VHPWR connector's two thermistor temperatures (PMD2X only), in Celsius.

    Ts[] comes from TH_GetTs() (see PMD2-FW's th.c/th_lut.h), which is also
    in tenths of a degree C. 0x7FFF means the thermistor read out of range,
    i.e. nothing is plugged in.

    Returns:
        Dict[str, Optional[float]]: 'ts_in' and 'ts_out'; either is None if
            that thermistor is not connected.
    """
    def _value(raw: int) -> Optional[float]:
        return None if raw == PMD2X_TS_NOT_CONNECTED else raw / 10

    return {
        "ts_in": _value(sensors.Ts[PMD2X_TS_IN]),
        "ts_out": _value(sensors.Ts[PMD2X_TS_OUT]),
    }
