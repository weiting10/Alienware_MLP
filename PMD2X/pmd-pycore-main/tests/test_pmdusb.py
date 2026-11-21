# tests/test_pmdusb.py

from ctypes import Structure, sizeof
from unittest.mock import patch, MagicMock, Mock

from pmd_pycore.core import (
    get_pmdusb_ports,
    read_pmdusb_uart,
    read_pmdusb_welcome_string,
    read_pmdusb_id,
    read_pmdusb_sensors,
    translate_pmdusb_sensors,
    translate_pmdusb_aggregate_power,
    PMDUSBIdStruct,
    PMDUSBSensorStruct,
    PMDUSB_CMD,
    PMDUSB_VENDOR_ID,
    PMDUSB_PRODUCT_ID,
)


def test_imports():
    assert callable(get_pmdusb_ports)
    assert callable(read_pmdusb_uart)
    assert callable(read_pmdusb_welcome_string)
    assert callable(read_pmdusb_id)
    assert callable(read_pmdusb_sensors)
    assert callable(translate_pmdusb_sensors)
    assert callable(translate_pmdusb_aggregate_power)

    assert isinstance(PMDUSBIdStruct(), Structure)
    assert isinstance(PMDUSBSensorStruct(), Structure)


@patch("pmd_pycore.core.pmdusb.serial.tools.list_ports.comports")
def test_get_pmdusb_ports_filters_by_vid_pid(mock_comports):
    match = Mock(device="COM22", hwid="USB VID:PID=1A86:7523 SER=1")
    no_match = Mock(device="COM4", hwid="USB VID:PID=0483:5740 SER=2")
    mock_comports.return_value = [match, no_match]

    ports = get_pmdusb_ports()

    assert ports == [{"port": "COM22"}]


def test_read_pmdusb_welcome_string():
    ser = MagicMock()
    ser.read.return_value = b"ElmorLabs PMD-USB"

    assert read_pmdusb_welcome_string(ser) == "ElmorLabs PMD-USB"


def test_read_pmdusb_id():
    id_struct = PMDUSBIdStruct(VendorId=PMDUSB_VENDOR_ID, ProductId=PMDUSB_PRODUCT_ID, FwVersion=6)
    ser = MagicMock()
    ser.read.return_value = bytes(id_struct)

    info = read_pmdusb_id(ser)

    assert info == {"VendorId": PMDUSB_VENDOR_ID, "ProductId": PMDUSB_PRODUCT_ID, "FwVersion": 6}


def test_read_pmdusb_id_wrong_vendor_returns_none():
    id_struct = PMDUSBIdStruct(VendorId=0x00, ProductId=PMDUSB_PRODUCT_ID, FwVersion=6)
    ser = MagicMock()
    ser.read.return_value = bytes(id_struct)

    assert read_pmdusb_id(ser) is None


def test_read_pmdusb_sensors():
    sensors = PMDUSBSensorStruct()
    sensors.Channels[0].Voltage = 1200
    sensors.Channels[0].Current = 50
    ser = MagicMock()
    ser.read.return_value = bytes(sensors)

    result = read_pmdusb_sensors(ser)

    assert isinstance(result, PMDUSBSensorStruct)
    assert result.Channels[0].Voltage == 1200


def test_read_pmdusb_sensors_incomplete_returns_none():
    ser = MagicMock()
    ser.read.return_value = b"\x01\x02"

    assert read_pmdusb_sensors(ser) is None


def test_translate_pmdusb_sensors():
    sensors = PMDUSBSensorStruct()
    sensors.Channels[0].Voltage = 1200  # 12.00 V
    sensors.Channels[0].Current = 50   # 5.0 A

    result = translate_pmdusb_sensors(sensors)

    assert result["PCIE1"] == {"voltage": 12.0, "current": 5.0, "power": 60.0}
    assert set(result.keys()) == {"PCIE1", "PCIE2", "EPS1", "EPS2"}


def test_translate_pmdusb_aggregate_power():
    sensors = PMDUSBSensorStruct()
    sensors.Channels[0].Voltage = 1200  # PCIE1: 12.0 V
    sensors.Channels[0].Current = 10    # 1.0 A -> 12.0 W
    sensors.Channels[1].Voltage = 1200  # PCIE2: 12.0 V
    sensors.Channels[1].Current = 10    # 1.0 A -> 12.0 W
    sensors.Channels[2].Voltage = 1200  # EPS1: 12.0 V
    sensors.Channels[2].Current = 20    # 2.0 A -> 24.0 W
    sensors.Channels[3].Voltage = 1200  # EPS2: 12.0 V
    sensors.Channels[3].Current = 0     # 0.0 A -> 0.0 W

    result = translate_pmdusb_aggregate_power(sensors)

    assert result == {"gpu": 24.0, "cpu": 24.0, "total": 48.0}
