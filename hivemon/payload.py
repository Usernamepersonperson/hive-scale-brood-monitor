"""Compact 14-byte uplink payload (LoRaWAN-sized, also fine for an NB-IoT-style datagram).

Byte layout (big endian):
  0      high nibble = format version (1); bit0 tilt/shock alarm, bit1 low battery, bit2 service mode
  1-2    weight, uint16, 10 g units (0 .. 655.35 kg)
  3-4    brood-nest temperature, int16, 0.01 C
  5-6    ambient temperature, int16, 0.01 C
  7      hive relative humidity, uint8, 0.5 % units
  8      max tilt deviation since last uplink, uint8, degrees
  9      max shock since last uplink, uint8, 0.05 g units
  10     battery, uint8, 2000 mV + 10 mV * value
  11     frame counter, uint8
  12-13  weight change over the last 30 minutes, int16, 10 g units
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

VERSION = 1
SIZE = 14


@dataclass
class Reading:
    weight_kg: float
    t_brood_c: float
    t_amb_c: float
    rh_pct: float
    tilt_deg: float = 0.0
    shock_g: float = 0.0
    battery_v: float = 3.6
    counter: int = 0
    delta30_kg: float = 0.0
    alarm: bool = False
    low_battery: bool = False
    service: bool = False


def _clip(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def encode(r: Reading) -> bytes:
    flags = (VERSION << 4) | (1 if r.alarm else 0) | (2 if r.low_battery else 0) | (4 if r.service else 0)
    return struct.pack(
        ">BHhhBBBBBh",
        flags,
        _clip(r.weight_kg * 100, 0, 65535),
        _clip(r.t_brood_c * 100, -32768, 32767),
        _clip(r.t_amb_c * 100, -32768, 32767),
        _clip(r.rh_pct * 2, 0, 255),
        _clip(r.tilt_deg, 0, 255),
        _clip(r.shock_g / 0.05, 0, 255),
        _clip((r.battery_v * 1000 - 2000) / 10, 0, 255),
        r.counter & 0xFF,
        _clip(r.delta30_kg * 100, -32768, 32767),
    )


def decode(b: bytes) -> dict:
    if len(b) != SIZE:
        raise ValueError(f"payload must be {SIZE} bytes, got {len(b)}")
    flags, w, tb, ta, rh, tilt, shock, bat, cnt, d30 = struct.unpack(">BHhhBBBBBh", b)
    if flags >> 4 != VERSION:
        raise ValueError(f"unsupported payload version {flags >> 4}")
    return {
        "alarm": bool(flags & 1), "low_battery": bool(flags & 2), "service": bool(flags & 4),
        "weight_kg": w / 100, "t_brood_c": tb / 100, "t_amb_c": ta / 100, "rh_pct": rh / 2,
        "tilt_deg": float(tilt), "shock_g": round(shock * 0.05, 2),
        "battery_v": (2000 + bat * 10) / 1000, "counter": cnt, "delta30_kg": d30 / 100,
    }


# ---------------------------------------------------------------------------------------------
# air time and battery estimate (assumed currents, not measurements)
# ---------------------------------------------------------------------------------------------

def lora_airtime_s(payload_bytes: int, sf: int, bw_hz: int = 125_000, cr: int = 1,
                   preamble: int = 8, crc: bool = True, explicit_header: bool = True) -> float:
    """Time on air of one LoRa frame (Semtech AN1200.13 formula). `payload_bytes` is the PHY payload."""
    if not 6 <= sf <= 12:
        raise ValueError("sf must be 6..12")
    t_sym = (2 ** sf) / bw_hz
    de = 1 if t_sym > 0.016 else 0
    ih = 0 if explicit_header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * (1 if crc else 0) - 20 * ih
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return (preamble + 4.25) * t_sym + n_payload * t_sym


LORAWAN_OVERHEAD = 13  # MHDR + DevAddr + FCtrl + FCnt + FPort + MIC


@dataclass
class PowerModel:
    """Typical-class assumptions for a duty-cycled hive node. Replace with your own measurements."""

    sleep_ua: float = 12.0        # MCU sleep + regulator + ADC power-down
    measure_ma: float = 6.0       # bridge excitation + ADC + temperature probes + MCU while measuring
    measure_s: float = 1.5
    tx_ma: float = 44.0           # sub-GHz transmitter at 14 dBm
    rx_ma: float = 12.0
    rx_s: float = 0.25            # two short class-A receive windows without downlink
    capacity_mah: float = 3000.0  # one Li-ion cell, usable
    derate: float = 0.8           # cold-weather / ageing derating
    self_discharge_per_year: float = 0.02


def battery_life_days(interval_min: float, sf: int, app_bytes: int = SIZE, pm: PowerModel | None = None) -> dict:
    pm = pm or PowerModel()
    toa = lora_airtime_s(app_bytes + LORAWAN_OVERHEAD, sf)
    cycle_s = interval_min * 60
    charge_mas = (pm.measure_ma * pm.measure_s + pm.tx_ma * toa + pm.rx_ma * pm.rx_s
                  + pm.sleep_ua / 1000 * max(cycle_s - pm.measure_s - toa - pm.rx_s, 0))
    avg_ma = charge_mas / cycle_s
    avg_ma += pm.capacity_mah * pm.self_discharge_per_year / (365 * 24)
    days = pm.capacity_mah * pm.derate / avg_ma / 24
    return {"interval_min": interval_min, "sf": sf, "airtime_ms": toa * 1000, "avg_ma": avg_ma,
            "life_days": days, "life_years": days / 365}
