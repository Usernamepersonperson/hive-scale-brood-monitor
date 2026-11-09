import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from hivemon import payload

ROOT = Path(__file__).resolve().parents[1]


def _reading(**kw):
    base = dict(weight_kg=48.25, t_brood_c=34.56, t_amb_c=-3.2, rh_pct=61.5, tilt_deg=7, shock_g=0.35,
                battery_v=3.87, counter=200, delta30_kg=-1.37)
    base.update(kw)
    return payload.Reading(**base)


def test_size_is_14_bytes():
    assert len(payload.encode(_reading())) == 14


def test_roundtrip_values():
    d = payload.decode(payload.encode(_reading(alarm=True, service=True)))
    assert d["weight_kg"] == pytest.approx(48.25)
    assert d["t_brood_c"] == pytest.approx(34.56)
    assert d["t_amb_c"] == pytest.approx(-3.2)
    assert d["rh_pct"] == pytest.approx(61.5)
    assert d["delta30_kg"] == pytest.approx(-1.37)
    assert d["battery_v"] == pytest.approx(3.87)
    assert d["alarm"] and d["service"] and not d["low_battery"]


def test_clipping_not_wrapping():
    d = payload.decode(payload.encode(_reading(weight_kg=900, t_amb_c=-500, rh_pct=300)))
    assert d["weight_kg"] == pytest.approx(655.35)
    assert d["t_amb_c"] == pytest.approx(-327.68)
    assert d["rh_pct"] == 127.5


def test_negative_weight_clips_to_zero():
    assert payload.decode(payload.encode(_reading(weight_kg=-4)))["weight_kg"] == 0


def test_counter_wraps():
    assert payload.decode(payload.encode(_reading(counter=257)))["counter"] == 1


def test_decode_rejects_bad_length_and_version():
    with pytest.raises(ValueError):
        payload.decode(b"\x10\x00")
    with pytest.raises(ValueError):
        payload.decode(bytes(14))


def test_airtime_known_values():
    # Semtech AN1200.13 / common calculators: 51-byte PHY payload at SF12 BW125 is 2.466 s (hand-derived from the formula)
    assert payload.lora_airtime_s(51, 12) == pytest.approx(2.4658, abs=0.001)
    assert payload.lora_airtime_s(10, 7) == pytest.approx(0.0412, abs=0.0005)


def test_airtime_grows_with_sf():
    t = [payload.lora_airtime_s(27, sf) for sf in range(7, 13)]
    assert t == sorted(t)


def test_airtime_rejects_bad_sf():
    with pytest.raises(ValueError):
        payload.lora_airtime_s(10, 5)


def test_battery_longer_with_slower_reporting():
    a = payload.battery_life_days(15, 9)["life_days"]
    b = payload.battery_life_days(60, 9)["life_days"]
    assert b > a > 100


def test_battery_sf12_costs_more_than_sf7():
    assert payload.battery_life_days(15, 12)["avg_ma"] > payload.battery_life_days(15, 7)["avg_ma"]


def test_vectors_for_js_decoder_and_node_run():
    vecs = []
    for i, r in enumerate([_reading(), _reading(weight_kg=0, t_amb_c=-20.5, delta30_kg=2.0, alarm=True),
                           _reading(weight_kg=120.01, battery_v=3.0, low_battery=True, counter=255)]):
        b = payload.encode(r)
        vecs.append({"hex": b.hex(), "decoded": payload.decode(b)})
        assert i >= 0
    (ROOT / "decoder" / "vectors.json").write_text(json.dumps(vecs, indent=1))
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    out = subprocess.run([node, str(ROOT / "decoder" / "test.js")], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "decoder ok" in out.stdout
    assert not math.isnan(vecs[0]["decoded"]["weight_kg"])
