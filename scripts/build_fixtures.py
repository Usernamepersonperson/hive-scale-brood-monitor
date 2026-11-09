"""Rebuild the small data fixtures in hivemon/data from the public sources.

Run from the repo root:  python scripts/build_fixtures.py [--audio-dir DIR]

Sources (all downloaded at build time, nothing proprietary):
  * MeteoSwiss SwissMetNet hourly open data (SMA, BER, ENG), https://data.geo.admin.ch
  * Zenodo 5148318 (CC BY 4.0): EFSA-funded field data, hive-scale weight + beekeeper log
  * Zenodo 20399470 (CC BY 4.0): hive temperature/humidity/weight, Apis mellifera, Brazil
  * Zenodo 1321278 (CC BY 4.0): annotated beehive audio (queen present / absent)
The audio step needs the WAV files downloaded first (see docs/DATA.md); it stores only features.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "hivemon" / "data"
UA = {"User-Agent": "hive-scale-brood-monitor/0.1 (fixture build)"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 - fixed https sources
        return r.read()


def zenodo(rec: int, name: str) -> bytes:
    return fetch(f"https://zenodo.org/api/records/{rec}/files/{urllib.parse.quote(name)}/content")


def meteoswiss() -> None:
    cols = {"tre200h0": "temp_c", "rre150h0": "precip_mm", "sre000h0": "sun_min",
            "gre000h0": "glob_rad_wm2", "fu3010h0": "wind_ms"}
    out = []
    for st in ("sma", "ber", "eng"):
        url = (f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/{st}/"
               f"ogd-smn_{st}_h_historical_2020-2029.csv")
        df = pd.read_csv(io.BytesIO(fetch(url)), sep=";", encoding="latin1")
        df["time_utc"] = pd.to_datetime(df["reference_timestamp"], format="%d.%m.%Y %H:%M")
        df = df[(df.time_utc >= "2023-01-01") & (df.time_utc < "2025-01-01")]
        df = df.rename(columns=cols)
        df["station"] = st.upper()
        out.append(df[["station", "time_utc", *cols.values()]])
    res = pd.concat(out)
    res.to_csv(DATA / "meteoswiss_hourly_2023_2024.csv.gz", index=False, float_format="%.2f")
    print("meteoswiss", res.shape)


def mustb() -> None:
    z = zipfile.ZipFile(io.BytesIO(zenodo(5148318, "Table_V_Colony_inspection.zip")))
    t5 = pd.read_csv(z.open("Table_V_Colony_inspection.csv"), low_memory=False)
    w = t5[t5.paramText == "Colony weight"].copy()
    w = w[w.sampY.isin([2019, 2020])]
    w["time"] = pd.to_datetime(dict(year=w.sampY, month=w.sampM, day=w.sampD, hour=w.sampT))
    w = w.rename(columns={"siteNo": "site", "hiveNo": "hive", "resVal": "weight_kg"})
    w = w[["site", "hive", "time", "weight_kg"]].sort_values(["site", "hive", "time"])
    w.to_csv(DATA / "mustb_weight_hourly.csv.gz", index=False)
    ev = pd.read_csv(io.BytesIO(zenodo(5148318, "Table_IV_Colony_management.csv")))
    ev["date"] = pd.to_datetime(dict(year=ev.eventY, month=ev.eventM, day=ev.eventD))
    ev = ev.rename(columns={"siteNo": "site", "hiveNo": "hive"})
    ev = ev[["site", "hive", "date", "eventType", "inoutType", "inoutQuantity", "eventDescription"]]
    ev["eventDescription"] = ev["eventDescription"].fillna("").str.replace('"', "").str.strip()
    ev.to_csv(DATA / "mustb_events.csv", index=False)
    print("mustb", w.shape, ev.shape)


def ufc() -> None:
    frames = []
    for name, tag in (("dadosColmeia1Apis.csv", "A1"), ("dadosColmeia2Apis.csv", "A2")):
        d = pd.read_csv(io.BytesIO(zenodo(20399470, name)), parse_dates=["time"])
        d["hive"] = tag
        frames.append(d)
    d = pd.concat(frames).sort_values(["hive", "time"])
    d = d.rename(columns={"temperature": "t_brood", "ext_temperature": "t_ext",
                          "humidity": "rh_in", "ext_humidity": "rh_ext", "weight": "weight_kg"})
    d = d[["hive", "time", "weight_kg", "t_brood", "rh_in", "t_ext", "rh_ext"]]
    d.to_csv(DATA / "ufc_hives.csv.gz", index=False, float_format="%.3f")
    print("ufc", d.shape)


def audio(audio_dir: Path) -> None:
    from hivemon import acoustics

    rows = acoustics.extract_dir(audio_dir)
    rows.to_csv(DATA / "audio_features.csv", index=False, float_format="%.5f")
    print("audio", rows.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir")
    ap.add_argument("--skip-net", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if not a.skip_net:
        meteoswiss()
        mustb()
        ufc()
    if a.audio_dir:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        audio(Path(a.audio_dir))
