"""MeteoSwiss SwissMetNet hourly open data (fixture), resampled to the device cadence.

Source: MeteoSwiss (hourly SwissMetNet measurements, open government data, attribution required).
Stations in the fixture: SMA (Zurich / Fluntern, 556 m region), BER (Bern / Zollikofen, 553 m),
ENG (Engelberg, 1036 m).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import pandas as pd

STATIONS = {"SMA": "Zurich / Fluntern", "BER": "Bern / Zollikofen", "ENG": "Engelberg"}
STEP_MIN = 15
TZ = "Europe/Zurich"
COLS = ["temp_c", "precip_mm", "sun_min", "glob_rad_wm2", "wind_ms"]


@lru_cache(maxsize=1)
def load_hourly() -> pd.DataFrame:
    with resources.files("hivemon").joinpath("data/meteoswiss_hourly_2023_2024.csv.gz").open("rb") as f:
        df = pd.read_csv(f, parse_dates=["time_utc"], compression="gzip")
    return df


def station_year(station: str, year: int, step_min: int = STEP_MIN) -> pd.DataFrame:
    """Local-time (Europe/Zurich) weather at `step_min` resolution for one calendar year.

    Temperature, radiation and wind are linearly interpolated; precipitation (mm per hour) is
    spread evenly over the quarter-hours of its hour. Short gaps are filled, long gaps raise.
    """
    df = load_hourly()
    df = df[df.station == station.upper()]
    if df.empty:
        raise KeyError(f"unknown station {station!r}; have {sorted(STATIONS)}")
    s = df.set_index(df.time_utc.dt.tz_localize("UTC")).sort_index()[COLS]
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min(), s.index.max(), freq="1h", tz="UTC")
    s = s.reindex(full)
    if s[["temp_c", "glob_rad_wm2"]].isna().sum().max() > 72:
        raise ValueError("too many missing hours in weather fixture")
    s = s.interpolate(limit=12, limit_direction="both")
    s["precip_mm"] = s["precip_mm"].fillna(0.0)
    s["sun_min"] = s["sun_min"].fillna(0.0)
    s.index = s.index.tz_convert(TZ)
    s = s[(s.index.year == year)]
    if s.empty:
        raise KeyError(f"year {year} not in fixture (2023, 2024)")
    k = 60 // step_min
    idx = pd.date_range(s.index[0], s.index[-1] + pd.Timedelta(minutes=60 - step_min), freq=f"{step_min}min")
    out = s[["temp_c", "glob_rad_wm2", "wind_ms"]].reindex(idx).interpolate()
    out["precip_mm"] = (s["precip_mm"].reindex(idx).ffill() / k).fillna(0.0)
    out["sun_min"] = s["sun_min"].reindex(idx).ffill().fillna(0.0)
    return out
