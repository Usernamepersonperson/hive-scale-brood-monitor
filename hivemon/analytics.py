"""Hive analytics on a regular, local-time-indexed series.

All functions take pandas objects with a tz-aware (or naive local) DatetimeIndex and never look
at simulator truth columns. Weights are in kg, temperatures in deg C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------------------------
# 1. daily weight decomposition: nectar flow vs consumption vs evaporation
# ---------------------------------------------------------------------------------------------

MORNING_H, EVENING_H = 5, 21     # foragers in the box at both times
NIGHT_EVAP_SHARE = 0.7           # assumed share of daily evaporation that happens at night
QUIET_GAIN_KG = 0.2              # a day with less gross gain than this is "quiet"


def _at_hour(w: pd.Series, hour: int) -> pd.Series:
    """Median of the samples in the hour starting at `hour`, indexed by local date."""
    sel = w[w.index.hour == hour]
    return sel.groupby(sel.index.normalize()).median()


def daily_decomposition(weight: pd.Series, max_gap_days: int = 2) -> pd.DataFrame:
    """Split each day into nectar intake, consumption and evaporation (kg/day).

    day change   dW_day   = W(21:00) - W(05:00)         foraging gain minus losses over 16 h
    night change dW_night = W(05:00, next day) - W(21:00)  losses only (8 h)
    consumption rate c (kg/h) = rolling median of night loss rate on quiet days (little gain)
    evaporation = excess night loss above c, scaled by the assumed night share
    nectar = net 24 h change + consumption + evaporation
    """
    m = _at_hour(weight, MORNING_H)
    e = _at_hour(weight, EVENING_H)
    df = pd.DataFrame({"w_morning": m, "w_evening": e})
    df["day_change"] = df["w_evening"] - df["w_morning"]
    df["night_change"] = df["w_morning"].shift(-1) - df["w_evening"]
    hours_night = 24 - (EVENING_H - MORNING_H)
    df["night_loss_rate"] = -df["night_change"] / hours_night
    quiet = df["day_change"] < QUIET_GAIN_KG
    c = df["night_loss_rate"].where(quiet).clip(lower=0)
    df["cons_rate"] = c.rolling(14, min_periods=3).median().ffill(limit=max_gap_days * 7).bfill(limit=14)
    df["cons_rate"] = df["cons_rate"].fillna(0.0)
    df["consumption_kg"] = df["cons_rate"] * 24
    excess = (df["night_loss_rate"] - df["cons_rate"]).clip(lower=0) * hours_night
    df["evaporation_kg"] = excess / NIGHT_EVAP_SHARE
    df["net_24h_kg"] = df["w_morning"].shift(-1) - df["w_morning"]
    df["nectar_kg"] = (df["net_24h_kg"] + df["consumption_kg"] + df["evaporation_kg"]).clip(lower=0)
    return df


# ---------------------------------------------------------------------------------------------
# 2. swarm detection
# ---------------------------------------------------------------------------------------------

@dataclass
class SwarmParams:
    drop_kg: float = 0.9          # minimum drop vs the reference window
    max_drop_kg: float = 4.0      # larger steps are handling / theft, not a swarm
    ref_from_min: int = 150       # reference window: median of samples from t-150 ... t-45 min
    ref_to_min: int = 45
    confirm_samples: int = 2      # samples in a row that must be below the reference
    hours: tuple[float, float] = (9.5, 17.0)   # local hours in which swarms are accepted
    min_temp_c: float | None = 13.0            # ambient temperature gate (skipped if no series)
    fast_frac: float = 0.7        # share of the drop that must have happened within ref_to_min
    refractory_h: float = 24.0
    shock_g: float = 0.5          # handling suppresses the alert for `handling_h` hours
    handling_h: float = 1.0


def _step_minutes(idx: pd.DatetimeIndex) -> float:
    return float(np.median(np.diff(idx)) / pd.Timedelta(minutes=1))


def detect_swarms(weight: pd.Series, p: SwarmParams | None = None, t_amb: pd.Series | None = None,
                  shock: pd.Series | None = None, service: pd.Series | None = None,
                  mask: pd.Series | None = None) -> pd.DataFrame:
    """Causal swarm-drop detector. Returns one row per alert: time, drop_kg, delay-relevant fields.

    The alert time is the sample at which the drop has persisted for `confirm_samples` samples,
    so a real swarm is reported one or two device intervals after it happened. `mask` (bool
    series, True = suppress) lets the caller exclude logged handling days when no accelerometer
    is available (used for the real-data run).
    """
    p = p or SwarmParams()
    step = _step_minutes(weight.index)
    w = weight.astype(float)
    a, b = max(1, int(round(p.ref_to_min / step))), max(2, int(round(p.ref_from_min / step)))
    # reference = rolling median of the window [t-b, t-a]
    ref = w.shift(a).rolling(b - a + 1, min_periods=max(2, (b - a + 1) // 2)).median()
    drop = ref - w
    fast = (w.shift(a) - w) >= p.fast_frac * p.drop_kg     # a swarm leaves within minutes, robbing does not
    recent_big = (drop > p.max_drop_kg).astype(int).rolling(b, min_periods=1).max() > 0  # step: handling
    below = (drop >= p.drop_kg) & (drop <= p.max_drop_kg) & fast & ~recent_big
    k = max(1, p.confirm_samples)
    persist = below.rolling(k).sum() >= k
    hr = w.index.hour + w.index.minute / 60
    ok = persist & (hr >= p.hours[0]) & (hr <= p.hours[1])
    if t_amb is not None and p.min_temp_c is not None:
        ok &= t_amb.reindex(w.index).ffill() >= p.min_temp_c
    hs = max(1, int(round(p.handling_h * 60 / step)))
    if shock is not None:
        handled = (shock.reindex(w.index).fillna(0) >= p.shock_g).astype(int).rolling(hs, min_periods=1).max() > 0
        ok &= ~handled
    if service is not None:
        ok &= ~(service.reindex(w.index).fillna(False).astype(int).rolling(hs, min_periods=1).max() > 0)
    if mask is not None:
        ok &= ~mask.reindex(w.index).fillna(False)
    alerts, last = [], None
    for t in w.index[ok.to_numpy()]:
        if last is not None and (t - last) < pd.Timedelta(hours=p.refractory_h):
            continue
        d = float(drop.loc[t])
        alerts.append({"time": t, "drop_kg": d})
        last = t
    return pd.DataFrame(alerts, columns=["time", "drop_kg"])


# ---------------------------------------------------------------------------------------------
# 3. honey-harvest readiness
# ---------------------------------------------------------------------------------------------

def harvest_readiness(dec: pd.DataFrame, min_gain_kg: float = 6.0, flow_end_kg: float = 0.6,
                      evap_done_kg: float = 0.25, days: int = 3) -> pd.DataFrame:
    """Flag days on which supers look ready: enough gain, flow over, ripening finished.

    gain       = rise of the evening weight over the running minimum since the last big step down
    flow over  = mean nectar of the last `days` days below `flow_end_kg`
    ripe       = mean evaporation of the last `days` days below `evap_done_kg`
    """
    w = dec["w_evening"]
    drops = w.diff() < -4.0  # a harvest or other removal resets the baseline
    base = w.copy()
    run_min = []
    cur = np.inf
    for v, reset in zip(w.to_numpy(), drops.to_numpy(), strict=True):
        if reset or not np.isfinite(cur):
            cur = v
        cur = min(cur, v)
        run_min.append(cur)
    base[:] = run_min
    out = pd.DataFrame(index=dec.index)
    out["gain_kg"] = w - base
    out["flow_over"] = dec["nectar_kg"].rolling(days).mean() < flow_end_kg
    out["ripe"] = dec["evaporation_kg"].rolling(days).mean() < evap_done_kg
    out["ready"] = (out["gain_kg"] >= min_gain_kg) & out["flow_over"] & out["ripe"]
    return out


# ---------------------------------------------------------------------------------------------
# 4. winter food reserve forecast
# ---------------------------------------------------------------------------------------------

def reserve_forecast(w_morning: pd.Series, tare_kg: float, reserve_min_kg: float = 3.0,
                     window_days: int = 14) -> pd.DataFrame:
    """Days of stores left at the current consumption rate.

    stores = morning weight - tare; rate = Theil-Sen slope of the last `window_days` mornings.
    Rate floors at 5 g/day. The forecast is a straight line: it ignores the consumption rise when
    brood rearing restarts, so it is optimistic in January/February (see README).
    """
    rows = []
    vals = w_morning.dropna()
    for i in range(window_days, len(vals)):
        seg = vals.iloc[i - window_days:i + 1]
        x = np.arange(len(seg), dtype=float)
        slope = float(stats.theilslopes(seg.to_numpy(), x)[0])  # kg/day
        rate = max(-slope, 0.005)
        stores = float(seg.iloc[-1] - tare_kg)
        rows.append({"date": vals.index[i], "stores_kg": stores, "rate_kg_day": rate,
                     "days_left": max(0.0, (stores - reserve_min_kg) / rate)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["stores_kg", "rate_kg_day", "days_left"])


# ---------------------------------------------------------------------------------------------
# 5. brood presence from brood-nest temperature stability
# ---------------------------------------------------------------------------------------------

BROOD_TEMP_RANGE = (32.0, 36.5)  # deg C, plausible regulated brood-nest band (literature ~34-35)


def daily_brood_features(t_brood: pd.Series) -> pd.DataFrame:
    g = t_brood.groupby(t_brood.index.normalize())
    f = pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.count()})
    f = f[f["n"] >= 0.8 * f["n"].median()]
    return f


def brood_presence(t_brood: pd.Series, max_std: float = 1.0, smooth_days: int = 3) -> pd.DataFrame:
    """Daily brood flag: mean inside 32-36.5 C and a 3-day median of the daily std below `max_std`."""
    f = daily_brood_features(t_brood)
    f["std_s"] = f["std"].rolling(smooth_days, min_periods=1).median()
    f["mean_s"] = f["mean"].rolling(smooth_days, min_periods=1).median()
    f["brood"] = (f["std_s"] <= max_std) & f["mean_s"].between(*BROOD_TEMP_RANGE)
    return f


def brood_break_alerts(flags: pd.Series, min_days: int = 5, season: tuple[int, int] = (100, 260)) -> list:
    """Dates on which brood has been absent for `min_days` in a row inside the brood season."""
    out, run = [], 0
    for d, v in flags.items():
        doy = d.day_of_year
        if season[0] <= doy <= season[1] and not v:
            run += 1
            if run == min_days:
                out.append(d)
        else:
            run = 0
    return out


# ---------------------------------------------------------------------------------------------
# 6. theft / tipping alarm from the accelerometer summary
# ---------------------------------------------------------------------------------------------

@dataclass
class TiltParams:
    tilt_deg: float = 12.0
    shock_g: float = 1.8
    theft_drop_frac: float = 0.6   # weight falls by this fraction within 2 samples -> "theft"
    refractory_h: float = 6.0


def tilt_alarms(tilt: pd.Series, shock: pd.Series, weight: pd.Series | None = None,
                service: pd.Series | None = None, p: TiltParams | None = None) -> pd.DataFrame:
    p = p or TiltParams()
    hit = (tilt >= p.tilt_deg) | (shock >= p.shock_g)
    if service is not None:
        hit &= ~service.reindex(tilt.index).fillna(False)
    rows, last = [], None
    for t in tilt.index[hit.to_numpy()]:
        if last is not None and (t - last) < pd.Timedelta(hours=p.refractory_h):
            continue
        kind = "tip"
        if weight is not None:
            j = weight.index.get_loc(t)
            before = weight.iloc[max(0, j - 6):max(1, j)].median()
            after = weight.iloc[j + 1:j + 3].median() if j + 1 < len(weight) else np.nan
            if before > 5 and np.isfinite(after) and after < (1 - p.theft_drop_frac) * before:
                kind = "theft"
        rows.append({"time": t, "kind": kind})
        last = t
    return pd.DataFrame(rows, columns=["time", "kind"])


# ---------------------------------------------------------------------------------------------
# 7. robbing hint
# ---------------------------------------------------------------------------------------------

def robbing_hint(dec: pd.DataFrame, loss_kg: float = 1.0, days: int = 2,
                 season: tuple[int, int] = (170, 280)) -> list:
    """Dates with a net daytime loss >= `loss_kg` for `days` consecutive days in the dearth season."""
    out, run = [], 0
    for d, v in dec["day_change"].items():
        if season[0] <= d.day_of_year <= season[1] and v <= -loss_kg:
            run += 1
            if run == days:
                out.append(d)
        else:
            run = 0
    return out
