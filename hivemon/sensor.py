"""Load-cell temperature compensation learned from the hive's own quiet nights.

A bar-type load cell reads  W_raw = W * (1 + a*dT) + z*dT,  dT = T_cell - T_ref.
Between consecutive night hours the true weight changes slowly (consumption, evaporation), while
T_cell often moves by 1-3 K, so the slope of hourly weight change against hourly cell-temperature
change estimates k = a*W + z (kg per K). The corrected reading is W_raw - k * (T_cell - T_ref).
k is re-estimated on a rolling window because the span term scales with the load.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

T_REF = 20.0
NIGHT = (23, 4)  # local hours: night window used for the fit


def _night_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    h = idx.hour
    return (h >= NIGHT[0]) | (h < NIGHT[1])


def estimate_k(weight: pd.Series, t_cell: pd.Series, shock: pd.Series | None = None,
               max_pairs: int = 4000) -> float:
    """Robust (Theil-Sen) slope of hourly weight change vs hourly cell-temperature change, kg/K.

    Returns 0.0 when there is too little temperature variation to fit (|dT| < 0.3 K everywhere).
    """
    df = pd.DataFrame({"w": weight, "t": t_cell})
    if shock is not None:
        df.loc[shock.reindex(df.index).fillna(0) > 0.4, ["w", "t"]] = np.nan
    h = df.resample("1h").mean()
    dw, dt = h["w"].diff(), h["t"].diff()
    ok = _night_mask(h.index) & dw.notna() & dt.notna() & (dw.abs() < 0.5) & (dt.abs() >= 0.3)
    if ok.sum() < 20:
        return 0.0
    # remove the systematic night profile (evaporation and cooling both decay through the night):
    # subtract the per-hour-of-night median from both series so only night-to-night weather varies
    hod = h.index.hour
    dw_c = dw[ok] - dw[ok].groupby(hod[ok.to_numpy()]).transform("median")
    dt_c = dt[ok] - dt[ok].groupby(hod[ok.to_numpy()]).transform("median")
    keep = dt_c.abs() >= 0.2
    if keep.sum() < 20:
        return 0.0
    x, y = dt_c[keep].to_numpy(), dw_c[keep].to_numpy()
    if len(x) > max_pairs:
        sel = np.random.default_rng(0).choice(len(x), max_pairs, replace=False)
        x, y = x[sel], y[sel]
    return float(stats.theilslopes(y, x)[0])


def compensate(weight: pd.Series, t_cell: pd.Series, k: float, t_ref: float = T_REF) -> pd.Series:
    return weight - k * (t_cell - t_ref)


def compensate_rolling(weight: pd.Series, t_cell: pd.Series, shock: pd.Series | None = None,
                       window_days: int = 28) -> pd.Series:
    """Piecewise compensation: k is fitted per `window_days` block (using that block and its predecessor)."""
    out = weight.copy()
    days = weight.index.normalize()
    start = days.min()
    n_blocks = int((days.max() - start).days // window_days) + 1
    prev_k = 0.0
    for b in range(n_blocks):
        lo = start + pd.Timedelta(days=b * window_days)
        hi = lo + pd.Timedelta(days=window_days)
        fit = (days >= lo - pd.Timedelta(days=window_days)) & (days < hi)
        k = estimate_k(weight[fit], t_cell[fit], shock[fit] if shock is not None else None)
        if k == 0.0:
            k = prev_k
        prev_k = k
        blk = (days >= lo) & (days < hi)
        out[blk] = weight[blk] - k * (t_cell[blk] - T_REF)
    return out
