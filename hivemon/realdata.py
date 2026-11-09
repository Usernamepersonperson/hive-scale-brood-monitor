"""Evaluation on REAL open data (small derived fixtures, attribution in README / docs/DATA.md).

* Hive-scale weight + beekeeper log: Zenodo 10.5281/zenodo.5148318 (CC BY 4.0), field colonies in
  Denmark and Portugal, 2019-2020, hourly weight; two beekeeper-logged swarms.
* Brood-nest temperature, humidity, weight: Zenodo 10.5281/zenodo.20399470 (CC BY 4.0), two Apis
  mellifera hives, Ceara, Brazil, 10-minute readings; no inspection labels.
"""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

from hivemon import analytics

LOG_HANDLING = {"INPUT", "OUTPUT", "SAMPLE", "OTHER", "CLINICAL SIGNS", "QUEEN LOSS"}


def _csv(name: str, **kw) -> pd.DataFrame:
    with resources.files("hivemon").joinpath(f"data/{name}").open("rb") as f:
        return pd.read_csv(f, **kw)


def load_mustb_weight() -> pd.DataFrame:
    d = _csv("mustb_weight_hourly.csv.gz", parse_dates=["time"], compression="gzip")
    return d[(d["time"] >= "2019-03-01") & (d["time"] < "2020-12-31")]


def load_mustb_events() -> pd.DataFrame:
    return _csv("mustb_events.csv", parse_dates=["date"])


def load_ufc() -> pd.DataFrame:
    d = _csv("ufc_hives.csv.gz", compression="gzip")
    d["time"] = pd.to_datetime(d["time"], format="mixed")
    return d


def hourly_series(w: pd.DataFrame, site: int, hive: int) -> pd.Series:
    s = w[(w.site == site) & (w.hive == hive)].drop_duplicates("time").set_index("time")["weight_kg"].sort_index()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="1h"))


def real_swarm_eval(drop_kg: float = 0.9) -> dict:
    """Run the swarm detector on every real hive; compare with the two beekeeper-logged swarms.

    No accelerometer exists in this data set, so alerts on a day with any other logged handling
    event of that hive (inspection samples, feeding, treatment, queen events) are suppressed,
    mimicking the device's shock/service-mode suppression. Hours are not gated because the
    time-of-day convention of the data set is not documented.
    """
    w = load_mustb_weight()
    ev = load_mustb_events()
    swarm_ev = ev[ev.eventType == "SWARM"]
    p = analytics.SwarmParams(drop_kg=drop_kg, hours=(0.0, 24.0), min_temp_c=None, ref_from_min=360,
                              ref_to_min=120, confirm_samples=1, fast_frac=0.7)
    n_hive_seasons = 0
    alerts_all, alerts_open = [], []
    hits = []
    for (site, hive), _ in w.groupby(["site", "hive"]):
        s = hourly_series(w, site, hive)
        if s.notna().sum() < 24 * 30:
            continue
        n_hive_seasons += max(1, round(s.notna().sum() / (24 * 245)))
        h_ev = ev[(ev.site == site) & (ev.hive == hive) & ev.eventType.isin(LOG_HANDLING)]
        busy_days = set(h_ev["date"].dt.normalize())
        mask = pd.Series(s.index.normalize().isin(busy_days), index=s.index)
        al = analytics.detect_swarms(s, p, mask=mask)
        al_raw = analytics.detect_swarms(s, p)
        for _, a in al_raw.iterrows():
            alerts_all.append({"site": site, "hive": hive, **a.to_dict()})
        for _, a in al.iterrows():
            alerts_open.append({"site": site, "hive": hive, **a.to_dict()})
        for _, e in swarm_ev[(swarm_ev.site == site) & (swarm_ev.hive == hive)].iterrows():
            near = al_raw[(al_raw.time >= e.date - pd.Timedelta(days=1)) & (al_raw.time <= e.date + pd.Timedelta(days=1))]
            hits.append({"date": e.date, "site": site, "hive": hive, "detected": len(near) > 0,
                         "drop_kg": float(near.drop_kg.max()) if len(near) else float("nan"),
                         "alert_time": near.time.iloc[0] if len(near) else None})
    return {
        "n_hives": int(w.groupby(["site", "hive"]).ngroups),
        "hive_seasons_approx": int(n_hive_seasons),
        "logged_swarms": len(swarm_ev), "logged_swarms_detected": int(sum(h["detected"] for h in hits)),
        "swarm_hits": hits,
        "alerts_total": len(alerts_all), "alerts_not_on_logged_handling_day": len(alerts_open),
        "alerts_per_hive_season_not_on_handling_day": len(alerts_open) / max(1, n_hive_seasons),
    }


def real_forecast_eval(horizon_days: int = 30,
                       origins: tuple[str, ...] = ("2019-09-15", "2019-10-01", "2019-10-15", "2019-11-01",
                                                   "2019-11-15", "2019-12-01")) -> dict:
    """Real food-reserve forecast check: straight-line weight extrapolation vs weight 30 days later.

    Pools several autumn origins. Hive-origins with a logged feeding / honey / comb / bee input or
    output between 14 days before the origin and the target are skipped, because no forecast can
    know about them. Reports weight error in kg (tare of these hives is unknown, so not stores).
    """
    w = load_mustb_weight()
    ev = load_mustb_events()
    skip_types = ["Sucrose", "Honey", "Super + 10 frames", "Food comb", "Brood comb",
                  "Foundation comb", "Live bees", "Empty frame"]
    errs = []
    for o in origins:
        t0 = pd.Timestamp(o)
        t1 = t0 + pd.Timedelta(days=horizon_days)
        for (site, hive), _ in w.groupby(["site", "hive"]):
            s = hourly_series(w, site, hive)
            mm = s[s.index.hour == 5]
            m = mm.groupby(mm.index.normalize()).median()
            if m.loc[t0 - pd.Timedelta(days=14):t0].count() < 10 or m.loc[t1 - pd.Timedelta(days=2):t1].count() < 1:
                continue
            busy = ev[(ev.site == site) & (ev.hive == hive) & (ev.date >= t0 - pd.Timedelta(days=14))
                      & (ev.date <= t1) & ev.inoutType.isin(skip_types)]
            if len(busy):
                continue
            fc = analytics.reserve_forecast(m.loc[:t0], tare_kg=0.0, reserve_min_kg=0.0, window_days=14)
            if fc.empty:
                continue
            rate = fc["rate_kg_day"].iloc[-1]
            pred = float(m.loc[:t0].iloc[-1] - rate * horizon_days)
            actual = float(m.loc[t1 - pd.Timedelta(days=2):t1].median())
            errs.append({"origin": o, "site": site, "hive": hive, "err_kg": pred - actual,
                         "rate_g_day": rate * 1000,
                         "actual_rate_g_day": (m.loc[:t0].iloc[-1] - actual) / horizon_days * 1000})
    e = pd.DataFrame(errs)
    if e.empty:
        return {"n_forecasts": 0}
    return {"n_forecasts": int(len(e)), "n_hives": int(e.groupby(["site", "hive"]).ngroups),
            "horizon_days": horizon_days, "mae_kg": float(e.err_kg.abs().mean()),
            "bias_kg": float(e.err_kg.mean()),
            "median_actual_rate_g_day": float(e.actual_rate_g_day.median()),
            "median_forecast_rate_g_day": float(e.rate_g_day.median())}


def real_brood_temperature() -> dict:
    """Brood-nest temperature statistics and the brood flag on the two real Brazilian hives (no labels)."""
    u = load_ufc()
    out = {}
    for hive, g in u.groupby("hive"):
        t = g.set_index("time")["t_brood"].sort_index()
        f = analytics.brood_presence(t)
        out[hive] = {"days": int(len(f)), "median_daily_mean_c": float(f["mean"].median()),
                     "median_daily_std_c": float(f["std"].median()),
                     "share_days_in_34_35_5": float(((f["mean"] >= 34.0) & (f["mean"] <= 35.5)).mean()),
                     "share_days_flagged_brood": float(f["brood"].mean()),
                     "corr_brood_vs_ambient": float(np.corrcoef(
                         g["t_brood"].to_numpy(), g["t_ext"].to_numpy())[0, 1])}
    return out
