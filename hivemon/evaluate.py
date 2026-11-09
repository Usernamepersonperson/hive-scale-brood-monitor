"""Measured evaluation: every number in the README comes from these functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hivemon import analytics, sensor, sim

SEASON = (pd.Timestamp("03-01").day_of_year, pd.Timestamp("10-31").day_of_year)  # hive-season, 245 days


def prepare(res: sim.SimResult) -> pd.DataFrame:
    """Device view of a simulated hive with the load cell temperature-compensated."""
    if "prepared" not in res.meta:
        d = res.data.copy()
        d["weight_kg"] = sensor.compensate_rolling(d["weight_raw_kg"], d["t_cell"], d["shock_g"])
        res.meta["prepared"] = d
    return res.meta["prepared"]


def _events(res: sim.SimResult, kind: str) -> pd.DataFrame:
    e = res.events
    return e[e["type"] == kind]


def match_alerts(alerts: pd.DataFrame, events: pd.DataFrame, window_h: float = 3.0):
    """Match alert times to event start times. Returns (tp_delays_h, fn_count, fp_times)."""
    used, delays, fn = set(), [], 0
    for _, ev in events.iterrows():
        hit = None
        for j, a in alerts.iterrows():
            dt = (a["time"] - ev["start"]).total_seconds() / 3600
            if j not in used and -0.01 <= dt <= window_h:
                hit = j
                delays.append(dt)
                used.add(j)
                break
        if hit is None:
            fn += 1
    fp = [a["time"] for j, a in alerts.iterrows() if j not in used]
    return delays, fn, fp


def swarm_eval(cohort: list[sim.SimResult], params: analytics.SwarmParams | None = None) -> dict:
    tp_d, fn, fp, n_ev, n_seasons = [], 0, [], 0, 0
    fp_cause: dict[str, int] = {}
    miss_sizes, hit_sizes = [], []
    for res in cohort:
        d = prepare(res)
        season = d[(d.index.day_of_year >= SEASON[0]) & (d.index.day_of_year <= SEASON[1])]
        al = analytics.detect_swarms(season["weight_kg"], params, season["t_amb"], season["shock_g"],
                                     season["service_mode"])
        ev = _events(res, "swarm")
        delays, f, fps = match_alerts(al, ev)
        tp_d += delays
        fn += f
        fp += fps
        n_ev += len(ev)
        n_seasons += 1
        for _, e in ev.iterrows():
            (hit_sizes if any(abs((a - e["start"]).total_seconds()) < 3 * 3600 for a in al["time"])
             else miss_sizes).append(e["magnitude_kg"])
        for t in fps:
            near = res.events[(res.events["start"] - pd.Timedelta(hours=4) <= t)
                              & (res.events["end"] + pd.Timedelta(hours=4) >= t)]
            kinds = [k for k in near["type"] if k != "swarm"]
            c = kinds[0] if kinds else "unexplained"
            fp_cause[c] = fp_cause.get(c, 0) + 1
    tp = len(tp_d)
    return {
        "n_hive_seasons": n_seasons, "n_swarms": n_ev, "tp": tp, "fn": fn, "fp": len(fp),
        "precision": tp / (tp + len(fp)) if tp + len(fp) else float("nan"),
        "recall": tp / n_ev if n_ev else float("nan"),
        "delay_median_min": float(np.median(tp_d) * 60) if tp_d else float("nan"),
        "delay_p90_min": float(np.quantile(tp_d, 0.9) * 60) if tp_d else float("nan"),
        "false_alerts_per_hive_season": len(fp) / n_seasons if n_seasons else float("nan"),
        "missed_swarm_sizes_kg": [round(x, 2) for x in miss_sizes],
        "false_alert_causes": fp_cause,
    }
