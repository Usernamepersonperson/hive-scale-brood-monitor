"""Remaining measurements on the simulated cohort, and the single `run_all` entry point."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

from hivemon import acoustics, analytics, payload, realdata, sim
from hivemon.evaluate import SEASON, prepare, swarm_eval


def sensor_eval(cohort: list[sim.SimResult]) -> dict:
    """Load-cell temperature compensation: high-passed error against truth, quiet samples only."""
    raw, comp = [], []
    for res in cohort:
        d = prepare(res)
        quiet = d["shock_g"] < 0.3
        for col, acc in (("weight_raw_kg", raw), ("weight_kg", comp)):
            e = d[col] - d["true_weight_kg"]
            hp = e - e.rolling(96 * 7, center=True, min_periods=10).median()
            acc.append(hp[quiet].dropna().abs())
    r, c = pd.concat(raw), pd.concat(comp)
    return {"p90_abs_error_raw_kg": float(r.quantile(0.9)),
            "p90_abs_error_compensated_kg": float(c.quantile(0.9)),
            "median_abs_error_raw_kg": float(r.median()),
            "median_abs_error_compensated_kg": float(c.median())}


def decomposition_eval(cohort: list[sim.SimResult]) -> dict:
    errs: dict[str, list] = {"nectar": [], "consumption": [], "evaporation": []}
    flow_err, cx, cy = [], [], []
    for res in cohort:
        d = prepare(res)
        dec = analytics.daily_decomposition(d["weight_kg"])
        sh = (d["shock_g"] > 0.4).groupby(d.index.normalize()).any()
        t = d.copy()
        t.index = t.index - pd.Timedelta(hours=5)
        tr = t[["true_nectar_kg", "true_cons_kg", "true_evap_kg"]].groupby(t.index.normalize()).sum()
        j = dec.join(tr, how="inner")
        j = j[(j.index.day_of_year >= SEASON[0]) & (j.index.day_of_year <= SEASON[1])]
        j = j[~sh.reindex(j.index).fillna(False).to_numpy()].dropna(subset=["nectar_kg"])
        for k, a, b in (("nectar", "nectar_kg", "true_nectar_kg"),
                        ("consumption", "consumption_kg", "true_cons_kg"),
                        ("evaporation", "evaporation_kg", "true_evap_kg")):
            errs[k].append((j[a] - j[b]).abs())
        fl = j[j["true_nectar_kg"] > 1.0]
        flow_err.append((fl["nectar_kg"] - fl["true_nectar_kg"]).abs())
        cx.append(j["nectar_kg"])
        cy.append(j["true_nectar_kg"])
    x, y = pd.concat(cx), pd.concat(cy)
    out = {f"mae_{k}_kg_day": float(pd.concat(v).mean()) for k, v in errs.items()}
    out["mae_nectar_flow_days_kg_day"] = float(pd.concat(flow_err).mean())
    out["nectar_corr"] = float(np.corrcoef(x, y)[0, 1])
    out["mean_true_nectar_kg_day"] = float(y.mean())
    return out


def harvest_eval(cohort: list[sim.SimResult]) -> dict:
    tp = fp = fn = tn = premature = flagged = 0
    for res in cohort:
        d = prepare(res)
        dec = analytics.daily_decomposition(d["weight_kg"])
        rd = analytics.harvest_readiness(dec)
        ev = d[d.index.hour == 21]
        ev.index = ev.index.normalize()
        j = rd.join(ev[["true_unripe_kg", "true_store_kg"]], how="inner")
        j = j[(j.index.day_of_year >= 152) & (j.index.day_of_year <= 304)]
        gain = j["true_store_kg"] - j["true_store_kg"].cummin()
        truth = (j["true_unripe_kg"] <= 0.5) & (gain >= 6.0)
        pred = j["ready"].astype(bool)
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())
        premature += int((pred & (j["true_unripe_kg"] > 1.0)).sum())
        flagged += int(pred.sum())
    n = tp + fp + fn + tn
    return {"day_accuracy": (tp + tn) / n, "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"), "flagged_days": flagged,
            "premature_flag_share": premature / flagged if flagged else float("nan")}


def reserve_eval(cohort: list[sim.SimResult], horizon: int = 30) -> dict:
    rows = []
    for res in cohort:
        d = prepare(res)
        w05 = d["weight_kg"][d.index.hour == 5]
        w05.index = w05.index.normalize()
        w05 = w05.groupby(level=0).median()
        fc = analytics.reserve_forecast(w05, res.meta["tare_kg"], 3.0)
        busy = res.events[res.events["type"].isin(["harvest", "feeding"])]
        for md in ("01-15", "02-15", "03-01", "10-15", "11-15"):
            t0 = pd.Timestamp(f"{res.year}-{md}", tz=d.index.tz)
            t1 = t0 + pd.Timedelta(days=horizon)
            if t0 not in fc.index or t1 not in w05.index:
                continue
            if ((busy["start"] >= t0) & (busy["start"] <= t1)).any():
                continue
            r = fc.loc[t0]
            pred_w = w05.loc[t0] - r["rate_kg_day"] * horizon
            rows.append({"origin": md, "err_kg": float(pred_w - w05.loc[t1])})
    e = pd.DataFrame(rows)
    by = e.groupby("origin")["err_kg"].agg(n="size", mae_kg=lambda x: float(x.abs().mean()), bias_kg="mean")
    return {"n_forecasts": int(len(e)), "horizon_days": horizon, "mae_kg": float(e.err_kg.abs().mean()),
            "bias_kg": float(e.err_kg.mean()), "by_origin": by.round(3).to_dict("index")}


def brood_eval(cohort: list[sim.SimResult]) -> dict:
    tp = fp = fn = tn = 0
    delays, fp_alerts, qn, n_other = [], 0, 0, 0
    for res in cohort:
        d = prepare(res)
        f = analytics.brood_presence(d["t_brood"])
        truth = d["true_brood"].groupby(d.index.normalize()).mean().reindex(f.index) > 0.05
        pred = f["brood"]
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())
        al = analytics.brood_break_alerts(f["brood"])
        q = res.events[res.events["type"] == "queen_loss"]
        if len(q):
            qn += 1
            q0 = q["start"].iloc[0].normalize()
            after = [a for a in al if a >= q0]
            if after:
                delays.append((after[0] - q0).days)
        else:
            n_other += 1
            fp_alerts += len(al)
    n = tp + fp + fn + tn
    return {"day_accuracy": (tp + tn) / n, "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "queenless_hives": qn, "queenless_alerted": len(delays),
            "queenless_delay_median_days": float(np.median(delays)) if delays else float("nan"),
            "break_alerts_per_other_hive_year": fp_alerts / max(n_other, 1)}


def tilt_eval(cohort: list[sim.SimResult]) -> dict:
    tp = n_ev = fp = kinds_ok = 0
    delays = []
    for res in cohort:
        d = prepare(res)
        al = analytics.tilt_alarms(d["tilt_dev_deg"], d["shock_g"], d["weight_kg"], d["service_mode"])
        ev = res.events[res.events["type"].isin(["theft", "tipping"])]
        used: set = set()
        for _, e in ev.iterrows():
            n_ev += 1
            for j, a in al.iterrows():
                dt = (a["time"] - e["start"]).total_seconds() / 3600
                if j not in used and -0.01 <= dt <= 1.0:
                    used.add(j)
                    tp += 1
                    delays.append(dt * 60)
                    kinds_ok += int(a["kind"] == ("theft" if e["type"] == "theft" else "tip"))
                    break
        fp += len(al) - len(used)
    return {"events": n_ev, "detected": tp, "recall": tp / n_ev if n_ev else float("nan"),
            "kind_correct": kinds_ok, "false_alarms": fp, "false_alarms_per_hive_year": fp / len(cohort),
            "delay_median_min": float(np.median(delays)) if delays else float("nan")}


def robbing_eval(cohort: list[sim.SimResult]) -> dict:
    tp = n_ev = fp = 0
    for res in cohort:
        d = prepare(res)
        dec = analytics.daily_decomposition(d["weight_kg"])
        sh = (d["shock_g"] > 0.4).groupby(d.index.normalize()).any()
        dec = dec[~sh.reindex(dec.index).fillna(False).to_numpy()]
        hits = analytics.robbing_hint(dec)
        used: set = set()
        for _, e in res.events[res.events["type"] == "robbing"].iterrows():
            n_ev += 1
            m = [h for h in hits if e["start"].normalize() <= h <= e["end"].normalize() + pd.Timedelta(days=1)]
            if m:
                tp += 1
                used.update(m)
        fp += len([h for h in hits if h not in used])
    return {"events": n_ev, "detected": tp, "recall": tp / n_ev if n_ev else float("nan"),
            "false_alerts": fp, "false_alerts_per_hive_year": fp / len(cohort)}


def alerts_per_season(cohort: list[sim.SimResult]) -> dict:
    """All alert types the device would raise per simulated hive-season (Mar-Oct)."""
    sw = fp_tilt = 0
    for res in cohort:
        d = prepare(res)
        s = d[(d.index.day_of_year >= SEASON[0]) & (d.index.day_of_year <= SEASON[1])]
        sw += len(analytics.detect_swarms(s["weight_kg"], None, s["t_amb"], s["shock_g"], s["service_mode"]))
        sw += 0
        fp_tilt += len(analytics.tilt_alarms(s["tilt_dev_deg"], s["shock_g"], s["weight_kg"], s["service_mode"]))
    n = len(cohort)
    return {"swarm_alerts_per_hive_season": sw / n, "tilt_theft_alerts_per_hive_season": fp_tilt / n,
            "total_per_hive_season": (sw + fp_tilt) / n}


def run_all(per_cell: int = 8, seed: int = 7, verbose: bool = False) -> tuple[dict, list[sim.SimResult]]:
    cohort = sim.simulate_cohort(per_cell=per_cell, seed=seed, p_swarm=0.4)
    if verbose:
        print(f"simulated {len(cohort)} hive-years", flush=True)
    res = {
        "cohort": {"hive_years": len(cohort), "seed": seed, "stations": ["SMA", "BER", "ENG"],
                   "years": [2023, 2024],
                   "swarms": int(sum((r.events["type"] == "swarm").sum() for r in cohort))},
        "swarm_sim": swarm_eval(cohort),
        "sensor": sensor_eval(cohort),
        "decomposition": decomposition_eval(cohort),
        "harvest": harvest_eval(cohort),
        "reserve_sim": reserve_eval(cohort),
        "brood_sim": brood_eval(cohort),
        "tilt": tilt_eval(cohort),
        "robbing": robbing_eval(cohort),
        "alerts": alerts_per_season(cohort),
        "swarm_real": realdata.real_swarm_eval(),
        "reserve_real": realdata.real_forecast_eval(),
        "brood_real_unlabelled": realdata.real_brood_temperature(),
        "battery": [payload.battery_life_days(i, sf) for i in (15, 60) for sf in (7, 9, 12)],
    }
    with resources.files("hivemon").joinpath("data/audio_features.csv").open("rb") as f:
        res["acoustics_queen"] = acoustics.evaluate_holdout(pd.read_csv(f))
    return res, cohort
