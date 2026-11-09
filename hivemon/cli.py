"""Command line: `python -m hivemon demo` reproduces every number and figure in the README."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from hivemon import measures

ROOT = Path(__file__).resolve().parents[1]


def _fmt(v, nd=3):
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.{nd}g}"
    return str(v)


def results_markdown(r: dict) -> str:
    s, sr = r["swarm_sim"], r["swarm_real"]
    rows = [
        ("Swarm alert, simulated", f"precision {s['precision']:.2f}, recall {s['recall']:.2f}, "
         f"median delay {s['delay_median_min']:.0f} min (p90 {s['delay_p90_min']:.0f}) on {s['n_swarms']} swarms"),
        ("Swarm alert, real hives", f"{sr['logged_swarms_detected']} of {sr['logged_swarms']} logged swarms; "
         f"{sr['alerts_not_on_logged_handling_day']} other alerts in ~{sr['hive_seasons_approx']} hive-seasons "
         "(unlabelled, not all false)"),
        ("Alerts per simulated hive-season", f"{r['alerts']['total_per_hive_season']:.2f} "
         f"(swarm {r['alerts']['swarm_alerts_per_hive_season']:.2f}, tilt/theft "
         f"{r['alerts']['tilt_theft_alerts_per_hive_season']:.2f})"),
        ("Daily nectar estimate error", f"{r['decomposition']['mae_nectar_kg_day']:.2f} kg/day MAE "
         f"({r['decomposition']['mae_nectar_flow_days_kg_day']:.2f} on flow days), r = "
         f"{r['decomposition']['nectar_corr']:.2f}"),
        ("Load-cell compensation", f"p90 error {r['sensor']['p90_abs_error_raw_kg']:.3f} to "
         f"{r['sensor']['p90_abs_error_compensated_kg']:.3f} kg"),
        ("Harvest readiness, simulated", f"day accuracy {r['harvest']['day_accuracy']:.2f}, precision "
         f"{r['harvest']['precision']:.2f}, recall {r['harvest']['recall']:.2f}"),
        ("30-day weight forecast, simulated", f"MAE {r['reserve_sim']['mae_kg']:.2f} kg, bias "
         f"{r['reserve_sim']['bias_kg']:+.2f} kg (n={r['reserve_sim']['n_forecasts']})"),
        ("30-day weight forecast, real", f"MAE {r['reserve_real'].get('mae_kg', float('nan')):.2f} kg, bias "
         f"{r['reserve_real'].get('bias_kg', float('nan')):+.2f} kg (n={r['reserve_real'].get('n_forecasts', 0)})"),
        ("Brood presence, simulated", f"day accuracy {r['brood_sim']['day_accuracy']:.2f}, precision "
         f"{r['brood_sim']['precision']:.2f}, recall {r['brood_sim']['recall']:.2f}"),
        ("Queenless brood-break alert", f"{r['brood_sim']['queenless_alerted']} of {r['brood_sim']['queenless_hives']} "
         f"hives, median {_fmt(r['brood_sim']['queenless_delay_median_days'])} days after queen loss"),
        ("Theft / tipping alarm", f"recall {r['tilt']['recall']:.2f}, {r['tilt']['false_alarms']} false alarms "
         f"({r['tilt']['false_alarms_per_hive_year']:.2f} per hive-year)"),
        ("Robbing hint", f"recall {r['robbing']['recall']:.2f}, {r['robbing']['false_alerts']} false alerts"),
        ("Queen presence from sound", f"hive-held-out balanced accuracy "
         f"{r['acoustics_queen']['mean_window_bacc']:.2f} (chance 0.5): does not transfer, not shipped"),
    ]
    out = ["| Measurement | Result |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows]
    out += ["", "Battery estimate (assumed currents, not measured):", "",
            "| Interval | SF | Airtime | Life |", "|---|---|---|---|"]
    out += [f"| {b['interval_min']:.0f} min | {b['sf']} | {b['airtime_ms']:.0f} ms | {b['life_years']:.1f} years |"
            for b in r["battery"]]
    return "\n".join(out) + "\n"


def demo(per_cell: int, figures: bool, outdir: Path) -> dict:
    r, cohort = measures.run_all(per_cell=per_cell, verbose=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(r, indent=1, default=str))
    md = results_markdown(r)
    (outdir / "results.md").write_text(md)
    print(md)
    if figures:
        from hivemon import figures

        figures.make_all(r, cohort, outdir)
        print(f"figures written to {outdir}")
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hivemon")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="simulate, evaluate on real data, write docs/results and figures")
    d.add_argument("--per-cell", type=int, default=8, help="simulated hives per station-year (default 8, 48 total)")
    d.add_argument("--no-figures", action="store_true")
    d.add_argument("--out", default=str(ROOT / "docs"))
    dec = sub.add_parser("decode", help="decode a hex uplink payload")
    dec.add_argument("hex")
    a = ap.parse_args(argv)
    if a.cmd == "decode":
        from hivemon import payload

        print(json.dumps(payload.decode(bytes.fromhex(a.hex)), indent=1))
        return 0
    demo(a.per_cell, not a.no_figures, Path(a.out))
    return 0
