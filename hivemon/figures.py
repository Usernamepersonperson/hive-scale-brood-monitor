"""docs/figure.png (1600x900) and docs/post-figure.png (960x700, dark, phone-sized)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hivemon import analytics, sim  # noqa: E402
from hivemon.evaluate import SEASON, prepare, swarm_eval  # noqa: E402

BG, GRID, TEXT, MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
ACCENT, GREEN, AMBER, RED, GREY, VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"
matplotlib.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
SOURCE = "Simulated hive driven by MeteoSwiss SwissMetNet weather (Bern / Zollikofen)"


def _style(ax, size=12):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=size)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def pick_showcase(cohort: list[sim.SimResult]) -> sim.SimResult:
    """A clean swarm-only hive whose swarm the detector caught, with a believable peak weight (~65 kg)."""
    best, score = None, -1e9
    for r in cohort:
        s = r.scenario
        if not s.swarm or s.theft or s.tipping or s.queenless or s.robbing:
            continue
        sw = r.events[r.events["type"] == "swarm"]
        if sw.empty or not (r.events["type"] == "harvest").any():
            continue
        d, al, _ = _season(r)
        t0 = sw["start"].iloc[0]
        if not ((al["time"] >= t0) & (al["time"] <= t0 + pd.Timedelta(hours=3))).any():
            continue
        sc = -abs(float(d["weight_kg"].max()) - 65.0) + 2.0 * sw["magnitude_kg"].iloc[0]
        if sc > score:
            best, score = r, sc
    return best or cohort[0]


def _season(res: sim.SimResult):
    d = prepare(res)
    d = d[(d.index.day_of_year >= SEASON[0]) & (d.index.day_of_year <= SEASON[1])]
    al = analytics.detect_swarms(d["weight_kg"], None, d["t_amb"], d["shock_g"], d["service_mode"])
    dec = analytics.daily_decomposition(d["weight_kg"])
    return d, al, dec


def _season_panel(ax, res, d, al, dec, size=12, big=False):
    _style(ax, size)
    day = d["weight_kg"].resample("h").median()
    ax.plot(day.index, day.values, color=ACCENT, lw=1.6 if big else 1.2, zorder=3)
    flow = dec[dec["nectar_kg"] >= 1.0].index
    for t in flow:
        ax.axvspan(t, t + pd.Timedelta(days=1), color=GREEN, alpha=0.20, lw=0, zorder=1)
    ev = res.events
    for _, h in ev[ev["type"] == "harvest"].iterrows():
        ax.axvline(h["start"], color=GREY, lw=1.2, ls=":")
        ax.text(h["start"] + pd.Timedelta(days=3), d["weight_kg"].max() * 0.93, "honey harvest", color=MUTED,
                fontsize=size, ha="left", va="top")
    if len(al):
        a = al.iloc[0]
        t = a["time"]
        y = float(d["weight_kg"].loc[t])
        ax.scatter([t], [y], s=140, color=RED, zorder=5, edgecolor=TEXT, linewidth=1.2)
        ax.annotate(f"swarm detected\n{t:%d %b %H:%M}, -{a['drop_kg']:.1f} kg", xy=(t, y),
                    xytext=(-60 if big else -140, -105), textcoords="offset points", color=TEXT, fontsize=size + 1,
                    fontweight="bold", arrowprops={"arrowstyle": "-|>", "color": RED, "lw": 2},
                    bbox={"boxstyle": "round,pad=0.35", "fc": BG, "ec": RED})
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("hive weight, kg", color=MUTED, fontsize=size)
    ax.set_xlim(day.index[0], day.index[-1])


def post_figure(res, d, al, dec, path: Path) -> None:
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.095, 0.15, 0.87, 0.65])
    _season_panel(ax, res, d, al, dec, size=13, big=True)
    ax.text(0.985, 0.50, "green = nectar flow days (from the scale)", transform=ax.transAxes,
            color=GREEN, fontsize=13, va="bottom", ha="right")
    if len(al):
        t = al.iloc[0]["time"]
        ins = ax.inset_axes([0.05, 0.50, 0.30, 0.42])
        _style(ins, 11)
        z = d["weight_kg"][t - pd.Timedelta(hours=5):t + pd.Timedelta(hours=5)]
        ins.plot(z.index, z.values, color=ACCENT, lw=2)
        ins.axvline(t, color=RED, lw=2)
        ins.set_title(f"swarm day, {t:%d %b}, kg", color=TEXT, fontsize=12)
        ins.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        ins.xaxis.set_major_locator(mdates.HourLocator(interval=4, tz=z.index.tz))
    fig.text(0.04, 0.93, "A season of hive weight, nectar flows and one swarm", color=TEXT, fontsize=19,
             fontweight="bold")
    fig.text(0.04, 0.875, "SIMULATED hive, real MeteoSwiss weather, March to October", color=MUTED, fontsize=14)
    fig.text(0.04, 0.045, "Source: MeteoSwiss (weather input). Model: hive-scale-brood-monitor.", color=MUTED,
             fontsize=13)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def main_figure(res, d, al, dec, results: dict, cohort, path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    fig.text(0.03, 0.955, "Open-source hive monitor: scale + brood-nest temperature, measured", color=TEXT,
             fontsize=22, fontweight="bold")
    fig.text(0.03, 0.915, "Simulated hives on real MeteoSwiss weather (a, b, c); real open hive-scale data (d)",
             color=MUTED, fontsize=14)
    # a: season
    ax = fig.add_axes([0.055, 0.56, 0.60, 0.30])
    _season_panel(ax, res, d, al, dec, size=11)
    ax.set_title("a  Season weight, flows (green), swarm alert (red)", color=TEXT, fontsize=13, loc="left")
    # b: decomposition for 40 days around the main flow
    ax = fig.add_axes([0.72, 0.56, 0.26, 0.30])
    _style(ax, 11)
    j = dec[(dec.index >= "2023-05-01") & (dec.index < "2023-06-15")] if dec.index[0].year == 2023 else \
        dec[(dec.index >= "2024-05-01") & (dec.index < "2024-06-15")]
    x = np.arange(len(j))
    ax.bar(x, j["nectar_kg"], color=GREEN, label="nectar in")
    ax.bar(x, -j["consumption_kg"], color=AMBER, label="consumption")
    ax.bar(x, -j["evaporation_kg"], bottom=-j["consumption_kg"], color=VIOLET, label="evaporation")
    ax.plot(x, j["net_24h_kg"], color=TEXT, lw=1.4, label="net 24 h")
    ax.set_xticks([])
    ax.set_ylabel("kg per day", color=MUTED, fontsize=11)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper right", ncol=2)
    ax.set_title("b  Daily weight split (6 weeks)", color=TEXT, fontsize=13, loc="left")
    # c: threshold sweep
    ax = fig.add_axes([0.055, 0.09, 0.40, 0.33])
    _style(ax, 11)
    ths = [0.5, 0.7, 0.9, 1.1, 1.4, 1.8]
    pr = []
    for th in ths:
        r = swarm_eval(cohort, analytics.SwarmParams(drop_kg=th, fast_frac=0.7))
        pr.append((r["precision"], r["recall"]))
    ax.plot(ths, [p for p, _ in pr], color=ACCENT, marker="o", lw=2, label="precision")
    ax.plot(ths, [r for _, r in pr], color=AMBER, marker="o", lw=2, label="recall")
    ax.axvline(0.9, color=GREY, ls=":")
    ax.set_xlabel("alert threshold, kg dropped within 2 h", color=MUTED, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="lower left")
    ax.set_title("c  Swarm alert threshold (simulated, swarms 1-3 kg)", color=TEXT, fontsize=13, loc="left")
    # d: real data swarm
    ax = fig.add_axes([0.56, 0.09, 0.42, 0.33])
    _style(ax, 11)
    from hivemon import realdata

    w = realdata.load_mustb_weight()
    s = realdata.hourly_series(w, 5, 2)["2019-04-15":"2019-05-02"]
    ax.plot(s.index, s.values, color=ACCENT, lw=1.8)
    for h in results["swarm_real"]["swarm_hits"]:
        if h["alert_time"] is not None:
            ax.axvline(pd.Timestamp(h["alert_time"]), color=RED, lw=2)
    ax.set_ylabel("weight, kg", color=MUTED, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title("d  Real hive, two logged swarms, both alerted (red)", color=TEXT, fontsize=13, loc="left")
    fig.text(0.03, 0.02, "Sources: MeteoSwiss SwissMetNet (weather); Zenodo 10.5281/zenodo.5148318, CC BY 4.0 "
             "(real hive-scale data). Simulated panels are labelled; see README for limits.", color=MUTED, fontsize=11)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def make_all(results: dict, cohort: list[sim.SimResult], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    res = pick_showcase(cohort)
    d, al, dec = _season(res)
    post_figure(res, d, al, dec, outdir / "post-figure.png")
    main_figure(res, d, al, dec, results, cohort, outdir / "figure.png")
