"""Documented hive simulator driven by real MeteoSwiss weather. SIMULATED DATA.

One call simulates one hive for one calendar year at 15-minute resolution and returns what the
device would report (raw load-cell reading with temperature drift, brood-nest temperature,
hive humidity, ambient temperature, accelerometer summary) next to the hidden truth and labelled
events. See docs/SIMULATOR.md for every equation and parameter; the numbers are plausible
beekeeping magnitudes chosen by the author, not fitted to a particular apiary.

Scenarios (all can co-occur): swarm, robbing, queenless, theft, tipping, plus always-on routine
beekeeping (inspections, super added, honey harvest, autumn feeding) and weather-driven
nectar flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hivemon import weather

DT_H = weather.STEP_MIN / 60.0
STEPS_PER_DAY = 24 * 60 // weather.STEP_MIN
T_REF = 20.0  # load-cell calibration temperature, deg C

# (centre day-of-year, half-width days, peak kg/day for a strong colony in a perfect weather day)
FLOWS = (
    (105, 8, 1.8),   # fruit bloom / dandelion
    (128, 9, 3.7),   # rape / maple
    (162, 11, 2.9),  # meadow, bramble
    (178, 7, 2.4),   # linden
    (205, 14, 2.1),  # honeydew (patchy, scaled by a per-hive random draw)
)


@dataclass
class Scenario:
    """What happens to one simulated hive. Defaults are the no-incident baseline."""

    swarm: bool = False
    robbing: bool = False
    queenless: bool = False
    theft: bool = False
    tipping: bool = False
    vigour: float = 1.0
    stores_jan: float = 12.0   # kg of stores in the brood nest on 1 Jan
    autumn_target: float = 14.0  # kg of stores the beekeeper feeds up to in late summer
    zero_tc_g_per_k: float = 8.0
    span_tc_per_k: float = 0.0003


@dataclass
class SimResult:
    data: pd.DataFrame
    events: pd.DataFrame
    scenario: Scenario
    station: str
    year: int
    meta: dict = field(default_factory=dict)


def random_scenario(rng: np.random.Generator, p_swarm=0.30, p_robbing=0.10, p_queenless=0.10,
                    p_theft=0.04, p_tipping=0.05) -> Scenario:
    return Scenario(
        swarm=bool(rng.random() < p_swarm),
        robbing=bool(rng.random() < p_robbing),
        queenless=bool(rng.random() < p_queenless),
        theft=bool(rng.random() < p_theft),
        tipping=bool(rng.random() < p_tipping),
        vigour=float(np.clip(rng.normal(1.0, 0.18), 0.55, 1.5)),
        stores_jan=float(rng.uniform(4.5, 16.0)),
        autumn_target=float(rng.uniform(7.0, 20.0)),
        zero_tc_g_per_k=float(rng.normal(8.0, 4.0)),
        span_tc_per_k=float(np.clip(rng.normal(0.0003, 0.00012), 0.0, 0.0008)),
    )


def brood_curve(doy: np.ndarray) -> np.ndarray:
    """Seasonal brood-nest activity 0..1: starts early February, peak May-July, ends late October."""
    up = 1 / (1 + np.exp(-(doy - 80) / 14.0))
    down = 1 / (1 + np.exp((doy - 262) / 14.0))
    return np.clip(up * down, 0.0, 1.0)


def nectar_potential(doy: np.ndarray, altitude_m: float, amp_scale: np.ndarray) -> np.ndarray:
    """Per-day nectar flow strength (kg/day for a strong colony in perfect weather)."""
    shift = 3.0 * max(0.0, altitude_m - 550.0) / 100.0
    f = np.zeros_like(doy, dtype=float)
    for (c, w, a), s in zip(FLOWS, amp_scale, strict=True):
        f += a * s * np.exp(-0.5 * ((doy - (c + shift)) / w) ** 2)
    return f


def _suitability(w: pd.DataFrame) -> np.ndarray:
    """Foraging suitability per step in 0..1 from temperature, light, rain and wind."""
    t = w["temp_c"].to_numpy()
    rad = w["glob_rad_wm2"].to_numpy()
    rain = w["precip_mm"].to_numpy()
    wind = w["wind_ms"].to_numpy()
    s = (1 / (1 + np.exp(-(t - 12.0) / 2.5))) * np.clip(rad / 450.0, 0, 1)
    s *= np.where(rain > 0.05, 0.1, 1.0)
    s *= np.clip(1.0 - np.maximum(wind - 5.0, 0.0) / 8.0, 0.0, 1.0)
    return s


def _pick_swarm_step(rng, w: pd.DataFrame, suit: np.ndarray) -> int | None:
    """A warm, dry, bright midday step between mid-April and the end of June."""
    idx = w.index
    hour = idx.hour + idx.minute / 60
    doy = idx.day_of_year
    ok = (doy >= 112) & (doy <= 180) & (hour >= 10.5) & (hour <= 14.5) & (suit > 0.55)
    cand = np.flatnonzero(ok)
    return int(rng.choice(cand)) if cand.size else None


def simulate_hive(station: str, year: int, scenario: Scenario | None = None, seed: int = 0,
                  altitude_m: float | None = None) -> SimResult:
    rng = np.random.default_rng(seed)
    sc = scenario or Scenario()
    w = weather.station_year(station, year)
    n = len(w)
    idx = w.index
    doy = idx.day_of_year.to_numpy()
    hour = (idx.hour + idx.minute / 60).to_numpy()
    day_i = np.asarray(pd.factorize(idx.normalize())[0])
    alt = altitude_m if altitude_m is not None else {"SMA": 556, "BER": 553, "ENG": 1036}.get(station, 550)
    suit = _suitability(w)
    t_amb = w["temp_c"].to_numpy()
    rad = w["glob_rad_wm2"].to_numpy()
    rain = w["precip_mm"].to_numpy()
    wind = w["wind_ms"].to_numpy()

    # ---- daily drivers ------------------------------------------------------------------
    n_days = int(day_i.max()) + 1
    doy_d = np.array([doy[day_i == d][0] for d in range(n_days)])
    brood_target = brood_curve(doy_d.astype(float))
    amp_scale = rng.lognormal(0.0, 0.3, size=len(FLOWS))
    amp_scale[-1] *= rng.choice([0.0, 0.5, 1.0])  # honeydew is patchy
    flow_d = nectar_potential(doy_d.astype(float), alt, amp_scale)

    events = []
    # swarm
    swarm_step = swarm_loss = None
    if sc.swarm:
        swarm_step = _pick_swarm_step(rng, w, suit)
        if swarm_step is not None:
            swarm_loss = float(rng.uniform(1.0, 3.0))
            events.append(("swarm", idx[swarm_step], idx[min(n - 1, swarm_step + int(rng.integers(1, 3)))],
                           swarm_loss))
    swarm_day = int(day_i[swarm_step]) if swarm_step is not None else None
    # queen loss
    qloss_day = int(rng.integers(100, 200)) if sc.queenless else None
    if qloss_day is not None:
        events.append(("queen_loss", idx[qloss_day * STEPS_PER_DAY], idx[qloss_day * STEPS_PER_DAY], 0.0))
    # robbing: 2-6 consecutive dearth days
    rob_days: list[int] = []
    if sc.robbing:
        d0 = int(rng.integers(205, 255))
        rob_days = list(range(d0, d0 + int(rng.integers(2, 7))))
        events.append(("robbing", idx[d0 * STEPS_PER_DAY], idx[min(n - 1, (rob_days[-1] + 1) * STEPS_PER_DAY - 1)],
                       0.0))
    # theft / tipping
    theft_step = tip_step = None
    if sc.theft:
        theft_step = int(rng.integers(60 * STEPS_PER_DAY, 300 * STEPS_PER_DAY))
        events.append(("theft", idx[theft_step], idx[theft_step], 0.0))
    if sc.tipping:
        tip_step = int(rng.integers(40 * STEPS_PER_DAY, 330 * STEPS_PER_DAY))
        events.append(("tipping", idx[tip_step], idx[tip_step + 1], 0.0))
    # routine beekeeping, all at midday on a dry day
    insp_steps = []
    for d in range(95, 280, int(rng.integers(8, 12))):
        s0 = d * STEPS_PER_DAY + int(rng.integers(11 * 4, 14 * 4))
        if s0 < n - 6:
            insp_steps.append(s0)
    super_day = int(rng.integers(108, 118))
    harvest_day = int(rng.integers(196, 212))
    feed_day = int(rng.integers(228, 246))
    harvest_step = harvest_day * STEPS_PER_DAY + 12 * 4
    feed_step = feed_day * STEPS_PER_DAY + 11 * 4
    super_step = super_day * STEPS_PER_DAY + 11 * 4
    for s in insp_steps:
        events.append(("inspection", idx[s], idx[s + 2], 0.0))

    # ---- state & outputs ----------------------------------------------------------------
    tare = 22.0 + rng.normal(0, 1.5)
    H = sc.stores_jan  # ripened honey / syrup in kg
    U = 0.0            # unripe nectar in kg
    super_kg = 0.0
    out = {k: np.zeros(n) for k in (
        "true_weight_kg", "true_nectar_kg", "true_cons_kg", "true_evap_kg", "true_store_kg",
        "true_unripe_kg", "true_brood", "weight_raw_kg", "t_cell", "t_brood", "rh_in", "t_amb",
        "tilt_dev_deg", "shock_g")}
    forager_ema = 0.0
    rain_ema = 0.0
    cell_t = t_amb[0]
    t_noise = 0.0
    creep = 0.0
    swarm_ramp = []  # (step, kg) increments of the swarm departure
    if swarm_step is not None:
        dur = int(rng.integers(1, 3))
        swarm_ramp = [(swarm_step + k, swarm_loss * (k + 1) / dur - swarm_loss * k / dur) for k in range(dur)]
    swarm_inc = {s: kg for s, kg in swarm_ramp}
    cum_removed_swarm = 0.0
    offset_manip = 0.0  # temporary reading offset during handling
    manip_left = 0
    manip_shock = 0.0
    rng.normal(1.5, 0.6)  # (unused rest-tilt draw, kept so seeded runs stay reproducible)
    theft_done = False
    tip_active = 0
    insp_set = set(insp_steps)
    insp_service = {s: rng.random() < 0.85 for s in insp_steps}
    service_until = -1
    in_service_flag = np.zeros(n, dtype=bool)
    removed_by_theft = False
    sm_cap = 25.0  # kg of stores above which the beekeeper harvests
    # pre-drawn noise (keeps the loop cheap and the run reproducible)
    n_w1, n_w2 = rng.normal(0, 0.015, n), rng.normal(0, 0.012, n) * wind
    n_creep = np.cumsum(rng.normal(0, 0.0004, n))
    n_tamb, n_tb = rng.normal(0, 0.15, n), rng.normal(0, 1.0, n)
    n_rh, n_tilt, n_shock = rng.normal(0, 1.2, n), np.abs(rng.normal(0, 0.25, n)), np.abs(rng.normal(0, 0.03, n))
    diurnal = np.sin(2 * np.pi * (hour - 14) / 24)
    n_w1, n_w2, n_creep, n_tamb, n_tb, n_rh, n_tilt, n_shock, diurnal = (
        a.tolist() for a in (n_w1, n_w2, n_creep, n_tamb, n_tb, n_rh, n_tilt, n_shock, diurnal))
    sup = suit.tolist()
    bt_l, flow_l, day_l = brood_target.tolist(), flow_d.tolist(), day_i.tolist()
    ta, rd, rn, wd, hr = t_amb.tolist(), rad.tolist(), rain.tolist(), wind.tolist(), hour.tolist()

    for i in range(n):
        d = day_l[i]
        b_t = bt_l[d]
        # brood & strength
        eff = b_t
        if swarm_day is not None and d >= swarm_day:
            gap = d - swarm_day
            eff *= 0.55 + 0.45 * min(1.0, gap / 25.0)
        if qloss_day is not None and d >= qloss_day:
            eff *= max(0.0, 1.0 - (d - qloss_day) / 21.0)
        lagged = bt_l[max(0, d - 21)]
        strength = (0.15 + 0.85 * lagged) * sc.vigour
        if swarm_day is not None and d >= swarm_day:
            strength *= 0.55 + 0.45 * min(1.0, (d - swarm_day) / 35.0)
        if qloss_day is not None and d >= qloss_day:
            strength *= max(0.15, 1.0 - 0.8 * (d - qloss_day) / 60.0)
        out["true_brood"][i] = eff
        # foraging
        forager_ema += (sup[i] - forager_ema) * 0.25
        nectar = flow_l[d] * strength * sup[i] / 32.0
        # consumption
        cons_day = 0.015 + 0.22 * eff * (0.4 + 0.6 * min(strength, 1.2)) + 0.003 * max(0.0, 10.0 - ta[i]) * (0.5 + eff)
        cons = cons_day * DT_H / 24.0
        # evaporation / ripening
        tau = 14.0 if (hr[i] < 5.5 or hr[i] >= 20.0) else 48.0
        ripened = U * DT_H / tau
        evap = 0.56 * ripened
        U += nectar - ripened
        H += 0.44 * ripened
        take = min(cons, max(H, 0.0))
        H -= take
        # robbing
        rob = 0.0
        if d in rob_days and 9.5 <= hr[i] <= 17.5 and sup[i] > 0.2:
            rob = float(rng.uniform(0.06, 0.15))  # kg per 15 min
            H -= min(rob, max(H, 0.0))
        # swarm departure
        sw = swarm_inc.get(i, 0.0)
        cum_removed_swarm += sw
        # routine beekeeping
        step_shock = 0.0
        step_tilt = 0.0
        if i == super_step:
            super_kg += 3.2
        if i == harvest_step:
            take_h = max(0.0, H - 9.0) if H > sm_cap * 0.5 else 0.0
            H -= take_h
            if take_h > 0:
                events.append(("harvest", idx[i], idx[i + 3], float(take_h + super_kg)))
            super_kg = 0.0
            manip_left, manip_shock, offset_manip = 3, (1.1 if rng.random() > 0.1 else 0.0), float(rng.uniform(-2, 4))
        if i == feed_step:
            add = max(0.0, sc.autumn_target - H)
            H += add
            if add > 0:
                events.append(("feeding", idx[i], idx[i + 3], float(add)))
            manip_left, manip_shock, offset_manip = 2, 0.9, float(rng.uniform(-1, 2))
        if i in insp_set:
            manip_left = 2
            manip_shock = float(rng.uniform(0.6, 1.6)) if rng.random() > 0.1 else 0.0  # 10 %: shock missed
            offset_manip = float(rng.uniform(-6.0, 3.0)) if rng.random() < 0.6 else float(rng.uniform(-1, 1))
            if insp_service[i]:
                service_until = i + 3
            if rng.random() < 0.03:
                step_tilt = float(rng.uniform(12, 20))
        in_service_flag[i] = i <= service_until
        manip_now = 0.0
        if manip_left > 0:
            manip_now = offset_manip
            step_shock = max(step_shock, manip_shock + float(rng.normal(0, 0.1)))
            step_tilt = max(step_tilt, float(abs(rng.normal(2.5, 1.0))))
            manip_left -= 1
        # theft / tipping
        tilt_extra = 0.0
        if theft_step is not None and i == theft_step:
            theft_done = True
        if theft_done and not removed_by_theft:
            tilt_extra = float(rng.uniform(15, 40))
            step_shock = max(step_shock, float(rng.uniform(2.0, 4.5)))
            removed_by_theft = True
        if tip_step is not None and i == tip_step:
            tip_active = 2
        if tip_active > 0:
            tilt_extra = max(tilt_extra, float(rng.uniform(14, 35)))
            step_shock = max(step_shock, float(rng.uniform(1.8, 3.5)))
            tip_active -= 1
        if wd[i] > 10:
            step_shock = max(step_shock, float(min(1.0, abs(rng.normal(0.2, 0.15)) * wd[i] / 10)))
        # true weight
        rain_ema += (rn[i] - rain_ema) * (DT_H / 6.0)
        w_true = (tare + super_kg + 3.5 * strength * 0.4 + H + U
                  - 0.7 * min(strength, 1.2) * min(1.0, forager_ema * 1.5)
                  + 0.2 * rain_ema - cum_removed_swarm)
        out["true_weight_kg"][i] = w_true
        out["true_nectar_kg"][i] = nectar
        out["true_cons_kg"][i] = take
        out["true_evap_kg"][i] = evap
        out["true_store_kg"][i] = H
        out["true_unripe_kg"][i] = U
        # sensor
        cell_t += (ta[i] + 0.012 * rd[i] - cell_t) * (DT_H / 3.0)
        creep = n_creep[i]
        wm = w_true
        if removed_by_theft:
            wm = max(0.0, w_true - (tare + super_kg + H + U + 2.0))  # scale without hive
        elif tip_active > 0:
            wm = w_true * float(rng.uniform(0.2, 0.9))
        dT = cell_t - T_REF
        raw = (wm + manip_now) * (1 + sc.span_tc_per_k * dT) + sc.zero_tc_g_per_k * dT / 1000.0
        raw += creep + n_w1[i] + n_w2[i]
        out["weight_raw_kg"][i] = raw
        out["t_cell"][i] = cell_t
        out["t_amb"][i] = ta[i] + 0.006 * rd[i] + n_tamb[i]
        # brood nest temperature & humidity
        regulated = min(1.0, max(0.0, eff / 0.30))
        t_noise = 0.85 * t_noise + n_tb[i] * (0.12 + 0.9 * (1 - regulated))
        nest = 34.5 + 0.015 * (ta[i] - 15.0) * regulated + 0.12 * diurnal[i]
        cluster = 22.0 + 0.45 * ta[i] + 3.5 * diurnal[i]
        tb = regulated * nest + (1 - regulated) * min(34.0, max(cluster, ta[i] + 3))
        out["t_brood"][i] = round((tb + t_noise) / 0.0625) * 0.0625
        out["rh_in"][i] = min(99.0, max(25.0, 52 + 18 * regulated + 3 * t_noise + n_rh[i]))
        out["tilt_dev_deg"][i] = max(step_tilt, tilt_extra) + n_tilt[i]
        out["shock_g"][i] = step_shock + n_shock[i]
    df = pd.DataFrame(out, index=idx)
    df["service_mode"] = in_service_flag
    ev = pd.DataFrame(events, columns=["type", "start", "end", "magnitude_kg"]).sort_values("start")
    ev = ev.reset_index(drop=True)
    return SimResult(df, ev, sc, station, year, {"altitude_m": alt, "seed": seed, "tare_kg": tare})


def simulate_cohort(stations=("SMA", "BER", "ENG"), years=(2023, 2024), per_cell: int = 10,
                    seed: int = 7, **kw) -> list[SimResult]:
    """Hive-seasons across stations and years with randomised scenarios."""
    rng = np.random.default_rng(seed)
    out = []
    for st in stations:
        for yr in years:
            for _ in range(per_cell):
                sc = random_scenario(rng, **kw)
                out.append(simulate_hive(st, yr, sc, seed=int(rng.integers(1 << 30))))
    return out
