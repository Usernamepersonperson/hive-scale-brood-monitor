import numpy as np
import pandas as pd
import pytest

from hivemon import analytics, sensor

TZ = "Europe/Zurich"


def _flat(days=3, level=40.0):
    return pd.Series(level, index=pd.date_range("2023-06-01", periods=days * 96, freq="15min", tz=TZ))


# ---- swarm -----------------------------------------------------------------------------------

def test_swarm_step_detected_with_small_delay():
    w = _flat()
    t = pd.Timestamp("2023-06-02 12:00", tz=TZ)
    w[w.index >= t] -= 1.6
    al = analytics.detect_swarms(w)
    assert len(al) == 1
    assert 0 <= (al.time.iloc[0] - t) / pd.Timedelta(minutes=1) <= 30
    assert al.drop_kg.iloc[0] == pytest.approx(1.6, abs=0.01)


def test_swarm_below_threshold_ignored():
    w = _flat()
    w[w.index >= pd.Timestamp("2023-06-02 12:00", tz=TZ)] -= 0.5
    assert analytics.detect_swarms(w).empty


def test_swarm_big_step_is_not_a_swarm():
    w = _flat()
    w[w.index >= pd.Timestamp("2023-06-02 12:00", tz=TZ)] -= 9.0
    assert analytics.detect_swarms(w).empty


def test_swarm_at_night_ignored():
    w = _flat()
    w[w.index >= pd.Timestamp("2023-06-02 02:00", tz=TZ)] -= 1.8
    assert analytics.detect_swarms(w).empty


def test_swarm_gradual_loss_ignored():
    w = _flat()
    t0 = pd.Timestamp("2023-06-02 10:00", tz=TZ)
    ramp = ((w.index - t0) / pd.Timedelta(hours=1)).to_numpy().clip(0, 6) * 0.3
    assert analytics.detect_swarms(w - ramp).empty


def test_swarm_single_sample_spike_ignored():
    w = _flat()
    w.iloc[w.index.get_loc(pd.Timestamp("2023-06-02 12:00", tz=TZ))] -= 2.0
    assert analytics.detect_swarms(w).empty


def test_swarm_suppressed_by_shock():
    w = _flat()
    t = pd.Timestamp("2023-06-02 12:00", tz=TZ)
    w[w.index >= t] -= 1.6
    shock = pd.Series(0.0, index=w.index)
    shock[(w.index >= t - pd.Timedelta(minutes=15)) & (w.index <= t)] = 1.0
    assert analytics.detect_swarms(w, shock=shock).empty


def test_swarm_cold_gate():
    w = _flat()
    w[w.index >= pd.Timestamp("2023-06-02 12:00", tz=TZ)] -= 1.6
    cold = pd.Series(5.0, index=w.index)
    assert analytics.detect_swarms(w, t_amb=cold).empty


def test_swarm_nan_gaps_do_not_crash():
    w = _flat()
    w.iloc[20:30] = np.nan
    assert analytics.detect_swarms(w).empty


def test_swarm_refractory_one_alert():
    w = _flat(days=2)
    t = pd.Timestamp("2023-06-01 12:00", tz=TZ)
    w[w.index >= t] -= 1.5
    w[w.index >= t + pd.Timedelta(hours=3)] -= 1.5
    assert len(analytics.detect_swarms(w)) == 1


# ---- decomposition ---------------------------------------------------------------------------

def _synthetic_days(nectar=2.0, cons_rate=0.01, evap_night=0.4, days=10):
    idx = pd.date_range("2023-06-01", periods=days * 96, freq="15min", tz=TZ)
    w = np.zeros(len(idx))
    cur = 40.0
    for i, t in enumerate(idx):
        h = t.hour + t.minute / 60
        cur -= cons_rate / 4
        if 7 <= h < 19:
            cur += nectar / 12 / 4
        if h >= 21 or h < 5:
            cur -= evap_night / 8 / 4
        w[i] = cur
    return pd.Series(w, index=idx)


def test_decomposition_recovers_net_change_exactly():
    dec = analytics.daily_decomposition(_synthetic_days())
    # +2.0 nectar - 0.24 consumption - 0.4 night evaporation per day
    assert dec["net_24h_kg"].iloc[3] == pytest.approx(1.36, abs=1e-6)
    assert dec["nectar_kg"].iloc[3] == pytest.approx(2.0, abs=0.6)


def test_decomposition_nonnegative_nectar():
    dec = analytics.daily_decomposition(_synthetic_days(nectar=0.0))
    assert (dec["nectar_kg"].dropna() >= 0).all()


def test_decomposition_consumption_estimate():
    dec = analytics.daily_decomposition(_synthetic_days(nectar=0.0, evap_night=0.0, cons_rate=0.02))
    assert dec["consumption_kg"].iloc[-3] == pytest.approx(0.48, abs=0.05)


# ---- harvest readiness -----------------------------------------------------------------------

def test_harvest_ready_after_flow_and_ripening():
    days = pd.date_range("2023-06-01", periods=20, freq="D", tz=TZ)
    dec = pd.DataFrame({"w_evening": np.r_[np.linspace(30, 42, 12), np.full(8, 42.0)],
                        "nectar_kg": np.r_[np.full(12, 2.0), np.full(8, 0.1)],
                        "evaporation_kg": np.r_[np.full(12, 0.8), np.full(8, 0.05)]}, index=days)
    rd = analytics.harvest_readiness(dec)
    assert not rd["ready"].iloc[:10].any()
    assert rd["ready"].iloc[-1]


def test_harvest_not_ready_while_evaporating():
    days = pd.date_range("2023-06-01", periods=15, freq="D", tz=TZ)
    dec = pd.DataFrame({"w_evening": np.linspace(30, 45, 15), "nectar_kg": np.full(15, 0.1),
                        "evaporation_kg": np.full(15, 1.0)}, index=days)
    assert not analytics.harvest_readiness(dec)["ready"].any()


# ---- reserve forecast ------------------------------------------------------------------------

def test_reserve_forecast_linear_loss_exact():
    d = pd.date_range("2023-11-01", periods=40, freq="D", tz=TZ)
    w = pd.Series(30.0 - 0.05 * np.arange(40), index=d)
    fc = analytics.reserve_forecast(w, tare_kg=20.0, reserve_min_kg=3.0)
    last = fc.iloc[-1]
    stores = (30 - 0.05 * 39) - 20
    assert last["rate_kg_day"] == pytest.approx(0.05, abs=1e-6)
    assert last["days_left"] == pytest.approx((stores - 3) / 0.05, abs=1e-3)


def test_reserve_forecast_floor_rate_when_weight_rises():
    d = pd.date_range("2023-11-01", periods=30, freq="D", tz=TZ)
    fc = analytics.reserve_forecast(pd.Series(np.linspace(30, 32, 30), index=d), 20.0)
    assert (fc["rate_kg_day"] >= 0.005).all()


def test_reserve_forecast_short_series_empty():
    d = pd.date_range("2023-11-01", periods=5, freq="D", tz=TZ)
    assert analytics.reserve_forecast(pd.Series(1.0, index=d), 0.0).empty


# ---- brood presence --------------------------------------------------------------------------

def _temps(kind, days=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-06-01", periods=days * 96, freq="15min", tz=TZ)
    if kind == "brood":
        return pd.Series(34.5 + rng.normal(0, 0.2, len(idx)), index=idx)
    return pd.Series(22 + rng.normal(0, 4.0, len(idx)), index=idx)


def test_brood_flagged_for_stable_34_5():
    assert analytics.brood_presence(_temps("brood"))["brood"].all()


def test_brood_not_flagged_for_unstable_cluster():
    assert not analytics.brood_presence(_temps("cluster"))["brood"].any()


def test_brood_not_flagged_when_stable_but_cold():
    t = _temps("brood") - 12.0
    assert not analytics.brood_presence(t)["brood"].any()


def test_brood_break_alert_after_five_days():
    idx = pd.date_range("2023-06-01", periods=12, freq="D", tz=TZ)
    flags = pd.Series([True] * 4 + [False] * 8, index=idx)
    al = analytics.brood_break_alerts(flags)
    assert len(al) == 1 and al[0] == idx[8]


def test_brood_break_out_of_season_ignored():
    idx = pd.date_range("2023-12-01", periods=12, freq="D", tz=TZ)
    assert analytics.brood_break_alerts(pd.Series(False, index=idx)) == []


# ---- tilt / theft ----------------------------------------------------------------------------

def _tilt_frame():
    idx = pd.date_range("2023-06-01", periods=96, freq="15min", tz=TZ)
    return pd.Series(1.0, index=idx), pd.Series(0.02, index=idx), pd.Series(40.0, index=idx)


def test_tilt_alarm_tip():
    tilt, shock, w = _tilt_frame()
    tilt.iloc[40] = 25
    al = analytics.tilt_alarms(tilt, shock, w)
    assert list(al.kind) == ["tip"]


def test_tilt_alarm_theft_when_weight_vanishes():
    tilt, shock, w = _tilt_frame()
    tilt.iloc[40] = 30
    w.iloc[41:] = 0.5
    assert list(analytics.tilt_alarms(tilt, shock, w).kind) == ["theft"]


def test_tilt_service_mode_mutes():
    tilt, shock, w = _tilt_frame()
    tilt.iloc[40] = 25
    svc = pd.Series(False, index=tilt.index)
    svc.iloc[39:42] = True
    assert analytics.tilt_alarms(tilt, shock, w, svc).empty


def test_tilt_quiet_no_alarm():
    tilt, shock, w = _tilt_frame()
    assert analytics.tilt_alarms(tilt, shock, w).empty


# ---- robbing ---------------------------------------------------------------------------------

def test_robbing_hint_two_days():
    d = pd.date_range("2023-08-01", periods=6, freq="D", tz=TZ)
    dec = pd.DataFrame({"day_change": [-0.1, -1.5, -1.8, -0.1, -0.1, -0.1]}, index=d)
    assert analytics.robbing_hint(dec) == [d[2]]


# ---- sensor compensation ---------------------------------------------------------------------

def test_compensation_removes_synthetic_temperature_artifact():
    idx = pd.date_range("2023-06-01", periods=96 * 30, freq="15min", tz=TZ)
    rng = np.random.default_rng(1)
    hours = idx.hour + idx.minute / 60
    t = 15 + 8 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 0.5, len(idx))
    true = 40 - 0.01 * np.arange(len(idx)) / 96
    raw = pd.Series(true + 0.03 * (t - 20), index=idx)
    k = sensor.estimate_k(raw, pd.Series(t, index=idx))
    assert k == pytest.approx(0.03, abs=0.01)
    comp = sensor.compensate(raw, pd.Series(t, index=idx), k)
    assert (comp - true).abs().mean() < (raw - true).abs().mean()


def test_estimate_k_flat_temperature_returns_zero():
    idx = pd.date_range("2023-06-01", periods=96 * 5, freq="15min", tz=TZ)
    assert sensor.estimate_k(pd.Series(40.0, index=idx), pd.Series(20.0, index=idx)) == 0.0
