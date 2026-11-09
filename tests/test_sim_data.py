import numpy as np
import pandas as pd
import pytest

from hivemon import acoustics, realdata, sim, weather


@pytest.fixture(scope="module")
def hive():
    return sim.simulate_hive("BER", 2023, sim.Scenario(swarm=True), seed=3)


def test_weather_fixture_covers_stations_and_years():
    for st in weather.STATIONS:
        w = weather.station_year(st, 2023)
        assert len(w) in (35040, 35036)
        assert w["temp_c"].between(-30, 42).all()
        assert (w["precip_mm"] >= 0).all()


def test_weather_unknown_station_and_year():
    with pytest.raises(KeyError):
        weather.station_year("XXX", 2023)
    with pytest.raises(KeyError):
        weather.station_year("SMA", 1999)


def test_weather_step_is_15_minutes():
    w = weather.station_year("SMA", 2024)
    assert (np.diff(w.index) == pd.Timedelta(minutes=15)).mean() > 0.99


def test_engelberg_colder_than_bern():
    assert weather.station_year("ENG", 2023)["temp_c"].mean() < weather.station_year("BER", 2023)["temp_c"].mean()


def test_brood_curve_seasonality():
    doy = np.array([10, 150, 330], dtype=float)
    b = sim.brood_curve(doy)
    assert b[1] > 0.9 and b[0] < 0.1 and b[2] < 0.1


def test_nectar_potential_peaks_in_may_and_shifts_with_altitude():
    d = np.arange(1, 366, dtype=float)
    lo = sim.nectar_potential(d, 500, np.ones(5))
    hi = sim.nectar_potential(d, 1500, np.ones(5))
    assert 100 < d[lo.argmax()] < 190
    assert hi[:140].sum() < lo[:140].sum()


def test_sim_is_reproducible():
    a = sim.simulate_hive("SMA", 2023, sim.Scenario(), seed=11).data["weight_raw_kg"]
    b = sim.simulate_hive("SMA", 2023, sim.Scenario(), seed=11).data["weight_raw_kg"]
    assert a.equals(b)


def test_swarm_event_labelled_and_size_in_range(hive):
    ev = hive.events[hive.events["type"] == "swarm"]
    assert len(ev) == 1
    assert 1.0 <= ev["magnitude_kg"].iloc[0] <= 3.0
    t = ev["start"].iloc[0]
    assert 10 <= t.hour <= 15


def test_swarm_drop_is_in_true_weight(hive):
    ev = hive.events[hive.events["type"] == "swarm"].iloc[0]
    tw = hive.data["true_weight_kg"]
    before = tw[ev["start"] - pd.Timedelta(minutes=30)]
    after = tw[ev["end"] + pd.Timedelta(minutes=30)]
    assert before - after > 0.7 * ev["magnitude_kg"]


def test_mass_balance_components_nonnegative(hive):
    d = hive.data
    assert (d["true_nectar_kg"] >= 0).all() and (d["true_cons_kg"] >= 0).all() and (d["true_evap_kg"] >= 0).all()
    assert (d["true_store_kg"] >= -1e-9).all()


def test_brood_nest_temperature_when_brood_present(hive):
    d = hive.data
    m = d["true_brood"] > 0.5
    assert abs(d.loc[m, "t_brood"].mean() - 34.5) < 0.5
    assert d.loc[m, "t_brood"].std() < 0.8


def test_queenless_hive_loses_brood():
    r = sim.simulate_hive("SMA", 2023, sim.Scenario(queenless=True), seed=5)
    q = r.events[r.events["type"] == "queen_loss"]["start"].iloc[0]
    late = r.data.loc[q + pd.Timedelta(days=40)]["true_brood"]
    assert late < 0.05


def test_theft_event_zeroes_scale():
    r = sim.simulate_hive("SMA", 2023, sim.Scenario(theft=True), seed=2)
    t = r.events[r.events["type"] == "theft"]["start"].iloc[0]
    assert r.data["weight_raw_kg"][t + pd.Timedelta(hours=2)] < 2.0
    assert r.data["tilt_dev_deg"][t] > 10


def test_random_scenarios_vary():
    rng = np.random.default_rng(0)
    s = [sim.random_scenario(rng) for _ in range(50)]
    assert 0 < sum(x.swarm for x in s) < 50


def test_mustb_fixture_is_clean():
    w = realdata.load_mustb_weight()
    assert w["time"].min() >= pd.Timestamp("2019-03-01")
    assert w["weight_kg"].between(5, 150).all()
    assert w.groupby(["site", "hive"]).ngroups >= 50


def test_mustb_logged_swarms_present():
    ev = realdata.load_mustb_events()
    assert (ev.eventType == "SWARM").sum() == 2


def test_real_swarm_detection_finds_both_logged_swarms():
    r = realdata.real_swarm_eval()
    assert r["logged_swarms"] == 2 and r["logged_swarms_detected"] == 2


def test_real_brood_temperature_summary():
    r = realdata.real_brood_temperature()
    assert set(r) == {"A1", "A2"}
    assert 33 < r["A1"]["median_daily_mean_c"] < 37


def test_acoustic_features_shape_and_labels():
    import importlib.resources as ir

    with ir.files("hivemon").joinpath("data/audio_features.csv").open("rb") as f:
        feat = pd.read_csv(f)
    assert set(feat.queenless) == {0, 1} and set(feat.hive) == {"H1", "H3"}
    assert feat[acoustics.FEATURES].notna().all().all()


def test_acoustic_window_features_pure_tone():
    sr = 8000
    t = np.arange(sr * 4) / sr
    f = acoustics.window_features(np.sin(2 * np.pi * 250 * t), sr)
    assert abs(f["peak_hz"] - 250) < 5
    assert f["b200_300"] > 0.9


def test_acoustic_holdout_needs_two_hives():
    df = pd.DataFrame({"hive": ["H1"] * 4, "queenless": [0, 0, 1, 1], "file": list("aabb"),
                       **{k: [0.1, 0.2, 0.3, 0.4] for k in acoustics.FEATURES}})
    with pytest.raises(ValueError):
        acoustics.evaluate_holdout(df)


def test_parse_name_queen_label():
    m = acoustics.parse_name("Hive3_12_07_2017_NO_QueenBee_H3_audio___15_00_00.wav")
    assert m["hive"] == "H3" and m["queenless"] == 1
    assert acoustics.parse_name("Hive1_12_06_2018_QueenBee_H1_audio___15_00_00.wav")["queenless"] == 0
