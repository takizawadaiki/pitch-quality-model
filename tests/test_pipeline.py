"""Checks that the code does what the write-up says it does. Run with: python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from pitchq import evaluate, model, prepare, synthetic


@pytest.fixture(scope="module")
def raw():
    return synthetic.make(n_pitchers=70, pitches_per_season=650)


@pytest.fixture(scope="module")
def prepared(raw):
    return prepare.add_features(raw)


@pytest.fixture(scope="module")
def fitted(prepared, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    booster = model.fit(prepared, out)
    return booster, model.score(booster, prepared), out


# ---- no information from after the pitch may reach the model
def test_features_contain_no_outcomes():
    assert not set(config.FEATURES) & set(config.OUTCOME_COLUMNS)
    assert config.TARGET not in config.FEATURES


def test_seasons_are_strictly_ordered():
    assert max(config.TRAIN_SEASONS) < config.TUNE_SEASON < config.SCORE_SEASON < config.TARGET_SEASON


def test_features_ignore_results(raw):
    """Scrambling every outcome column must leave the model inputs unchanged."""
    shuffled = raw.copy()
    rng = np.random.default_rng(0)
    for col in ["delta_run_exp", "events", "woba_value", "woba_denom", "estimated_woba_using_speedangle"]:
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    a, b = prepare.add_features(raw), prepare.add_features(shuffled)
    pd.testing.assert_frame_equal(a[config.FEATURES], b[config.FEATURES])


# ---- handedness
def test_left_and_right_handers_are_mirrored():
    """A left-hander who is the exact mirror image of a right-hander gets identical inputs."""
    base = dict(game_year=2021, pitch_type="FF", stand="R", release_speed=95.0, pfx_z=1.3,
                release_spin_rate=2300.0, release_extension=6.5, release_pos_z=6.0)
    right = dict(base, pitcher=1, p_throws="R", pfx_x=-0.6, release_pos_x=-2.0, spin_axis=205.0)
    left = dict(base, pitcher=2, p_throws="L", pfx_x=0.6, release_pos_x=2.0, spin_axis=155.0, stand="L")
    out = prepare.add_features(pd.DataFrame([right, left]))
    cols = [c for c in config.FEATURES if c != "same_hand"]
    np.testing.assert_allclose(out.loc[0, cols].astype(float), out.loc[1, cols].astype(float))
    assert out.loc[0, "hmov"] > 0                       # arm side is positive


def test_nullable_column_types_are_handled(raw):
    """Downloads can use pandas' nullable types; the inputs must come out as plain floats either way."""
    sample = raw.head(4000)
    plain, nullable = prepare.add_features(sample), prepare.add_features(sample.convert_dtypes())
    assert all(str(t) == "float64" for t in nullable[config.FEATURES].dtypes)
    np.testing.assert_allclose(plain[config.FEATURES].to_numpy(), nullable[config.FEATURES].to_numpy())
    assert str(nullable[config.TARGET].dtype) == "float64"


def test_fastball_is_its_own_reference(prepared):
    primary = prepared.loc[prepared["pitch_type"] == "FF"].groupby(["pitcher", "game_year"])["velo_diff"].mean()
    assert primary.abs().max() < 1e-6


def test_excluded_pitch_types_are_dropped(raw):
    odd = raw.head(50).assign(pitch_type="PO")
    assert prepare.add_features(odd).empty


def test_run_value_direction_guard(prepared):
    prepare.check_run_value_direction(prepared)          # passes as generated
    flipped = prepared.assign(delta_run_exp=-prepared["delta_run_exp"])
    with pytest.raises(SystemExit):
        prepare.check_run_value_direction(flipped)


# ---- pitcher statistics, checked by hand
def test_plate_appearance_stats_by_hand():
    events = ["strikeout", "strikeout", "walk", "home_run", "field_out", "hit_by_pitch", "single", "stolen_base_2b", None, None]
    woba = [0, 0, .69, 2.0, 0, .72, .88, np.nan, np.nan, np.nan]
    df = pd.DataFrame(dict(pitcher=1, game_year=2025, xrv=0.01, delta_run_exp=0.02, events=events,
                           woba_value=woba, woba_denom=[1, 1, 1, 1, 1, 1, 1, np.nan, np.nan, np.nan],
                           estimated_woba_using_speedangle=[np.nan, np.nan, np.nan, 1.5, .1, np.nan, .5, np.nan, np.nan, np.nan]))
    row = evaluate.pitcher_seasons(df).iloc[0]
    assert row["n_pitches"] == 10 and row["pa"] == 7     # the stolen base is not a plate appearance
    assert row["k_bb"] == pytest.approx((2 - 1) / 7)
    assert row["fip_core"] == pytest.approx((13 * 1 + 3 * (1 + 1) - 2 * 2) / 7)
    assert row["woba"] == pytest.approx((.69 + 2.0 + .72 + .88) / 7)
    assert row["xwoba"] == pytest.approx((0 + 0 + .69 + 1.5 + .1 + .72 + .5) / 7)
    assert row["stuff"] == pytest.approx(-1.0) and row["rv100"] == pytest.approx(-2.0)


def test_correlation_interval_contains_estimate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    c = evaluate.corr_with_interval(x, x * .5 + rng.normal(size=200), np.random.default_rng(2))
    assert c["lo"] < c["r"] < c["hi"] and 0.2 < c["r"] < 0.7


def test_incremental_value_detects_a_real_extra_signal():
    """If the rating truly carries extra information, the error must fall; if it is noise, it must not."""
    rng = np.random.default_rng(3)
    n = 300
    base = rng.normal(size=(n, 3))
    signal, noise = rng.normal(size=n), rng.normal(size=n)
    y = base @ np.array([.5, .3, .2]) + .8 * signal + rng.normal(scale=.5, size=n)
    cols = evaluate.RESULT_STATS

    def pair(stuff):
        d = pd.DataFrame(base, columns=cols).assign(stuff=stuff)
        for t in evaluate.TARGETS:
            d[t + "_next"] = y
        return d
    helpful = evaluate.incremental_value(pair(signal), np.random.default_rng(4))["k_bb"]
    useless = evaluate.incremental_value(pair(noise), np.random.default_rng(4))["k_bb"]
    assert helpful["delta_rmse"] < 0 and helpful["delta_rmse_ci"][1] < 0
    assert useless["delta_rmse_ci"][0] < 0 < useless["delta_rmse_ci"][1] or useless["delta_rmse"] >= 0


# ---- the whole pipeline, on data with a planted answer
def test_model_trains_only_on_training_seasons(fitted):
    report = json.loads((fitted[2] / "model_report.json").read_text())
    assert report["train_seasons"] == config.TRAIN_SEASONS
    roles = {int(s): v["role"] for s, v in report["pitch_level"].items()}
    assert roles[config.SCORE_SEASON] == "unseen" and roles[config.TARGET_SEASON] == "unseen"


def test_model_beats_a_constant_on_unseen_seasons(fitted):
    report = json.loads((fitted[2] / "model_report.json").read_text())
    for season in (config.SCORE_SEASON, config.TARGET_SEASON):
        assert report["pitch_level"][str(season)]["r2"] > 0


def test_ratings_recover_the_planted_quality(fitted):
    scored = fitted[1]
    season = scored.loc[scored["game_year"] == config.SCORE_SEASON]
    by_pitcher = season.groupby("pitcher").agg(rating=("xrv", "mean"), truth=("true_xrv", "mean"))
    assert np.corrcoef(by_pitcher["rating"], by_pitcher["truth"])[0, 1] > 0.9


def test_evaluation_writes_complete_results(fitted):
    _, scored, out = fitted
    metrics = evaluate.run(scored, {}, out)
    for name in ("metrics.json", "summary.md", "leaderboard.csv", "pitch_types.csv", "pitcher_seasons.csv",
                 "rating_vs_next_season.png", "predictors_of_next_season.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0
    final = metrics["final"]
    assert (final["this_season"], final["next_season"]) == (config.SCORE_SEASON, config.TARGET_SEASON)
    assert final["correlations"]["k_bb"]["stuff"]["r"] > 0.3        # planted: better stuff, more strikeouts
    assert metrics["reliability"]["stuff"]["r"] > metrics["reliability"]["rv100"]["r"]
    board = pd.read_csv(out / "leaderboard.csv")
    assert board["stuff_plus"].is_monotonic_decreasing and (board["n_pitches"] >= config.MIN_PITCHES).all()
