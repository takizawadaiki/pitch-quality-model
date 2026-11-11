"""Fit the pitch-level model and score pitches.

The model answers one question: given only how a pitch left the hand, what change in run
expectancy should we expect? It is trained on earlier seasons and never sees later ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import config


def fit(df: pd.DataFrame, results_dir: str | Path = config.RESULTS_DIR) -> lgb.Booster:
    """Train on TRAIN_SEASONS, stop on TUNE_SEASON, save the model and a short report."""
    rows = df.dropna(subset=[config.TARGET])
    train = rows.loc[rows["game_year"].isin(config.TRAIN_SEASONS)]
    tune = rows.loc[rows["game_year"] == config.TUNE_SEASON]
    if train.empty or tune.empty:
        raise SystemExit("Training or tuning season has no rows. Check config.py and the download.")
    # Guard against the one mistake that would invalidate everything: training on the future.
    assert train["game_year"].max() < config.TUNE_SEASON < config.SCORE_SEASON < config.TARGET_SEASON

    booster = lgb.train(
        dict(config.LGBM_PARAMS, seed=config.SEED),
        lgb.Dataset(train[config.FEATURES], label=train[config.TARGET]),
        num_boost_round=config.MAX_ROUNDS,
        valid_sets=[lgb.Dataset(tune[config.FEATURES], label=tune[config.TARGET])],
        callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    report = dict(
        train_seasons=config.TRAIN_SEASONS, tune_season=config.TUNE_SEASON,
        train_pitches=int(len(train)), tune_pitches=int(len(tune)),
        rounds=int(booster.best_iteration), features=config.FEATURES,
        pitch_level=pitch_level_fit(booster, rows, float(train[config.TARGET].mean())),
        importance_gain=dict(sorted(
            zip(config.FEATURES, (float(v) for v in booster.feature_importance("gain"))),
            key=lambda kv: -kv[1])),
    )
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(results_dir / "model.txt"), num_iteration=booster.best_iteration)
    (results_dir / "model_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return booster


def pitch_level_fit(booster: lgb.Booster, rows: pd.DataFrame, train_mean: float) -> dict:
    """Error on single pitches, by season, against always guessing the training average.

    Expect tiny gains: one pitch is almost all noise. The model earns its keep when thousands
    of pitches are averaged for a pitcher, which evaluate.py tests.
    """
    out = {}
    for season, part in rows.groupby("game_year"):
        y = part[config.TARGET].to_numpy()
        pred = booster.predict(part[config.FEATURES], num_iteration=booster.best_iteration)
        rmse_model = float(np.sqrt(np.mean((y - pred) ** 2)))
        rmse_mean = float(np.sqrt(np.mean((y - train_mean) ** 2)))
        role = ("train" if season in config.TRAIN_SEASONS
                else "tune" if season == config.TUNE_SEASON else "unseen")
        out[str(int(season))] = dict(role=role, pitches=int(len(part)), rmse_model=rmse_model,
                                     rmse_constant=rmse_mean,
                                     r2=float(1 - (rmse_model / rmse_mean) ** 2))
    return out


def score(booster: lgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    """Add 'xrv': the expected run value of each pitch from its shape alone (batting side)."""
    out = df.copy()
    out["xrv"] = booster.predict(out[config.FEATURES], num_iteration=booster.best_iteration)
    return out


def load(results_dir: str | Path = config.RESULTS_DIR) -> lgb.Booster:
    path = Path(results_dir) / "model.txt"
    if not path.exists():
        raise SystemExit("No trained model found. Run: python run.py train")
    return lgb.Booster(model_file=str(path))
