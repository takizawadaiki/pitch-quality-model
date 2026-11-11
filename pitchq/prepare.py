"""Turn raw Statcast rows into model inputs.

Everything computed here describes the pitch as it leaves the hand. Nothing here may use what
happened afterwards (location, call, contact, result); tests/test_pipeline.py enforces that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

PHYSICAL = ["release_speed", "pfx_x", "pfx_z", "release_spin_rate", "spin_axis",
            "release_extension", "release_pos_x", "release_pos_z"]
NUMERIC_RESULTS = ["delta_run_exp", "woba_value", "woba_denom", "estimated_woba_using_speedangle"]


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean rows and add the columns named in config.FEATURES. Returns a new frame."""
    df = raw.loc[raw["pitch_type"].isin(config.PITCH_TYPES)].copy()
    # Statcast downloads can arrive with pandas' nullable column types, which the model library
    # does not accept. Make every number a plain float, with NaN for missing.
    for col in PHYSICAL + NUMERIC_RESULTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df = df.dropna(subset=PHYSICAL + ["p_throws", "stand", "pitcher", "game_year"])
    df["p_throws"] = df["p_throws"].astype(str)
    df["stand"] = df["stand"].astype(str)
    df["pitch_type"] = df["pitch_type"].astype(str)
    df["game_year"] = df["game_year"].astype(int)
    df["pitcher"] = df["pitcher"].astype(int)

    # Statcast reports horizontal numbers from the catcher's view, so a right-hander's arm side
    # is negative and a left-hander's is positive. Flip right-handers so "arm side" is positive
    # for everyone, and a lefty and his mirror-image righty get identical inputs.
    flip = np.where(df["p_throws"] == "R", -1.0, 1.0)
    df["hmov"] = df["pfx_x"] * 12.0 * flip              # feet -> inches
    df["vmov"] = df["pfx_z"] * 12.0
    df["release_x_adj"] = df["release_pos_x"] * flip
    df["spin_axis_adj"] = np.where(df["p_throws"] == "R", df["spin_axis"], 360.0 - df["spin_axis"])
    df["same_hand"] = (df["p_throws"] == df["stand"]).astype(int)

    # Each pitch relative to that pitcher's own main fastball in the same season.
    fb = primary_fastball(df)
    df = df.merge(fb, on=["pitcher", "game_year"], how="left")
    df["velo_diff"] = df["release_speed"] - df["fb_velo"]
    df["hmov_diff"] = df["hmov"] - df["fb_hmov"]
    df["vmov_diff"] = df["vmov"] - df["fb_vmov"]
    df[config.FEATURES] = df[config.FEATURES].astype("float64")
    return df.drop(columns=["fb_velo", "fb_hmov", "fb_vmov"]).reset_index(drop=True)


def primary_fastball(df: pd.DataFrame) -> pd.DataFrame:
    """Average velocity and movement of each pitcher-season's most-thrown fastball.

    A pitcher with no fastball at all falls back to his most-thrown pitch.
    """
    counts = (df.groupby(["pitcher", "game_year", "pitch_type"]).size().rename("n").reset_index())
    counts["is_fb"] = counts["pitch_type"].isin(config.FASTBALLS)
    # Prefer fastballs, then the largest count; ties broken by pitch-type name so it is repeatable.
    counts = counts.sort_values(["pitcher", "game_year", "is_fb", "n", "pitch_type"],
                                ascending=[True, True, False, False, True])
    main = counts.drop_duplicates(["pitcher", "game_year"])[["pitcher", "game_year", "pitch_type"]]
    picked = df.merge(main, on=["pitcher", "game_year", "pitch_type"], how="inner")
    return (picked.groupby(["pitcher", "game_year"])
            .agg(fb_velo=("release_speed", "mean"), fb_hmov=("hmov", "mean"), fb_vmov=("vmov", "mean"))
            .reset_index())


def check_run_value_direction(df: pd.DataFrame) -> None:
    """delta_run_exp must be from the batting side: strikeouts negative, home runs positive.

    If Statcast ever changes the sign convention, every rating would silently flip. Stop instead.
    """
    ended = df.dropna(subset=["events", config.TARGET])
    k = ended.loc[ended["events"] == "strikeout", config.TARGET].mean()
    hr = ended.loc[ended["events"] == "home_run", config.TARGET].mean()
    if not (k < 0 < hr):
        raise SystemExit(f"Unexpected run-value signs (strikeout {k:.3f}, home run {hr:.3f}). "
                         "Check the delta_run_exp definition before trusting any output.")
