"""Run the study, one stage at a time or all at once.

    python run.py all                 download, prepare, train, evaluate (real Statcast data)
    python run.py download            fetch any months not yet on disk
    python run.py train               fit the model on the training seasons
    python run.py evaluate            score every pitch and judge the ratings
    python run.py all --synthetic     same pipeline on made-up data, to check the code works
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from pitchq import download, evaluate, model, prepare, synthetic


def prepared(args) -> pd.DataFrame:
    """Raw pitches -> cleaned rows with model inputs. Cached as one parquet file."""
    out = Path(config.PREPARED_DIR) / ("synthetic.parquet" if args.synthetic else "pitches.parquet")
    if out.exists() and not args.refresh:
        return pd.read_parquet(out)
    raw = synthetic.make() if args.synthetic else download.load_raw(config.ALL_SEASONS)
    df = prepare.add_features(raw)
    prepare.check_run_value_direction(df)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Prepared {len(df):,} pitches from {df['game_year'].nunique()} seasons "
          f"({len(raw) - len(df):,} rows dropped for missing tracking data or excluded pitch types).")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["download", "prepare", "train", "evaluate", "all"])
    parser.add_argument("--synthetic", action="store_true", help="use made-up data (for testing the code only)")
    parser.add_argument("--refresh", action="store_true", help="rebuild the prepared file even if it exists")
    args = parser.parse_args()
    results = Path("results_synthetic" if args.synthetic else config.RESULTS_DIR)   # never mix the two

    if args.stage in ("download", "all") and not args.synthetic:
        download.download_all()
    if args.stage == "download":
        return
    df = prepared(args)
    if args.stage == "prepare":
        return
    booster = model.fit(df, results) if args.stage in ("train", "all") else model.load(results)
    if args.stage == "train":
        return
    scored = model.score(booster, df)
    qualified = scored.groupby("pitcher").size()
    names = {} if args.synthetic else download.pitcher_names(qualified[qualified >= config.MIN_PITCHES].index)
    metrics = evaluate.run(scored, names, results)
    print((results / "summary.md").read_text())
    print(f"Wrote {results}/metrics.json, summary.md, leaderboard.csv, pitch_types.csv and two figures.")
    if args.synthetic:
        print("\nThese numbers come from made-up data. They show the code runs; they are not findings.")


if __name__ == "__main__":
    main()
