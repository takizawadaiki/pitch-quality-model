"""Download pitch-level Statcast data, one month at a time, and keep it as parquet files.

Files that already exist are skipped, so an interrupted download can simply be run again.
Source: Baseball Savant (baseballsavant.mlb.com) through the pybaseball package.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

import config

# Regular seasons fall inside these months. Spring training and the postseason are removed
# afterwards with game_type == "R", so generous bounds are safe.
MONTHS = [(3, 15, 3, 31), (4, 1, 4, 30), (5, 1, 5, 31), (6, 1, 6, 30),
          (7, 1, 7, 31), (8, 1, 8, 31), (9, 1, 9, 30), (10, 1, 10, 10)]


def month_file(raw_dir: Path, season: int, month: int) -> Path:
    return raw_dir / f"statcast_{season}_{month:02d}.parquet"


def download_season(season: int, raw_dir: str | Path = config.RAW_DIR, pause: float = 2.0) -> list[Path]:
    """Fetch one season. Returns the list of month files (downloaded or already present)."""
    from pybaseball import statcast  # imported here so tests do not need pybaseball installed

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for m1, d1, m2, d2 in MONTHS:
        out = month_file(raw_dir, season, m1)
        files.append(out)
        if out.exists():
            print(f"  {out.name}: already downloaded")
            continue
        start, end = date(season, m1, d1), date(season, m2, d2)
        if start > date.today():
            raise SystemExit(f"{season} has not been played yet. Check the seasons in config.py.")
        print(f"  {out.name}: downloading {start} to {end} ...", flush=True)
        frame = statcast(start_dt=start.isoformat(), end_dt=end.isoformat(), verbose=False)
        if frame is None or frame.empty:
            raise SystemExit(f"No data returned for {start} to {end}. Baseball Savant may be "
                             "blocking this network; try again later or from another connection.")
        missing = [c for c in config.RAW_COLUMNS if c not in frame.columns]
        if missing:
            raise SystemExit(f"Statcast no longer returns these columns: {missing}")
        frame = frame.loc[frame["game_type"] == "R", config.RAW_COLUMNS].reset_index(drop=True)
        frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.strftime("%Y-%m-%d")
        tmp = out.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.rename(out)                      # a half-written file never looks finished
        print(f"    kept {len(frame):,} regular-season pitches")
        time.sleep(pause)                    # be polite to the server
    return files


def download_all(seasons=config.ALL_SEASONS, raw_dir: str | Path = config.RAW_DIR) -> None:
    raw_dir = Path(raw_dir)
    counts = {}
    for season in seasons:
        print(f"Season {season}")
        files = download_season(season, raw_dir)
        counts[str(season)] = int(sum(len(pd.read_parquet(f, columns=["pitcher"])) for f in files))
    manifest = dict(
        source="Baseball Savant Statcast search, via pybaseball.statcast",
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        regular_season_pitches=counts,
        columns=config.RAW_COLUMNS,
    )
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.RESULTS_DIR, "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Pitches per season:", counts)


def load_raw(seasons, raw_dir: str | Path = config.RAW_DIR) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(f for s in seasons for f in raw_dir.glob(f"statcast_{s}_*.parquet"))
    if not files:
        raise SystemExit(f"No raw files for {list(seasons)} in {raw_dir}. Run: python run.py download")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def pitcher_names(ids) -> dict[int, str]:
    """MLBAM id -> 'First Last'. Falls back to the id if the lookup is unavailable."""
    ids = [int(i) for i in ids]
    try:
        from pybaseball import playerid_reverse_lookup
        table = playerid_reverse_lookup(ids, key_type="mlbam")
        return {int(r.key_mlbam): f"{str(r.name_first).title()} {str(r.name_last).title()}"
                for r in table.itertuples()}
    except Exception as error:  # offline, or the register is unavailable
        print(f"Name lookup unavailable ({type(error).__name__}); using ids.")
        return {}
