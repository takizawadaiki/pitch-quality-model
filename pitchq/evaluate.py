"""Judge the ratings at the level that matters: the pitcher, and the following season.

Three questions, each answered with an interval rather than a single number:
  1. Does this season's rating track next season's results, and how does that compare with
     simply using this season's results?                                   (correlations)
  2. Does the rating add anything once this season's results are known?    (incremental value)
  3. How much of the rating is signal rather than luck?                    (reliability)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Plate appearances that count. Steals, pickoffs, wild pitches and similar are not included.
PA_EVENTS = {
    "strikeout", "strikeout_double_play", "walk", "intent_walk", "hit_by_pitch",
    "single", "double", "triple", "home_run", "field_out", "force_out",
    "grounded_into_double_play", "double_play", "triple_play", "fielders_choice",
    "fielders_choice_out", "field_error", "sac_fly", "sac_fly_double_play", "sac_bunt",
    "sac_bunt_double_play", "catcher_interf",
}
STRIKEOUTS = {"strikeout", "strikeout_double_play"}

# Results-based measures a team already has, used as the benchmark the rating must beat.
RESULT_STATS = ["k_bb", "fip_core", "xwoba"]
PREDICTORS = ["stuff", "k_bb", "fip_core", "woba", "xwoba", "rv100"]
TARGETS = ["k_bb", "xwoba", "rv100"]
LABELS = {
    "stuff": "Pitch quality rating (this model)",
    "k_bb": "Strikeout minus walk rate",
    "fip_core": "FIP components per batter (13·HR + 3·(BB+HBP) − 2·K)",
    "woba": "wOBA allowed",
    "xwoba": "Expected wOBA allowed",
    "rv100": "Runs prevented per 100 pitches",
}


# ------------------------------------------------------------------ pitcher-season table
def pitcher_seasons(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher-season: the rating, and what actually happened."""
    g = scored.groupby(["pitcher", "game_year"])
    table = g.agg(n_pitches=("xrv", "size"), xrv=("xrv", "mean"),
                  rv=(config.TARGET, "mean")).reset_index()
    table["stuff"] = -100.0 * table["xrv"]        # runs prevented per 100 pitches, from shape
    table["rv100"] = -100.0 * table["rv"]         # runs prevented per 100 pitches, actual

    pa = scored.loc[scored["events"].isin(PA_EVENTS)].copy()
    pa["k"] = pa["events"].isin(STRIKEOUTS)
    pa["bb"] = pa["events"] == "walk"             # unintentional walks only
    pa["hbp"] = pa["events"] == "hit_by_pitch"
    pa["hr"] = pa["events"] == "home_run"
    counted = pa["woba_denom"].fillna(0) > 0
    pa["woba_num"] = pa["woba_value"].where(counted, 0.0).fillna(0.0)
    pa["woba_den"] = counted.astype(float)
    # Expected wOBA: the batted-ball estimate when the ball was put in play, the actual value otherwise.
    pa["xwoba_num"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"]).where(counted, 0.0).fillna(0.0)
    sums = (pa.groupby(["pitcher", "game_year"])
            [["k", "bb", "hbp", "hr", "woba_num", "woba_den", "xwoba_num"]].sum()
            .join(pa.groupby(["pitcher", "game_year"]).size().rename("pa")).reset_index())
    sums["k_bb"] = (sums["k"] - sums["bb"]) / sums["pa"]
    sums["fip_core"] = (13 * sums["hr"] + 3 * (sums["bb"] + sums["hbp"]) - 2 * sums["k"]) / sums["pa"]
    sums["woba"] = sums["woba_num"] / sums["woba_den"].replace(0, np.nan)
    sums["xwoba"] = sums["xwoba_num"] / sums["woba_den"].replace(0, np.nan)
    keep = ["pitcher", "game_year", "pa", "k_bb", "fip_core", "woba", "xwoba"]
    return table.drop(columns=["xrv", "rv"]).merge(sums[keep], on=["pitcher", "game_year"], how="left")


def add_stuff_plus(table: pd.DataFrame) -> pd.DataFrame:
    """Scale to 100 = average, 10 points = one standard deviation, fixed on training seasons."""
    ref = table.loc[table["game_year"].isin(config.TRAIN_SEASONS) & (table["n_pitches"] >= config.MIN_PITCHES), "stuff"]
    out = table.copy()
    out["stuff_plus"] = 100.0 + 10.0 * (out["stuff"] - ref.mean()) / ref.std(ddof=0)
    return out


def season_pair(table: pd.DataFrame, this_year: int, next_year: int) -> pd.DataFrame:
    """Pitchers with enough pitches in both seasons; next season's columns end in '_next'."""
    ok = table.loc[table["n_pitches"] >= config.MIN_PITCHES]
    a = ok.loc[ok["game_year"] == this_year]
    b = ok.loc[ok["game_year"] == next_year, ["pitcher", "stuff"] + TARGETS]
    pair = a.merge(b, on="pitcher", suffixes=("", "_next"))
    return pair.dropna(subset=PREDICTORS + [t + "_next" for t in TARGETS]).reset_index(drop=True)


# ------------------------------------------------------------------ 1. correlations
def corr_with_interval(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Pearson r with a 95% interval from resampling pitchers."""
    n = len(x)
    idx = rng.integers(0, n, size=(config.BOOTSTRAP_REPS, n))
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(axis=1, keepdims=True)
    ys = ys - ys.mean(axis=1, keepdims=True)
    reps = (xs * ys).sum(axis=1) / np.sqrt((xs ** 2).sum(axis=1) * (ys ** 2).sum(axis=1))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(r=float(np.corrcoef(x, y)[0, 1]), lo=float(lo), hi=float(hi))


def correlations(pair: pd.DataFrame, rng: np.random.Generator) -> dict:
    out = {}
    for target in TARGETS:
        y = pair[target + "_next"].to_numpy(float)
        out[target] = {p: corr_with_interval(pair[p].to_numpy(float), y, rng) for p in PREDICTORS}
    out["stuff_to_stuff"] = corr_with_interval(pair["stuff"].to_numpy(float), pair["stuff_next"].to_numpy(float), rng)
    return out


# ------------------------------------------------------------------ 2. incremental value
def out_of_fold_sq_errors(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-pitcher squared error of a linear model that never saw that pitcher, averaged over repeats."""
    n = len(y)
    errors = np.zeros(n)
    for _ in range(config.CV_REPEATS):
        order = rng.permutation(n)
        for fold in np.array_split(order, config.CV_FOLDS):
            train = np.setdiff1d(order, fold, assume_unique=True)
            mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
            sd[sd == 0] = 1.0
            A = np.column_stack([np.ones(len(train)), (X[train] - mu) / sd])
            beta, *_ = np.linalg.lstsq(A, y[train], rcond=None)
            pred = np.column_stack([np.ones(len(fold)), (X[fold] - mu) / sd]) @ beta
            errors[fold] += (y[fold] - pred) ** 2
    return errors / config.CV_REPEATS


def incremental_value(pair: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Predict next season from this season's results, with and without the rating."""
    out = {}
    for target in TARGETS:
        y = pair[target + "_next"].to_numpy(float)
        base = out_of_fold_sq_errors(pair[RESULT_STATS].to_numpy(float), y, rng)
        full = out_of_fold_sq_errors(pair[RESULT_STATS + ["stuff"]].to_numpy(float), y, rng)
        idx = rng.integers(0, len(y), size=(config.BOOTSTRAP_REPS, len(y)))
        deltas = np.sqrt(full[idx].mean(axis=1)) - np.sqrt(base[idx].mean(axis=1))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        var = float(np.var(y))
        out[target] = dict(
            rmse_results_only=float(np.sqrt(base.mean())), rmse_with_rating=float(np.sqrt(full.mean())),
            delta_rmse=float(np.sqrt(full.mean()) - np.sqrt(base.mean())),   # negative = the rating helps
            delta_rmse_ci=[float(lo), float(hi)],
            r2_results_only=float(1 - base.mean() / var), r2_with_rating=float(1 - full.mean() / var),
        )
    return out


# ------------------------------------------------------------------ 3. reliability
def split_half(scored: pd.DataFrame, season: int, rng: np.random.Generator) -> dict:
    """Split each pitcher's games into alternating halves and compare the two halves.

    A rating that is mostly skill gives nearly the same answer in both halves; results that are
    mostly luck do not. Spearman-Brown projects the half-season figure to a full season.
    """
    part = scored.loc[scored["game_year"] == season, ["pitcher", "game_pk", "xrv", config.TARGET]].copy()
    games = part[["pitcher", "game_pk"]].drop_duplicates().sort_values(["pitcher", "game_pk"])
    games["half"] = games.groupby("pitcher").cumcount() % 2
    part = part.merge(games, on=["pitcher", "game_pk"])
    g = part.groupby(["pitcher", "half"]).agg(n=("xrv", "size"), stuff=("xrv", "mean"), rv=(config.TARGET, "mean"))
    wide = g.unstack("half").dropna()
    wide = wide.loc[(wide[("n", 0)] >= config.MIN_PITCHES / 2) & (wide[("n", 1)] >= config.MIN_PITCHES / 2)]
    out = dict(season=int(season), pitchers=int(len(wide)))
    for name, col in (("stuff", "stuff"), ("rv100", "rv")):
        c = corr_with_interval(wide[(col, 0)].to_numpy(float), wide[(col, 1)].to_numpy(float), rng)
        c["full_season"] = float(2 * c["r"] / (1 + c["r"])) if c["r"] > -1 else float("nan")
        out[name] = c
    return out


# ------------------------------------------------------------------ outputs
def leaderboards(scored: pd.DataFrame, table: pd.DataFrame, names: dict, results_dir: Path) -> None:
    season = table.loc[(table["game_year"] == config.SCORE_SEASON) & (table["n_pitches"] >= config.MIN_PITCHES)].copy()
    season["name"] = season["pitcher"].map(names).fillna(season["pitcher"].astype(str))
    cols = ["name", "pitcher", "n_pitches", "stuff_plus", "stuff", "rv100", "k_bb", "xwoba"]
    season.sort_values("stuff_plus", ascending=False)[cols].round(3).to_csv(results_dir / "leaderboard.csv", index=False)

    part = scored.loc[scored["game_year"] == config.SCORE_SEASON]
    by_type = part.groupby(["pitcher", "pitch_type"]).agg(
        n=("xrv", "size"), velo=("release_speed", "mean"), hmov=("hmov", "mean"),
        vmov=("vmov", "mean"), stuff=("xrv", lambda s: -100.0 * s.mean())).reset_index()
    by_type = by_type.loc[by_type["n"] >= 100]
    by_type["name"] = by_type["pitcher"].map(names).fillna(by_type["pitcher"].astype(str))
    (by_type.sort_values("stuff", ascending=False)
     [["name", "pitcher", "pitch_type", "n", "velo", "hmov", "vmov", "stuff"]].round(2)
     .to_csv(results_dir / "pitch_types.csv", index=False))


def figures(pair: pd.DataFrame, corr: dict, results_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, mark = "#1c1f1a", "#6b7264", "#2a78d6"
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    ax.scatter(pair["stuff_plus"], pair["xwoba_next"], s=16, color=mark, alpha=.75, edgecolor="white", linewidth=.4)
    slope, intercept = np.polyfit(pair["stuff_plus"], pair["xwoba_next"], 1)
    xs = np.array([pair["stuff_plus"].min(), pair["stuff_plus"].max()])
    ax.plot(xs, intercept + slope * xs, color=ink, linewidth=1.2, linestyle="--")
    c = corr["xwoba"]["stuff"]
    ax.set_title(f"{config.SCORE_SEASON} rating vs {config.TARGET_SEASON} expected wOBA allowed\n"
                 f"r = {c['r']:.2f}  (95% interval {c['lo']:.2f} to {c['hi']:.2f}),  {len(pair)} pitchers",
                 fontsize=10, color=ink, loc="left")
    ax.set_xlabel(f"Pitch quality rating, {config.SCORE_SEASON} (100 = average)", color=muted)
    ax.set_ylabel(f"Expected wOBA allowed, {config.TARGET_SEASON}", color=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e6e8e1", linewidth=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(results_dir / "rating_vs_next_season.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    rows = sorted(PREDICTORS, key=lambda p: abs(corr["k_bb"][p]["r"]))
    for i, p in enumerate(rows):
        c = corr["k_bb"][p]
        lo, hi = sorted((abs(c["lo"]), abs(c["hi"]))) if c["lo"] * c["hi"] > 0 else (0.0, max(abs(c["lo"]), abs(c["hi"])))
        ax.plot([lo, hi], [i, i], color=muted, linewidth=2, solid_capstyle="round")
        ax.scatter([abs(c["r"])], [i], s=44, color=mark if p == "stuff" else ink, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[p].split(" (")[0] for p in rows], fontsize=8, color=ink)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Strength of link with {config.TARGET_SEASON} strikeout-minus-walk rate (|r|, 95% interval)", color=muted, fontsize=8)
    ax.set_title("Which this-season number says most about next season?", fontsize=10, color=ink, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e6e8e1", linewidth=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(results_dir / "predictors_of_next_season.png")
    plt.close(fig)


def summary_markdown(metrics: dict) -> str:
    """The numbers in one readable table. Interpretation belongs in the README, written by a person."""
    f, lines = metrics["final"], []
    lines += [f"# Results: {config.SCORE_SEASON} ratings judged on {config.TARGET_SEASON}", "",
              f"{f['pitchers']} pitchers with at least {config.MIN_PITCHES} pitches in both seasons.", "",
              "## 1. Link with next season (Pearson r, 95% interval)", "",
              "| This season | " + " | ".join(f"Next-season {t}" for t in TARGETS) + " |",
              "|---|" + "---|" * len(TARGETS)]
    for p in PREDICTORS:
        cells = [f"{f['correlations'][t][p]['r']:+.2f} ({f['correlations'][t][p]['lo']:+.2f} to {f['correlations'][t][p]['hi']:+.2f})" for t in TARGETS]
        lines.append(f"| {LABELS[p]} | " + " | ".join(cells) + " |")
    lines += ["", "Lower is better for wOBA, xwOBA and FIP components, so a useful predictor of those shows a negative r.",
              "", "## 2. Does the rating add to this season's results?", "",
              "Out-of-sample error predicting next season from k_bb, fip_core and xwoba, with and without the rating.",
              "A negative change whose interval excludes zero means the rating adds information.", "",
              "| Next-season target | RMSE, results only | RMSE, with rating | Change (95% interval) |", "|---|---|---|---|"]
    for t in TARGETS:
        v = f["incremental"][t]
        lines.append(f"| {t} | {v['rmse_results_only']:.4f} | {v['rmse_with_rating']:.4f} | {v['delta_rmse']:+.4f} ({v['delta_rmse_ci'][0]:+.4f} to {v['delta_rmse_ci'][1]:+.4f}) |")
    r = metrics["reliability"]
    lines += ["", "## 3. Signal versus luck", "",
              f"Split-half agreement within {r['season']} ({r['pitchers']} pitchers), projected to a full season:",
              f"- Rating: {r['stuff']['full_season']:.2f}",
              f"- Actual runs prevented per 100 pitches: {r['rv100']['full_season']:.2f}",
              f"- Rating, year to year: r = {f['correlations']['stuff_to_stuff']['r']:.2f}", ""]
    return "\n".join(lines)


def run(scored: pd.DataFrame, names: dict | None = None, results_dir: str | Path = config.RESULTS_DIR) -> dict:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)
    table = add_stuff_plus(pitcher_seasons(scored))

    metrics = dict(config=dict(train_seasons=config.TRAIN_SEASONS, tune_season=config.TUNE_SEASON,
                               score_season=config.SCORE_SEASON, target_season=config.TARGET_SEASON,
                               min_pitches=config.MIN_PITCHES, result_stats=RESULT_STATS))
    final_pair = None
    for name, (y0, y1) in (("development", (config.TUNE_SEASON, config.SCORE_SEASON)),
                           ("final", (config.SCORE_SEASON, config.TARGET_SEASON))):
        pair = season_pair(table, y0, y1)
        if len(pair) < 30:
            raise SystemExit(f"Only {len(pair)} pitchers qualify for {y0}->{y1}; need at least 30.")
        metrics[name] = dict(this_season=y0, next_season=y1, pitchers=int(len(pair)),
                             correlations=correlations(pair, rng), incremental=incremental_value(pair, rng))
        if name == "final":
            final_pair = pair
    metrics["development"]["note"] = (f"{config.TUNE_SEASON} decided when training stopped, so this pair is "
                                      "slightly optimistic. The 'final' pair is the test.")
    metrics["reliability"] = split_half(scored, config.SCORE_SEASON, rng)

    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (results_dir / "summary.md").write_text(summary_markdown(metrics))
    table.round(5).to_csv(results_dir / "pitcher_seasons.csv", index=False)
    leaderboards(scored, table, names or {}, results_dir)
    figures(final_pair, metrics["final"]["correlations"], results_dir)
    return metrics
