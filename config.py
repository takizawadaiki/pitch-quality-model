"""Every choice that defines the study lives here, so a reader can see them in one place.

Change these BEFORE looking at the final-test numbers, and write down why in RESEARCH_PLAN.md.
"""

# --- Seasons -------------------------------------------------------------------------------
# The model only ever learns from TRAIN_SEASONS. TUNE_SEASON decides when training stops.
# Ratings for SCORE_SEASON are then judged against what happened in TARGET_SEASON.
TRAIN_SEASONS = [2021, 2022, 2023]
TUNE_SEASON = 2024
SCORE_SEASON = 2025
TARGET_SEASON = 2026
ALL_SEASONS = TRAIN_SEASONS + [TUNE_SEASON, SCORE_SEASON, TARGET_SEASON]

# --- Sample --------------------------------------------------------------------------------
# A pitcher-season counts only with at least this many pitches (in BOTH seasons of a pair).
MIN_PITCHES = 500
# Pitch types kept. Pitchouts, eephus pitches, knuckleballs and unclassified pitches are dropped.
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CH", "FS"]
FASTBALLS = ["FF", "SI", "FC"]

# --- Model inputs --------------------------------------------------------------------------
# Only what the ball does out of the hand. No location, no count, no results, no pitch-type label.
FEATURES = [
    "release_speed",       # mph
    "hmov",                # horizontal movement, inches, arm side positive for both hands
    "vmov",                # induced vertical movement, inches
    "release_spin_rate",   # rpm
    "spin_axis_adj",       # degrees, mirrored for left-handers
    "release_extension",   # feet in front of the rubber
    "release_x_adj",       # feet, arm side positive
    "release_pos_z",       # feet above the ground
    "velo_diff",           # mph relative to the pitcher's own primary fastball
    "hmov_diff",           # inches relative to that fastball
    "vmov_diff",           # inches relative to that fastball
    "same_hand",           # 1 when pitcher and batter are same-handed
]
# What the model predicts: change in run expectancy on the pitch, from the batting team's side.
TARGET = "delta_run_exp"
# Columns that describe what happened AFTER release. A test fails if any of these enters FEATURES.
OUTCOME_COLUMNS = [
    "delta_run_exp", "events", "description", "woba_value", "woba_denom",
    "estimated_woba_using_speedangle", "launch_speed", "launch_angle", "plate_x", "plate_z",
    "balls", "strikes", "zone", "type",
]

# --- Model settings (fixed; not searched) --------------------------------------------------
LGBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=400,   # the target is very noisy, so leaves must be large
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
)
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50

# --- Evaluation ----------------------------------------------------------------------------
BOOTSTRAP_REPS = 2000
CV_FOLDS = 5
CV_REPEATS = 20
SEED = 7

# --- Paths ---------------------------------------------------------------------------------
RAW_DIR = "data/raw"
PREPARED_DIR = "data/prepared"
RESULTS_DIR = "results"

# Columns requested from Statcast. Everything else is discarded at download time.
RAW_COLUMNS = [
    "game_date", "game_year", "game_type", "game_pk", "at_bat_number", "pitch_number",
    "pitcher", "batter", "p_throws", "stand", "pitch_type",
    "release_speed", "pfx_x", "pfx_z", "release_spin_rate", "spin_axis",
    "release_extension", "release_pos_x", "release_pos_z",
    "delta_run_exp", "events", "description", "woba_value", "woba_denom",
    "estimated_woba_using_speedangle",
]
