# Research plan

Written 11 November 2026, before any real data was run. Changes after that date belong in the
log at the bottom, with the reason.

## Question

Can a rating built only from how a pitch leaves the hand (velocity, movement, spin, release)
tell a club something about a pitcher's next season that his results this season do not?

## Why it matters

Results over one season are noisy. If pitch shape carries stable information, a club can
evaluate a pitcher on fewer innings: a mid-season call-up, a reliever, a trade target with a
bad ERA, or a pitcher who has changed his arsenal.

## Data

Pitch-level Statcast, regular seasons 2021 to 2026, from Baseball Savant through `pybaseball`.
2021 is the first full season used because ball tracking has been on one system since 2020 and
2020 was shortened.

## Design, fixed in advance

| Role | Seasons | Used for |
|---|---|---|
| Train | 2021, 2022, 2023 | Fitting the model |
| Tune | 2024 | Deciding when training stops. Nothing else |
| Score | 2025 | Ratings are computed here |
| Target | 2026 | Ratings are judged against these results, once |

- **Unit of prediction:** one pitch. **Unit of evaluation:** one pitcher-season.
- **Target:** `delta_run_exp`, the change in run expectancy on the pitch (batting side).
- **Inputs:** the twelve columns in `config.FEATURES`. No location, count, pitch-type label or
  result. Horizontal values are mirrored so left- and right-handers share one scale.
- **Model:** gradient-boosted trees (LightGBM), settings fixed in `config.py`, not searched.
- **Sample:** pitchers with at least 500 pitches in both seasons of a pair.

## What will be reported

1. **Link with next season.** Correlation of the 2025 rating with 2026 strikeout-minus-walk
   rate, expected wOBA allowed and runs prevented per 100 pitches, beside the same correlations
   for 2025 results. 95% intervals from resampling pitchers.
2. **Incremental value.** Out-of-sample error predicting 2026 from 2025 results alone, and from
   2025 results plus the rating. The change in error, with an interval.
3. **Reliability.** Split-half agreement of the rating within 2025, against the same figure for
   actual runs prevented.

## What would count as support, and what would not

- **Supports the idea:** the rating's link with next season is at least as strong as the
  results-based measures, and adding it lowers out-of-sample error with an interval that
  excludes zero for at least one target.
- **Does not support it:** the interval for the change in error includes zero for every target.
  That is still a finding and will be reported as one.

## Known limits

- The run-value target depends on the count; the model is not told the count.
- No location, so this measures stuff, not command.
- Pitchers who throw 500 pitches in consecutive seasons are a selected, healthier, better group.
- Movement is not adjusted for ballpark or altitude. The ball itself changes between seasons.
- Batter quality and defence are not adjusted.
- FIP proper needs innings and ERA needs earned runs; neither is in pitch-level data. The
  benchmark uses FIP's components per batter faced instead.

## Decisions that are mine to revisit

Each of these is a default, not a conclusion. Change one at a time, on the development pair
(2024 to 2025) only, and log it below before looking at the final pair again.

- [ ] Minimum pitches (500). Separate thresholds for starters and relievers?
- [ ] Use a count-neutral run value, or model swing, whiff and contact outcomes separately.
- [ ] Add release-angle or arm-angle inputs.
- [ ] Separate models for fastballs, breaking balls and offspeed.
- [ ] Adjust movement for ballpark.
- [ ] Add a second test pair by retraining with one more season.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-11 | Plan written | |
