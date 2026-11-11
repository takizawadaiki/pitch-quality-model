# Results: 2025 ratings judged on 2026

323 pitchers with at least 500 pitches in both seasons.

## 1. Link with next season (Pearson r, 95% interval)

| This season | Next-season k_bb | Next-season xwoba | Next-season rv100 |
|---|---|---|---|
| Pitch quality rating (this model) | +0.41 (+0.31 to +0.50) | -0.40 (-0.49 to -0.30) | +0.31 (+0.20 to +0.41) |
| Strikeout minus walk rate | +0.55 (+0.46 to +0.62) | -0.43 (-0.53 to -0.33) | +0.30 (+0.19 to +0.41) |
| FIP components per batter (13·HR + 3·(BB+HBP) − 2·K) | -0.36 (-0.45 to -0.26) | +0.36 (+0.26 to +0.47) | -0.24 (-0.35 to -0.13) |
| wOBA allowed | -0.30 (-0.41 to -0.20) | +0.29 (+0.18 to +0.40) | -0.19 (-0.31 to -0.08) |
| Expected wOBA allowed | -0.40 (-0.49 to -0.30) | +0.43 (+0.33 to +0.52) | -0.29 (-0.40 to -0.18) |
| Runs prevented per 100 pitches | +0.28 (+0.17 to +0.39) | -0.30 (-0.40 to -0.19) | +0.20 (+0.08 to +0.31) |

Lower is better for wOBA, xwOBA and FIP components, so a useful predictor of those shows a negative r.

## 2. Does the rating add to this season's results?

Out-of-sample error predicting next season from k_bb, fip_core and xwoba, with and without the rating.
A negative change whose interval excludes zero means the rating adds information.

| Next-season target | RMSE, results only | RMSE, with rating | Change (95% interval) |
|---|---|---|---|
| k_bb | 0.0497 | 0.0481 | -0.0015 (-0.0030 to -0.0001) |
| xwoba | 0.0297 | 0.0288 | -0.0009 (-0.0017 to -0.0001) |
| rv100 | 0.7838 | 0.7694 | -0.0144 (-0.0333 to +0.0042) |

## 3. Signal versus luck

Split-half agreement within 2025 (453 pitchers), projected to a full season:
- Rating: 0.99
- Actual runs prevented per 100 pitches: 0.30
- Rating, year to year: r = 0.89
