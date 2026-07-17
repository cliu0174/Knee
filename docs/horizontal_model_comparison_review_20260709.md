# Horizontal Model Comparison Review

Date: 2026-07-09

## Purpose

This step runs a first horizontal comparison including traditional models, traditional model variants, and two recent adapted frontier baselines under the same development/external protocol.

Main protocol:

- Development cohort: official train + primary test.
- External validation cohort: official secondary test.
- Target: knee cycle.
- Main feature setting: N=150, top40 Local DeltaQ sensitive-region features.

## Models Compared

Traditional models:

- Ridge
- ElasticNet
- SVR-RBF
- kNN
- RandomForest
- ExtraTrees
- Gaussian Process
- XGBoost

Input variants:

- raw/DeltaQ
- Local DeltaQ

Adapted frontier baselines:

- CNN-XGBoost inspired by Safavi et al. 2024.
- TCN-ECA inspired by Cui et al. 2025.

These frontier baselines are adapted reproducible baselines, not exact reimplementations.

## External Validation Ranking

| Rank | Model | Input | External MAE | External RMSE | External MAPE | External R2 |
|---:|---|---|---:|---:|---:|---:|
| 1 | ExtraTrees | Local DeltaQ | 80.90 | 117.35 | 9.43% | 0.770 |
| 2 | XGBoost | Local DeltaQ | 90.90 | 121.24 | 11.34% | 0.754 |
| 3 | RandomForest | Local DeltaQ | 95.60 | 127.78 | 11.68% | 0.727 |
| 4 | XGBoost | raw/DeltaQ | 103.55 | 132.18 | 13.18% | 0.708 |
| 5 | Adapted CNN-XGBoost | CNN embedding + raw/DeltaQ | 106.83 | 138.56 | 13.51% | 0.679 |
| 6 | RandomForest | raw/DeltaQ | 107.68 | 154.53 | 12.80% | 0.601 |
| 7 | Ridge | Local DeltaQ | 112.92 | 147.97 | 14.62% | 0.634 |
| 8 | ElasticNet | Local DeltaQ | 130.31 | 176.13 | 15.49% | 0.482 |
| 9 | kNN | Local DeltaQ | 132.35 | 236.23 | 13.48% | 0.068 |
| 10 | GaussianProcess | raw/DeltaQ | 139.40 | 181.63 | 17.82% | 0.449 |
| 11 | GaussianProcess | Local DeltaQ | 152.19 | 287.85 | 14.87% | -0.384 |
| 12 | Ridge | raw/DeltaQ | 166.13 | 186.58 | 22.41% | 0.418 |
| 13 | ElasticNet | raw/DeltaQ | 167.10 | 187.59 | 22.55% | 0.412 |
| 14 | ExtraTrees | raw/DeltaQ | 187.65 | 251.94 | 19.89% | -0.061 |
| 15 | Adapted TCN-ECA | Local DeltaQ sequence | 189.12 | 209.69 | 24.78% | 0.265 |
| 16 | kNN | raw/DeltaQ | 232.93 | 377.04 | 22.95% | -1.375 |
| 17 | SVR-RBF | Local DeltaQ | 249.22 | 358.58 | 25.26% | -1.148 |
| 18 | SVR-RBF | raw/DeltaQ | 322.25 | 406.45 | 34.61% | -1.760 |

## ExtraTrees Seed Robustness

Because `ExtraTrees + Local DeltaQ` unexpectedly outperformed the current XGBoost main model, it was rerun with 30 random seeds.

External secondary summary:

| Metric | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| MAE | 79.67 | 1.29 | 76.07 | 79.59 | 83.14 |
| RMSE | 117.38 | 1.11 | 115.29 | 117.41 | 119.62 |
| MAPE | 9.21% | 0.20% | 8.65% | 9.20% | 9.74% |
| R2 | 0.770 | 0.004 | 0.761 | 0.770 | 0.778 |

This indicates that the ExtraTrees result is stable and not a single random-seed accident.

## Interpretation

The strongest conclusion is now sharper:

> Local DeltaQ features consistently improve external-batch prediction across tree-based models, and ExtraTrees provides the strongest current readout.

The previous main model, Local DeltaQ + XGBoost, remains a strong boosting baseline, but it is no longer the best model in the horizontal comparison.

The adapted frontier baselines did not outperform the explicit Local DeltaQ tree ensembles in this initial implementation. This supports the manuscript angle that compact degradation-sensitive feature construction can be more effective than heavier learned representations on this small dataset.

## Self-review

What passed:

- Multiple traditional model families were compared under the same development/external protocol.
- Raw/DeltaQ and Local DeltaQ variants were both included.
- Two recent frontier-style baselines were implemented in adapted form.
- The best result was rerun across 30 random seeds and remained stable.

What needs caution:

- XGBoost was run with a fast top20 candidate search in this comparison, so the previous development/external XGBoost result should still be cited separately.
- The CNN-XGBoost and TCN-ECA baselines are adapted implementations, not exact reproductions of the original papers.
- kNN shows near-zero development error and poor external performance, indicating overfitting/memorization.
- The TCN-ECA baseline likely needs more careful sequence construction and tuning before being treated as a definitive deep-learning result.

## Stage Decision

Promote the main method from:

> Top40 Local DeltaQ + XGBoost

to:

> Top40 Local DeltaQ sensitive-region representation with tree-ensemble readout, with ExtraTrees as the current best external-validation model and XGBoost as the strong boosting baseline.

For manuscript wording, the contribution should remain on the Local DeltaQ representation, not on ExtraTrees itself.

## Next Step

Update the paper story and final experiment table:

1. Main result: Local DeltaQ + ExtraTrees.
2. Strong baseline: Local DeltaQ + XGBoost.
3. Raw/DeltaQ baseline: XGBoost / RandomForest.
4. Frontier adapted baselines: CNN-XGBoost and TCN-ECA.
5. Robustness: ExtraTrees 30-seed stability and random-split benchmark.

