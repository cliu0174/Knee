# LDQ-ET Hyperparameter Optimization Review

Date: 2026-07-09

## Purpose

After horizontal comparison showed that `ExtraTrees + Local DeltaQ` outperformed the current XGBoost readout, this step performs dedicated hyperparameter optimization for the LDQ-ET main model.

Main protocol:

- Development cohort: official train + primary test.
- External validation: official secondary test.
- Hyperparameters are selected only by cross-validation on the development cohort.
- Secondary is used only for final external validation.

## Model

Method name:

> LDQ-ET: Local DeltaQ sensitive-region representation with ExtraTrees readout.

Feature set:

- N=150 raw/DeltaQ features.
- Top40 Local DeltaQ sensitive-region features.

## Search Space

ExtraTrees hyperparameters:

- `n_estimators`: 500, 800, 1200
- `max_features`: sqrt, 0.35, 0.5, 0.75, 1.0
- `min_samples_leaf`: 1, 2, 3, 4
- `min_samples_split`: 2, 4, 8
- `max_depth`: None, 6, 10, 16
- `bootstrap`: False, True

The full grid is large, so a quick HPO run sampled 30 candidates and included the previous horizontal-comparison anchor:

```text
n_estimators=700, max_features=sqrt, min_samples_leaf=2,
min_samples_split=2, max_depth=None, bootstrap=False
```

## Selected Configuration

Selected candidate:

```text
et_hpo_0000
```

Parameters:

```text
n_estimators = 1200
max_features = 0.5
min_samples_leaf = 1
min_samples_split = 2
max_depth = 6
bootstrap = False
```

Development CV:

```text
dev_cv_mae_mean = 42.20
dev_cv_mae_std = 10.54
dev_cv_mae_max = 57.62
dev_selection_score = 49.02
```

## External Validation

The selected configuration was retrained on the full development cohort and evaluated on the secondary external batch across 30 random seeds.

External secondary summary:

| Metric | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| MAE | 77.23 | 0.89 | 75.04 | 77.41 | 78.48 |
| RMSE | 111.07 | 0.93 | 108.94 | 111.19 | 112.68 |
| MAPE | 9.04% | 0.12% | 8.77% | 9.05% | 9.22% |
| R2 | 0.794 | 0.004 | 0.788 | 0.793 | 0.802 |

Development cohort summary:

| Metric | Mean | Std |
|---|---:|---:|
| MAE | 10.66 | 0.06 |
| RMSE | 14.15 | 0.08 |
| MAPE | 2.37% | 0.01% |
| R2 | 0.997 | ~0.000 |

## Comparison With Previous Main Results

| Model | External MAE | External RMSE | External MAPE | External R2 |
|---|---:|---:|---:|---:|
| Raw/DeltaQ + XGBoost | 108.27 | 143.23 | 13.84% | 0.657 |
| Local DeltaQ + XGBoost | 89.03 | 121.07 | 10.96% | 0.755 |
| Local DeltaQ + ExtraTrees anchor | 79.67 ± 1.29 | 117.38 ± 1.11 | 9.21% ± 0.20% | 0.770 ± 0.004 |
| LDQ-ET HPO selected | 77.23 ± 0.89 | 111.07 ± 0.93 | 9.04% ± 0.12% | 0.794 ± 0.004 |

Relative to the raw/DeltaQ + XGBoost baseline, LDQ-ET improves:

- MAE by 31.04 cycles.
- RMSE by 32.16 cycles.
- MAPE by 4.80 percentage points.
- R2 by 0.137.

Relative to Local DeltaQ + XGBoost, LDQ-ET improves:

- MAE by 11.80 cycles.
- RMSE by 10.00 cycles.
- MAPE by 1.92 percentage points.
- R2 by 0.039.

## Interpretation

The HPO result supports promoting LDQ-ET as the current main model.

The selected ExtraTrees readout is shallow (`max_depth=6`) but uses many randomized trees and half of the features per split. This is consistent with the task characteristics:

- small sample size,
- high-dimensional local DeltaQ features,
- nonlinear interactions,
- need for robust external-batch generalization,
- feature redundancy after top-k local segment selection.

The method contribution remains the Local DeltaQ representation. ExtraTrees is a strong and stable readout, not a novel algorithm by itself.

## Self-review

What passed:

- Hyperparameters were selected only from the development cohort.
- Secondary batch was held out for external validation.
- The final model was evaluated over 30 random seeds.
- The improvement over XGBoost and raw/DeltaQ baselines is consistent across MAE, RMSE, MAPE, and R2.

Remaining limitations:

- The HPO used a quick 30-candidate sampled search, not the full grid, because larger searches exceeded the current runtime budget.
- The development performance is very high, so overfitting risk should be acknowledged. External validation mitigates this concern but does not remove it entirely.
- Random-split benchmark for the HPO configuration was attempted but timed out due to the 1200-tree configuration; existing random-split results with `xgb_hpo_0160` remain available as supplementary evidence.

## Stage Decision

Promote LDQ-ET as the primary model:

> Local DeltaQ sensitive-region representation + ExtraTrees readout.

Recommended headline result:

```text
External secondary MAE = 77.23 ± 0.89 cycles
External secondary RMSE = 111.07 ± 0.93 cycles
External secondary MAPE = 9.04 ± 0.12%
External secondary R2 = 0.794 ± 0.004
```

## Next Step

Update the paper story and final tables:

1. Main method: LDQ-ET.
2. Main baseline: raw/DeltaQ + XGBoost.
3. Strong readout baseline: Local DeltaQ + XGBoost.
4. Horizontal comparison: traditional models + adapted frontier baselines.
5. Robustness: 30-seed LDQ-ET stability and random-split benchmark.

