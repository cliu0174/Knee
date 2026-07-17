# Random Split Benchmark Review

Date: 2026-07-09

## Purpose

The user asked whether the three Severson batches can be merged and randomly split for evaluation. This step tests that protocol as a supplementary benchmark.

Important boundary:

Random splitting is feasible, but it mixes batch distributions between train and test. It evaluates same-distribution interpolation, not cross-batch external validation. Therefore, it should not replace the external secondary-batch result in the main manuscript.

## Model

Model:

> Top40 Local DeltaQ sensitive-region features + XGBoost

Candidate:

- `xgb_hpo_0160`

This is the same candidate selected in the development/external protocol.

## Protocols Tested

### 1. Unstratified random split

- Merge all valid cells.
- Randomly split into 75% train and 25% test.
- Repeat 100 times.

### 2. Batch-stratified random split

- Merge all valid cells.
- Randomly split into 75% train and 25% test while preserving batch proportions.
- Repeat 100 times.

## Results

### Unstratified random split

| Metric | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| MAE | 60.58 | 12.76 | 33.20 | 59.81 | 97.09 |
| RMSE | 95.80 | 25.32 | 41.58 | 93.01 | 152.32 |
| MAPE | 9.03% | 1.61% | 5.97% | 9.08% | 14.77% |
| R2 | 0.872 | 0.068 | 0.637 | 0.886 | 0.967 |

Best repeat:

- MAE = 35.84
- RMSE = 47.66
- MAPE = 5.97%
- R2 = 0.967

### Batch-stratified random split

| Metric | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| MAE | 62.42 | 12.18 | 36.09 | 61.53 | 103.15 |
| RMSE | 99.90 | 24.55 | 50.28 | 101.98 | 157.39 |
| MAPE | 9.23% | 1.53% | 6.20% | 9.24% | 13.44% |
| R2 | 0.870 | 0.059 | 0.656 | 0.879 | 0.955 |

Best repeat:

- MAE = 50.54
- RMSE = 73.20
- MAPE = 7.60%
- R2 = 0.955

## Comparison With External-Batch Protocol

Main external-batch result:

- External secondary MAE = 89.03
- External secondary RMSE = 121.07
- External secondary MAPE = 10.96%
- External secondary R2 = 0.755

Random split result:

- Mean R2 is about 0.87.
- Best random repeats can exceed R2 = 0.95.

Interpretation:

The high R2 is real under random splitting, but it reflects a much easier protocol. Since train and test both contain cells from the same batches, the model sees batch-specific degradation patterns during training.

## Stage Decision

Use random split results as a supplementary benchmark or comparison against studies that use random train/test splits.

Do not use random split as the main claim if the paper emphasizes cross-batch generalization.

Recommended manuscript wording:

> Under a repeated random-split benchmark, the Local DeltaQ XGBoost model achieved an average R2 of approximately 0.87 and reached above 0.95 in the best split. However, because random splitting mixes cells from the same experimental batches across train and test sets, we report the independent secondary-batch result as the main external-validation metric.

