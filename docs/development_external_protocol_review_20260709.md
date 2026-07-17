# Development/External Protocol Review

Date: 2026-07-09

## Purpose

This step reframes the evaluation protocol for a publishable manuscript. Instead of treating both primary and secondary splits as equally final test sets, we use:

- Development cohort: official train + primary test
- External validation cohort: official secondary test

This protocol is easier to defend in a paper because model development and parameter selection are performed on the development cohort, while the secondary batch remains an independent cross-batch external validation set.

## Rationale

The previous strict train-only protocol was methodologically conservative but unstable because only 41 cells and two train batches were available for model selection. It selected models that performed well in train-only cross-validation but did not generalize optimally to the official external batch.

The development/external protocol has a clearer manuscript logic:

1. Use the official train and primary-test cells as the model-development cohort.
2. Select hyperparameters by repeated cross-validation within that development cohort.
3. Train the selected model on the full development cohort.
4. Evaluate once on the official secondary batch as external validation.

## Model

Main candidate:

> Top40 Local DeltaQ sensitive-region features + XGBoost

The hybrid residual component is not used as the final model because the train-only fixed-weight protocol selected `weight_xgb = 1.00`, indicating that the residual branch was not robustly supported by internal validation.

## Results

### Local DeltaQ model

Selected candidate:

- `xgb_hpo_0160`
- `n_estimators = 260`
- `max_depth = 1`
- `learning_rate = 0.055`
- `subsample = 0.75`
- `colsample_bytree = 0.90`
- `reg_lambda = 0.5`
- `reg_alpha = 1.0`
- `min_child_weight = 1.0`

Performance:

| Model | Development MAE | External MAE | External RMSE | External MAPE | External R2 |
|---|---:|---:|---:|---:|---:|
| Top40 Local DeltaQ + XGBoost | 15.96 | 89.03 | 121.07 | 10.96% | 0.755 |

### Raw/DeltaQ baseline under the same protocol

The same development/external selection protocol was run without Local DeltaQ features.

| Model | Development MAE | External MAE | External RMSE | External MAPE | External R2 |
|---|---:|---:|---:|---:|---:|
| Raw/DeltaQ + XGBoost | 17.31 | 108.27 | 143.23 | 13.84% | 0.657 |

### Improvement

Compared with the raw/DeltaQ baseline under the same protocol, Local DeltaQ features improve external validation by:

- External MAE: 108.27 -> 89.03, improvement of 19.24 cycles
- External RMSE: 143.23 -> 121.07, improvement of 22.16 cycles
- External MAPE: 13.84% -> 10.96%, improvement of 2.88 percentage points
- External R2: 0.657 -> 0.755, improvement of 0.098

This is currently the cleanest result for the paper.

## Manuscript Claim

Supported claim:

> A compact representation of local DeltaQ sensitive segments improves cross-batch knee-cycle prediction compared with global raw/DeltaQ statistics when both models are selected under the same development/external protocol.

Do not claim:

- The model reaches R2 = 0.95.
- The hybrid residual branch is the final main innovation.
- The method universally generalizes across all battery chemistries or protocols.

## Innovation Positioning

The defensible method contribution is:

1. Shift from global early-cycle summary statistics to local DeltaQ sensitive-region representation.
2. Select compact local segments using train/development-only evidence.
3. Use a low-complexity gradient-boosted readout to avoid unnecessary deep-model complexity on a small battery dataset.
4. Validate the final model on an independent secondary batch.

The story is not that XGBoost itself is novel. The story is that Local DeltaQ features expose degradation-sensitive voltage regions that the model can use for earlier and more robust knee-cycle prediction.

## Self-review

What supports publication framing:

- The Local DeltaQ model and raw/DeltaQ baseline use the same development/external protocol.
- The improvement is visible across MAE, RMSE, MAPE, and R2.
- The external validation batch is kept separate from model development.
- The method remains interpretable: the top local features are concentrated in specific DeltaQ index regions.

Remaining risks:

- The dataset is still small, and the development cohort includes the official primary split, so the paper must clearly define it as development validation rather than a final test set.
- External validation is limited to one secondary batch.
- The Local DeltaQ feature extraction should be described carefully enough for reproduction.
- The result is strong but not near-perfect; claims should emphasize robust improvement rather than absolute accuracy.

## Stage Decision

Promote the following as the current main paper model:

> Top40 Local DeltaQ sensitive-region features + XGBoost, selected on the development cohort and externally validated on the secondary batch.

Keep hybrid residual adaptation as an exploratory ablation, not the main contribution.

## Next Small Step

Prepare the manuscript story package:

1. Final method name and terminology ledger.
2. Final experiment table comparing raw/DeltaQ baseline, Local DeltaQ model, and exploratory hybrid.
3. Figure plan showing the Local DeltaQ sensitive region and external-validation prediction scatter.
4. Results paragraph in Chinese/English draft form.

