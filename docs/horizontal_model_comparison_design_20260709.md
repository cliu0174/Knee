# Horizontal Model Comparison Design

Date: 2026-07-09

## Goal

Build a publication-ready horizontal model comparison for early knee-cycle prediction on the Severson/MIT-Stanford battery dataset.

The comparison should include:

1. Traditional models.
2. Traditional model variants.
3. Two recent frontier-style models from 2024-2026 literature that are structurally clear and feasible to reproduce.
4. Our proposed Local DeltaQ model.

## Evaluation Protocols

Use two levels of evaluation:

### Main protocol

Development/external protocol:

- Development cohort: official train + primary test.
- External validation: official secondary test.
- This is the main manuscript protocol.

### Supplementary protocol

Repeated random split:

- Merge all three batches.
- 75/25 train/test split.
- 100 repeats.
- Use only as a same-distribution benchmark for comparison with papers that use random splits.

## Existing Main Result

Current main model:

> Top40 Local DeltaQ sensitive-region features + XGBoost

Development/external result:

| Model | External MAE | External RMSE | External MAPE | External R2 |
|---|---:|---:|---:|---:|
| Raw/DeltaQ + XGBoost | 108.27 | 143.23 | 13.84% | 0.657 |
| Top40 Local DeltaQ + XGBoost | 89.03 | 121.07 | 10.96% | 0.755 |

Random-split benchmark:

| Protocol | Mean MAE | Mean MAPE | Mean R2 | Best R2 |
|---|---:|---:|---:|---:|
| Unstratified random split | 60.58 | 9.03% | 0.872 | 0.967 |
| Batch-stratified random split | 62.42 | 9.23% | 0.870 | 0.955 |

## Traditional Models To Include

These models are simple, reproducible, and expected by reviewers.

| Family | Model | Input variants | Purpose |
|---|---|---|---|
| Linear | Ridge / ElasticNet | raw/DeltaQ; Local DeltaQ | Tests whether improvement is feature-driven rather than model-driven |
| Kernel | SVR-RBF | raw/DeltaQ; Local DeltaQ | Traditional nonlinear baseline |
| Distance | kNN regression | raw/DeltaQ; Local DeltaQ | Local similarity baseline |
| Tree ensemble | RandomForest / ExtraTrees | raw/DeltaQ; Local DeltaQ | Nonlinear non-boosting baseline |
| Boosting | XGBoost | raw/DeltaQ; Local DeltaQ | Strong tabular baseline and our main readout |
| Probabilistic | Gaussian Process Regression | compact top-k features only | Uncertainty-aware small-data baseline; may be slow on full features |

Recommended table grouping:

1. Traditional models on raw/DeltaQ.
2. Same model families on Local DeltaQ.
3. Our selected Local DeltaQ + XGBoost.

This directly tests whether the contribution is the representation or just the regressor.

## Recent Frontier Models Screened

### Selected Frontier Model 1: CNN-XGBoost / COM-CNN-XGBoost

Source:

- Safavi et al., 2024, *Journal of Energy Storage*, DOI: `10.1016/j.est.2024.113176`.

Reported idea:

- Convert early discharge capacity data into an image-like representation.
- Use CNN to extract learned features.
- Concatenate CNN features with handcrafted early-cycle features and charging-policy features.
- Feed the combined vector into XGBoost.
- Coati Optimization Method is used for CNN hyperparameter tuning.

Reported performance:

- Best reported RUL result around RMSE = 106 cycles and MAPE = 7.5%.

Why selected:

- Uses the Severson 124-cell dataset.
- Uses first 100 cycles.
- Model structure is clear.
- Can be adapted to our task by replacing the RUL/cycle-life label with knee-cycle label.
- XGBoost readout is compatible with our existing pipeline.

Planned reproduction:

- Implement a simplified CNN-XGBoost:
  - Input: early-cycle local DeltaQ or Qdlin matrix.
  - CNN: 1D/2D convolutional encoder.
  - Feature vector: CNN embedding + raw/DeltaQ features.
  - Regressor: XGBoost with the same development/external protocol.
- Do not reproduce Coati optimization initially; use a small fixed/random hyperparameter grid.

Role in paper:

- Recent deep feature extraction baseline.
- Shows whether learned CNN features beat our explicit Local DeltaQ representation.

### Selected Frontier Model 2: TCN/ECA-MCCA Style Joint Knee-Life Model

Source:

- Cui et al., 2025, *Symmetry*, DOI: `10.3390/sym17030351`.

Reported idea:

- Predict cycle-to-knee (CTK) and cycle life jointly.
- Use temporal convolutional structures and attention modules.
- Compare temporal, spatial, and spatiotemporal fusion models.

Reported CTK result:

- TECAN CTK prediction: RMSE = 88.25 cycles, MAE = 66.18 cycles, MAPE = 13.57%, R2 = 86.56%.
- TECAN-ECA-MCCA CTK prediction: RMSE = 89.52 cycles, MAE = 68.22 cycles, MAPE = 14.35%, R2 = 86.93%.
- N=100 TECAN CTK result: RMSE = 86.92 cycles, MAPE = 14.27%, R2 = 86.33%.

Why selected:

- Same Stanford/Severson-style dataset.
- Directly predicts CTK/knee-related target.
- 2025 paper, recent enough for frontier comparison.
- TCN/attention structure is implementable in PyTorch.
- More comparable than BatteryGPT because it still works around early-cycle prediction rather than long-horizon trajectory generation from 30% lifetime data.

Planned reproduction:

- Implement a simplified TCN-attention regressor:
  - Input: early-cycle DeltaQ/Qdlin sequence or local-segment sequence.
  - TCN blocks: dilated 1D convolution + residual connection.
  - ECA-style channel attention.
  - Output: knee-cycle regression.
- Optional second head for cycle life if labels are already available, but main comparison should use knee-cycle output for fairness.

Role in paper:

- Recent sequence/attention baseline.
- Tests whether learned temporal attention outperforms explicit Local DeltaQ segments.

## Frontier Models Not Selected As Main Reproducible Baselines

### BatteryGPT

Source:

- DOI: `10.1038/s41467-025-66819-0`.

Reason not selected:

- Uses MIT LIB 46-cell diagnostic data rather than the full 124-cell Severson split used here.
- Uses first 30% lifetime data in its strongest setting, which is much later than fixed N=100/N=150 early-cycle prediction.
- Predicts full lifecycle SOH trajectory, then derives knee/EOL.
- Requires a GPT-style sequence-generation pipeline, high implementation cost, and a different task formulation.

Use in paper:

- Literature discussion only.
- Not a fair horizontal baseline.

### AE-FCNN+ Knee Interval Model

Source:

- Yu et al., 2025, *Journal of Power Sources*, DOI: `10.1016/j.jpowsour.2025.236438`.

Reported result:

- Knee interval prediction: AE-FCNN+ MAE = 56.5, RMSE = 74.74, MAPE = 7.1%.

Reason not selected as primary frontier baseline:

- Predicts knee interval rather than single knee-cycle label.
- Label definition differs from our current piecewise/Kneedle knee-cycle labels.
- Still useful as an optional third frontier baseline if time allows.

Use in paper:

- Literature comparison or optional future extension.

## Recommended Final Horizontal Comparison Table

Target table: `Table 1. Model comparison under development/external protocol`.

Rows:

1. Ridge, raw/DeltaQ.
2. SVR-RBF, raw/DeltaQ.
3. RandomForest, raw/DeltaQ.
4. XGBoost, raw/DeltaQ.
5. Ridge, Local DeltaQ.
6. RandomForest, Local DeltaQ.
7. XGBoost, Local DeltaQ.
8. CNN-XGBoost, learned early-cycle representation.
9. TCN-ECA, learned temporal-attention representation.
10. Proposed Local DeltaQ + XGBoost.

Columns:

- Feature/input type.
- Parameter-selection protocol.
- Development MAE.
- External MAE.
- External RMSE.
- External MAPE.
- External R2.
- Notes on reproducibility / model size.

## Fairness Rules

1. All models use the same labels.
2. All models use the same development/external split.
3. Hyperparameters are selected only on the development cohort.
4. Secondary batch is used once as external validation.
5. Random split results are reported separately as supplementary evidence.
6. Frontier reproductions should be clearly marked as "adapted implementation" unless exact official code is used.

## Next Step

Implement `scripts/run_horizontal_model_comparison.py`.

Initial implementation order:

1. Traditional models and variants.
2. CNN-XGBoost adapted baseline.
3. TCN-ECA adapted baseline.
4. Produce `results/horizontal_model_comparison_N150/`.
5. Write `docs/horizontal_model_comparison_review_20260709.md`.

