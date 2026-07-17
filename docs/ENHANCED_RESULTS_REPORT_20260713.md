# Enhanced research results report (2026-07-13)

## Executive result

The project now has a leakage-resistant development experiment, an immutable frozen main model, one protocol-controlled secondary-batch execution, observation-budget and data-efficiency experiments, feature/model/region ablations, empirical prediction intervals, seven figure bundles, and machine validation.

The strict evidence does **not** support the planned contiguous-region mechanism as a new main method. It also does not support a clear independent Local DeltaQ gain over raw+global descriptors. The defensible main model is region-balanced LDQ-ET because it minimized the predeclared development selection score and then reproduced the historical anchor level on the secondary batch.

## 1. Historical anchor reproduction

Output: `results/reproduction_ldq_et_N150_20260713/`

The original 30-seed LDQ-ET top40 result was reproduced exactly:

| Metric | Mean | Seed SD |
|---|---:|---:|
| MAE | 77.2256 | 0.8919 |
| RMSE | 111.0655 | 0.9314 |
| MAPE (%) | 9.0420 | 0.1166 |
| R2 | 0.79388 | 0.00345 |

This is a historical anchor, not the newly selected main configuration. The old top40 ranking was fixed outside development folds and the repository had previously viewed secondary-batch top-k/architecture results.

## 2. Strict nested selector comparison at N150

Output: `results/nested_contiguous_ldq_N150_20260713/`

Protocol: 83 development cells; 5 outer folds repeated twice; 3-fold inner selection; all imputation, selector fitting and model choice inside training folds; 365 Local DeltaQ candidates.

| Selector | Outer MAE mean ± SD | RMSE | q90 error | Feature Jaccard | Selection score |
|---|---:|---:|---:|---:|---:|
| region-balanced | 42.30 ± 7.93 | 64.50 | 88.27 | 0.545 | **48.752** |
| bootstrap-stable | **42.19 ± 8.80** | 65.09 | 89.44 | 0.792 | 48.766 |
| mRMR | 42.89 ± 8.61 | 65.30 | 90.89 | **0.805** | 49.616 |
| contiguous band | 43.13 ± 8.80 | 65.73 | 90.41 | 0.623 | 50.151 |
| Pearson top-k | 44.07 ± 9.44 | 68.00 | 97.89 | 0.638 | 50.961 |

Region-balanced versus bootstrap-stable per-cell absolute-error difference was 0.10 cycles (95% bootstrap CI -1.34 to 1.39; sign-flip p=0.890). The score gap is not substantive evidence of superiority; it is only the predeclared rule used to choose one reproducible model.

### Contiguous mechanism decision

The contiguous candidate did not win N150 and did not form a consistent advantage in paired tests. It also failed to win the formal 5×2 N50/N100 budget runs. It is retained as a negative result, not promoted as a new method.

## 3. Common-cohort observation budgets

Outputs:

- `results/nested_contiguous_ldq_N50_commonN150_5x2_20260713/`
- `results/nested_contiguous_ldq_N100_commonN150_5x2_20260713/`
- `results/nested_contiguous_ldq_N150_20260713/`

All budgets use the same N150-eligible cohort, so the window comparison is not confounded by changing cell eligibility.

| Observed cycles | Development-selected selector | MAE mean ± SD | RMSE mean | Selection score |
|---:|---|---:|---:|---:|
| 50 | bootstrap-stable | 59.75 ± 23.70 | 98.69 | 73.763 |
| 100 | bootstrap-stable | 46.54 ± 9.76 | 68.74 | 53.583 |
| 150 | region-balanced | 42.30 ± 7.93 | 64.50 | 48.752 |

The first single-5-fold N50 screen briefly ranked contiguous first. The matched 5×2 repeat reversed that ordering, demonstrating why the repeated result, not the initial screen, is reported.

## 4. Frozen main model and secondary-informed batch result

Frozen file: `results/nested_contiguous_ldq_N150_20260713/frozen_config.json`

- selector: region-balanced top40;
- maximum per region: 3;
- frozen selected regions: 14;
- readout: 1200-tree ExtraTrees, max_features 0.5, depth 6, leaf 1;
- seeds: 123–152;
- feature file SHA256 and software versions saved before evaluation.

| Metric | 30-seed mean ± SD | Mean-prediction cell-bootstrap point | 95% cell-bootstrap CI |
|---|---:|---:|---:|
| MAE | 76.57 ± 0.72 | 76.39 | 51.89–103.17 |
| RMSE | 113.87 ± 0.92 | 113.79 | 75.05–148.14 |
| MAPE (%) | 8.99 ± 0.10 | 8.97 | 6.34–12.07 |
| R2 | 0.783 ± 0.003 | 0.784 | 0.601–0.887 |

Frozen minus historical-anchor paired absolute error: -0.61 cycles (95% CI -3.93 to 3.32; p=0.778). This is parity, not a significant improvement.

Tail results: median 43.92; q90 190.61; q95 265.16; maximum 376.17 cycles; 10/40 cells above 100 cycles; 3/40 above 200 cycles.

The external modeling pass completed and wrote all predictions. A post-processing coding error interrupted only the comparison-writing step. A new `finalize` recovery phase rebuilt comparison/metadata from the already saved predictions without reopening the feature table or refitting models. A second external model run was therefore avoided.

## 5. Feature-group, model-family and region ablations

Output: `results/nested_ablation_learning_N150_20260713/`

### Seven feature groups

| Group | Outer MAE mean ± SD |
|---|---:|
| raw only | 89.32 ± 22.21 |
| global DeltaQ only | 59.71 ± 14.22 |
| Local DeltaQ only | 53.62 ± 11.17 |
| raw + global | 42.33 ± 11.13 |
| raw + local | **42.16 ± 8.75** |
| global + local | 54.32 ± 11.15 |
| raw + global + local | 42.39 ± 8.42 |

The full feature set does not improve mean development MAE over raw+global. Raw+local and raw+global are nearly interchangeable in mean error. The result supports correlated alternative representations, not a unique Local DeltaQ increment.

### Representative readouts

| Model | Outer MAE mean ± SD |
|---|---:|
| compact ExtraTrees | **46.24 ± 10.01** |
| RandomForest | 51.45 ± 13.06 |
| XGBoost | 53.73 ± 13.03 |
| RBF-SVR | 103.29 ± 30.04 |
| RidgeCV | 690.20 ± 1924.26 |

Ridge instability is consistent with the highly collinear feature representation. The formal ExtraTrees configuration in the main nested experiment performs better than the compact screen (42.30 versus 46.24 cycles).

### Region contribution

| Control | Outer MAE mean ± SD |
|---|---:|
| all-region Pearson top40 | 42.49 ± 8.25 |
| core 0.525–0.650 only | 43.69 ± 11.70 |
| exclude core | 43.82 ± 11.25 |
| selected set minus core | 42.70 ± 9.47 |
| random 40 features, three repeats | 46.31 pooled mean |
| core columns permuted | approximately 98 pooled mean |

Core-region permutation is a predictive-reliance diagnostic under an out-of-distribution intervention. It is not causal. The removal controls show that the core region is replaceable by correlated neighboring/other-region features.

## 6. Data efficiency, uncertainty and complexity

Development subsampling MAE decreased monotonically:

- 25% (18.0 mean training cells): 62.43 cycles;
- 50% (33.4): 54.44;
- 75% (48.4): 46.07;
- 100% (66.4): 42.80.

The development-calibrated nominal 90% residual interval had a constant width of 203.94 cycles. Secondary empirical coverage was 77.5%; early 84.6%; mid 84.6%; late 64.3%. The interval is undercalibrated under shift.

Mean model complexity over 30 seeds: 1.18 s fit, 4.56 ms/cell inference, 5.02 MB serialized model.

## 7. Figures and validation

- Seven editable SVG/PDF/600-dpi PNG figure bundles: `manuscript_ionics/figures/fig1_protocol_nested_workflow.*` through `fig7_uncertainty_tail_and_complexity.*`.
- Nineteen source-data CSVs: `manuscript_ionics/figure_source_data/`.
- Figure export and manual visual QA: `manuscript_ionics/figure_qa/figure_qa_report.md`.
- Machine validation: `results/enhancement_validation_20260713/validation_report.md` (PASS).

## 8. Final claim boundary

Supported:

1. Strict nested local-region selection can retain a lightweight knee-cycle point predictor at historical-anchor accuracy.
2. ExtraTrees is the strongest tested small-sample readout in this project.
3. Local regions carry predictive information but are strongly redundant and substitutable.
4. Longer observation windows and more development cells improve performance.
5. Batch-shift uncertainty calibration is currently inadequate.

Not supported:

1. A novel contiguous/physics-consistent selector that consistently beats the baseline.
2. A statistically significant secondary-batch improvement over the historical top40 anchor.
3. A clear independent Local DeltaQ increment beyond raw+global descriptors.
4. A unique causal/electrochemical 0.525–0.650 region.
5. Untouched external confirmation or deployment-ready prediction intervals.
