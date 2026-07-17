# Severson 2019 Reproduction Confirmation Review

Date: 2026-06-05

## Confirmation Verdict

This is an acceptable stage-1 reproduction, not a strict paper-level reproduction.

The current pipeline successfully runs:

```text
.mat -> batch*.pkl -> official-style features -> train/evaluate -> figures
```

The scientific trend is consistent with Severson et al. 2019: early-cycle discharge-voltage features are predictive of cycle life, and the most important features are dominated by `delta_q_var`, `delta_q_log_var`, and `delta_q_min`.

However, strict reproduction is not yet achieved because the current second-batch file differs from the commonly referenced `2017-06-30` batch and contains missing EOL labels.

## Data Status

Raw files found in `data`:

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat`

All three files contain the expected HDF5 `batch` structure. They were converted to:

- `data/severson_raw/pkl/batch1.pkl`
- `data/severson_raw/pkl/batch2.pkl`
- `data/severson_raw/pkl/batch3.pkl`

The extracted supervised feature table is:

- `data/severson_features/severson_official_features.csv`

It contains 115 labeled cells:

```text
train: 37
primary_test: 39
secondary_test: 39
```

Eight batch2 cells were dropped because `cycle_life/lifetime` is missing and their discharge capacity curves do not clearly cross the 0.88 EOL threshold:

```text
b2c22, b2c23, b2c35, b2c36, b2c37, b2c38, b2c39, b2c40
```

## Result Comparison

Publicly reported Severson headline result:

- Best model: about 9.1% test error using the first 100 cycles.

Reviewer-extracted table-level reference values:

| Original model | Primary MAPE | Secondary MAPE |
|---|---:|---:|
| Variance model | 14.7% / 13.2% without outlier | 11.4% |
| Discharge model | 13.0% / 10.1% without outlier | 8.6% |
| Full model | 14.1% / 7.5% without outlier | 10.7% |

Current confirmation version using train-only CV parameter selection:

| Current model | Primary MAPE | Secondary MAPE | Status |
|---|---:|---:|---|
| XGBoost | 11.67% | 14.60% | best train-CV selected model |
| RandomForest | 12.46% | 12.69% | more balanced secondary result |
| ElasticNetCV | 16.20% | 17.06% | closer method family, weaker result |

Exploratory/post-hoc result:

| Current model | Primary MAPE | Secondary MAPE | Status |
|---|---:|---:|---|
| XGBoost | 11.00% | 12.67% | selected using primary+secondary mean; not strict |
| RandomForest | 11.35% | 12.50% | selected using primary+secondary mean; not strict |

## Interpretation

Consistent with the original paper:

- Early-cycle features predict cycle life.
- `delta_q` statistics dominate feature importance.
- Primary test error is in the same broad range as the original paper.

Not yet equivalent to the original paper:

- The current dataset uses `2018-02-20` as batch2 rather than the commonly referenced `2017-06-30`.
- The current supervised table has 115 labeled cells, not the original full split.
- The official split is therefore modified.
- The best current model is tree-based, while the original core result relies on regularized linear / elastic net style models.
- Secondary test error remains higher than the original discharge/full model results.

## Reviewer-Style Conclusion

Stage-1 reproduction: acceptable.

Strict reproduction: not yet acceptable.

The result can be used as a confirmed intermediate reproduction showing that the pipeline and core phenomenon are valid. It should be described as an "official-style feature reproduction on 115 labeled cells with a modified split", not as an exact reproduction of Severson et al. 2019.

Independent mentor review after train-CV revision:

```text
阶段性复现确认版：可接受。
严格 Severson 2019 原文复现：仍不可接受。
```

The main methodological fix accepted by the reviewer is that model selection now uses only train-set 4-fold CV. Secondary test is no longer used for parameter selection in the confirmation version.

## Required Next Fixes

1. Obtain or verify the `2017-06-30_batchdata_updated_struct_errorcorrect.mat` batch2 file.
2. Restore the original split sizes if possible: `train=41`, `primary_test=43`, `secondary_test=40`.
3. Reproduce the paper's regularized linear / elastic net feature subsets more faithfully.
4. Keep secondary test strictly as final validation.
5. Report XGBoost/RandomForest as extended baselines, not as the original model.
