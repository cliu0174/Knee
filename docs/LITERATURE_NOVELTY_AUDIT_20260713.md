# Literature and novelty audit (2026-07-13)

## Scope and search rule

This audit is claim-oriented rather than a broad narrative review. Searches covered (i) early lithium-ion battery prognosis and knee prediction, (ii) ordered/functional interval selection, (iii) stability selection, and (iv) predictive uncertainty for RUL. Bibliographic metadata below was checked against publisher or journal pages and DOI records. The audit distinguishes a reusable method from the contribution of applying it to this dataset and task.

## Nearest-work matrix

| Work | Closest overlap | What it prevents us from claiming | Defensible distinction in this project |
|---|---|---|---|
| Severson et al. (2019), `10.1038/s41560-019-0356-8` | The same 124-cell fast-charging dataset; early-cycle discharge-curve differences; life prediction | Early-cycle curve differences or this dataset are not new | Absolute piecewise-linear knee cycle is the target; the protocol exposes a chronology-aware secondary batch |
| Hu et al. (online 2025; volume 17, 2026), `10.1038/s41467-025-66819-0` | Early full-trajectory, knee and EOL prediction with BatteryGPT | Early knee prediction itself and forecasting from partial life are not new | A lightweight, non-generative estimator using localized cycle-9-to-early-window discharge-capacity changes; substantially lower model complexity |
| Servien and Vialaneix (2024), `10.1002/sam.11705` | Random-forest interval selection for functional regression | Tree-based selection of consecutive functional-domain intervals is not new | Our ordered units are physically indexed voltage/SOH regions in an early $\Delta Q$ map, and selection is audited inside nested folds for a knee-point/batch-transfer task |
| Tibshirani et al. (2005), `10.1111/j.1467-9868.2005.00490.x` | Sparsity plus smoothness over meaningfully ordered features | Neighbour smoothing/contiguity regularization is not a new general principle | The present contiguous selector is a deliberately simple domain-specific candidate, not a new optimization theory |
| Abramowicz et al. (2023), `10.1111/biom.13669` | Domain selection for functional data with formal error control | “Functional domain selection” cannot be presented as invented here | We perform predictive feature selection, not formal pointwise inference or FWER-controlled domain discovery |
| Meinshausen and Bühlmann (2010), `10.1111/j.1467-9868.2010.00740.x` | Subsampling-based stability selection | Bootstrap selection frequency is not methodologically new | We use stability frequency as a diagnostic/baseline and report Jaccard/core-region behavior; no finite-sample false-discovery guarantee is claimed |
| Javanmardi and Hüllermeier (2023), `10.36001/ijphm.2023.v14i2.3417` | Conformal RUL prediction intervals | Conformal uncertainty for prognostics is not new | Any interval reported here is an empirical, cross-fitted residual interval for the knee target and is labelled separately from the point model |
| Sanches et al. (2026), `10.1016/j.rineng.2026.111218` | Same 124-cell LFP dataset; single-cycle statistical features; quantile ExtraTrees; conformalized SoH/RUL | ExtraTrees, lightweight deployment, single-cycle statistics and conformalized tree ensembles are not novel individually | Localized early $\Delta Q$ regions, absolute knee-cycle prediction, nested feature selection and chronology-aware batch reporting remain the defensible combination |

## Claim decisions

### Claims retained

1. The study evaluates whether localized early $\Delta Q$ descriptors contain transferable information about the absolute degradation knee.
2. A compact ExtraTrees pipeline provides a reproducible accuracy--complexity reference under a chronology-aware batch split.
3. Strict nesting quantifies how much apparent development performance changes when feature selection is repeated inside each training fold.
4. Region concentration, selection stability, tail error, data-budget behavior and external-batch uncertainty are treated as evidence about robustness, not merely a single headline MAE.

### Claims weakened or removed

1. **Removed:** “the first interval/region selection method for battery curves.”
2. **Removed:** “the first use of ExtraTrees/conformal prediction for lightweight battery prognosis.”
3. **Weakened:** “external validation” is reported as a **secondary-informed, protocol-controlled batch evaluation**, because this batch had already been viewed in historical architecture and top-$k$ exploration.
4. **Weakened:** a selected voltage/SOH band is a predictive association and interpretability clue; it is not direct electrochemical-mechanism proof.
5. **Weakened:** repeated random seeds describe algorithmic variation conditional on one batch split; they do not create 30 independent external datasets.

## Citation metadata corrections

The BatteryGPT record is assigned to *Nature Communications* volume 17, article 126 (2026), while preserving the DOI suffix and noting its online publication on 5 December 2025 in prose when chronology matters. New verified entries were added to `manuscript_ionics/references.bib` for stability selection, fused lasso, functional interval/domain selection, conformal RUL intervals, and the 2026 QERT battery study.

## Remaining evidence gap

No literature search can establish novelty by absence. Before submission, the final title/abstract claims should be checked once more against newly indexed 2026 battery-knee and functional-region-selection papers. The safest contribution language is therefore “a protocol-controlled study of localized early $\Delta Q$ regions for knee prediction,” not “a novel universal framework.”
