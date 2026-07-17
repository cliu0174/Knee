# Full citation audit

Audit date: 2026-07-15

## Scope

- Sources checked: `main.tex` and `references.bib`.
- Unique cited references: 32.
- BibTeX entries after cleanup: 32.
- Missing citation keys: 0.
- Uncited BibTeX entries: 0.
- Duplicate DOIs: 0.
- Citation groups containing more than two references: 0.

Metadata were checked against DOI/publisher records and, where available, locally stored full texts. Claim support was assessed from the cited article's title, abstract, methods, and relevant full-text passages rather than from title similarity alone.

## Reference-by-reference assessment

| Key | Metadata | Manuscript use | Claim match |
|---|---|---|---|
| `severson2019data` | Verified | Dataset provenance, early DeltaQ precedent, cycle-10 convention | Direct primary support |
| `attia2020closed` | Verified | Closed-loop fast-charging optimization | Direct primary support; removed from dataset-provenance statements |
| `zhang2020impedance` | Verified | Impedance spectra as early prognostic signals | Direct primary support |
| `rhyu2025formation` | Verified | Formation-stage Q(V) feature design | Direct primary support |
| `qi2025hybrid` | Verified | Decomposition-based early capacity/RUL prediction | Direct method-context support; wording narrowed from “multi-resolution learning” to “signal decomposition” |
| `strange2023online` | Verified | Cycle-by-cycle lifetime updating | Direct primary support |
| `jiang2023predicting` | Verified | Discharge-voltage-curve lifetime prediction and global descriptors | Direct primary support |
| `lu2023deepsoh` | Verified | SOH transfer without additional target degradation experiments | Direct primary support |
| `wang2024pinn` | Verified | Physics-informed degradation prognosis across datasets | Direct method-context support |
| `zhang2025batlinet` | Verified | Inter-cell lifetime prediction across ageing conditions | Direct primary support |
| `li2024varying` | Verified | Degradation-informed early features under varying usage | Direct primary support |
| `tan2024magnet` | Verified | Domain-generalized degradation forecasting | Direct primary support |
| `fan2024relaxation` | Verified | Joint knee-point and RUL prediction from relaxation voltage | Direct primary support |
| `zhao2024historyfree` | Verified | History-free degradation-stage assessment and life prediction | Direct primary support |
| `hu2025batterygpt` | Verified | Transformer-based SOH trajectory, knee-point, and EOL prediction | Direct primary support; 2026 issue year with online publication in 2025 is correctly represented |
| `fuhrmann2024dva` | Verified | Differential-voltage localization, inhomogeneity, and knee prediction | Direct primary support; mechanistic claim kept bounded |
| `wang2025ica` | Verified | Incremental-capacity features for SOH estimation | Direct method-context support; wording limited to battery health rather than unique mechanism |
| `yu2025knee` | Verified | Curvature-based knee-interval definition and early prediction | Direct primary support |
| `li2025degradationmodes` | Verified | Non-uniqueness and importance of degradation-mode attribution | Direct primary support |
| `tibshirani2005fused` | Verified | Ordered sparsity and smoothness through fused lasso | Direct foundational support |
| `meinshausen2010stability` | Verified | Resampling-based stability selection | Direct foundational support |
| `servien2024interval` | Verified | Random-forest interval selection for functional predictors | Direct closest-method support |
| `pang2025distributionfree` | Verified | Distribution-free SOH interval estimation | Direct primary support |
| `sanches2026single` | Verified | Single-cycle statistical descriptors, quantile ExtraTrees, and conformal SoH/RUL intervals | Direct closest-method support |
| `yeh2016matrix` | Verified | Matrix-profile definition, motifs, and discords | Direct foundational support |
| `safavi2024early` | Verified | CNN-XGBoost early RUL prediction | Direct architecture-context support |
| `cui2025joint` | Verified | Neural/hybrid joint cycle-life and knee-point prediction | Direct architecture-context support |
| `geurts2006extremely` | Verified | Extremely Randomized Trees algorithm | Direct foundational support |
| `chen2016xgboost` | Verified | XGBoost baseline | Direct foundational support |
| `breiman2001random` | Verified | Random Forest baseline | Direct foundational support |
| `javanmardi2023conformal` | Verified | Conformal intervals for RUL estimation | Direct methodological support; manuscript explicitly avoids claiming an exact conformal guarantee under batch shift |
| `wang2024capacity` | Verified | Incremental-capacity/capacity-gradient degradation stages and knee prediction | Direct primary support; separated from the DVA-inhomogeneity clause |

## Corrections applied in this audit

1. Restricted the dataset-source citation to Severson et al.; Attia et al. remains cited only for closed-loop fast-charging optimization.
2. Compressed the cell-level leakage statement to one sentence.
3. Replaced the broad “multi-resolution learning” description of Qi et al. with the directly supported “signal decomposition.”
4. Split the three-reference architecture citation and the three-reference future-work citation so that every citation group contains at most two papers.
5. Separated differential-voltage inhomogeneity evidence from incremental-capacity/capacity-gradient degradation-stage evidence.
6. Removed the uncited `tao2024review` entry from the manuscript bibliography.

## Remaining boundaries

- Literature citations motivate feature localization but do not establish a unique electrochemical mechanism for the present normalized Local DeltaQ regions.
- Transfer-learning and domain-generalization studies support future extensions, not demonstrated generalization of the present model to new chemistries or operating domains.
- The residual intervals reported here are empirical batch-shift diagnostics and are not presented as exact split-conformal intervals.
