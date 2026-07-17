# Full citation and claim audit

Date: 2026-07-14  
Manuscript: `manuscript_ionics/main.tex`  
Reference database: `manuscript_ionics/references.bib`

## Scope and method

- All `\cite{}` occurrences in the manuscript were mapped to their surrounding claims.
- Every DOI was resolved through Crossref. Title, first author, journal or proceedings source, and bibliographic year were compared with the BibTeX record.
- Claim support was checked against official publisher abstracts and pages. Locally stored full-text PDFs were used where available, especially for the battery-prognosis literature.
- Methodological references were checked against their original articles rather than secondary descriptions.
- Two real but unused BibTeX entries (`jia2023identification` and `abramowicz2023domain`) were removed so that the database contains only the 32 works cited by the manuscript.

## Metadata result

- Cited works checked: **32**
- DOI resolved: **32/32**
- Title matched: **32/32**
- First author matched: **32/32**
- Journal or proceedings source matched: **32/32**
- Retraction, fabricated DOI, or unresolvable record detected: **0**

Three records have normal online-first versus issue-year differences. `hu2025batterygpt` was published online in December 2025 and assigned to *Nature Communications* volume 17 in 2026. The fused-lasso and functional-domain records likewise have Crossref online dates preceding their issue years. The manuscript retains the final volume or issue year.

## Claim-level assessment

| Key | Manuscript role | Evidence checked | Assessment |
|---|---|---|---|
| `tao2024review` | Data-driven state estimation and dataset design | *Ionics* publisher page and abstract | Appropriate review-context support. The revised text no longer attributes a specific generalization result to this review. |
| `severson2019data` | Early discharge-curve prediction, dataset provenance, and DeltaQ precedent | Nature Energy page and local full text | Strong direct support. |
| `attia2020closed` | Closed-loop fast-charging optimization using early lifetime prediction; dataset provenance | Nature page | Strong direct support. |
| `zhang2020impedance` | Prognostic information from impedance spectra | Nature Communications page | Strong direct support; the paper uses the full EIS spectrum. |
| `rhyu2025formation` | Formation-stage predictive Q(V) features | Local *Joule* full text | Strong direct support. |
| `qi2025hybrid` | Multi-resolution early capacity and RUL prediction | *Ionics* publisher abstract | Strong direct support. |
| `strange2023online` | Cycle-by-cycle lifetime updating and ensembling | Local full text and publisher metadata | Strong direct support. |
| `jiang2023predicting` | Machine-learning descriptors from discharge-voltage curves | Local full text and publisher metadata | Strong direct support. |
| `lu2023deepsoh` | SOH estimation without target-battery degradation labels | Nature Communications page | Strong direct support. |
| `wang2024pinn` | Stable SOH prognosis across battery types and charge/discharge protocols | Nature Communications page | Strong direct support. |
| `zhang2025batlinet` | Inter-cell lifetime prediction across protocols, temperatures, and chemistries | Nature Machine Intelligence page | Strong direct support. |
| `li2024varying` | Degradation-informed features and extrapolation under varying use | Local *Cell Reports Physical Science* full text | Strong direct support. |
| `tan2024magnet` | Domain-generalized degradation forecasting under distribution shift | Local full text | Strong direct support. |
| `fan2024relaxation` | Joint knee-point and RUL prediction from relaxation voltage | Local full text | Strong direct support. |
| `zhao2024historyfree` | History-free degradation-stage assessment and life prediction | Local full text | Strong direct support. |
| `hu2025batterygpt` | Full SOH trajectory, knee-point, and EOL prediction from early data | Nature Communications page and local full text | Strong direct support. |
| `fuhrmann2024dva` | DVA-based knee prediction and electrode inhomogeneity | Local full text | Strong direct support. The discussion now uses “electrode inhomogeneity” rather than the broader “electrode imbalance”. |
| `wang2025ica` | Incremental-capacity health indicators for SOH estimation | *Ionics* publisher abstract | Strong direct support. |
| `yu2025knee` | Curvature-based early knee-interval prediction | Local full text | Strong direct support. |
| `li2025degradationmodes` | Non-unique mechanisms behind similar capacity and power fade | Nature Communications page | Strong direct boundary support. |
| `tibshirani2005fused` | Sparsity plus smoothness across ordered coefficients | Original article metadata and method description | Strong method support. The revised wording follows the original objective rather than the looser phrase “ordered sparsity”. |
| `meinshausen2010stability` | Resampling-based control of variable selection | Original article metadata and method description | Strong method support. The revised wording avoids an unsupported generic reproducibility claim. |
| `servien2024interval` | Random-forest interval selection for functional regression | Original article metadata and publisher record | Strong direct method support. |
| `pang2025distributionfree` | Distribution-free SOH prediction intervals | *Ionics* publisher abstract | Strong direct support. |
| `sanches2026single` | Quantile ExtraTrees and conformalized SOH/RUL intervals | ScienceDirect publisher abstract | Strong direct support. |
| `safavi2024early` | CNN-based early RUL modelling | Local full text and publisher metadata | Appropriate direct support for the sequence-model comparison. |
| `cui2025joint` | Attention and temporal-convolution modules for joint life and knee prediction | Local full text | Strong direct support for the sequence-model comparison. |
| `geurts2006extremely` | Extremely Randomized Trees definition and readout rationale | Original Machine Learning article | Strong direct method support. Manuscript performance claims remain based on the present experiments. |
| `chen2016xgboost` | XGBoost comparison model | Original KDD paper | Strong direct method support. |
| `breiman2001random` | Random Forest comparison model | Original Machine Learning article | Strong direct method support. |
| `javanmardi2023conformal` | Conformal RUL intervals and the distinction from empirical residual bands | Original article and DOI record | Strong direct methodological support; the manuscript explicitly avoids claiming an exact conformal guarantee. |
| `wang2024capacity` | Degradation-stage analysis and knee-point prediction | Local full text | Strong direct support. |

## Text corrections made after the audit

1. Restricted the *Ionics* review citation to data-driven modelling and dataset design, which are directly covered by the review.
2. Replaced “ordered sparsity organizes correlated predictors” with the original fused-lasso role: sparsity plus smoothness across ordered coefficients.
3. Replaced the generic claim that stability selection “improves reproducibility” with its direct resampling-based variable-selection role.
4. Replaced “electrode imbalance” with “electrode inhomogeneity” in the DVA discussion.
5. Reorganized the Introduction into application bottleneck, established capabilities, generalization, knee/localized evidence, structured-selection gap, and present study.

## Overall conclusion

All 32 cited works are real and independently retrievable. After the wording corrections above, each citation supports the claim to which it is attached at either a direct experimental/methodological level or an explicitly identified review-context level. No citation is retained solely because its title appears related.
