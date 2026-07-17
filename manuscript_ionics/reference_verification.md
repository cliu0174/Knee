# Reference verification report

Verification date: 2026-07-13

## Scope and checks

- Parsed every entry in `references.bib` through the manuscript BibTeX build.
- Checked that every citation key used by `main.tex` resolves after BibTeX and repeated LaTeX compilation.
- Verified title, journal, year, DOI, volume/article number, and author list for the newly cited 2024--2025 papers against the locally supplied publisher PDFs.
- Cross-checked selected publisher records for recent or previously abbreviated entries. No metadata-only paper was used to support a manuscript claim.
- Re-ran a claim-oriented nearest-work search for ordered functional-region selection, stability selection, conformal prognostics, and 2026 ExtraTrees battery prognostics. The resulting novelty boundaries are recorded in `docs/LITERATURE_NOVELTY_AUDIT_20260713.md`.

## Corrections made

| Citation key | Correction |
|---|---|
| `yao2022two` | Corrected authors to Jia Yao, Kody Powell, and Tao Gao. |
| `strange2023online` | Corrected Rasheed Ibraheem's initial/name and expanded the three-author list. |
| `jiang2023predicting` | Corrected authors to Yinfeng Jiang and Wenxiang Song. |
| `jia2023identification` | Added the complete five-author list. |
| `safavi2024early` | Corrected Journal of Energy Storage volume from 99 to 98 and added the complete author list. |
| `wang2024capacity` | Corrected first author from `Y. Wang` to Teng Wang; added all authors, volume 3, and article 100171. |
| `cui2025joint` | Corrected first author from `Y. Cui` to Xinru Cui and added the complete author list. |
| `hu2025batterygpt` | Expanded the complete author list and corrected the final issue metadata to *Nature Communications* volume 17, article 126 (2026); the publisher page records online publication on 5 December 2025. |
| `yu2025knee` | Corrected first author from `X. Yu` to Yongjiang Yu; added all authors, volume 633, and article 236438. |

## Newly added, full-text-verified references

| Citation key | DOI | Evidence used in manuscript | Support grade |
|---|---|---|---|
| `fan2024relaxation` | 10.1016/j.energy.2024.130900 | Relaxation-voltage features and joint knee/RUL prediction | Strong method-context support |
| `fuhrmann2024dva` | 10.1016/j.jpowsour.2024.235210 | Local differential-voltage structure, inhomogeneity, and knee risk | Strong mechanism-context support |
| `zhao2024historyfree` | 10.1016/j.ensm.2024.103441 | Degradation-stage recognition and physics-similarity sample selection | Strong method-context support |
| `tan2024magnet` | 10.1016/j.ensm.2024.103725 | Domain shift and domain-generalized degradation forecasting | Strong method-context support |
| `li2024varying` | 10.1016/j.xcrp.2024.101891 | Degradation-informed features and hierarchical extrapolation across usage conditions | Strong method-context support |
| `rhyu2025formation` | 10.1016/j.joule.2025.101884 | Systematic, interpretable early feature design | Strong method-context support |

## Newly added, publisher/DOI-verified nearest-work references

| Citation key | DOI | Evidence used in manuscript | Support grade |
|---|---|---|---|
| `meinshausen2010stability` | 10.1111/j.1467-9868.2010.00740.x | Subsampling-based stability selection and its claim boundary | Primary-method support |
| `tibshirani2005fused` | 10.1111/j.1467-9868.2005.00490.x | Smoothness/sparsity for meaningfully ordered predictors | Primary-method support |
| `servien2024interval` | 10.1002/sam.11705 | Random-forest interval selection for functional regression | Closest-method support |
| `abramowicz2023domain` | 10.1111/biom.13669 | Functional-domain selection and formal error-control distinction | Primary-method support |
| `javanmardi2023conformal` | 10.36001/ijphm.2023.v14i2.3417 | Conformal prediction intervals in RUL prognostics | Primary-method support |
| `sanches2026single` | 10.1016/j.rineng.2026.111218 | Same 124-cell dataset, single-cycle statistics, quantile ExtraTrees and conformal SoH/RUL prediction | Closest battery-method support |

## Claim boundaries

- Fuhrmann et al. support the general claim that restricted differential-voltage regions can reflect electrode-capacity inhomogeneity and lithium-plating risk. They do **not** prove that the normalized LDQ region at 0.525--0.650 in the present dataset has the same mechanism.
- Tan et al. and Zhao et al. motivate domain-generalization and similarity-selection extensions. They do **not** validate those modules for the present two-development-batch dataset.
- Rhyu et al. and Li et al. support the value of physically/degradation-informed feature design. Their targets and datasets differ from the present knee-cycle task, so they are used for method context rather than direct performance comparison.
- Servien and Vialaneix prevent a generic novelty claim for tree-based interval selection. The manuscript can claim only the battery-specific representation/protocol combination, subject to the new nested evidence.
- Sanches et al. prevent novelty claims for ExtraTrees, lightweight inference, single-cycle statistics, or conformalized tree ensembles on this dataset. Their SoH/RUL targets and descriptors differ from the present localized early-$\Delta Q$ knee-cycle task.
- The cross-fitted residual interval added in this repository is an empirical batch-shift diagnostic. It is not described as exact split-conformal or conditional coverage.

## Build status

The bibliography compiles with `sn-mathphys-num.bst`. The final MiKTeX build produced a 20-page A4 PDF. The log contains no undefined citations, undefined cross-references, overfull boxes, or LaTeX errors. Remaining messages are underfull-box warnings caused by deliberate wrapping in narrow table columns. All pages were rendered through Poppler and visually inspected; the revised abstract, discussion, conclusion, tables, figures, references, and hyperlinks remain inside the page area.
