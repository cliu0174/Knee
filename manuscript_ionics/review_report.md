# Manuscript Review Report

Manuscript: `main.tex`

Path: `D:/Projects/Besttery/manuscript_ionics/main.tex`

Review mode: simulated professional peer review, focusing on manuscript logic, evidence strength, technical soundness, reproducibility, and likely reviewer concerns.

## Review Setup

> **Status note (2026-07-13):** Sections 1--7 below preserve the original pre-revision reviewer simulation. They should not be read as a description of the current manuscript state. The authoritative resolution status is recorded in the revision update at the end of this file. The manuscript retains one main model, LDQ-ET top40; dual-view, MoE, and other fusion experiments are exploratory and are not part of the confirmatory paper claim.

- **Input scope:** Full LaTeX manuscript, corresponding PDF text extraction, figure file list, bibliography file, and LaTeX compilation log.
- **Assessment boundary:** This review is based on the local manuscript package. Individual references were not externally verified online. Figure readability comments are based on the source files, PDF text extraction, and compilation signals; the final visual PDF should still be checked manually.
- **Shared manuscript claim summary:** The manuscript proposes Local DeltaQ sensitive-region features combined with an Extremely Randomized Trees readout for early capacity knee-cycle prediction of lithium-ion batteries. Using the first 150 cycles from the corrected Severson/MIT-Stanford fast-charging dataset, the proposed LDQ-ET top40 model reports external secondary-batch performance of MAE 77.23 cycles, MAPE 9.04%, and R2 0.794 across 30 random seeds.
- **Visible evidence base:** The manuscript provides an external-batch validation protocol, Raw/DeltaQ and Local DeltaQ comparisons, horizontal model comparisons, top-k ablation, 30-seed robustness, and split-protocol analysis.
- **Missing materials affecting confidence:** The current draft lacks full reproducibility details for knee-label generation, DeltaQ construction, local-region definition, feature-ranking protocol, baseline tuning, selected-region stability, and code/data release.

## Reviewer 1

### Overall Assessment

The manuscript addresses a meaningful and practical battery-prognostics problem: predicting the capacity knee point before accelerated degradation becomes visible. The central design choice, preserving local DeltaQ responses rather than compressing the whole discharge response into global descriptors, is reasonable and potentially useful. The use of the secondary batch as an external validation cohort is a clear strength.

However, the paper currently does not provide enough methodological detail for a reader to reproduce the reported results. The largest technical weakness is that the knee-labeling procedure is described only as a "consistent curve-based knee detection procedure." Because the prediction target is algorithmically generated, ambiguity in this step directly affects the validity of the whole study.

### Who Would Be Interested, and Why

The work would interest researchers in battery health management, early degradation prediction, BMS-oriented prognostics, and small-sample machine learning for electrochemical systems. It is especially relevant to readers concerned with cross-batch generalization rather than random-split performance.

### Major Strengths

- The paper identifies a practical target distinct from conventional cycle-life or RUL prediction.
- The external secondary-batch validation protocol is stronger than a mixed random split.
- The representation-vs-readout narrative is clear: Local DeltaQ improves XGBoost, and ExtraTrees provides a further but smaller gain.
- The discussion appropriately avoids claiming a unique electrochemical mechanism for each local feature.

### Major Concerns

- The knee-point labeling method is underspecified.
- The DeltaQ construction does not state the exact cycle pair or pairs, interpolation grid, voltage/capacity coordinate range, normalization, smoothing, or missing-curve handling.
- The local-region design lacks enough detail: number of candidate regions, region width, overlap ratio, candidate descriptor count, and selected feature list are not reported.
- It is unclear whether feature ranking and top-k selection are performed entirely inside model-selection folds or whether any external validation feedback influenced the final top40 choice.

### Technical Failings That Need to Be Addressed

1. Provide a complete knee-labeling algorithm, including smoothing, fitting, thresholds, parameters, and edge-case handling.
2. Add a table defining all raw, global DeltaQ, and Local DeltaQ features.
3. Clarify whether correlation-based feature ranking uses only official training cells, the full development cohort, or fold-specific training subsets during cross-validation.
4. Demonstrate that the secondary batch was not used directly or indirectly to choose top40, tune hyperparameters, or select the final model.
5. Report the distribution of true knee cycles in the development and external cohorts.

### Assessment Against Nature-Style Criteria

- **Originality:** Moderate. The work appears to combine a localized DeltaQ representation with a low-data tree ensemble in a carefully separated validation protocol. The representation is plausible, but the novelty relative to prior local curve-feature or incremental-capacity analyses needs sharper positioning.
- **Scientific importance:** Field-relevant and practical, but not yet shown to have broad scientific importance beyond battery prognostics.
- **Interdisciplinary readership:** The current framing is understandable, but the appeal is mostly to battery and ML-for-energy readers.
- **Technical soundness:** Promising, but not fully established until labeling, feature construction, and selection leakage concerns are resolved.
- **Readability for nonspecialists:** Generally readable, although the Local DeltaQ construction remains too abstract.

### Recommendation Posture

Promising, but major methodological clarification is required before the central performance claims are fully established.

## Reviewer 2

### Overall Assessment

The manuscript is strongest when it presents itself as a rigorous representation and validation study rather than as a new high-capacity model. The reported external-batch improvement is meaningful, but the experimental evidence should be made more internally consistent and statistically persuasive.

At present, some numerical inconsistencies between the main results and the broader model-comparison table may raise questions. For example, similar model names and input representations appear with different MAE and R2 values across tables. These differences may be legitimate if they come from different seeds, hyperparameter settings, or experiment versions, but the manuscript must explain them explicitly.

### Who Would Be Interested, and Why

Researchers comparing feature-engineering and deep-learning approaches for battery lifetime prediction would find this work useful. It may also interest practitioners who prefer lightweight, stable models for small battery datasets.

### Major Strengths

- The paper reports several complementary evaluations rather than relying on a single result table.
- The top-k ablation supports the claim that moderate local-region coverage is beneficial.
- The paper distinguishes external-batch performance from random-split performance, which is an important methodological point in this field.
- The manuscript avoids overclaiming that deep models are generally unsuitable.

### Major Concerns

- Baseline fairness is not sufficiently documented.
- The adapted deep-learning baselines are described only at a high level, making it difficult to judge whether they are competitive implementations.
- The table-to-table differences in model performance need explanation.
- The top40 choice may be perceived as partly selected from external-test behavior unless the selection protocol is made stricter and clearer.

### Technical Failings That Need to Be Addressed

1. Report all main baselines as mean plus standard deviation across repeated seeds or splits where applicable.
2. Add paired statistical testing or bootstrap confidence intervals over the 40 external cells.
3. Include a naive baseline, such as predicting the development-set mean or median knee cycle.
4. Provide per-cell external errors or an error distribution plot.
5. Explain the difference between the LDQ-ET result in the main table and the ExtraTrees Local DeltaQ result in the model-comparison table.
6. State whether every model was tuned with the same development-only protocol and comparable search budget.

### Assessment Against Nature-Style Criteria

- **Originality:** The feature representation and external validation framing are useful, but the model itself is not the primary novelty.
- **Scientific importance:** The result is important within early battery prognostics, especially for small-sample external-batch prediction.
- **Interdisciplinary readership:** The manuscript currently remains specialized.
- **Technical soundness:** The central result is plausible, but stronger statistical reporting and baseline documentation are needed.
- **Readability for nonspecialists:** The performance narrative is clear; the experimental protocol should be made more transparent.

### Recommendation Posture

Technically interesting, but the significance and robustness claims should be strengthened with clearer baseline protocols, statistical uncertainty, and consistency checks.

## Reviewer 3

### Overall Assessment

The manuscript is readable and appropriately cautious in its electrochemical claims. The idea that local discharge-curve changes may carry useful early degradation information is intuitive. However, the manuscript currently uses the word "interpretable" more strongly than the evidence supports.

The paper would be more convincing if it showed where the selected Local DeltaQ regions lie along the voltage or normalized capacity coordinate, how stable those regions are across resampling, and whether they correspond to physically meaningful curve changes.

### Who Would Be Interested, and Why

Battery degradation researchers, battery-data scientists, and BMS engineers may be interested if the selected regions can be connected to interpretable degradation signatures. Without that connection, the paper remains mostly a predictive machine-learning study.

### Major Strengths

- The introduction motivates knee-point prediction clearly.
- The discussion acknowledges limitations in physical mechanism interpretation.
- The manuscript distinguishes primary external validation from supplementary random-split results.
- The limitations section is honest and useful.

### Major Concerns

- The interpretability claim is underdeveloped.
- The representative knee examples do not establish the full knee-label distribution.
- The selected local regions are not visualized.
- Figure readability should be manually checked in the final PDF. PDF text extraction suggests that some figure labels may be crowded or hard to parse, especially in workflow and model-comparison figures.

### Technical Failings That Need to Be Addressed

1. Add a heatmap or annotated curve showing the top Local DeltaQ regions.
2. Report feature importance, permutation importance, or SHAP-like analysis for the final model.
3. Show whether top regions are stable across bootstrap samples or cross-validation folds.
4. Add a knee-cycle distribution figure for development and external cohorts.
5. Improve figure label spacing and ensure all plots remain readable after journal-column scaling.

### Assessment Against Nature-Style Criteria

- **Originality:** The local-feature framing is promising but needs clearer differentiation from existing curve-based feature methods.
- **Scientific importance:** The study is useful for applied prediction, but physical insight is not yet established.
- **Interdisciplinary readership:** The work could reach a broader energy-materials audience if interpretability and degradation relevance are strengthened.
- **Technical soundness:** The predictive evidence is encouraging; interpretability evidence is insufficient.
- **Readability for nonspecialists:** The high-level story is accessible, but the technical representation requires more concrete visualization.

### Recommendation Posture

Readable and promising, but the claims about interpretability and local degradation sensitivity need stronger visual and analytical support.

## Cross-Review Synthesis

### Consensus Strengths

- The manuscript has a clear practical target: early capacity knee-point prediction.
- The external secondary-batch validation protocol is the strongest part of the study.
- The Local DeltaQ representation is motivated by a plausible limitation of global discharge-curve descriptors.
- The paper's central comparison separates representation gain from readout-model gain.
- The discussion is appropriately cautious about physical interpretation and deployment limitations.

### Consensus Technical Risks

- The knee label is not reproducible from the current description.
- The Local DeltaQ feature construction is underdefined.
- Feature selection and top-k choice may be questioned unless the development-only protocol is made explicit.
- Baseline fairness and table-to-table consistency require clarification.
- The interpretability claim is not yet supported by sufficient evidence.

### Where Emphasis Differs Across Reviewers

- Reviewer 1 emphasizes reproducibility and target-definition validity.
- Reviewer 2 emphasizes experimental fairness, statistical support, and consistency of reported results.
- Reviewer 3 emphasizes interpretability, figure quality, and broader readability.

### Broad-Interest / Significance Readout

The manuscript appears suitable for a domain journal focused on batteries, electrochemical energy systems, or applied battery prognostics if the technical issues are addressed. Under a stricter Nature-style lens, the current draft reads as a solid field-specific advance rather than a broad, far-reaching scientific result.

### Most Important Issues to Resolve Before a Strong Case Is Established

1. Fully specify and validate the knee-labeling procedure.
2. Fully define the Local DeltaQ representation and feature-ranking pipeline.
3. Clarify all model-selection steps and ensure the secondary batch is used only once for final validation.
4. Add statistical uncertainty and per-cell error analysis for external validation.
5. Visualize selected regions and support the interpretability claim.
6. Reconcile numerical differences across result tables.
7. Replace all manuscript placeholders before submission.

## Risk / Unsupported Claims

- The phrase "lightweight and interpretable route" is only partly supported. The model is lightweight, but interpretability needs additional evidence.
- The term "sensitive regions" is plausible but should be supported by selected-region maps, stability analysis, or feature-importance evidence.
- Claims about deep-learning baselines should remain limited to the specific small-sample, external-batch setting used here.
- The current Data Availability and Code Availability statements are weak for a data-driven paper. Public scripts, processed labels, and split definitions would substantially improve confidence.
- Funding, author contribution, and supplementary information placeholders must be completed before submission.
- Recent references, especially 2024 and 2025 entries with abbreviated author lists, should be verified for full bibliographic accuracy.

## Suggested Revision Priorities

### High Priority

1. Add a reproducible knee-labeling subsection with equations, parameters, and sensitivity analysis.
2. Add a Local DeltaQ implementation table: cycle pairs, interpolation grid, region width, overlap, descriptors, candidate count, and selected count.
3. Clarify the exact feature-selection protocol and prevent any perception of external-validation leakage.
4. Make Table 2 and Table 3 numerically consistent or explain why values differ.
5. Report uncertainty for all major baselines, not only LDQ-ET.

### Medium Priority

1. Add selected-region visualization and feature-importance analysis.
2. Add external per-cell error distribution and worst-case examples.
3. Add naive baseline and possibly a simple linear/regularized model baseline in the main table.
4. Improve adapted deep-baseline descriptions so reviewers can judge fairness.
5. Add knee-cycle distribution comparison between development and external cohorts.

### Low Priority

1. Polish figure spacing and labels after final PDF rendering.
2. Replace placeholders in declarations.
3. Strengthen data/code availability wording.
4. Verify bibliography completeness and DOI accuracy.

## Authoritative post-revision audit (2026-07-13)

This section supersedes the earlier progress snapshot. The paper now reports one formal main model: a region-balanced Local DeltaQ top40 selector with a prespecified ExtraTrees readout. All imputation, candidate scoring, stability estimation, region balancing, selector comparison, and model comparison are fitted inside the relevant development folds.

### Resolved issues

- **Leakage-resistant selection:** five selector families were compared by strict 5-fold $\times$ 2 nested development evaluation. Region-balanced selection achieved the best predeclared score (48.752), narrowly ahead of bootstrap-stable selection (48.766); their paired difference was not significant.
- **Honest secondary-batch status:** because historical repository experiments had viewed this batch, it is labelled a secondary-informed, protocol-controlled batch evaluation rather than untouched external confirmation.
- **Exact historical anchor:** the prior LDQ-ET top40 result was reproduced at $77.23\pm0.89$ cycles MAE over 30 seeds.
- **Frozen result:** the new immutable region-balanced model reached $76.57\pm0.72$ cycles MAE, $113.87\pm0.92$ RMSE, $8.99\pm0.10\%$ MAPE, and $R^2=0.783\pm0.003$. Its cell-level improvement over the historical anchor was $-0.61$ cycles (95\% CI, $-3.93$ to $3.32$; $p=0.778$), which supports parity rather than superiority.
- **Observation budgets and data efficiency:** matched N50/N100/N150 experiments and 25/50/75/100\% repeated development subsampling were completed. Performance improved monotonically with more cycles and more development cells; the apparent contiguous-band win in the initial N50 screen did not reproduce in the matched 5-fold $\times$ 2 run.
- **Contribution controls:** seven feature-group ablations, five readout families, core-only/core-excluded/selected-minus-core controls, random bands, random features, and a reliance-only permutation diagnostic were added. Local DeltaQ did not show a clear independent mean-MAE increment over raw+global features, and no single coordinate band was necessary or sufficient.
- **Reliability and cost:** the secondary-batch q90 and q95 absolute errors were 190.61 and 265.16 cycles. A nominal 90\% development-calibrated residual interval covered only 77.5\% of secondary cells and 64.3\% of late-knee cells. Mean fit time was 1.18 s, inference 4.56 ms/cell, and serialized size 5.02 MB.
- **Literature and novelty:** the nearest interval-selection, stability-selection, ordered-smoothness, conformal-prognostics, and same-dataset ExtraTrees work is cited. The defensible contribution is the task-specific Local DeltaQ representation plus chronology-aware, leakage-resistant knee-cycle audit, not interval selection, ExtraTrees, or conformal inference in isolation.
- **Artifacts and QA:** seven new figure bundles are available as SVG/PDF/600-dpi PNG with 19 source CSV files. The manuscript compiles to a 20-page PDF with no overfull boxes, undefined citations, undefined cross-references, or LaTeX errors. All pages were rendered; the full contact sheet and key figure/table pages were visually inspected.

### Remaining major limitations

- The evidence comes from one fast-charging LFP dataset; a genuinely untouched chemistry, site, or cycling protocol is still required for a broad generalization claim.
- The target is an offline, algorithm-dependent absolute knee label. Piecewise-linear and Kneedle labels have nearly identical rank ordering ($\rho\approx0.998$) but differ by 32.85 cycles on average.
- The secondary batch cannot regain untouched-confirmation status, and its residual intervals under-cover under batch shift.
- The regional analyses show distributed redundancy and predictive reliance, not a unique electrochemical mechanism or a causal voltage-region assignment.
- Author/funding/contribution details and public repository/accession links remain submission-stage placeholders.

### Revised recommendation

The manuscript is now substantially more reproducible, transparent, and statistically defensible for a battery-prognostics or electrochemical-energy domain journal. It still does not support a broad Nature-level generalization or mechanism claim without genuinely independent validation and stronger uncertainty calibration.
