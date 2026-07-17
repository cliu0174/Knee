# Figure contract (2026-07-13)

Target output: journal-ready double-column figures (about 183 mm wide), Python/matplotlib backend only, editable SVG as the primary export plus PDF and 600-dpi PNG. All quantitative panels must link to a source CSV copied into `manuscript_ionics/figure_source_data/`.

## Figure 1 — Protocol and data flow

- Core conclusion: the evidence chain separates development-only nested selection, an immutable freeze, and secondary-informed batch evaluation.
- Archetype: schematic-led composite.
- Panel map: (a) 124-cell source and corrected labels; (b) common N50/N100/N150 development cohort; (c) nested selection; (d) frozen 30-seed evaluation.
- Evidence hierarchy: protocol isolation is the hero evidence; sample counts and historical-status warning are validation annotations.
- Statistics: 41/42/40 retained cells at N150; no hypothesis test.
- Reviewer risk: calling the secondary batch untouched despite historical repository use.

## Figure 2 — Strict nested selector comparison

- Core conclusion: region-balanced and bootstrap-stable selection are statistically indistinguishable development leaders, while the contiguous candidate does not justify promotion.
- Archetype: quantitative grid.
- Panel map: (a) fold MAE distributions; (b) development selection score; (c) feature-set Jaccard stability; (d) paired cell-level MAE differences with bootstrap CIs.
- Statistics: 5-fold outer CV repeated twice; cell-level paired bootstrap/sign-flip tests; mean and fold SD stated.
- Reviewer risk: presenting the 0.013 score gap as meaningful.

## Figure 3 — Observation budget and data efficiency

- Core conclusion: longer observation and more development cells improve performance only to the extent supported by common-cohort nested evidence.
- Archetype: quantitative grid.
- Panel map: (a) N50/N100/N150 nested MAE; (b) 25/50/75/100% learning curve; (c) RMSE or tail trend; (d) effective training-cell counts.
- Statistics: common N150-retained cohort; outer folds; repeated stratified subsampling.
- Reviewer risk: confounding observation budget with changing cell eligibility.

## Figure 4 — Selected regions and controlled region tests

- Core conclusion: predictive information is distributed across ordered LDQ regions; the previously highlighted 0.525–0.650 band is informative but not sufficient alone.
- Archetype: asymmetric quantitative composite.
- Panel map: (a) frozen feature counts across normalized coordinates; (b) fold selection frequency heat strip by descriptor/region; (c) core-only/excluded/full/random-band controls; (d) core permutation delta.
- Statistics: selection performed inside each outer training fold; randomized controls repeated within fold.
- Reviewer risk: treating a predictive region as mechanistic proof.

## Figure 5 — Frozen external performance

- Core conclusion: the protocol-frozen region-balanced LDQ-ET reproduces the original anchor-level accuracy on the shifted batch without a statistically significant paired gain.
- Archetype: quantitative grid.
- Panel map: (a) predicted versus observed with empirical intervals; (b) signed residual versus observed knee; (c) original versus frozen paired absolute error; (d) 30-seed metric distribution.
- Statistics: 40 external cells; 30 seeds; cell bootstrap CI and paired sign-flip test.
- Reviewer risk: treating seeds as independent external datasets.

## Figure 6 — Feature-group and model-family ablation

- Core conclusion: local LDQ contributes beyond raw/global summaries, and the ExtraTrees readout is a small-sample choice rather than the claimed methodological novelty.
- Archetype: quantitative grid.
- Panel map: (a) seven feature groups; (b) linear/kernel/tree/boosting families; (c) paired error deltas to the full group; (d) feature count.
- Statistics: identical development outer folds with local selection repeated in each training fold.
- Reviewer risk: comparing models under different tuning or feature-selection access.

## Figure 7 — Uncertainty, tail risk, and complexity

- Core conclusion: the point model is lightweight, but a development-calibrated 90% residual interval undercovers the shifted secondary batch, particularly late-knee cells.
- Archetype: quantitative grid.
- Panel map: (a) overall/early/mid/late interval coverage; (b) absolute-error ECDF and q90/q95; (c) worst-cell table/strip; (d) fit time, inference time and model size.
- Statistics: nominal 90%; empirical coverage; 40 cells; model complexity averaged across 30 seeds.
- Reviewer risk: implying conditional conformal guarantees under distribution shift.

## Image integrity and export QA

No microscopy or photographic manipulation is involved. Every panel is regenerated from CSV/JSON artifacts; no values are hand-entered. SVG text remains editable, PDF fonts use Type 42, and PNG previews are exported at 600 dpi. Final QA checks panel labels, clipping, legends, font readability at 183 mm, grayscale distinguishability, exact sample definitions, and source-data traceability.
