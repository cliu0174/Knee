# Figure QA report

Overall export check: **PASS**.

- Backend: Python/matplotlib only.
- Primary editable export: SVG with text nodes preserved.
- Secondary exports: PDF and 600-dpi PNG.
- Quantitative source data: CSV files in `figure_source_data/`.
- Image-integrity note: plots are regenerated from numeric artifacts; no photographic panels or local image adjustments are present.
- Statistical note: folds, seeds, cell bootstrap intervals, and the protocol-frozen secondary-batch status are stated in captions/manuscript text.

| figure                                       |   png_width_px |   png_height_px |   png_dpi_x | svg_exists   | pdf_exists   |   svg_text_nodes |   source_files |
|:---------------------------------------------|---------------:|----------------:|------------:|:-------------|:-------------|-----------------:|---------------:|
| fig1_protocol_nested_workflow                |           4222 |            2266 |     599.999 | True         | True         |               21 |              1 |
| fig2_strict_nested_selector_comparison       |           4299 |            3348 |     599.999 | True         | True         |               55 |              2 |
| fig3_observation_budget_and_data_efficiency  |           4309 |            3233 |     599.999 | True         | True         |               62 |              2 |
| fig4_local_region_contribution_and_controls  |           4288 |            3345 |     599.999 | True         | True         |               64 |              4 |
| fig5_frozen_secondary_batch_performance      |           4307 |            3414 |     599.999 | True         | True         |               66 |              3 |
| fig6_feature_group_and_model_family_ablation |           4309 |            3465 |     599.999 | True         | True         |               57 |              3 |
| fig7_uncertainty_tail_and_complexity         |           4308 |            3293 |     599.999 | True         | True         |               66 |              4 |