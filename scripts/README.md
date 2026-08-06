# Reusable script index

Run these scripts from the repository root (`D:\Projects\Besttery`), because their input and output paths are rooted there.  Exploratory, hypothetical-preview, and one-off regeneration scripts have been removed after the confirmed manuscript package was finalized.

## Data and labels

- `download_severson_raw_data.ps1`, `build_severson_pkl_from_mat.py`: obtain and normalize source data.
- `correct_soh_curves.py`, `detect_knee_points.py`: construct corrected SOH trajectories and knee labels.
- `export_soh_curves.py`, `plot_soh_raw_curves.py`: export and inspect SOH curves.

## Features

- `extract_early_knee_features.py`, `extract_extended_early_features.py`, `extract_local_deltaq_features.py`: extract early-cycle and Local DeltaQ representations.
- `extract_severson_official_features.py`, `extract_severson_supplementary_features.py`: reproduce published baseline feature sets.
- `select_local_deltaq_features.py`: select the local descriptors.

## Formal experiments and baselines

- `run_second_round_experiments.py`, `run_nested_contiguous_ldq.py`, `run_formal_414240_selector.py`: core frozen LDQ-ET selection/evaluation workflows.
- `run_formal_cycle_pair_scan.py`, `run_formal_label_robustness.py`, `run_formal_observation_feature_ablation.py`: formal robustness and observation-budget analyses.
- `run_matched_xgboost_full70.py`, `run_bai_inspired_xgbrf_baseline.py`, `reproduce_severson_baseline.py`, `reproduce_severson_linear_table1_models.py`: matched and published-baseline comparisons.
- `run_feature_ablation.py`, `train_knee_baselines.py`: shared feature-grouping, model-building, and metric helpers imported by the formal workflows; retain these even though they originated as exploratory analyses.

## Figures and QA

- `prepare_knee_label_argument_figures.py`, `prepare_chapter3_methodology_figure.py`, `prepare_formal_414240_figures.py`, `export_formal_414240_direct_panels.py`: formal Figure 1--7 generation/export support.
- `analyze_knee_label_stability.py`, `audit_knee_split_convention.py`, `validate_enhancement_outputs.py`, `create_pdf_contact_sheet.py`: validation and visual-QA utilities.

The approved submission PNGs are external confirmed assets stored beside `main.tex`; do not regenerate or overwrite them without a new scientific/content review.
