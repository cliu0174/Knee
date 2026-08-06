# Battery knee-cycle prediction — final project guide

Updated: 2026-08-06

This repository contains the final source package for the Ionics manuscript, *Stability-aware Local DeltaQ selection for early lithium-ion battery knee-point prediction under batch shift*, together with the reusable data, experiment, figure, and QA code.

## Submission-ready files

The confirmed manuscript source is [manuscript_ionics/main.tex](manuscript_ionics/main.tex).  It is byte-identical to the confirmed file supplied on 2026-08-06 (SHA-256: `5D1883C9BA18B9B6ABC2B315D574B8C62E7E48D47FE500561AF089BBDAD9EF17`).  The seven approved PNG figures are kept next to it.

[overleaf_package_current](overleaf_package_current) is the direct-upload folder.  Upload every file in that folder to a blank Overleaf project and select **pdfLaTeX** with `main.tex` as the main document.  It intentionally uses a flat layout:

- `main.tex`, `references.bib`, `sn-jnl.cls`, and `sn-mathphys-num.bst`
- `Figure1.png` through `Figure7.png`

The package was compiled locally with pdfLaTeX and BibTeX on 2026-08-06.  The final 28-page build had no LaTeX errors or unresolved citations.

## Scientific scope

The manuscript predicts absolute battery knee cycles from the first 150 cycles.  It constructs full-life knee labels offline, represents early discharge curves with Local DeltaQ features, selects stable local descriptors, and uses an ExtraTrees predictor.  The confirmed abstract reports MAEs of 38.23 cycles (same-batch primary cohort) and 64.15 cycles (secondary-batch cohort), with corresponding R² values of 0.957 and 0.868.

## Reusable code

Only reusable code is retained under [scripts](scripts).  The files remain flat deliberately: several use paths relative to the repository root, and moving them would make their recorded workflows non-reproducible.  [scripts/README.md](scripts/README.md) is the definitive grouped index.

- The two Figure 1--2-specific source scripts remain in `manuscript_ionics/figure_scripts/`, next to their figure-source data.

The scripts assume the existing project data layout.  They are reproducibility tools, not part of the Overleaf upload folder.

## Retention policy

Historical review notes, temporary renderings, hypothetical-figure previews, and incremental regeneration scripts are pruned in favour of this concise guide and the confirmed manuscript/package.  Raw data, experiment results, figure source assets, manuscript backups, and pre-existing user-modified files are deliberately left untouched.
