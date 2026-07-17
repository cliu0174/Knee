# Final PDF quality-assurance report

Date: 2026-07-14  
Source: `manuscript_ionics/main.tex`  
Compiled artifact: `output/pdf/Besttery_LDQ_ET_strict_nested_manuscript_20260714.pdf`

## Automated checks

- Engine: MiKTeX pdfTeX 1.40.28 through `latexmk` 4.88.
- Output: 20 A4 pages, 687,878 bytes, PDF 1.5.
- SHA-256: `1BBDBF9A81C7278484CFD0C65DD35D7FAA906FB354072138B562BFB178B44942`.
- BibTeX and cross-reference passes converged.
- Final `main.log`: no LaTeX errors, overfull boxes, undefined citations, undefined references, or multiply defined labels.
- Remaining underfull-box messages arise from intentional wrapping in narrow table columns and do not indicate clipping.

## Visual checks

- Rendered the 21-page pre-adjustment build at 110 dpi into `tmp/pdfs/main_ionics_references_20260714/`, then removed the isolated final reference line by setting zero inter-entry bibliography spacing and recompiled to 20 pages.
- Inspected the final reference pages 18--20 at full resolution from `tmp/pdfs/main_ionics_references_compact_20260714/`.
- Re-rendered and inspected the citation-audited, polished Introduction pages 2--3 at 130 dpi in `tmp/pdfs/main_citation_audited_intro_polished_20260714/`.
- Inspected Introduction pages 2--3 at full resolution. The section is text-only, the three contribution bullets remain together, and Section 2 begins cleanly after the paper-organization paragraph.
- Inspected reference pages 19--20 at full resolution. All added entries and DOI links are complete, legible, and contained within the printable area.
- Confirmed that all seven remaining manuscript figures occur after the Introduction and that tables and figures remain inside the printable area after repagination.
- No clipped captions, truncated axes, missing glyphs, rasterization failures, or objects outside the printable area were observed.

## Scientific presentation checks

- The Introduction now follows background, research status and limitations, model motivation, text-only algorithm structure, itemized contributions, and paper organization, with no figures or tables.
- The research-status discussion now cites 25 distinct papers. Four publisher-verified *Ionics* papers support review context, early capacity/RUL prediction, incremental-capacity health features, and distribution-free uncertainty estimation. Three claim-level substitutions were made only where the evidence was equivalent; non-equivalent mechanistic and conformal references were retained elsewhere.
- All 32 cited references were independently resolved by DOI; titles, first authors, and publication sources match Crossref records. Claim-level support and wording corrections are documented in `full_citation_claim_audit_20260714.md`.
- The strict nested selector table and figure agree on the selected region-balanced configuration.
- External parity, distributed-region controls, tail errors, interval coverage, and computational cost are presented with contribution-first framing.
- Figure captions distinguish predictive reliance diagnostics from causal or mechanistic claims.

Status: **PASS** for compilation and manuscript-layout QA. Administrative placeholders and genuinely independent external validation remain scientific/submission limitations, not PDF defects.
