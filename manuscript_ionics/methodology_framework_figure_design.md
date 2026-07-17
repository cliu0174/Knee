# LDQ-ET Methodology Framework Figure Design Specification

> 中文方案与英文图中文字实施规范

## 0. 复检后的设计结论

建议优先制作一张全页宽三面板主图，并将其放在第三章开头导语之后、Section 3.1 之前。图的主线不是泛化的“数据—模型—结果”，而是三个必须由审稿人一眼读懂的方法边界：

1. 完整 SOH 轨迹只用于离线构造 knee-cycle 标签，预测输入严格限于 cycles 1–150。
2. Local DeltaQ 的插补、bootstrap 打分、区域均衡选择及 ExtraTrees 配置选择均在相应 development training fold 内完成。
3. 40-cell secondary batch 仅在协议冻结并基于全部 83 个 development cells 拟合后进入单向评估流程，不向任何开发步骤反馈信息。

三面板分别承担：

- Panel (a)：输入窗口与离线标签的信息边界；
- Panel (b)：候选字典、训练折内 region-balanced 选择、70 维输入和嵌套开发；
- Panel (c)：协议冻结、全部开发电芯拟合和 secondary-batch evaluation。

本次复检还收紧了三处容易产生过度表述的措辞：

- secondary batch 统一称为 `protocol-frozen secondary-batch evaluation`，不称为 `independent external validation`；
- 不写 `evaluated once`，因为冻结配置在 30 个预设算法种子下重复拟合和评估；
- 被选中的 Local DeltaQ 区域只表述为 `distributed predictive information` 或 `predictive association`，不赋予唯一电化学机理含义。

以下正文保留英文作为精确的制图、caption 和 LaTeX 实施规格，避免后续生成图文件时发生术语漂移。

## 1. Purpose and status

This document specifies the proposed methodology framework figure for the manuscript *Region-balanced Local DeltaQ selection for reproducible early lithium-ion battery knee-point prediction under batch shift*.

- Status: design specification only; no figure asset or LaTeX reference has yet been created.
- Intended location: immediately after the opening paragraph of Section 3 (`Methodology`) and before Section 3.1 (`Region-balanced Local DeltaQ selection`).
- Recommended deliverable: one full-width, three-panel framework figure.
- Fallback: split the selector detail into a second figure only if the complete figure cannot retain a minimum final text size of 7.5 pt.
- Non-negotiable constraint: no experimental value, conclusion, method parameter, cohort definition, or protocol boundary may be changed during figure production.

## 2. Figure contract

### 2.1 Core conclusion

Using predictor inputs restricted to cycles 1–150, LDQ-ET performs fold-safe, region-balanced Local DeltaQ selection exclusively within the development cohort, freezes the complete protocol, and only then evaluates the fitted predictor on the held-out secondary batch.

### 2.2 Evidence chain

| Panel | Question answered | Required evidence |
|---|---|---|
| (a) Observation and information boundaries | What information is available to the predictor, and how is the target obtained? | Cycles 1–150 form the only predictor input; the full SOH trajectory is used only to construct the offline knee label. |
| (b) Fold-safe LDQ-ET development | How is the candidate dictionary converted into a reproducible predictor? | Candidate counts, training-fold-only preprocessing and selection, 40 + 30 feature assembly, ExtraTrees readout, and nested development. |
| (c) Protocol freezing and secondary-batch evaluation | When and how does the secondary batch enter the workflow? | The protocol is frozen and fitted on all 83 development cells before predictions are generated for the 40 secondary-batch cells. |

### 2.3 Figure archetype

The figure is a **schematic-led composite** with a cross-functional information-flow structure. It should have one dominant left-to-right pipeline, supported by a compact input/label boundary panel and a clearly isolated secondary-batch lane.

### 2.4 Principal review risks addressed

The figure must prevent the following misreadings:

1. The knee is already visible in the first 150 cycles.
2. Measurements after cycle 150 enter the predictor.
3. Secondary-batch cells contribute to imputation, scoring, feature selection, hyperparameter selection, or model fitting.
4. Local DeltaQ selection is performed before cross-validation rather than within each training fold.
5. The 365 Local DeltaQ candidates are all supplied to ExtraTrees.
6. Selected Local DeltaQ regions constitute a unique electrochemical mechanism map.
7. The secondary-batch experiment is an independent external clinical-style validation or a single stochastic model run.
8. The morphology descriptors are unconstrained while only spatial regions are capped.
9. The fold-specific selected sets used during nested assessment are the same object as the final frozen feature set.

## 3. Literature-derived design principles

The local `paper` collection contains 25 PDF files and 24 unique file records. One Hu file is an exact duplicate, whereas the same-named Yu and Wang files are different versions and were reviewed separately. The visual audit covered the following design families.

| Design family | Local examples | Elements worth adopting |
|---|---|---|
| Linear prediction pipelines | Jiang, Li, Yang, Yao | Unambiguous reading direction; explicit data-to-feature-to-model mapping; sparse annotations. |
| Partitioned multistage frameworks | Hu, Bai, Cai, Fu, Rhyu, Wang, Zhao | Titled stage containers; clear separation of data preparation, feature engineering, model development, and application. |
| Model-architecture diagrams | Cui, Fan, Tan, Yu, Safavi | Useful only for compact model or selector insets; avoid unnecessary layer-level detail for tree ensembles. |
| Algorithm flowcharts | Jia, Zhang | Suitable for deterministic traversal and capacity checks, but a full decision-tree flowchart would overemphasize implementation detail here. |
| Evidence-led schematics rather than end-to-end workflows | Severson, Fuhrmann, Schaeffer, Strange | Small data-derived plots can ground an abstract workflow, but performance panels should remain in Results. |

The strongest references for the proposed composition are Rhyu's feature-design partitioning, Jiang's candidate-to-selected-feature mapping, Li's high-level early-life pipeline, Zhao's staged information flow, and Tan's explicit domain separation.

The figure should avoid dense decorative roadmaps, gradients, three-dimensional blocks, excessive dashed containers, miniature performance charts, and neural-network-style layer stacks that do not correspond to the ExtraTrees implementation.

## 4. Recommended overall layout

### 4.1 Canvas and panel proportions

- Final width: approximately 170–178 mm, inserted at `\textwidth`.
- Target aspect ratio: approximately 1.8:1 to 2.0:1.
- Recommended arrangement:
  - Panel (a): full-width upper band, approximately 27–30% of figure height.
  - Panel (b): lower-left main panel, approximately 66–70% of lower-row width.
  - Panel (c): lower-right panel, approximately 30–34% of lower-row width.
- The main visual hierarchy should be Panel (b) > Panel (c) > Panel (a).
- Final figure text must remain at least 7.5 pt at manuscript size.
- The figure must communicate one visual thesis in a left-to-right reading pass: early-only inputs are transformed into a fold-safe, region-balanced predictor before one-way secondary-batch evaluation.
- Every arrow must have one unambiguous scientific meaning and a visible arrowhead. Arrows may not cross, loop backward, or pass through an unrelated container.
- Use no more than four visual levels: panel, stage container, functional block, and supporting annotation. Do not add a fifth level of nested boxes.
- A block must either define a data object, state a transformation, or establish a protocol boundary; decorative blocks are not permitted.
- Preserve at least one text-height of whitespace between adjacent functional blocks and at least two text-heights between panels. Empty space is part of the hierarchy, not unused canvas.
- The design must remain legible when reduced to one-column screen width and when printed in grayscale.

### 4.2 Wireframe

```text
┌────────────────────────────── Panel (a) ──────────────────────────────┐
│ Early observations (cycles 1–150) ───────────► predictor-input path  │
│ Complete SOH trajectory ─ ─ ─ ─ ─ ─ ─ ─ ─ ─► offline label only     │
└───────────────────────────────────────────────────────────────────────┘

┌──────────────────────── Panel (b) ────────────────────────┐ ┌ Panel (c) ┐
│ Candidate dictionary → fold-only selection → 70-D → ET   │ │ Freeze     │
│          inner selection and outer assessment             │ │ ↓ fit ↓    │
│          distributed Local DeltaQ association             │ │ secondary  │
└────────────────────────────────────────────────────────────┘ └───────────┘
```

## 5. Panel specifications

### 5.1 Panel (a): Observation and information boundaries

#### Scientific role

Panel (a) establishes the separation between predictor information and target construction. It should complement, rather than duplicate, the detailed SOH trajectories already shown in Figure 1.

#### Required visual elements

1. A compact schematic SOH trajectory.
2. A light-purple shaded interval from cycles 1 to 150.
3. A later dashed knee-cycle marker.
4. A solid blue arrow from the early interval to the feature dictionary.
5. A rust-colored dashed arrow from the complete trajectory to the offline knee label.
6. A short boundary statement showing that post-150 measurements do not enter the predictor.

#### Exact English labels

- `Development cohort (n = 83)`
- `Early observations: cycles 1–150`
- `Predictor input only`
- `Complete SOH trajectory`
- `Offline knee-cycle label, y`
- `Label construction only`
- `Post-150 measurements do not enter the predictor`

#### Information-flow rules

- The early-observation arrow must be solid.
- The complete-trajectory-to-label arrow must be dashed and must not connect to any feature or model block.
- The knee label may connect to training objectives and evaluation, because it is the supervised target, but the underlying post-150 measurements must not connect to predictor features.
- Do not imply that the knee is detected within the input window.

### 5.2 Panel (b): Fold-safe region-balanced LDQ-ET development

Panel (b) is the hero panel and contains four visual layers.

#### Layer B1: Candidate dictionary

Use three aligned blocks:

- `22 raw capacity/SOH descriptors`
- `8 global ΔQ descriptors`
- `365 Local ΔQ candidates`

Under the Local DeltaQ block, use smaller supporting text:

- `351 regional descriptors (39 regions × 9 descriptors)`
- `14 matrix-profile morphology descriptors`

The 22 raw/SOH and eight global DeltaQ descriptors should join a fixed-feature branch labeled:

- `30 fixed descriptors`

Only the 365 Local DeltaQ candidates should enter the region-balanced selector.

#### Layer B2: Training-fold-only Local DeltaQ selection

Place the complete selector inside a blue-outlined container titled:

- `Within the current development training fold only`

The selector flow is:

```text
Training-fold median imputation
→ Bootstrap stability scoring (B = 40)
→ Descending-score traversal with group cap c ∈ {2, 3}
→ 40 selected Local ΔQ descriptors
```

Use `training-fold median imputation`, not the broader phrase `development-set imputation`, because medians are re-estimated within the corresponding training subset at every nested split. Validation data receive the medians learned from their paired training subset.

The selector inset should show:

- gray overlapping Local DeltaQ candidate regions;
- dispersed blue selection markers across the normalized coordinate;
- a small separate morphology group;
- a visible cap marker applying to both each spatial region and the morphology group;
- the frozen outcome summarized only in the final-protocol part of the panel as `13 spatial regions + 3 morphology descriptors`.

Use a compact selector-side note:

- `One cap per spatial region and morphology group`

The fold-specific selected sets inside nested assessment must be visually distinguished from the final frozen set. For example, use neutral blue markers for fold-specific selection and a darker outlined summary box only after development-wide protocol freezing. Do not visually imply that one outer-fold selection is transferred directly to the secondary batch.

Add the interpretive boundary directly beside the inset:

- `Distributed predictive information`
- `Predictive association, not a unique mechanistic assignment`

Do not use a flame, reaction, electrode, or causal-mechanism icon on selected regions.

#### Layer B3: Feature assembly and ExtraTrees readout

The two branches should merge as:

```text
40 selected Local ΔQ descriptors
                +
30 fixed descriptors
                ↓
70-dimensional input
                ↓
ExtraTrees regressor
```

Use a compact three-tree line icon rather than a literal forest or neural-network layer stack.

The figure may state:

- `Two prespecified ExtraTrees capacities compared during development`

If space permits, place the configurations in small secondary text:

- `Compact: 500 trees; min leaf = 2`
- `Formal: 1200 trees; min leaf = 1`
- `max features = 0.5; min split = 2; max depth = 6; no bootstrap`

These settings must not become the dominant visual element; the full reproducibility detail remains in the text.

#### Layer B4: Nested development protocol

Use nested rectangular frames, not circular feedback arrows:

- Outer frame: `Outer 5-fold × 2 assessment`
- Inner frame: `Inner 3-fold configuration selection`

The inner frame contains selector and ExtraTrees configuration selection. The outer validation fold is used only for development performance assessment.

After outer assessment, show a distinct arrow to:

- `Development-wide 5-fold × 2 protocol freezing`

This second repeated cross-validation stage is used to fix the final configuration on the development cohort and should not be visually merged with the outer performance estimates.

Add a small line directly below the training-fold container:

- `Repeated independently for every inner and outer training split`

This annotation is required because the same imputation and selection operations occur at multiple nested levels. It prevents the broad container title from being misread as preprocessing performed once on the full development cohort.

### 5.3 Panel (c): Protocol freezing and secondary-batch evaluation

#### Scientific role

Panel (c) demonstrates that the held-out batch enters only after all data-dependent choices have been frozen.

#### Required flow

```text
Development-wide protocol freezing
→ Frozen configuration
→ Fit on all 83 development cells using 30 prespecified seeds
→ One-way evaluation gate
→ Secondary batch (n = 40)
→ Protocol-frozen knee-cycle predictions and performance metrics
```

The frozen-configuration block should list, in compact form:

- `Region cap c = 3`
- `40 Local DeltaQ feature names`
- `Formal ExtraTrees settings`
- `Feature-file hash`
- `30 prespecified seeds`

Use the wording `protocol-frozen secondary-batch evaluation`. Do not label this step `independent external validation`, because the 40 cells are the official secondary-test batch of the same public dataset. Do not write `evaluated once`, because the frozen estimator is fitted and evaluated under 30 prespecified algorithmic seeds.

Place a compact qualifier beside the seed label:

- `Algorithmic variation, not 30 external cohorts`

#### Secondary-batch exclusion statement

Place the secondary-batch lane behind a light-gray or very pale green field with a visible one-way boundary. Include the following text:

```text
No secondary-batch contribution to:
• imputation
• candidate scoring
• feature selection
• hyperparameter selection
• model fitting
```

No arrow may return from the secondary-batch lane to the development lane.

## 6. Canonical terminology ledger

| Concept | Canonical figure wording | Avoid |
|---|---|---|
| Method | `region-balanced Local DeltaQ–ExtraTrees (LDQ-ET)` | Unexplained variants such as `Local-ΔQ ET` or `RB-LDQ model` |
| Early input | `early observations: cycles 1–150` | `first 150 data points`, `early-life trajectory` without the cycle limit |
| Label | `offline knee-cycle label` | `future input`, `detected early knee` |
| Development cells | `development cohort (n = 83)` | `training set` when inner/outer folds are being discussed |
| Held-out cells | `secondary batch (n = 40)` | `external cohort`, `independent external validation set` |
| Evaluation | `protocol-frozen secondary-batch evaluation` | `cross-validation on the secondary batch` |
| Fixed features | `30 fixed descriptors` | `30 selected baseline features` |
| Local candidates | `365 Local DeltaQ candidates` | `365 input features` |
| Final representation | `70-dimensional input` | `70 selected Local DeltaQ features` |
| Interpretation | `distributed predictive information` and `predictive association` | `mechanistic hotspot`, `unique degradation mechanism`, `causal region` |
| Group constraint | `one cap per spatial region and morphology group` | `spatial cap only`, `unconstrained morphology group` |

## 7. Visual style specification

### 7.1 Palette

Use the palette already established by the manuscript's current Figure 1.

| Role | Color | Suggested hex |
|---|---|---|
| Main predictor and feature flow | muted blue | `#2F6F8F` |
| Early observation window | muted purple | `#8B73C6` |
| Frozen protocol and secondary evaluation | muted green | `#5D8B62` |
| Offline label and information-boundary warning | muted rust | `#B65F4A` |
| Candidate regions and inactive structure | neutral gray | `#B7BDC3` |
| Panel/container fill | near-white tint of corresponding color | 5–10% visual saturation |

Color must be redundant with labels, line styles, and spatial grouping so the figure remains interpretable in grayscale.

### 7.2 Typography and geometry

- Typeface: Arial, Helvetica, or DejaVu Sans.
- Final text size: 7.5–8.5 pt; panel labels 9–10 pt bold.
- Stroke width: 0.8–1.0 pt; primary arrows 1.1–1.3 pt.
- Corners: slightly rounded rectangles, consistent radius throughout.
- Background: white.
- No shadows, gradients, glossy effects, or decorative 3D elements.
- Prefer direct labels to a detached legend.
- Use dashed lines only for offline-label paths, grouping boundaries, or prohibited information flow.

### 7.3 Arrow semantics

| Arrow style | Meaning |
|---|---|
| Solid blue | Predictor data or feature flow |
| Dashed rust | Offline label construction or information boundary |
| Solid green | Frozen deployment/evaluation flow |
| Thin gray | Structural association or nested-protocol containment |
| Crossed or terminated line | Explicitly prohibited information flow; use sparingly |

### 7.4 Top-tier visual communication constraints

The figure should read as a scientific argument, not as software documentation. Apply the following constraints during drawing and review:

- **One dominant claim.** The visual focal point is the fold-safe conversion of 365 Local DeltaQ candidates into a region-balanced 40-feature set. Panel (a) supplies the input boundary; Panel (c) supplies the evaluation boundary. Neither may compete with Panel (b).
- **Progressive disclosure.** At normal viewing size, readers should first recognize the three stages and one-way flow. Candidate counts, fold counts, and hyperparameters should be readable only on closer inspection.
- **Direct labeling over legends.** Encode meaning at the block or arrow itself. A detached legend may explain colors only if direct labels would become redundant.
- **No results in a method figure.** Do not add MAE, RMSE, scatterplots, error bars, ranked regions, or performance claims. Those belong to the Results figures.
- **No false precision.** Do not display a geographic-looking voltage map, electrode cartoon, or heatmap that implies spatial or mechanistic resolution beyond the normalized Local DeltaQ coordinate.
- **Minimal implementation detail.** The selector should be recognizable as deterministic descending-score traversal, but pseudocode, formulas, JSON syntax, file paths, and full hyperparameter lists must remain in the manuscript text or supplement.
- **Editorial discipline.** Use sentence case, one font family, consistent noun phrases, and no unexplained abbreviations other than those already defined in the paper. Avoid exclamation marks, marketing language, icons with decorative meaning, and redundant captions inside blocks.
- **Reproduction safety.** All values shown in the figure must be traceable to the frozen manuscript text; if a parameter changes, update the figure, caption, and checklist together.

## 8. Recommended figure text inventory

The following labels are approved for direct use. Text should be shortened only if its scientific meaning is preserved.

### Panel headings

- `(a) Observation and label boundaries`
- `(b) Fold-safe LDQ-ET development`
- `(c) Protocol-frozen secondary-batch evaluation`

### Core pipeline labels

- `Early observations: cycles 1–150`
- `Early capacity/SOH and discharge-curve measurements`
- `22 raw capacity/SOH descriptors`
- `8 global ΔQ descriptors`
- `365 Local ΔQ candidates`
- `Within the current development training fold only`
- `Training-fold median imputation`
- `Bootstrap stability scoring (B = 40)`
- `Descending-score traversal with region cap c ∈ {2, 3}`
- `40 selected Local DeltaQ descriptors`
- `30 fixed descriptors`
- `70-dimensional input`
- `ExtraTrees regressor`
- `Inner 3-fold configuration selection`
- `Outer 5-fold × 2 assessment`
- `Development-wide 5-fold × 2 protocol freezing`
- `Fit on all 83 development cells`
- `30 prespecified algorithmic seeds`
- `Algorithmic variation, not 30 external cohorts`
- `Secondary batch (n = 40)`

### Boundary labels

- `Complete SOH trajectory used only for offline label construction`
- `Post-150 measurements do not enter the predictor`
- `No secondary-batch contribution to preprocessing, selection, tuning, or fitting`
- `Distributed predictive information; predictive association only`
- `One cap per spatial region and morphology group`

## 9. Recommended caption

> **Methodological framework of the region-balanced Local DeltaQ–ExtraTrees (LDQ-ET) predictor.** (a) Predictor inputs are restricted to measurements from cycles 1–150, whereas the complete SOH trajectory is used only to construct the offline knee-cycle label. (b) Within each development training fold, the 365 Local DeltaQ candidates undergo training-fold median imputation, bootstrap stability scoring, and region-capped selection. The resulting 40 Local DeltaQ descriptors are concatenated with 30 fixed raw/SOH and global DeltaQ descriptors to form the 70-dimensional ExtraTrees input. Inner three-fold selection is nested within an outer five-fold cross-validation repeated twice, and a second five-fold cross-validation repeated twice across the complete development cohort freezes the final configuration. (c) The frozen pipeline is fitted to all 83 development cells under 30 prespecified algorithmic seeds and then evaluated on the 40-cell secondary batch. The secondary batch contributes to neither imputation, candidate scoring, feature selection, hyperparameter selection, nor model fitting. Selected Local DeltaQ regions represent distributed predictive information and predictive association, rather than a unique electrochemical mechanism.

## 10. LaTeX placement plan

After figure approval, insert the reference at the end of the opening paragraph of Section 3, currently immediately before `\subsection{Region-balanced Local DeltaQ selection}`.

Recommended LaTeX structure:

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_methodology_framework.pdf}
\caption{...}
\label{fig:methodology_framework}
\end{figure}
```

The introductory paragraph should then include one concise callout such as:

> Figure~\ref{fig:methodology_framework} summarizes the separation between early predictor inputs and offline label construction, the fold-safe region-balanced LDQ-ET development workflow, and the one-way transition to protocol-frozen secondary-batch evaluation.

The exact float placement should be finalized only after compiling and visually inspecting the affected manuscript pages.

## 11. Production and delivery plan

After approval of this specification:

1. Create an editable `fig_methodology_framework.drawio` source using custom geometry rather than a generic neural-network template.
2. Export a non-embedded preview PNG for visual review.
3. Check label clipping, minimum text size, panel balance, arrow direction, and grayscale interpretability.
4. Revise the same source file rather than producing multiple uncontrolled versions.
5. Export the approved figure as editable SVG and vector PDF; retain the `.drawio` source.
6. If a final embedded PNG is required, repair and validate the PNG after embedding the draw.io source.
7. Insert the vector PDF into LaTeX, compile with `latexmk`, and inspect the rendered page at manuscript scale.

Recommended deliverables:

- `figures/fig_methodology_framework.drawio`
- `figures/fig_methodology_framework.svg`
- `figures/fig_methodology_framework.pdf`
- `figures/fig_methodology_framework_preview.png`

## 12. Pre-production acceptance checklist

Before drawing begins, confirm that the specification satisfies all of the following:

- [x] Predictor input is explicitly limited to cycles 1–150.
- [x] Full-trajectory access is explicitly restricted to offline knee-label construction.
- [x] Candidate counts are 22 raw/SOH, eight global DeltaQ, and 365 Local DeltaQ.
- [x] Local DeltaQ decomposition is 351 regional plus 14 morphology descriptors.
- [x] Median imputation and Local DeltaQ scoring are training-fold only.
- [x] Bootstrap stability uses `B = 40`.
- [x] Region caps considered during development are `c ∈ {2, 3}`.
- [x] The frozen region cap is `c = 3`.
- [x] The morphology group is visibly subject to the same cap rule as each spatial region.
- [x] The final representation is 40 selected Local DeltaQ plus 30 fixed descriptors, totaling 70 dimensions.
- [x] Inner selection and outer development assessment are visually distinguishable.
- [x] Development-wide protocol freezing is visually separated from performance assessment.
- [x] Fold-specific selected sets are visually distinct from the final frozen set.
- [x] The frozen model is fitted to all 83 development cells under 30 prespecified seeds.
- [x] The seed annotation states that it quantifies algorithmic variation rather than independent external cohorts.
- [x] The 40-cell secondary batch enters only after protocol freezing.
- [x] No return arrow exists from the secondary batch to development.
- [x] The offline label reaches supervised training and evaluation only, never predictor-feature construction.
- [x] Region selection is described as predictive association and distributed information, not a unique mechanism.
- [x] No development or secondary-batch performance result is inserted into the methodology figure.
- [x] The wording does not overstate the secondary batch as independent external validation.
- [x] The figure remains readable in grayscale and all final text is at least 7.5 pt at manuscript scale.

## 13. Decision rule for one figure versus two

Retain the recommended single three-panel figure if all labels remain at least 7.5 pt and the nested-protocol frames are readable at `\textwidth`.

Split into two figures only if that condition fails:

1. **Overall LDQ-ET framework and protocol boundary:** observation window, candidate dictionary, 70-dimensional model input, freezing, and secondary-batch evaluation.
2. **Region-balanced Local DeltaQ selection:** overlapping regions, bootstrap score, deterministic descending traversal, region cap, and distributed selected features.

The split should be driven by legibility, not by a desire to add more visual detail.
