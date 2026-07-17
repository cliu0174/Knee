# SOH 平滑与单调修正阶段审查记录（2026-07-08）

## 本阶段目标

本阶段只完成一个小步：在已导出的 `raw SOH` 曲线基础上，生成：

- `soh_smooth`: 平滑后的 SOH 曲线。
- `soh_mono`: 单调退化约束后的 SOH 曲线。

本阶段不做 knee point 标定，也不做模型训练。

## 执行过程

新增脚本：

- `scripts/correct_soh_curves.py`

输入：

- `data/processed_cycles_20170630.csv`
- `results/soh_raw_curves/selected_20_cells.csv`

方法：

- 平滑方法：Savitzky-Golay filter
  - `window=21`
  - `polyorder=2`
- 单调修正方法：Isotonic Regression
  - 约束方向：单调不增

输出：

- `data/processed_cycles_soh_corrected_20170630.csv`
- `data/soh_correction_qc_20170630.csv`
- `results/soh_correction_curves/selected_20_soh_correction_grid.png`
- `results/soh_correction_curves/selected_20_individual/`

## 过程审查

1. 输入数据没有重新读取原始 `.mat` 或 `.pkl`，只基于上一阶段已经确认的循环级 CSV。
2. 修正过程没有使用 `cycle_life`、训练/测试标签或未来建模结果，因此不存在模型训练层面的信息泄漏。
3. 原始 `soh_raw` 没有被覆盖，而是新增 `soh_smooth` 和 `soh_mono` 两列，后续可以做 raw / smooth / mono 消融实验。
4. 非正容量占位点已经在上一阶段被设为 `soh_raw = NaN`，本阶段不会参与平滑和单调修正。
5. 平滑与单调修正作用于每个 cell 自身的完整退化曲线；这是为了生成 knee point 标签。后续早期预测模型只能使用前 N 个 cycle 的特征，不能使用完整曲线特征作为输入。

## 成果审查

输出规模：

- 修正后循环级记录：100501 rows, 19 columns。
- 修正 QC 表：124 rows, 26 columns。
- 20 个代表 cell 的总览图已生成。
- 20 张单 cell 对比图已生成。

关键指标：

| 指标 | raw | smooth | mono |
|---|---:|---:|---:|
| 平均单调违背次数 | 80.24 | 17.78 | 0.00 |
| 平均容量回升幅度 | 0.0843 | 0.0131 | 0.0000 |
| 最大单调违背次数 | 717 | 342 | 0 |

修正幅度：

- `mean_abs_smooth_minus_raw` 均值约 0.000275。
- `mean_abs_mono_minus_raw` 均值约 0.000318。
- `trend_fidelity_smooth_vs_raw` 中位数约 0.999985。
- `trend_fidelity_mono_vs_smooth` 中位数约 0.999987。

这说明修正主要是在消除局部上升和微小噪声，没有大幅改变原始退化趋势。

## 需要关注的样本

修正幅度最大的 cell：

- `b1c18`
- `b1c0`
- `b2c44`
- `b2c12`

其中 `b1c18` 的 `mean_abs_mono_minus_raw` 约 0.005437，明显高于多数 cell。这个样本在前面的 Severson full-model 复现中也表现为异常样本，因此后续 knee point 标定阶段应单独检查。

## 阶段结论

本阶段通过。

理由：

- 输出文件完整。
- 所有 cell 的 `soh_mono` 均满足单调不增约束。
- 修正幅度总体很小，趋势保真度高。
- raw / smooth / mono 三条曲线均被保留，支持后续消融实验。

下一小步建议：

> 在 `soh_mono` 曲线上实现分段线性 knee point 标定，并先对 20 个代表性 cell 画出 knee point 示意图，人工审查标签是否合理。
