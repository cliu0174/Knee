# N=100 早期特征提取阶段审查记录（2026-07-08）

## 本阶段目标

本阶段只完成一个小步：基于前 100 cycles 提取最小容量/SOH 特征组，用于后续 knee point 预测实验。

本阶段不训练模型。

## 执行过程

新增脚本：

- `scripts/extract_early_knee_features.py`

输入：

- `data/processed_cycles_soh_corrected_20170630.csv`
- `data/knee_labels_20170630.csv`

输出：

- `data/early_features_N100_20170630.csv`
- `data/early_features_N100_qc_20170630.csv`
- `results/early_features_N100/early_features_N100_qc.png`

特征窗口：

- `N=100`

特征组：

- 容量统计特征：mean, std, min, max, first, last, range, change, ratio。
- raw SOH 特征：mean, std, min, max, first, last, change, slope。
- smooth SOH 特征：last, change, slope。
- monotonic SOH 特征：last, change, slope。
- 容量回升/波动特征：regeneration count, amplitude, max increase。
- 修正距离特征：raw-mono mean absolute distance, smooth-mono mean absolute distance。

## 过程审查

1. 特征只使用 `cycle <= 100` 且 `is_positive_capacity=True` 的循环点。
2. `knee_cycle`、`relative_knee`、`cycle_life` 等目标/元信息被保留在同一表中，但不属于后续建模特征列。
3. 没有使用 100 cycles 之后的 SOH、容量、温度或模型结果作为输入特征。
4. `knee_after_window` 被显式标记，后续回归任务可排除 `knee_cycle <= 100` 的样本。

## 成果审查

输出规模：

- 124 rows, 47 columns。
- 特征 QC 表：36 个特征列。
- QC 图已生成。

数据完整性：

- 特征缺失列：无。
- 常数列：`N100_window_complete`，所有样本均为 True，后续建模时应丢弃。
- `knee_after_window=True`: 123 个样本。
- `knee_after_window=False`: 1 个样本，即 `b2c1`。

窗口完整性：

- `N100_n_observed` 均值约 99.67。
- 最小为 99，最大为 100。
- 由于部分 cell 有起始非正容量占位点，99/100 是合理现象。

初步相关性检查：

与 `knee_cycle` 绝对相关性较高的特征包括：

- `N100_soh_raw_first`: 约 0.5825。
- `N100_soh_raw_min`: 约 0.3242。
- `N100_raw_regen_count`: 约 -0.3187。
- `N100_soh_smooth_change`: 约 0.2618。
- `N100_soh_mono_last`: 约 0.2510。

这些相关性只用于 QC，不代表模型结果。

## 风险与注意事项

1. `N100_soh_raw_first` 相关性较高，可能反映初始容量差异或归一化基准差异。它不是标签泄漏，但后续应做“包含/排除初始容量水平特征”的消融。
2. `b2c1` 的 knee_cycle=94，小于 N=100。后续 N=100 回归任务应排除它，分类任务可保留为极早期风险样本。
3. 当前特征仍是最小容量/SOH 特征组，尚未加入温度、充电时间、电压曲线特征。

## 阶段结论

本阶段通过。

理由：

- 特征表完整，无缺失。
- 特征窗口严格限制在前 100 cycles。
- 已显式标记 knee 是否晚于预测窗口。
- 产物可直接进入下一小步的基线模型训练。

下一小步建议：

> 先训练最小容量特征组的基线模型：Ridge、SVR、RandomForest、XGBoost。只做 N=100，且回归任务排除 `b2c1`。训练后必须审查是否存在明显过拟合、split 泄漏、以及全样本/高置信样本结果差异。
