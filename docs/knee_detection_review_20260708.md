# Knee Point 分段线性标定阶段审查记录（2026-07-08）

## 本阶段目标

本阶段只完成一个小步：基于上一阶段得到的 `soh_mono` 曲线，为每个 cell 自动标定 knee point。

本阶段不做早期特征提取，也不训练预测模型。

## 执行过程

新增脚本：

- `scripts/detect_knee_points.py`

输入：

- `data/processed_cycles_soh_corrected_20170630.csv`
- `results/soh_raw_curves/selected_20_cells.csv`

方法：

- 主方法：分段线性拟合。
- 遍历候选分割点 `t`，最小化：

```text
SSE(1:t) + SSE(t:end)
```

- 候选范围：
  - `min_frac=0.2`
  - `max_frac=0.9`
  - `min_segment_points=30`

输出：

- `data/knee_labels_20170630.csv`
- `results/knee_detection/selected_20_knee_detection_grid.png`
- `results/knee_detection/selected_20_individual/`
- `results/knee_detection/knee_label_distribution_qc.png`

## 过程审查

1. 标定只使用 `soh_mono`，没有使用早期预测模型或测试集结果。
2. 分段线性拟合使用完整退化曲线，这是为了生成监督标签；后续早期预测模型不能使用完整曲线作为输入。
3. 原始 `soh_raw` 和 `soh_smooth` 只用于图中对照，不参与 knee point 目标函数。
4. 候选分割点限制在 `[20%, 90%]` 曲线长度范围内，避免起始噪声和末端失效区直接支配标签。
5. 所有 cell 都保留标签质量指标 `label_confidence`，后续可以做高置信样本消融。

## 成果审查

输出规模：

- knee 标签：124 rows, 17 columns。
- 20 个代表 cell 的 knee 图已生成。
- 20 张单 cell knee 图已生成。
- 标签分布 QC 图已生成。

全量统计：

| 指标 | 数值 |
|---|---:|
| 平均 knee cycle | 624.06 |
| knee cycle 中位数 | 578.50 |
| 平均 relative knee | 0.779 |
| relative knee 中位数 | 0.781 |
| 平均 knee SOH | 0.944 |
| 平均 label confidence | 0.955 |

质量检查：

- `status=ok`: 124/124。
- 所有 cell 的第二段斜率都比第一段更负，符合“进入加速退化阶段”的定义。
- 只有一个 cell 的 `knee_cycle <= 100`：
  - `b2c1`, `cycle_life=148`, `knee_cycle=94`。
- 因此后续 `N=100` 的回归任务应排除 `b2c1`，或将其作为极早期风险分类样本。

## 需要关注的样本

低置信度或异常样本：

- `b1c18`: label confidence 约 0.797，前面 SOH 修正阶段也显示其修正幅度最大。
- `b2c3`: label confidence 约 0.813。
- `b2c0`: label confidence 约 0.827。
- `b2c1`: knee cycle 约 94，小于 100-cycle 早期窗口。

这些样本后续建议在图上人工复核，并在建模阶段做：

- 全样本版本。
- 排除 `knee_cycle <= N` 的回归版本。
- 高置信样本版本。

## 阶段结论

本阶段通过，但带有审慎条件。

通过理由：

- 标签生成完整。
- 分段后第二段均更陡，符合加速退化定义。
- knee point 相对位置集中在寿命的约 0.78 附近，整体分布合理。
- 标签置信度整体较高。

审慎条件：

- 分段线性 knee point 是一种算法标签，不是人工真值。
- `label_confidence` 偏高可能受到单调修正后曲线形状影响，后续必须做标定方法消融。
- `b1c18`、`b2c1` 等样本需要单独处理或报告。

下一小步建议：

> 对 20 个代表 cell 的 knee 图进行人工筛查后，再开始提取前 50/100/150 cycles 的早期特征。优先先做 `N=100` 的最小容量特征组，不急着加入所有温度/电压特征。
