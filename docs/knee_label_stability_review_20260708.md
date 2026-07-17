# Knee Label 稳定性与候选模型标签鲁棒性审查

日期：2026-07-08

## 本阶段目标

当前所有 knee cycle 预测结果都依赖 knee point 标签。如果标签本身不稳定，模型结果就难以支撑论文。因此本阶段验证：

- 当前 piecewise linear 标签是否稳定？
- 与 Kneedle、退化速率阈值方法相比，标签差异多大？
- 两个候选主模型在不同标签定义下是否仍然稳定？

## 实验设置

新增脚本：

- `scripts/analyze_knee_label_stability.py`

输入：

- `data/processed_cycles_soh_corrected_20170630.csv`
- `data/early_features_N100_extended_20170630.csv`
- `data/early_features_N150_extended_20170630.csv`

输出：

- `results/knee_label_stability/knee_labels_by_method.csv`
- `results/knee_label_stability/label_method_comparison.csv`
- `results/knee_label_stability/candidate_model_label_robustness.csv`
- `results/knee_label_stability/label_method_scatter.png`
- `results/knee_label_stability/candidate_label_robustness_mae.png`
- `results/knee_label_stability/run_metadata.json`

## 对比标签方法

1. `piecewise`：当前主方法，分段线性拟合。
2. `kneedle`：基于归一化曲线到首尾连线的最大距离。
3. `rate_threshold`：基于局部退化速率超过早期基线阈值。

三种方法均使用同样的候选范围：

- min fraction：0.2
- max fraction：0.9
- minimum segment points：30

## 标签方法一致性

以 `piecewise` 作为参考：

### Kneedle

- n=124。
- mean absolute difference：32.85 cycles。
- median absolute difference：31.00 cycles。
- mean signed difference：-32.85 cycles。
- Pearson：0.9985。
- Spearman：0.9978。
- within ±50 cycles：87.90%。
- within ±100 cycles：99.19%。

判断：

- Kneedle 与 piecewise 高度一致。
- Kneedle 系统性略早，但整体差异很小。
- 这说明当前主标签不是由单一算法偶然产生，而是与另一种常用拐点思路高度一致。

### Rate Threshold

- n=124。
- mean absolute difference：69.37 cycles。
- median absolute difference：34.50 cycles。
- mean signed difference：-68.82 cycles。
- Pearson：0.9249。
- Spearman：0.9227。
- within ±50 cycles：59.68%。
- within ±100 cycles：81.45%。

判断：

- 速率阈值法整体偏早。
- 部分长寿命 cell 偏差很大，例如 `b1c0` 相比 piecewise 提前 857 cycles。
- 该方法更像粗糙早期加速检测，不适合作为主标签，但可作为保守对照。

## 早期窗口内标签检查

三种方法均成功输出 124 个标签。

对 N=100/N=150 窗口：

- piecewise：1 个 cell 在窗口内，即此前已识别的极早 knee 样本。
- Kneedle：1 个 cell 在窗口内。
- rate_threshold：2 个 cell 在窗口内，分别是 `b2c0` 和 `b2c1`。

这说明 rate_threshold 更容易把 knee 标到过早位置。

## 候选模型标签鲁棒性

候选模型 A：性能候选

- N=150
- raw_deltaq
- XGBoost

候选模型 B：早期预警候选

- N=100
- raw_deltaq
- RandomForest

### Piecewise 标签

N=150 raw_deltaq XGBoost：

- primary MAE：65.67。
- secondary MAE：103.29。

N=100 raw_deltaq RandomForest：

- primary MAE：65.93。
- secondary MAE：113.77。

### Kneedle 标签

N=150 raw_deltaq XGBoost：

- primary MAE：70.47。
- secondary MAE：103.00。

N=100 raw_deltaq RandomForest：

- primary MAE：66.55。
- secondary MAE：109.11。

判断：

- 两个候选模型在 piecewise 和 Kneedle 下都稳定。
- N=150 XGBoost 的 secondary 几乎不变。
- 这支持主标签方法具有鲁棒性。

### Rate Threshold 标签

N=150 raw_deltaq XGBoost：

- primary MAE：66.83。
- secondary MAE：146.92。

N=100 raw_deltaq RandomForest：

- primary MAE：78.43。
- secondary MAE：159.55。

判断：

- 速率阈值标签下两个模型均变差。
- N=150 XGBoost 仍明显优于 N=100 RandomForest。
- 这进一步支持 N=150 raw_deltaq XGBoost 作为主模型候选。

## 阶段结论

本阶段通过。

关键结论：

- Piecewise 与 Kneedle 标签高度一致，说明主标签可靠性较强。
- Rate threshold 作为对照方法偏早、偏粗糙，不适合作为主标签。
- 两个候选模型在 piecewise/Kneedle 标签下均稳定。
- N=150 raw_deltaq XGBoost 在不同标签场景下整体更稳，已经更适合作为论文唯一主力模型。
- N=100 raw_deltaq RandomForest 可降级为早期窗口/可解释性对照，而不是并列主模型。

## 对论文主线的影响

建议论文主线更新为：

> 主方法：基于 piecewise/Kneedle 一致性验证的 knee label，使用 N=150 raw_deltaq XGBoost 进行早期 knee cycle 预测。

同时保留：

> N=100 raw_deltaq RandomForest 作为更早预警窗口和可解释模型对照。

这样论文不会在两个主模型之间摇摆，同时仍然能讨论提前量、可解释性和工程部署权衡。

## 下一步建议

下一步可以进入最终实验整理阶段：

1. 固定主模型为 N=150 raw_deltaq XGBoost。
2. 固定主要对照为 N=100 raw_deltaq RandomForest。
3. 汇总最终论文表格：
   - baseline 对比表。
   - 特征组合搜索表。
   - 标签稳定性表。
   - 主模型预测性能表。
4. 开始生成论文图表与方法流程图。
