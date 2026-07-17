# N=50/100/150 早期窗口敏感性审查记录（2026-07-08）

## 本阶段目标

本阶段只完成一个小步：在不调参的前提下，比较早期窗口长度 `N=50/100/150` 对 knee_cycle 预测的影响。

## 执行过程

复用脚本：

- `scripts/extract_early_knee_features.py`
- `scripts/train_knee_baselines.py`

新增输出：

- `data/early_features_N50_20170630.csv`
- `data/early_features_N150_20170630.csv`
- `results/knee_baselines_N50_raw_only/`
- `results/knee_baselines_N150_raw_only/`
- `results/window_sensitivity_N50_N100_N150/metrics_all_windows.csv`
- `results/window_sensitivity_N50_N100_N150/window_sensitivity_mae.png`

统一设置：

- 目标：`knee_cycle`
- 特征：`raw_only`
- 模型：RidgeCV, SVR-RBF, RandomForest, XGBoost
- 不做新调参
- 仍使用 official split

## 过程审查

1. 三个窗口均使用 raw-only 特征，没有使用 `soh_mono` 或完整曲线修正特征作为输入。
2. `N=50` 中所有样本的 knee 都晚于窗口，因此使用 124 个样本。
3. `N=100` 和 `N=150` 中 `b2c1` 的 knee 已经早于窗口，因此回归任务使用 123 个样本。
4. 三个窗口均使用 22 个 raw-only 特征，保证横向比较公平。
5. 本阶段只比较窗口长度，不调整模型超参数。

## 核心结果

### Primary Test

| Window | Best model | MAE cycles | RMSE cycles | MAPE | R2 |
|---:|---|---:|---:|---:|---:|
| N=50 | RandomForest | 125.43 | 199.10 | 26.06% | 0.494 |
| N=100 | RandomForest | 123.46 | 196.45 | 19.93% | 0.488 |
| N=150 | XGBoost | 102.43 | 165.68 | 16.33% | 0.636 |

### Secondary Test

| Window | Best model | MAE cycles | RMSE cycles | MAPE | R2 |
|---:|---|---:|---:|---:|---:|
| N=50 | XGBoost | 283.75 | 370.75 | 29.98% | -1.297 |
| N=100 | RandomForest | 268.21 | 363.55 | 27.91% | -1.208 |
| N=150 | RandomForest | 247.71 | 338.77 | 25.79% | -0.918 |

## 结论解释

1. 窗口越长，整体预测效果越好，尤其 primary test。
2. `N=150` 是当前 raw-only 设置下的最佳窗口。
3. secondary test 仍然 R2 为负，说明跨 batch 泛化仍不足。
4. `N=150` 虽然更准，但预警提前量比 `N=50/100` 更短，因此不能简单说 N 越大越好。
5. 当前结果支持论文中做“窗口长度敏感性分析”，但还不足以作为最终方法性能。

## 风险与注意事项

- `N=150` 使用更接近 knee 的信息，性能提升符合预期，但工程预警价值要和提前量一起讨论。
- XGBoost 在 primary 上最好，但训练误差低，仍需警惕过拟合。
- RandomForest 在 secondary 上相对稳，是当前更稳健的主候选。
- secondary 长寿命 batch3 cell 仍被低估，后续需要加入更丰富特征或做 batch 泛化讨论。

## 阶段结论

本阶段通过。

推荐当前阶段性表述：

> Early-window sensitivity shows that extending the observation window from 50 to 150 cycles improves knee-cycle prediction, with the best primary-test MAE decreasing to about 102 cycles and the best secondary-test MAE decreasing to about 248 cycles. However, secondary-test R2 remains negative, indicating that raw capacity/SOH features alone are insufficient for robust cross-batch generalization.

下一小步建议：

> 不继续调参。下一步应扩展特征类型，在 N=100 或 N=150 上加入充电时间、温度和 Severson-style `Delta Q(V)` 特征，再观察 secondary 泛化是否改善。
