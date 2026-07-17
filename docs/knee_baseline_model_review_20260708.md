# N=100 Knee Point 基线模型阶段审查记录（2026-07-08）

## 本阶段目标

本阶段只完成一个小步：使用 `N=100` 最小早期容量/SOH 特征组训练基线模型，验证 knee point 预测任务是否初步可行。

本阶段不做超参数大规模搜索，不作为最终模型结论。

## 执行过程

新增脚本：

- `scripts/train_knee_baselines.py`

输入：

- `data/early_features_N100_20170630.csv`

有效建模设置：

- 目标：`knee_cycle`
- 特征集：`raw_only`
- 排除样本：`knee_after_window=False`，即 `b2c1`
- 使用样本数：123/124
- 特征数：22
- split：沿用 Severson official split
  - train: 41
  - primary_test: 42
  - secondary_test: 40

模型：

- RidgeCV
- SVR-RBF
- RandomForest
- XGBoost

输出：

- `results/knee_baselines_N100_raw_only/metrics.csv`
- `results/knee_baselines_N100_raw_only/predictions.csv`
- `results/knee_baselines_N100_raw_only/feature_importance.csv`
- `results/knee_baselines_N100_raw_only/knee_baseline_predictions.png`

## 过程审查

1. 已排除 `b2c1`，因为其 `knee_cycle=94`，早于 100-cycle 预测窗口。
2. 有效版本使用 `raw_only` 特征，不使用 `soh_smooth`、`soh_mono`、raw-mono correction distance 等完整曲线修正特征。
3. 发现并标记了一个重要风险：`soh_mono` 是基于完整寿命曲线生成的，适合用于 knee label 标定，但不适合作为早期预测输入。后续正式建模应优先使用 `raw_only` 或“仅用前 N cycle 重新计算的平滑特征”。
4. 模型只在 train split 拟合，primary/secondary 只做评估。
5. `cycle_life`、`knee_cycle`、`relative_knee`、`label_confidence` 等元信息没有进入 `raw_only` feature columns。

## 成果审查：有效 raw-only knee_cycle 版本

| Model | Split | MAE cycles | RMSE cycles | MAPE | R2 |
|---|---:|---:|---:|---:|---:|
| RidgeCV | train | 97.62 | 171.19 | 16.84% | 0.475 |
| RidgeCV | primary | 2045.59 | 10437.52 | 317.93% | -1444.26 |
| RidgeCV | secondary | 359.35 | 432.23 | 39.41% | -2.12 |
| SVR-RBF | train | 109.38 | 227.32 | 16.26% | 0.074 |
| SVR-RBF | primary | 137.81 | 276.14 | 17.64% | -0.012 |
| SVR-RBF | secondary | 384.05 | 451.68 | 42.60% | -2.41 |
| RandomForest | train | 51.46 | 113.41 | 8.82% | 0.769 |
| RandomForest | primary | 123.46 | 196.45 | 19.93% | 0.488 |
| RandomForest | secondary | 268.21 | 363.55 | 27.91% | -1.21 |
| XGBoost | train | 21.49 | 32.64 | 4.45% | 0.981 |
| XGBoost | primary | 146.92 | 222.59 | 25.06% | 0.343 |
| XGBoost | secondary | 269.46 | 368.94 | 27.97% | -1.27 |

## 与 naive baseline 对比

训练集均值 naive baseline：

- primary MAE: 177.94 cycles
- secondary MAE: 319.53 cycles

训练集中位数 naive baseline：

- primary MAE: 168.60 cycles
- secondary MAE: 400.43 cycles

有效模型对比：

- RandomForest primary MAE 123.46，优于 naive。
- RandomForest secondary MAE 268.21，优于 train-mean naive，但 R2 仍为负，说明 secondary 分布泛化仍不足。
- XGBoost primary/secondary MAE 与 RandomForest 接近，但 train error 极低，过拟合风险更高。

## 诊断性 relative_knee 目标

另外运行了 `relative_knee` 目标的 raw-only 诊断版本：

- `results/knee_baselines_N100_relative_raw_only/`

该版本在 target space 中表现较稳定，例如 XGBoost primary relative MAPE 约 2.26%，secondary relative MAPE 约 5.70%。

但是注意：

> 将 relative_knee 预测值换算回 knee_cycle 需要真实 `cycle_life`，这在早期预测场景中不可用。因此 relative_knee 的 cycle-space 指标不能作为部署性能，只能说明“相对拐点位置”更容易被早期特征学习。

## 主要问题

1. Absolute `knee_cycle` 预测在 secondary test 上明显偏差较大，尤其对长寿命 batch3 cell 普遍低估。
2. RidgeCV 对 `b1c18`、`b1c0` 等异常样本外推失控，不适合作为当前主模型。
3. XGBoost 训练集误差极低，存在明显过拟合风险。
4. 仅使用 N=100 raw capacity/SOH 特征，对跨 batch 泛化仍不够。

## 阶段结论

本阶段通过，但结论是“初步可行，不足以作为最终模型”。

推荐当前有效 baseline：

- RandomForest raw-only knee_cycle
  - primary MAE 123.46 cycles
  - secondary MAE 268.21 cycles

推荐写法：

> 使用前 100 cycles 的最小容量/SOH 特征可以在 primary test 上初步预测 knee_cycle，并优于简单训练集均值基线；但在 secondary test 上泛化不足，说明仅靠容量统计特征还不能稳定跨批次预测，需要进一步加入更丰富的早期电压、充电时间、温度或 Severson-style ΔQ(V) 特征，并进行高置信样本与不同窗口长度消融。

下一小步建议：

> 不急着调参。先扩展 `N=50/150` 窗口并复用同一 raw-only 特征与模型流程，判断窗口长度是否显著影响 primary/secondary 泛化。
