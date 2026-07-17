# LDQ-ET top-k Local DeltaQ 消融审查（2026-07-09）

## 1. 本轮补跑目的

本轮实验用于回答：在已经确定 `Local DeltaQ sensitive-region representation + ExtraTrees readout` 作为当前主线模型后，Local DeltaQ 敏感区间数量是否已经做过消融，以及主线采用 `top40` 是否有足够依据。

实验只改变 Local DeltaQ 的敏感区间数量，其他条件保持一致：

- 数据协议：official train + primary_test 作为 development cohort，secondary_test 作为 external validation。
- 输入窗口：前 150 cycles。
- 目标：knee_cycle。
- 模型：ExtraTreesRegressor。
- 超参数：沿用 `results/extratrees_ldq_hpo_N150_quick30/metadata.json` 中选定配置。
- 随机种子复核：30 seeds。
- 输出目录：`results/ldq_et_topk_ablation_N150/`。

选定 ExtraTrees 配置：

```text
n_estimators=1200
max_features=0.5
min_samples_leaf=1
min_samples_split=2
max_depth=6
bootstrap=False
```

## 2. 消融设置

| 设置 | 特征总数 | 说明 |
|---|---:|---|
| top20 | 50 | raw/DeltaQ 基础特征 + 20 个 Local DeltaQ 敏感区间特征 |
| top40 | 70 | raw/DeltaQ 基础特征 + 40 个 Local DeltaQ 敏感区间特征 |
| top60 | 90 | raw/DeltaQ 基础特征 + 60 个 Local DeltaQ 敏感区间特征 |
| top80 | 110 | raw/DeltaQ 基础特征 + 80 个 Local DeltaQ 敏感区间特征 |

## 3. External secondary 结果

| Local DeltaQ top-k | MAE (cycles) | RMSE (cycles) | MAPE (%) | R2 |
|---:|---:|---:|---:|---:|
| 20 | 80.95 ± 0.91 | 117.94 ± 0.84 | 9.48 ± 0.12 | 0.768 ± 0.003 |
| 40 | 77.23 ± 0.89 | 111.07 ± 0.93 | 9.04 ± 0.12 | 0.794 ± 0.003 |
| 60 | 76.35 ± 0.72 | 111.31 ± 0.62 | 9.03 ± 0.10 | 0.793 ± 0.002 |
| 80 | 77.67 ± 0.75 | 113.63 ± 0.81 | 9.18 ± 0.11 | 0.784 ± 0.003 |

## 4. 结果解释

1. `top20 -> top40` 带来明确提升，说明 Local DeltaQ 敏感区间数量过少时，局部退化信息覆盖不足。

2. `top40` 与 `top60` 形成性能平台：
   - top60 的 MAE 最低：76.35 cycles；
   - top40 的 RMSE 和 R2 略优：RMSE 111.07 cycles，R2 0.794；
   - 两者差异非常小，均明显优于 top20 和 top80。

3. `top80` 性能回落，说明继续增加低排名局部区间可能引入冗余或 batch-specific 噪声。

4. Development cohort 上的训练拟合随 top-k 增加持续变好，但 external secondary 并不同步单调提升。这一点支持论文中的一个重要说法：Local DeltaQ 不是越多越好，而需要控制敏感区间规模以获得跨 batch 泛化。

## 5. 审查结论

本轮消融通过。它能支撑以下论文表述：

- Local DeltaQ 敏感区间表示是有效模块，因为 top40/top60 显著优于 top20，并明显强于 raw/DeltaQ baseline。
- 敏感区间规模存在合理平台，过宽的 top80 不再提升，提示该方法需要受控的局部区域选择。
- 主线默认采用 top40 是可解释且稳妥的，因为它兼顾了此前 HPO 选择、较低 RMSE、最高 R2 和较少特征数量。
- top60 可作为补充结果报告为最低 MAE 设置，但不建议把主线切换到 top60，除非论文特别强调 MAE 单指标最优。

## 6. 对当前故事线的影响

当前主线仍建议写作：

> We propose a Local DeltaQ sensitive-region representation and use ExtraTrees as a stable low-data readout. The ablation over selected local regions shows a clear improvement from sparse local coverage to a moderate region set, while excessive regions do not further improve external-batch generalization.

中文论文说法可以是：

> 本文提出局部 DeltaQ 敏感区间表征，并采用 ExtraTrees 作为小样本条件下的稳健读出器。敏感区间数量消融表明，适度数量的局部区间能够显著提升外部 batch 泛化性能，但过多区间会引入冗余或批次相关噪声，因此方法的关键并非简单增加特征，而是构造受控的退化敏感局部表示。

## 7. 后续建议

下一步可以把 LDQ-ET 论文证据整理成三组主表：

1. 横向模型对比：raw/DeltaQ 与 Local DeltaQ 下的传统模型、树模型、前沿 adapted baseline。
2. 模块消融：raw/DeltaQ、Local DeltaQ top-k、ExtraTrees readout、XGBoost readout。
3. 鲁棒性分析：30-seed 稳定性、same-distribution random split、external secondary batch 验证。

