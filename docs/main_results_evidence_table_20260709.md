# 主结果证据总表与文件整理记录（2026-07-09）

## 1. 当前论文主线定稿

当前主线模型固定为：

> **LDQ-ET top40：Local DeltaQ sensitive-region representation + ExtraTrees readout**

主协议固定为：

- Development cohort：official train + primary_test，共 83 个有效 cell。
- External validation：secondary_test，共 40 个有效 cell。
- 输入窗口：前 150 cycles。
- 预测目标：knee_cycle。
- Local DeltaQ 设置：top40 敏感区间。

主模型配置：

```text
n_estimators=1200
max_features=0.5
min_samples_leaf=1
min_samples_split=2
max_depth=6
bootstrap=False
```

## 2. 正文主结果表建议

| 模型/方案 | 输入表征 | External MAE | External RMSE | External MAPE | External R2 | 建议用途 |
|---|---|---:|---:|---:|---:|---|
| Raw/DeltaQ + XGBoost | 原始早期统计 + DeltaQ | 108.27 | 143.23 | 13.84% | 0.657 | 基础 baseline |
| Local DeltaQ + XGBoost | Local DeltaQ top40 | 89.03 | 121.07 | 10.96% | 0.755 | 表征有效性对照 |
| **LDQ-ET top40** | **Local DeltaQ top40** | **77.23 ± 0.89** | **111.07 ± 0.93** | **9.04 ± 0.12%** | **0.794 ± 0.003** | **正文主模型** |

主线改善幅度：

- 相比 Raw/DeltaQ + XGBoost：MAE 降低 31.04 cycles，R2 提升 0.137。
- 相比 Local DeltaQ + XGBoost：MAE 降低 11.80 cycles，R2 提升 0.039。

## 3. 横向模型对比表建议

来自 `results/horizontal_model_comparison_N150_fast/`。该表用于证明：在同一数据协议下，Local DeltaQ + 树集成读出器是当前最稳的组合。

| 排名 | 模型 | 输入 | External MAE | External RMSE | External MAPE | External R2 | 建议用途 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | ExtraTrees | Local DeltaQ | 80.90 | 117.35 | 9.43% | 0.770 | 横向对比正文 |
| 2 | XGBoost | Local DeltaQ | 90.90 | 121.24 | 11.34% | 0.754 | 横向对比正文 |
| 3 | RandomForest | Local DeltaQ | 95.60 | 127.78 | 11.68% | 0.727 | 横向对比正文 |
| 4 | XGBoost | Raw/DeltaQ | 103.55 | 132.18 | 13.18% | 0.708 | 横向对比正文 |
| 5 | Adapted CNN-XGBoost | CNN embedding + Raw/DeltaQ | 106.83 | 138.56 | 13.51% | 0.679 | 前沿 adapted baseline |
| 15 | Adapted TCN-ECA | Local DeltaQ sequence | 189.12 | 209.69 | 24.78% | 0.265 | 补充/负结果 |

注意：横向对比的 ExtraTrees 是 fast setting 下的固定模型结果，最终主模型应引用 HPO 后的 LDQ-ET top40 结果，即 MAE 77.23、R2 0.794。

## 4. 消融实验表建议

### 4.1 Local DeltaQ 模块消融

| 对照 | External MAE | External RMSE | External MAPE | External R2 | 结论 |
|---|---:|---:|---:|---:|---|
| Raw/DeltaQ + XGBoost | 108.27 | 143.23 | 13.84% | 0.657 | 无 Local DeltaQ |
| Local DeltaQ + XGBoost | 89.03 | 121.07 | 10.96% | 0.755 | Local DeltaQ 显著改善 |
| LDQ-ET top40 | 77.23 ± 0.89 | 111.07 ± 0.93 | 9.04 ± 0.12% | 0.794 ± 0.003 | 表征 + 稳健读出器最佳 |

### 4.2 top-k 敏感区间数量消融

| Local DeltaQ top-k | 特征总数 | External MAE | External RMSE | External MAPE | External R2 | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 20 | 50 | 80.95 ± 0.91 | 117.94 ± 0.84 | 9.48 ± 0.12% | 0.768 ± 0.003 | 局部覆盖不足 |
| **40** | **70** | **77.23 ± 0.89** | **111.07 ± 0.93** | **9.04 ± 0.12%** | **0.794 ± 0.003** | **主线默认** |
| 60 | 90 | 76.35 ± 0.72 | 111.31 ± 0.62 | 9.03 ± 0.10% | 0.793 ± 0.002 | MAE 略优，补充报告 |
| 80 | 110 | 77.67 ± 0.75 | 113.63 ± 0.81 | 9.18 ± 0.11% | 0.784 ± 0.003 | 冗余/噪声增加 |

结论口径：

- top40/top60 构成稳定性能平台。
- top40 的 RMSE/R2 略优、特征更少、且与 HPO 主配置一致，因此作为主线。
- top60 可在补充材料中报告为最低 MAE 设置，但不作为默认模型。

## 5. 鲁棒性与补充结果

| 实验 | 结果摘要 | 建议用途 |
|---|---|---|
| LDQ-ET top40 30-seed robustness | External MAE 77.23 ± 0.89，R2 0.794 ± 0.003 | 正文或补充，证明随机种子稳定 |
| Same-distribution random split | unstratified mean R2 0.872，best R2 0.967；batch-stratified mean R2 0.870，best R2 0.955 | 补充材料，用于对比随机划分文献 |
| Hybrid residual / OOD gate | train-only 选择退化为 pure XGBoost，未形成稳定收益 | 探索记录，不作为主贡献 |
| Adapted deep baselines | CNN-XGBoost 未超过 LDQ-ET，TCN-ECA 初版较差 | 补充材料，支持小样本下物理启发表征更稳 |

## 6. 正文/补充/归档建议

### 正文应放

- LDQ-ET top40 主结果。
- Raw/DeltaQ + XGBoost 与 Local DeltaQ + XGBoost 对照。
- 横向模型对比前几名。
- top-k 消融。
- 30-seed 稳定性。

### 补充材料应放

- 随机合并划分结果。
- 完整横向模型对比表。
- Adapted CNN-XGBoost 和 TCN-ECA 实现细节。
- XGBoost readout 与 ExtraTrees readout 对照。

### 仅作为探索记录

- Hybrid residual / OOD-gated residual。
- Diagnostic candidate `xgb_hpo_0212`。
- 早期 N50/N100 窗口尝试。
- 早期 Severson linear table reproduction。

## 7. 本轮文件清理记录

已删除的高确定性无用内容：

- 空结果目录：
  - `results/extratrees_ldq_hpo_N150/`
  - `results/extratrees_ldq_hpo_N150_fast/`
  - `results/fixed_hybrid_weight_top40_N150/`
  - `results/hybrid_hpo_local_deltaq_N150/`
  - `results/local_deltaq_xgb_finalization_N150/`
  - `results/random_split_local_deltaq_N150/`
- Python 缓存目录：
  - `scripts/__pycache__/`
- 异常生成目录：
  - `%SystemDrive%/`

暂不删除但建议后续人工确认的大文件：

- `data/*.mat` 原始文件，总体占用约 9.6GB。
- 当前主线实际依赖的是已处理的 2017-06-30 pkl/csv 文件；若磁盘压力较大，可以把 `.mat` 原始文件移动到 D 盘归档目录或外置盘，而不是直接删除。

## 8. 下一步建议

下一步可以开始生成论文写作骨架：

1. 方法章节：Local DeltaQ sensitive-region representation + ExtraTrees readout。
2. 实验章节：数据协议、baselines、evaluation metrics。
3. 结果章节：主结果、横向对比、消融、鲁棒性。
4. 讨论章节：为什么随机划分 R2 更高、为什么 external secondary 更可信、为什么不用复杂深度模型作为主线。

