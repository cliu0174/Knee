# Severson 2019 Supplementary Table 1 线性复现确认版（2026-07-08）

## 本轮目标

在已恢复 `2017-06-30` batch2、并得到原文规模 `train=41 / primary=43 / secondary=40` 的基础上，本轮进一步靠近 Severson et al. 2019 原文主线：

1. 按 Supplementary Table 1 重建三类 regression feature set。
2. 使用 `log10(cycle life)` 作为预测目标。
3. 使用训练集拟合 scaler 与 regularized linear / ElasticNet 模型。
4. 输出 train、primary、primary 去异常、secondary 指标，并保留预测图和系数表。

参考来源：
- 论文主页：https://www.nature.com/articles/s41560-019-0356-8
- Supplementary Information：https://static-content.springer.com/esm/art%3A10.1038%2Fs41560-019-0356-8/MediaObjects/41560_2019_356_MOESM1_ESM.pdf
- 官方处理代码：https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
- 数据来源：https://data.matr.io/1

## 新增脚本与输出

新增脚本：
- `scripts/extract_severson_supplementary_features.py`
- `scripts/reproduce_severson_linear_table1_models.py`

新增特征表：
- `data/severson_features_20170630/severson_supplementary_table1_features_20170630.csv`

新增结果目录：
- `results/severson_20170630_table1_linear_models_sparse/metrics.csv`
- `results/severson_20170630_table1_linear_models_sparse/coefficients.csv`
- `results/severson_20170630_table1_linear_models_sparse/predictions.csv`
- `results/severson_20170630_table1_linear_models_sparse/table1_linear_model_predictions.png`

## 特征重建情况

Supplementary Table 1 的候选特征被重建为 3 组：

### 1. Variance model

- `dq_log_var`: `log10(var(Delta Q_100-10(V)))`

### 2. Discharge candidates

包括 `Delta Q_100-10(V)` 统计特征与放电容量衰减曲线特征：

- `dq_min`, `dq_mean`, `dq_var`, `dq_log_var`, `dq_skewness`, `dq_kurtosis`, `dq_value_at_2v`
- `qd_slope_2_100`, `qd_intercept_2_100`
- `qd_slope_91_100`, `qd_intercept_91_100`
- `qd_cycle_2`, `qd_max_minus_cycle_2`, `qd_cycle_100`

### 3. Full candidates

在 discharge candidates 上增加：

- `charge_time_avg_first_5`
- `temp_max_2_100`, `temp_min_2_100`, `temp_integral_2_100`
- `ir_cycle_2`, `ir_min_2_100`, `ir_diff_100_2`

其中 `temp_integral_2_100` 最初用 raw time trace 积分时出现极端异常，后改为 summary-level 近似：`sum(Tavg * chargetime)`，避免单个原始时间戳/温度接触异常支配 full model。

## 本轮结果

| Feature set | Model | Split | n | RMSE cycles | MAPE |
|---|---|---:|---:|---:|---:|
| variance | linear | train | 41 | 122.03 | 16.08% |
| variance | linear | primary | 43 | 136.51 | 17.04% |
| variance | linear | primary no b2c1 | 42 | 136.76 | 15.43% |
| variance | linear | secondary | 40 | 211.22 | 11.86% |
| discharge candidates | ElasticNetCV | train | 41 | 75.19 | 9.29% |
| discharge candidates | ElasticNetCV | primary | 43 | 348.16 | 27.42% |
| discharge candidates | ElasticNetCV | primary no b2c1 | 42 | 337.01 | 17.37% |
| discharge candidates | ElasticNetCV | secondary | 40 | 223.96 | 12.89% |
| full candidates raw | ElasticNetCV | train | 41 | 54.61 | 6.15% |
| full candidates raw | ElasticNetCV | primary | 43 | 787639.73 | 17432.37% |
| full candidates raw | ElasticNetCV | secondary | 40 | 300.25 | 18.00% |
| full candidates clean4 | ElasticNetCV | primary | 39 | 114.22 | 14.22% |
| full candidates clean4 | ElasticNetCV | secondary | 40 | 300.25 | 18.00% |

## 异常样本说明

原文 Fig. 3 caption / Methods 提到 full model analysis 排除了 4 个温度探针接触或测量噪声异常 cell。本轮用训练集 z-score 对 full candidates 做自动异常检查，排在最前的 4 个 cell 为：

- `b1c18`: `qd_max_minus_cycle_2` 约 897 个训练标准差。
- `b1c0`: `qd_max_minus_cycle_2` 约 229 个训练标准差。
- `b1c41`: `dq_kurtosis` 约 52.6 个训练标准差。
- `b2c24`: `dq_kurtosis` 约 35.0 个训练标准差。

因此报告中同时保留：

- raw full：不排除任何 cell，显示模型会被极端异常严重拉爆。
- clean4 full：排除上述 4 个 full-model 异常 cell 后评估 primary，作为更接近原文 full model exclusion 的口径。

## 与原文对比

原文 Table 1 / main text 的代表结果大致为：

- Variance model: primary RMSE 138 cycles，primary MPE 14.7%，secondary MPE 11.4%。
- Discharge model: primary RMSE 91 cycles，primary MPE 13.0%，secondary MPE 8.6%。
- Full model: primary MPE 可降至约 7.5%，secondary MPE 约 10.7%。

本轮判断：

1. Variance model 已经非常接近原文 RMSE 和 secondary MPE；primary MAPE 仍略高，但量级一致。
2. Discharge candidates 的 train 很好，但 primary 明显差，说明我们的特征选择/特征定义仍未完全等同原文 selected six features。
3. Full raw 暴露出异常样本会导致严重外推；clean4 后 primary 恢复到 14.22%，但仍达不到原文 full model 约 7.5% 的水平。
4. Secondary 在 full clean4 下为 18.00%，弱于原文 10.7%，说明 IR/temperature 特征定义、异常 cell 身份、或 ElasticNet 超参数选择流程仍存在差距。

## 导师式确认结论

本轮可确认的成果：

- 数据 batch 与 split 已恢复到原文规模。
- Supplementary Table 1 候选特征已经系统重建。
- Variance model 的核心结论复现较好：`Delta Q_100-10(V)` 的 log variance 是强预测信号。
- 原文提到的 full model 异常排除现象在本复现中也确实出现，且排除 4 个极端样本后 primary 指标明显恢复。

尚不能声称完全复现原文：

- 原文 modeling code 没有公开，只能从 Supplementary Table 1 和正文重建。
- 4 个 excluded cell 的官方身份未在当前可读材料中直接给出，本轮 clean4 是基于训练分布自动识别的近似口径。
- Discharge/full model 的特征选择结果与原文仍有差距，尤其 secondary test。

推荐论文写法：

> 我们完成了对 Severson et al. 2019 数据划分、核心 `Delta Q_100-10(V)` 特征和 regularized linear 建模思想的近似复现。单特征 variance model 在 primary/secondary 上与原文结果处于同一量级，支持早期放电电压曲线差分可预测循环寿命的核心结论。进一步的 discharge/full model 复现显示，模型对 Supplementary Table 1 特征定义、异常样本排除和正则化选择流程高度敏感，因此当前结果应作为方法学确认版，而非逐数值严格复现。

## 下一步建议

1. 精确定位原文排除的 4 个 high-noise cell 身份。
2. 对 `cycle_a=10/9`、`cycle_b=100/99` 做索引敏感性实验，解决 MATLAB/Python cycle indexing 差异。
3. 对 Supplementary Table 1 的“选中特征”与“候选特征”做更严格拆分，手动实现原文 six-feature discharge 与 nine-feature full。
4. 尝试联系或查找 Braatz modeling code，以确定 ElasticNet 超参数、Monte Carlo sampling 和 excluded cell 清单。

## 2026-07-08 追加尝试：cycle index 与训练集内特征选择

### 为什么继续尝试

上一版 `cycle_a=10`, `cycle_b=100` 能验证核心思想，但 variance model 与原文仍有小差距，discharge/full model 差距更明显。考虑到原文 MATLAB 代码与 Python pickle key 之间存在 1-index / 0-index 差异，本轮增加 cycle index 敏感性实验。

### Cycle index 敏感性实验

比较三种构造：

| 特征文件 | Delta Q 构造 |
|---|---|
| `severson_table1_features_c9_c99.csv` | cycle 9 到 cycle 99 |
| `severson_table1_features_c10_c100.csv` | cycle 10 到 cycle 100 |
| `severson_table1_features_c10_c101.csv` | cycle 10 到 cycle 101 |

最接近原文的是 `c9_c99`。其 variance model 结果为：

| Model | Split | RMSE cycles | MAPE |
|---|---:|---:|---:|
| variance linear | train | 103.57 | 14.12% |
| variance linear | primary | 137.87 | 14.75% |
| variance linear | primary no b2c1 | 138.30 | 13.19% |
| variance linear | secondary | 195.86 | 11.42% |

这个结果与原文 Table 1 的 variance model 基本重合：primary RMSE 约 138、primary MPE 14.7%、secondary MPE 11.4%。因此，后续最可信版本采用 `cycle_a=9`, `cycle_b=99`。

### 训练集内特征选择

为了尽量靠近原文 “discharge model 6 features / full model 9 features”，本轮在训练集内做 4-fold CV 贪心特征选择，不使用 primary/secondary 调参。

输出目录：

- `results/severson_table1_sensitivity_c9_c99_feature_selection/all_metrics.csv`
- `results/severson_table1_sensitivity_c9_c99_feature_selection/selection_summary.json`

#### Discharge greedy6

选中特征：

- `dq_log_var`
- `qd_max_minus_cycle_2`
- `dq_value_at_2v`
- `dq_mean`
- `qd_cycle_100`
- `qd_cycle_2`

结果：

| Split | RMSE cycles | MAPE |
|---|---:|---:|
| train | 64.79 | 8.36% |
| primary | 284.04 | 23.20% |
| primary no b2c1 | 263.63 | 11.82% |
| secondary | 165.58 | 7.82% |

解释：secondary 已接近/略优于原文 discharge secondary MPE 8.6%，但 primary 受 `b2c1` 和若干外推异常影响较大；去掉 `b2c1` 后 primary MAPE 降至 11.82%，接近原文 discharge primary 13.0%。

#### Full greedy9

选中特征：

- `dq_log_var`
- `ir_cycle_2`
- `charge_time_avg_first_5`
- `qd_max_minus_cycle_2`
- `temp_max_2_100`
- `temp_integral_2_100`
- `dq_skewness`
- `qd_cycle_2`
- `ir_diff_100_2`

结果：

| Split | RMSE cycles | MAPE |
|---|---:|---:|
| train | 60.38 | 6.92% |
| primary | 258.59 | 13.48% |
| primary no b2c1 | 260.15 | 10.88% |
| primary clean4 | 89.45 | 11.03% |
| secondary | 220.72 | 12.56% |

解释：full greedy9 的 primary clean4 已较接近原文 full primary，但 secondary 仍高于原文 full secondary 10.7%。这说明 full model 的 exact excluded cells、温度/IR 特征定义与原文 modeling code 仍未完全对齐。

### 最终确认结论

本轮已经把复现结果推进到可接受确认版：

1. `c9_c99` 的 variance model 与原文几乎一致，是当前最强证据。
2. `Delta Q_100-10(V)` 的 log variance 作为早期寿命预测核心特征，已经被可靠复现。
3. Discharge/full model 可通过训练集内特征选择进一步靠近原文，但无法在 primary 与 secondary 上同时逐数字对齐。
4. 差距主要来自未公开 modeling code、exact feature selection、4 个 excluded cell 官方身份、以及温度/IR 特征定义细节。

因此建议定性为：

> 本项目成功复现了 Severson et al. 2019 的核心数据划分、核心早期放电曲线特征和 variance model 级别结果；进一步的 discharge/full model 达到趋势一致和部分指标接近，但由于原文建模代码与异常样本清单未公开，当前不声称逐数字严格复现。
