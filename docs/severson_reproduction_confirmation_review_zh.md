# Severson 2019 复现导师审查确认版

日期：2026-06-05

## 确认结论

本复现可以作为**第一阶段复现确认版**接受，但还不能作为严格的论文级复现。

当前流程已经完整跑通：

```text
.mat 原始数据 -> batch*.pkl -> official-style 特征 -> 训练/评估 -> 图表输出
```

从科学结论上看，当前结果与 Severson et al. 2019 的核心思想一致：早期循环的放电电压曲线特征能够预测 cycle life，且最重要的特征集中在 `delta_q_var`、`delta_q_log_var`、`delta_q_min` 这类 `ΔQ_100-10(V)` 统计量上。

但是，目前还没有达到严格复现原文的程度，主要原因是当前第二批数据文件与常见原文版本 `2017-06-30` 不一致，并且该批数据中存在缺失 EOL 标签的 cell。

## 数据状态

当前在 `data` 文件夹中确认到三份原始 `.mat` 文件：

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat`

这三份文件都包含预期的 HDF5 `batch` 结构，并已转换为：

- `data/severson_raw/pkl/batch1.pkl`
- `data/severson_raw/pkl/batch2.pkl`
- `data/severson_raw/pkl/batch3.pkl`

最终提取出的监督学习特征表为：

- `data/severson_features/severson_official_features.csv`

该特征表包含 115 个有标签 cell：

```text
train: 37
primary_test: 39
secondary_test: 39
```

其中 batch2 有 8 个 cell 因为 `cycle_life/lifetime` 缺失，且放电容量曲线没有明确低于 0.88 EOL 阈值，所以被剔除：

```text
b2c22, b2c23, b2c35, b2c36, b2c37, b2c38, b2c39, b2c40
```

这个处理是合理的：相比强行插值或猜测 EOL，删除无可靠标签样本更稳妥。但这也意味着当前实验不是原文完整 124/123-cell 设置。

## 结果对比

Severson 原文公开摘要中的代表性结果：

- 使用前 100 cycles 预测 cycle life，最佳模型约为 9.1% test error。

导师根据原文表格整理的参考值：

| 原文模型 | Primary MAPE | Secondary MAPE |
|---|---:|---:|
| Variance model | 14.7% / 去异常后 13.2% | 11.4% |
| Discharge model | 13.0% / 去异常后 10.1% | 8.6% |
| Full model | 14.1% / 去异常后 7.5% | 10.7% |

当前确认版结果采用 **train-only 4-fold CV** 选择参数，即只在训练集内部调参，primary/secondary test 只用于最终评估。

| 当前模型 | Primary MAPE | Secondary MAPE | 说明 |
|---|---:|---:|---|
| XGBoost | 11.67% | 14.60% | train-CV 选出的最佳模型 |
| RandomForest | 12.46% | 12.69% | secondary 上更均衡 |
| ElasticNetCV | 16.20% | 17.06% | 方法族更接近原文，但效果较弱 |

探索性 post-hoc 结果如下：

| 当前模型 | Primary MAPE | Secondary MAPE | 说明 |
|---|---:|---:|---|
| XGBoost | 11.00% | 12.67% | 用 primary+secondary 平均 MAPE 选参，不作为严格结论 |
| RandomForest | 11.35% | 12.50% | 用 primary+secondary 平均 MAPE 选参，不作为严格结论 |

post-hoc 结果说明模型潜力存在，但因为使用了 secondary test 参与选参，因此不能作为确认版主结果。

## 结果解释

与原文一致的地方：

- 早期循环特征确实能预测 cycle life。
- `delta_q` 相关统计量主导特征重要性。
- primary test 误差与原文结果处在同一量级。

尚未达到严格复现的地方：

- 当前 batch2 使用的是 `2018-02-20`，而不是常见的 `2017-06-30`。
- 当前有标签样本数为 115，不是原文完整样本划分。
- 当前 split 是 modified split，不是 exact official split。
- 当前最佳模型是 XGBoost / RandomForest，而原文主线是 regularized linear / elastic net 风格模型。
- secondary test 误差仍高于原文 Discharge / Full model 的最佳结果。

## 导师式审查结论

**阶段性复现确认版：可接受。**

**严格 Severson 2019 原文复现：仍不可接受。**

当前结果可以作为一个确认过的中间复现版本，说明数据读取、特征提取、建模、评估、绘图流程成立，也说明论文核心现象成立。但在文档或论文中，应将其描述为：

```text
基于 115 个有标签 cell 的 official-style modified reproduction
```

不应写成：

```text
Severson et al. 2019 的 exact reproduction
```

导师认可的关键修正是：确认版模型选择已经改为只使用 train-set 4-fold CV，secondary test 不再参与参数选择。

## 后续必须优先修正的问题

1. 获取或验证 `2017-06-30_batchdata_updated_struct_errorcorrect.mat` 这个 batch2 文件。
2. 尽可能恢复原文 split 规模：`train=41`, `primary_test=43`, `secondary_test=40`。
3. 更忠实复现原文 regularized linear / elastic net 特征子集和建模流程。
4. secondary test 必须始终作为最终外部验证，不能参与调参。
5. XGBoost / RandomForest 应作为扩展 baseline 报告，不能称为 Severson 原文模型。

## 使用 2017-06-30 Batch2 后的新版确认

更新日期：2026-06-05

用户补充 `2017-06-30_batchdata_updated_struct_errorcorrect.mat` 后，重新进行了完整复现流程：

```text
2017-05-12 batch1
2017-06-30 batch2
2018-04-12 batch3
-> batch*.pkl
-> official-style features
-> train-CV 参数选择
-> official split 评估
```

新版结果可以从之前的：

```text
official-style modified reproduction
```

升级为：

```text
更接近 Severson 2019 原文数据划分的第一阶段确认版复现
```

但仍不能称为 exact reproduction。

### 新版数据状态

新版 pkl 输出目录：

- `data/severson_raw/pkl_20170630`

新版特征表：

- `data/severson_features_20170630/severson_official_features_20170630.csv`

新版样本与划分：

| 项目 | 数量 |
|---|---:|
| 总样本 | 124 |
| batch1 | 41 |
| batch2 | 43 |
| batch3 | 40 |
| train | 41 |
| primary_test | 43 |
| secondary_test | 40 |
| missing cycle_life | 0 |

这已经恢复到 Severson 原文 official split 规模。

### 新版主结果

新版结果目录：

- `results/severson_20170630_official_features_split_traincv_params`

全 primary / secondary test：

| 当前模型 | Primary MAPE | Secondary MAPE |
|---|---:|---:|
| RidgeCV | 13.87% | 22.13% |
| ElasticNetCV | 14.46% | 20.76% |
| RandomForest | 15.77% | 12.60% |
| XGBoost | 16.35% | 14.00% |
| GPR-RBF | 16.15% | 32.16% |

去除 primary 短寿命异常点 `b2c1` 后：

| 当前模型 | Primary no b2c1 MAPE |
|---|---:|
| GPR-RBF | 10.35% |
| RidgeCV | 11.11% |
| SVR-RBF | 11.78% |
| ElasticNetCV | 12.02% |
| RandomForest | 12.16% |
| XGBoost | 12.38% |

### 新版结果解释

与原文更接近的地方：

- 数据划分已恢复到 `train=41`, `primary_test=43`, `secondary_test=40`。
- Primary 全样本结果已接近原文量级，尤其 RidgeCV 13.87%、ElasticNetCV 14.46%，接近原文 variance/full model 的 14% 左右水平。
- 去除 primary 短寿命异常点 `b2c1` 后，RidgeCV 11.11%、ElasticNetCV 12.02%，进入原文去异常结果附近。
- 特征重要性仍由 `delta_q_var`, `delta_q_log_var`, `delta_q_min` 主导，支持 `ΔQ_100-10(V)` 是寿命预测核心信号。

仍存在的差距：

- ElasticNet/Ridge 的 secondary MAPE 仍较高，约 20%-22%。
- RandomForest secondary MAPE 12.60%，接近原文 variance model secondary 11.4%，但仍高于 discharge/full model 的最佳 secondary 结果。
- 当前 ElasticNetCV 是 sklearn 标准实现，不等同原文完整 regularized linear model 工作流。
- 当前特征是 official-style features，尚不能保证完全等同 Supplementary Table 1 的 exact feature set。
- 尚未复现原文 Monte Carlo sampling、特征子集选择和最终模型选择流程。

### 新版导师审查结论

子智能体导师审查后给出结论：

```text
小修后可接受，作为第一阶段确认版复现。
```

一句话判断：

> 新版已经从“修改版复现”升级为“原文 split 下的可信第一阶段复现”，科学结论一致，primary 数值接近，secondary 和 exact method 仍需继续追。

## 推荐写法

如果要在组会、开题或论文笔记中描述当前结果，建议使用：

> 本阶段完成了 Severson 2019 的 official-style modified reproduction。基于 115 个有标签 cell，使用前 100 cycles 的 `ΔQ_100-10(V)` 统计特征及早期容量、内阻、温度、充电时间特征，构建了 train-CV 参数选择下的寿命预测模型。结果显示，XGBoost 在 primary test 上达到 11.67% MAPE，RandomForest 在 secondary test 上达到 12.69% MAPE。特征重要性显示 `delta_q_var`、`delta_q_log_var` 和 `delta_q_min` 排名前列，与原文关于早期放电曲线差异可预测寿命的核心结论一致。但由于 batch2 数据版本、缺失标签样本和 split 规模与原文不完全一致，该结果应视为阶段性确认版，而非严格原文复现。
