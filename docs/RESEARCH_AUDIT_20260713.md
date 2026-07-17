# LDQ-ET 研究、代码与证据审计（2026-07-13）

## 1. 审计结论

当前可复现主线是 **LDQ-ET top40**：30 个基础描述符（22 raw/SOH + 8 global DeltaQ）、40 个 Local DeltaQ 描述符与 ExtraTrees 回归器。`results/reproduction_ldq_et_N150_20260713/` 的独立重跑复现了原正式结果：external secondary 的 30-seed MAE 为 77.2256±0.8919 cycles，RMSE 为 111.0655±0.9314 cycles，MAPE 为 9.0420±0.1166%，R² 为 0.7939±0.0035。

旧结果的外部预测本身可复现，但开发集模型选择并非严格嵌套：Local DeltaQ top40 在 41 个 official-train cells 上预先固定，然后用于 train+primary 的交叉验证。由于 official-train 中部分电池会进入交叉验证的验证折，其标签已经参与过 top40 排序，因此旧 development CV 不能作为“折内重选”的无偏估计。该问题不等同于 secondary 标签泄漏，但会使旧 HPO 的 development CV 偏乐观。

此外，secondary batch 已经在 top-k、稳定性变体和多种历史架构中被反复查看。后续冻结方案在 secondary 上的结果必须标记为 **secondary-informed protocol-controlled evaluation**，不能称为真正首次未触碰的外部确认。

## 2. 现行数据流

```text
corrected Severson MAT/PKL
        |
        +--> cycle-level capacity/Qdlin export
        |
        +--> SOH smoothing + monotone correction
        |          |
        |          +--> full-trajectory piecewise knee label
        |
        +--> first-N-cycle raw/SOH descriptors
        |
        +--> Qdlin(cycle N-1) - Qdlin(cycle 9)
                   |
                   +--> 8 global DeltaQ descriptors
                   |
                   +--> 39 overlapping regions x 9 statistics
                   +--> 14 motif/matrix-profile summaries
                              |
                              +--> 365 Local DeltaQ candidates
                                         |
                                         +--> feature selection
                                         +--> ExtraTrees readout
```

统计单位始终是一块电池。`knee_after_window=True` 后共有 123 cells：41 train、42 primary、40 secondary。完整寿命轨迹只生成监督标签；特征来自前 N 个循环。

## 3. 已核对的实现事实

| 项目 | 代码/数据事实 | 审计判断 |
|---|---|---|
| DeltaQ cycle pair | N150: 9→149；N100: 9→99；N50: 9→49 | 均在输入窗口内 |
| Local region | 50 grid points，stride 25 | 50% 重叠，共 39 个区域 |
| 区域统计 | mean/std/var/abs_mean/min/max/range/slope/energy | 351 个局部区域特征 |
| 非区域 LDQ | 14 个 matrix-profile/motif 描述符 | 与区域统计合计 365 候选 |
| 旧 top40 | official-train 绝对 Pearson 排序 | external 隔离，但非外层折内选择 |
| 旧 HPO | 5-fold、30 抽样候选、固定 top40 | 不满足本轮严格嵌套要求 |
| 外部随机性 | 30 seeds | 衡量模型随机性，不是样本 CI |
| 样本不确定性 | 40-cell bootstrap | 衡量当前 external 样本抽样不确定性 |
| 区域稳定性 | 41 train cells、1000 bootstrap | 稳定性证据，不是预测贡献或因果证据 |

## 4. 泄漏与测试反馈风险

1. **旧 development CV 的预选特征风险（已确认）**：top40 在 official train 全体上预选，随后 official train 又参与 train+primary CV。新实验必须在每个外层训练折内重新筛选。
2. **secondary 反馈风险（已确认）**：top20/40/60/80、stable/balanced 变体和历史探索模型均查看过 secondary。不得再称该批次为完全未触碰。
3. **标签生成边界（可接受）**：完整轨迹用于离线生成 knee 标签，但不进入模型输入。论文必须继续明确这一区分。
4. **随机混合划分乐观（已确认）**：混合三批后的 R² 显著高于外部批次协议，只能作为诊断对照。
5. **标签定义不确定性（部分完成）**：piecewise 与 Kneedle 排序高度一致，但现有模型稳健性比较主要使用 raw/global 基线，尚未对严格嵌套 LDQ 选择重复。

## 5. 统计缺口

- 原正式 LDQ-ET 尚缺与强基线的 cell-level 配对 bootstrap/置换检验；
- 原正式结果报告了 cell bootstrap CI，但缺少 q90 absolute error 和最大误差的主表整合；
- top-k、稳定性变体与横向模型比较多数直接读取 secondary，属于支持性/探索性证据；
- 尚缺严格嵌套的 50/100/150-cycle Local DeltaQ 比较；
- 尚缺 25/50/75/100% development-data 学习曲线；
- 尚缺核心区域只保留、遮蔽、随机等宽区域、随机特征和区域置换的折内比较；
- 尚缺开发校准、外部只评价的预测区间覆盖率与宽度。

## 6. 创新与正文缺口

现稿最稳妥的贡献是“局部曲线表征 + 小样本树读出 + 跨批次协议”，而不是 ExtraTrees 本身。当前“敏感区域”证据主要来自相关排序、特征重要性和 bootstrap 复现，尚不能区分：

1. 单变量相关；
2. 特征选择稳定性；
3. 相邻区域稳定性；
4. 条件预测贡献；
5. 电化学因果机制。

新机制应优先解决第 2–4 层，并把第 5 层保持为假设。正文还需补充真正的 nested validation、区域贡献、数据效率、输入预算、尾部误差、不确定性校准和证据分级。

## 7. 候选方向与筛选

| 方向 | 可证伪假设 | 小样本适用性 | 主要风险 | 决定 |
|---|---|---:|---|---|
| 折内 Pearson top-k | 旧结论在无预选泄漏时仍成立 | 高 | 性能下降 | 保留为严格基线 |
| Bootstrap-stable top-k | 训练样本扰动下的稳定排序改善外层误差 | 高 | 稳定不等于贡献 | 进入实验 |
| Region-balanced stability | 限制单一区域描述符重复可降低冗余 | 高 | 可能丢失真实核心带 | 进入实验 |
| 连续性约束核心带 + 互补特征 | 利用相邻窗口有序性可降低选择方差并保留局部贡献 | 高 | 核心带宽选择过拟合 | 进入严格嵌套实验 |
| mRMR | 显式惩罚冗余优于单变量相关 | 中高 | 相关矩阵在小样本下不稳 | 进入实验 |
| Functional PCA/wavelet | 低维函数基比固定统计更有效 | 中 | 基与维数选择容易过拟合 | 暂缓，待区域机制验证后 |
| Log/Box-Cox/relative target | 结构化目标降低晚-knee尾部误差 | 中 | 反变换偏差 | 作为后续补充 |
| Cross-conformal interval | 开发残差可给出有用风险区间 | 高 | batch shift 下覆盖不保证 | 进入可靠性实验 |
| Domain alignment/group DRO | 开发域不变表征改善跨批次泛化 | 低至中 | 只有两个开发批次，目标不可识别 | 暂缓 |

## 8. 新机制的审计定义

新脚本 `scripts/run_nested_contiguous_ldq.py` 实现五种折内选择器。连续性机制先在训练折 bootstrap 中计算每个 Local DeltaQ 特征的入选频率和平均绝对相关，再把同一区域的特征聚合，并用相邻区域 0.25/0.50/0.25 权重平滑。内层 CV 选择连续带宽与核心特征比例；最终 top-k 的核心部分来自一个连续区域带，其余来自带外互补特征。所有选择、缺失值中位数和 ExtraTrees 配置均在外层训练折内重新拟合。

开发选择分数预先固定为：

```text
mean outer-fold MAE
+ 0.10 x std outer-fold MAE
+ 0.10 x worst outer-fold MAE
+ predeclared method complexity penalty
```

复杂度惩罚在读取任何新 external 结果之前写入代码和 metadata。

## 9. 论文术语账本

| 规范术语 | 首次定义 | 不再使用的含混写法 |
|---|---|---|
| Local DeltaQ (LDQ) | region-wise descriptors of early-cycle DeltaQ | local degradation feature（未定义） |
| LDQ-ET top40 | 30 base descriptors + 40 LDQ descriptors + ExtraTrees | proposed ExtraTrees algorithm |
| development cohort | official train + primary test, 83 cells | training set（未说明范围） |
| external secondary batch | official secondary test, 40 cells | untouched test set（历史上已被查看） |
| seed standard deviation | across 30 model random seeds | confidence interval |
| cell-bootstrap 95% CI | resampling 40 external cells | seed variation |
| selected-region stability | recurrence under train-cell resampling | predictive contribution / mechanism |
| normalized curve coordinate | index/1000 along the interpolated Qdlin grid | voltage region（若未给出映射） |

## 10. 证据等级

| 等级 | 当前内容 |
|---|---|
| 正式可复现证据 | 原 LDQ-ET top40 30-seed external 结果；raw/global 与 Local DeltaQ XGBoost 主基线；external cell bootstrap；标签方法一致性 |
| 补充证据 | top-k 平台、横向模型比较、随机划分对照、区域 bootstrap 稳定性 |
| secondary-informed 探索 | stable/balanced top40、dual-view、MoE、factorized-knee、其他历史架构 |
| 待完成正式证据 | 严格嵌套连续区域机制、输入预算、数据效率、区域遮蔽/置换、预测区间 |

