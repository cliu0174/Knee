# 第四章实验设计与写作审计（2026-07-17）

## 1. 审计目标

本文件用于把本地锂离子电池早期寿命与 knee 预测论文的实验组织方式，映射到本项目已经冻结的 region-balanced Local DeltaQ--ExtraTrees（LDQ-ET）证据。借鉴范围仅包括研究问题拆分、对照层次、图表叙事和结果段落结构；不复制外部论文的模型、数据划分、数值或结论。

## 2. 本项目术语账本

| 规范术语 | 本文含义 | 不使用的替代表述 |
|---|---|---|
| development cohort | official train 与 primary-test 合并后的 83 个合格电芯 | training set（未说明范围） |
| secondary batch | 40 个 secondary-test 电芯 | untouched external test set |
| protocol-frozen, historically informed evaluation | 开发流程冻结后进行、但该批次曾被历史探索查看的评价 | independent confirmation |
| Local DeltaQ candidates | 351 个区域描述符与 14 个 morphology 描述符，共 365 个候选 | 365 independent regions |
| region-balanced selection | bootstrap stability score 加区域容量上限的折内选择 | electrochemical region discovery |
| LDQ-ET input | 30 个固定描述符加 40 个折内选择的 Local DeltaQ 描述符，共 70 维 | all 395 candidates |
| seed variability | 30 个预设随机种子产生的算法随机性 | sample confidence interval |
| predictive association | 当前数据与标签协议下的预测关联 | unique electrochemical mechanism |

## 3. 本地论文中可借鉴的实验范式

| 本地论文 | 可借鉴的实验设计 | 对 LDQ-ET 的对应方式 |
|---|---|---|
| Severson et al. (2019) | 区分训练、primary test 与后生成的 secondary test；比较逐步增加的数据流；设置朴素基线；分析观察循环位置 | 保留 development/secondary 边界；比较固定特征、Local DeltaQ 与完整表示；使用 N50/N100/N150 公共队列 |
| Jiang et al. (2023) | 在相同划分上比较完整特征与筛选子集、不同回归器；同时展示预测散点与残差尾部 | 使用相同外层折比较七组特征与五类 readout；外部图同时展示预测、残差和尾部误差 |
| Wang et al. (2024) | 单独评价 knee cycle；同一预处理下比较多个回归模型；报告平均与最大误差 | 以绝对 knee cycle 为唯一目标；同折比较 ExtraTrees、RandomForest、XGBoost、RBF-SVR 与 RidgeCV；报告 q90/q95 与最大误差 |
| Yu et al. (2025) | 将早期观察长度作为独立实验；增加实验数据验证；讨论更早输入与精度的权衡 | 将 observation budget 从选择器比较中独立出来；使用 N50/N100/N150 与数据效率曲线量化权衡 |
| Rhyu et al. (2025) | 嵌套交叉验证避免泄漏；比较自动特征与设计特征；同时关注平均与最差折；明确外推失效 | 使用 5-fold x 2 外层和 3-fold 内层；开发分数同时包含平均、标准差和最差折；报告晚-knee及区间覆盖失效 |
| Bai et al. (2026) | 在同一优化条件下比较树模型与其他基线；先表格给出指标，再用预测—观测图解释误差结构 | 模型族比较严格共享折与处理流程；主表给指标，外部图负责展示残差、配对误差与随机种子变化 |
| Cui et al. (2025) | 分开回答 knee 预测、观察长度和模块组合三个问题；承认复杂模型不必然带来单调收益 | 第四章分别设置选择器、观察预算、secondary评价和表示/readout消融；不把复杂度当作性能依据 |
| Strange et al. (2023) | 将总体指标与随寿命阶段变化的误差、模型组合和 knee 邻近性分析分开 | 总体 secondary 指标留在第四章，尾部、晚-knee覆盖和失败电芯留在第五章讨论边界 |

## 4. LDQ-ET 的主张—证据对应

| 主张 | 当前证据 | 状态 |
|---|---|---|
| region-balanced selector 是开发阶段最可靠的冻结选择 | 10 个外层折的 MAE、RMSE、q90、Jaccard、预设开发分数与配对区间 | 支持，但与相近选择器的差异不显著，必须写成开发选择而非普遍优越性 |
| 增加早期观察循环可降低误差与折间波动 | 公共 83-cell 队列上的 N50/N100/N150 嵌套结果 | 支持 |
| 当前开发样本量仍限制性能 | 25/50/75/100% 外层训练数据学习曲线 | 支持 |
| 冻结管线在 secondary batch 上保持历史锚点水平 | 40-cell、30-seed secondary 评价与逐电芯配对检验 | 支持“parity”，不支持显著提升或全新独立确认 |
| Local DeltaQ 含有有用但可替代的预测信息 | 七组特征消融、区域删除/随机/置换控制 | 支持分布式、冗余信息；不支持唯一机理或明确独立增益 |
| ExtraTrees 适合当前小样本结构化表示 | 相同折上的模型族比较与正式/紧凑 ExtraTrees 对照 | 支持“strongest tested readout”，不支持普遍优于所有深度模型 |
| 风险主要集中在晚-knee与尾部电芯 | q90/q95、最大误差、覆盖率与最困难电芯 | 支持，并应保留为限制边界 |

## 5. 建议的第四章结构

### 4.1 Strictly nested selector comparison

问题：在完全不读取 secondary batch 的条件下，哪一种选择规则应被冻结？

证据：Table 3 与 selector comparison figure。

写法：先说明比较问题和公平协议，再报告 region-balanced 的开发结果与配对不确定性，最后把结论限制为“development choice”。

### 4.2 Observation budget and development-data efficiency

问题：需要观察多少个早期循环，当前样本量是否已达到性能平台？

证据：N50/N100/N150 公共队列与 25/50/75/100% 学习曲线。

写法：先固定队列以排除 eligibility 混杂，再报告窗口与样本量趋势，不把 N150 写成普适最优窗口。

### 4.3 Protocol-frozen secondary-batch evaluation

问题：冻结后的完整 LDQ-ET 管线在后续批次上能否维持开发阶段确定的性能水平？

证据：30-seed 指标、cell bootstrap、历史锚点配对检验、预测与残差图。

写法：主结论为 parity 与可复现性；紧接着报告 q90/q95 和最大误差，避免只呈现平均性能。

### 4.4 Feature representation and readout-family comparison

问题：性能来自哪些表示组合，ExtraTrees 是否只是任意读出器？

证据：七组特征和五类 readout 的同折比较。

写法：明确完整表示没有显著降低相对 raw+global 的平均 MAE，优势主要体现在较低折间波动；ExtraTrees 结论限定为本项目测试范围。

## 6. 第一轮正文修改范围

1. 将现有 4.1 拆为 selector comparison 与 observation/data efficiency 两个小节。
2. 四个小节均采用“研究问题/公平条件 -> 关键结果 -> 不确定性或边界”的段落顺序。
3. 保留现有表格、图片、标签和所有实验数值。
4. 不加入跨论文数值排名，因为目标、标签、数据划分和评价指标不可直接比较。

## 7. 是否需要补跑实验

当前第四章的主证据链已经完整，不建议先增加新的模型搜索。优先级如下：

1. **无需补跑即可完善正文：** 四小节重组、claim-first 写法、选择差异不显著与 historically informed 边界。
2. **可由现有文件补充：** 完整开发集 finalization 中 `c=2/3` 与两种 ExtraTrees capacity 的选择分数；如正文过密，可放补充材料。
3. **只有审稿目标明确时再补跑：** 独立多来源/多化学体系验证。当前 Severson 数据不能替代这一证据。
4. **不建议继续搜索：** 已查看 secondary 结果后再挑选新的架构、特征数或融合权重，这会扩大测试反馈风险。

