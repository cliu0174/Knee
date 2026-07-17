# Besttery 项目从零实现与理解指南

> 文档定位：这是当前 **严格嵌套 region-balanced LDQ-ET top40** 正式主线的中文技术教程，而不是历史实验日志。它用于帮助第一次接触本项目的读者，从研究问题、数据、算法、代码、实验设计和结果解释逐层理解整个项目，也适合作为后续 AI 辅助问答的主上下文。
>
> 当前版本：2026-07-15。本文以现行代码、`manuscript_ionics/main.tex` 和非重叠分段规则下的正式结果目录为准。Dual-view、MoE、factorized-knee、STRIDE/Uzi 等路线是历史探索，不属于当前唯一主模型。

---

## 1. 先用一分钟理解这个项目

这个项目研究的问题是：

> 只观察一块锂离子电池最初 150 个循环的容量与放电曲线，能否提前预测它在未来第几个循环进入容量加速衰退阶段？

容量缓慢衰减转为快速衰减的位置称为 **capacity knee point**，其横坐标称为 **knee cycle**。它可以被理解为电池退化轨迹开始明显“拐弯”的时间。越早、越准确地预测 knee cycle，越有利于筛选电池、安排测试、进行寿命预警和维护决策。

项目的核心假设是：未来 knee 的早期信息可能并不均匀分布在整条放电曲线差分中。若只把整条曲线压缩成均值、方差等全局统计量，可能会稀释局部信号。因此，本项目：

1. 计算两个早期循环之间的放电容量曲线差分 `DeltaQ`；
2. 沿归一化放电坐标把 DeltaQ 切成相互重叠的局部区域；
3. 为每个区域计算均值、波动、斜率、能量等描述符；
4. 在每个训练折内部比较 Pearson、bootstrap-stable、region-balanced、contiguous 和 mRMR，并选出 40 个局部特征；
5. 将 40 个 Local DeltaQ 特征与 30 个基础特征拼接；
6. 用受控深度的 ExtraTrees 回归器预测绝对 knee cycle。

最终主模型称为：

> **严格嵌套 region-balanced LDQ-ET top40 = Local DeltaQ top40 representation + Extremely Randomized Trees**

它不是深度神经网络，也不是 mixture-of-experts。它是一个针对小样本、跨批次早期预测场景设计的“局部曲线表征 + 稳定树集成读出”框架。

---

## 2. 研究问题的严格定义

### 2.1 输入、输出和信息边界

对第 \(i\) 块电池，模型输入记为：

\[
X_i^{(150)}=\{\text{capacity and discharge-curve observations from cycles }1\ldots150\}.
\]

监督目标是该电池由完整退化轨迹离线计算得到的绝对 knee cycle：

\[
y_i=k_i.
\]

模型学习映射：

\[
\hat{k}_i=f_\theta\!\left(X_i^{(150)}\right).
\]

必须区分两个阶段：

- **训练标签生成阶段**可以查看完整寿命轨迹，因为需要知道真实 knee 在哪里；
- **模型训练和推理阶段**只允许使用前 150 个循环形成的输入特征。

因此，本项目不是用前 150 个循环“直接检测”已经发生的 knee，而是根据早期退化迹象预测未来 knee。

### 2.2 为什么剔除 knee 不晚于 150 的样本

若某块电池在第 150 个循环之前已经出现 knee，模型输入中实际上已经包含目标事件，这不再是严格的未来预测。因此正式实验使用：

\[
k_i>150.
\]

原始数据共有 124 块电池。执行该条件后，正式建模使用 123 块：

| 官方分组 | 角色 | 样本数 |
|---|---|---:|
| train | 特征筛选；同时属于开发队列 | 41 |
| primary test | 与 train 合并，用于模型开发和 HPO | 42 |
| secondary test | protocol-frozen 批次评价 | 40 |
| 合计 | 正式建模样本 | 123 |

开发队列共有 83 块，后续批次评价队列共有 40 块。由于仓库早期 top-k 和架构实验已经查看过 secondary 结果，这 40 块电池不再具备“完全未触碰外部确认”的资格。

### 2.3 为什么不把三批电池随机混合

三批电池并非同分布。真实 knee cycle 的中位数分别约为：

- train：428 cycles；
- primary test：429.5 cycles；
- secondary test：796.5 cycles。

secondary batch 的 knee 明显更晚。如果把三批随机混合，训练集和测试集会共享批次特征，结果会明显乐观。本项目仍保留随机混合划分作为历史诊断性对照；当前论文主证据由“83-cell development 严格嵌套评价”和“protocol-frozen secondary-batch 压力测试”共同构成。secondary 的观测值和标签均未用于确定正式 LDQ-ET 的特征选择器、特征数量、算法类别、超参数或拟合参数。正式模型开发完全基于 development cohort，未受到 secondary batch 的干扰。

---

## 3. 从原始数据到监督标签

### 3.1 数据来源和基本对象

项目使用修正后的 Severson/MIT-Stanford 2017-06-30 快充电池数据。每块电池包含：

- 循环编号；
- 充放电容量；
- 循环内的电压—容量曲线；
- 完整寿命等元数据；
- 官方 train、primary test、secondary test 分组。

项目不把单个循环当作独立样本。统计独立单位始终是“电池”。这是因为同一电池不同循环强相关，若把循环随机拆分到训练集和测试集，会产生严重的数据泄漏。

### 3.2 SOH 的定义与校正

容量健康状态可写成：

\[
\mathrm{SOH}_{i,t}=\frac{Q_{i,t}}{Q_{i,\mathrm{ref}}},
\]

其中 \(Q_{i,t}\) 是第 \(t\) 个循环的放电容量，\(Q_{i,\mathrm{ref}}\) 是参考初始容量。

实际容量轨迹包含测量噪声、恢复现象和异常小幅上升。`scripts/correct_soh_curves.py` 依次执行：

1. 对有效正容量点进行 Savitzky–Golay 平滑；
2. 若相应依赖不可用，则退化为移动平均；
3. 对平滑曲线执行单调递减的 isotonic regression；
4. 若 sklearn 方法不可用，则使用 PAVA 作为后备；
5. 保存 raw、smooth、monotone 三种曲线和修正幅度、再生次数等质量控制量。

这样做的目的不是声称真实电池容量绝对单调，而是为离线 knee 标注建立更稳健的趋势基线。原始容量和原始 SOH 仍被保留用于早期特征，因此趋势校正不是把所有输入强行改成单调曲线。

### 3.3 knee 标签如何计算

主标签由 `scripts/detect_knee_points.py` 的两段式分段线性拟合产生。设单调 SOH 轨迹为 \((t_j,s_j)\)，对每个允许的候选断点 \(b\)，分别拟合断点前后的直线：

\[
s_j \approx a_1t_j+c_1,\quad j\le b,
\]

\[
s_j \approx a_2t_j+c_2,\quad j>b.
\]

选择总残差平方和最小的断点：

\[
b^*=\arg\min_b\left(\mathrm{SSE}_{1:b}+\mathrm{SSE}_{b+1:n}\right),
\]

并令：

\[
k_i=t_{b^*}.
\]

为防止断点落在寿命轨迹极端位置，候选范围约束在轨迹的 20% 到 90%，且每段至少包含 30 个点。断点观测只计入前段，后段从下一观测开始，因此每个观测在目标函数中只贡献一个残差。

2026-07-15 的规则审计比较了旧版“边界点两侧重复计入”和现行“互不重叠”定义。124 个电芯中有 71 个 knee 标签改变，平均绝对变化 0.59 cycles，最大变化 2 cycles；122 个电芯的变化不超过 1 cycle，且 `knee_cycle > 150` 的合格样本集合完全不变。尽管标签位置变化很小，严格折内特征排序和开发误差仍发生了可测变化，因此正式实验已全部按互不重叠规则重跑。

标签拟合改善分数使用“一条直线”相对“两段直线”的拟合改善定义：

\[
C_i=\frac{\mathrm{SSE}_{\mathrm{single}}-\mathrm{SSE}_{\mathrm{piecewise}}}
{\mathrm{SSE}_{\mathrm{single}}}.
\]

\(C_i\) 越大，说明分段模型比单一线性衰减更能解释该轨迹，但它不是校准概率。124 个标签的 \(C_i\) 为 \(0.9559\pm0.0284\)（均值±标准差），中位数 0.9632，范围 0.7975--0.9879；正式流程未使用该分数删除样本、分配权重或作为预测特征。

### 3.4 标签稳定性意味着什么

项目还将分段线性 knee 与 Kneedle、衰减速率阈值等替代定义进行了比较。分段线性与 Kneedle 的结果具有很高的秩相关：

- 平均绝对差约 32.85 cycles；
- Spearman 相关系数约 0.9978；
- 87.90% 的电池差异在 ±50 cycles 内；
- 99.19% 的电池差异在 ±100 cycles 内。

这说明不同合理标签方法对“哪些电池 knee 更早或更晚”的排序较一致，但 knee 的绝对位置仍有算法依赖性。论文结果应表述为“预测由指定离线算法定义的 capacity knee”，不能把该标签说成唯一物理真值。

---

## 4. DeltaQ 为什么能包含早期退化信息

### 4.1 曲线差分的基本思想

仅看前 150 个循环内的容量下降幅度，信号可能非常弱。DeltaQ 比较两个早期循环在相同放电坐标上的容量响应：

\[
\Delta Q_i(u)=Q_{i,149}(u)-Q_{i,9}(u),
\]

其中 \(u\in[0,1]\) 是统一、归一化后的放电曲线坐标。代码默认比较 cycle 9 与 cycle 149，从而避免最初少数循环的形成期扰动，同时覆盖整个 150-cycle 观察窗口。

如果整条放电响应在早期已经发生位移、形变或局部波动，这些变化会反映在 \(\Delta Q_i(u)\) 中。Severson 类工作常把整条 DeltaQ 曲线压缩为方差等全局统计量。本项目进一步问：

> 如果与未来 knee 相关的变化只集中在曲线的一部分，整条曲线的全局压缩会不会把它平均掉？

Local DeltaQ 就是对这个问题的工程化回答。

### 4.2 全局 DeltaQ 特征

全局特征对完整 \(\Delta Q_i(u)\) 计算均值、样本标准差、总体方差、log variance、最小值、最大值、偏差校正的样本偏度和偏差校正的 Fisher excess kurtosis。当前正式向量保留 8 个 global DeltaQ 描述符。由于每条曲线固定为 \(L=1000\) 个点，样本标准差 \(s_{\Delta q}\) 与总体方差 \(v_{\Delta q}\) 满足 \(s_{\Delta q}^2=L v_{\Delta q}/(L-1)\)，log variance 又是方差的单调变换。三者作为冻结特征生成管线中的预设尺度编码保留，但不能解释成三个相互独立的信息来源。

全局描述符的优点是低维、稳健；缺点是丢失空间位置。例如，两条曲线可能具有相近方差，但一条在前半段变化，另一条在后半段变化。两者潜在退化含义和跨批次稳定性可能不同。

### 4.3 Local DeltaQ 的重叠窗口

将统一网格上的 DeltaQ 曲线切成长度为 50 个网格点、步长为 25 个点的重叠窗口。若网格长度为 1000，就得到 39 个局部区域：

\[
R_j=[u_j,u_j+0.05],\qquad u_{j+1}-u_j=0.025.
\]

50% 重叠使边界附近的局部变化不会因为一次硬切分而完全落入不同区域，也提供了多尺度近似：相邻窗口共同描述较宽的敏感带。

对每个局部区域计算 9 个描述符：

| 描述符 | 含义 |
|---|---|
| mean | 局部平均位移 |
| std | 局部波动强度 |
| variance | 波动能量的平方尺度 |
| absolute mean | 不考虑正负方向的平均变化 |
| min | 最强负向变化 |
| max | 最强正向变化 |
| range | 局部振幅范围 |
| slope | 区域内沿放电坐标的趋势 |
| energy | 局部平方能量 |

固定窗口宽度为 \(W=50\) 时，局部样本标准差与总体方差满足 \(s^2=Wv/(W-1)\)，因此二者是确定性的尺度变换。它们继续保留在预先定义的候选字典中，由每个训练折内部的特征选择决定是否进入局部特征块；正式 region-balanced 选择结果保留过 `std`，未保留 `variance`。这一设置用于保持现有特征生成与实验结果可复现，不应被表述为二者提供独立信息。

39 个区域 × 9 个描述符 = 351 个区域特征。脚本还从整条 DeltaQ 的 matrix-profile/形态统计中生成 14 个 `ldq` 描述符，因此候选 Local DeltaQ 特征总数为：

\[
351+14=365.
\]

实现位于 `scripts/extract_local_deltaq_features.py`。

### 4.4 Local DeltaQ 不是已经证明的电化学机理

局部放电曲线变化可能与极化、动力学变化、活性锂损失或活性材料损失有关，但本数据没有拆解电极贡献，也没有 post-mortem 证据。0.525–0.650 区域的重复入选只能支持：

> 在当前数据和标签定义下，这一归一化放电坐标带包含可重复的 knee 相关统计信息。

它不能支持“该区域唯一对应某个确定反应机制”。这一区分对论文可信度非常重要。

---

## 5. 完整特征向量如何形成

### 5.1 22 个 raw/SOH 描述符

`scripts/extract_early_knee_features.py` 在前 150 个循环内计算基础容量和 SOH 特征。正式模型从中使用 22 个 raw/SOH 描述符，主要包括：

- 放电容量的均值、标准差、最小值、最大值、首值、末值、极差、变化量和末/首比；
- raw SOH 的均值、标准差、最小值、最大值、首值、末值、变化量、斜率和截距；
- raw SOH 在相邻有效观测之间的再生次数、总再生幅度和最大上升；
- 有效观测数/窗口完整性等变量会在建模前按数值性和常数性规则处理。

再生事件使用固定阈值 \(\delta=10^{-4}\)。该值在 `scripts/extract_early_knee_features.py` 中作为数值容差使用，用于排除相邻有效观测之间可忽略的小幅波动；正式流程始终保持固定，未被纳入针对 knee 标签的超参数搜索。代码标识符 `raw_max_single_cycle_increase` 保留了历史命名，但数学定义不假定两个有效观测必然只相隔一个物理循环。

代码还生成 smooth/monotone 相关 QC 特征，但正式 `select_local_deltaq_features.py` 的入模过滤规则只保留既定 raw、global DeltaQ 和 Local DeltaQ 族，不能把“CSV 中存在”误认为“最终模型一定使用”。

### 5.2 top40 的严格折内筛选

历史版本曾只在 41 块 official train 电池上按绝对 Pearson 相关选 top40。该结果现在只作为可精确复现的历史锚点。正式主线在每个外层和内层训练折内重新完成：

1. 删除非数值或常数候选；
2. 用当前训练折中位数拟合缺失值插补；
3. 拟合候选评分、bootstrap 稳定性或冗余惩罚；
4. 选择 40 个 Local DeltaQ 描述符；
5. 拟合相应 ExtraTrees 配置；
6. 只在未参与上述步骤的验证折上计算误差。

五个候选选择器为：

- **Pearson top40**：按 \(|\mathrm{corr}(x_j,k)|\) 排序；
- **bootstrap-stable top40**：优先保留训练折重采样中反复出现的描述符；
- **region-balanced top40**：每个局部区域最多保留 3 个描述符，避免单一区域过度占据名额；
- **contiguous band**：对相邻区域得分平滑后选择连续候选带；
- **mRMR**：在目标相关性与候选间冗余之间折中。

最终输入仍为：

\[
\mathbf{x}_i=
[\underbrace{\mathbf{x}^{raw}_i}_{22},
\underbrace{\mathbf{x}^{global\Delta Q}_i}_{8},
\underbrace{\mathbf{x}^{local\Delta Q}_i}_{40}],
\]

总维度为 70。开发集 5 折×2 的预声明选择分数最终选中 region-balanced top40；它覆盖 14 个区域，每个区域最多 3 个描述符。

### 5.3 top40 容量与历史 top-k 结果如何理解

历史 secondary top-k 消融得到 top40–top60 平台，但它已经读取 secondary 标签，不能继续作为当前无偏选型依据：

| Local 特征数 | 历史 secondary MAE | 历史 secondary RMSE | 说明 |
|---:|---:|---:|---|
| 20 | 80.95 ± 0.91 | 117.94 ± 0.84 | 特征覆盖不足 |
| 40 | 77.23 ± 0.89 | 111.07 ± 0.93 | 历史锚点 |
| 60 | 76.35 ± 0.72 | 111.31 ± 0.62 | MAE 略低但维度更高 |
| 80 | 77.67 ± 0.75 | 113.63 ± 0.81 | 出现回落 |

当前严格实验固定 40 个 Local 特征，是为了保持候选选择器之间的容量、总输入维数和复杂度可比；选择器胜负完全由 development 折内证据决定。若要重新选择 20/40/60/80 的容量，应在 development 内重新嵌套，之后再到真正未触碰的新数据上一次确认。

---

## 6. 主模型 ExtraTrees 的结构和原理

### 6.1 一棵回归树做什么

回归树不断选择“特征 + 阈值”把样本空间划分为若干叶节点。叶节点输出落入该区域的训练标签均值。它可以表示诸如：

> 当局部 0.55 附近的 DeltaQ range 较大，且早期 SOH slope 更负时，预测 knee 更早。

这类非线性、带交互的条件不需要预先写成解析公式。

### 6.2 ExtraTrees 与普通决策树/随机森林的区别

单棵深树方差很大。ExtraTrees 构建许多随机化回归树，并平均它们的输出：

\[
\hat{k}(\mathbf{x})=\frac{1}{T}\sum_{t=1}^{T}h_t(\mathbf{x}).
\]

其关键随机性包括：

- 每次分裂只考虑一部分特征；
- 分裂阈值比普通随机森林更加随机化；
- 多棵去相关树的平均可降低方差。

在只有 83 个开发样本、70 个表格特征的场景中，ExtraTrees 比高参数量序列模型更容易控制复杂度，又能表达非线性与局部特征交互。

### 6.3 当前唯一主模型的固定配置

最终配置来自开发队列内的 HPO：

```text
n_estimators       = 1200
max_features       = 0.5
max_depth          = 6
min_samples_leaf   = 1
min_samples_split  = 2
bootstrap          = False
```

含义如下：

- 1200 棵树：通过更多随机树平均降低预测波动；
- 每次分裂考虑 50% 特征：在信息利用和树间去相关之间折中；
- 最大深度 6：限制小样本条件下的过拟合；
- 叶节点最少 1 个样本：保留局部拟合能力；
- 内节点至少 2 个样本可分裂；
- 不做 bootstrap：每棵树使用全部开发样本，随机性主要来自特征和阈值选择。

### 6.4 正式选择如何避免使用批次评价标签

当前正式入口是 `scripts/run_nested_contiguous_ldq.py`，而 `run_extratrees_ldq_hpo.py` 保留为历史配置搜索记录。正式流程为：

1. 将 train + primary 组成 83-cell development；
2. 固定外层 5 折×2，用于估计完整管线的泛化误差；
3. 在每个外层训练折内部执行 3 折选择，重新拟合插补、Local DeltaQ 选择器和候选 ExtraTrees 配置；
4. 用预声明分数选择方案：

\[
J=\overline{\mathrm{MAE}}+0.1\,\mathrm{SD}(\mathrm{MAE})
+0.1\,\mathrm{MAE}_{\mathrm{worst}}+\lambda C,
\]

其中 \(C\) 是预先定义的复杂度项；
5. 只在外层验证折评价，汇总 10 个外层折；
6. 根据 development 证据写入不可变 `frozen_config.json`；
7. 用全部 83 个 development 电池重训，在 30 个预指定种子下执行 secondary 批次评价。

所有选择、调参和拟合均不读取 secondary 指标或标签。secondary 始终与正式模型开发隔离，仅在完整流程冻结后用于批次评价。

---

## 7. 实验设计：每一层验证到底回答什么

### 7.1 六层证据结构

本项目同时使用多种分层、重复与重采样方法。它们回答不同问题，不能互相替代：

| 方法 | 重复对象 | 回答的问题 |
|---|---|---|
| 内层 3 折 | 当前外层训练电池 | 哪个选择器/配置可以进入该外层折？ |
| 外层 5 折×2 | 83 个 development 电池 | 完整选择管线在未参与选择的开发电池上表现如何？ |
| 30 个固定 seeds | 同一冻结数据与配置 | 结果是否依赖幸运的模型随机种子？ |
| development OOF residuals | 外层未见电池 | 在不使用批次标签时如何构造经验残差区间？ |
| secondary cell bootstrap | 40 个批次评价电池 | 当前有限批次样本下，指标抽样不确定性多大？ |
| 特征集合 Jaccard/区域计数 | 10 个外层折 | 选择器在训练样本扰动下是否稳定？ |

“均值 ± 标准差”必须写清重复对象：开发表中的标准差来自 10 个外层折，冻结批次表中的标准差来自 30 个模型种子。cell-bootstrap 95% CI 描述 40 个电池的样本不确定性，不能与种子波动互换。

### 7.2 正式对照如何组织

当前正式证据不再依赖一条“XGBoost 更换表征再更换读出器”的单线故事，而是拆成三组公平对照：

1. **选择器对照**：Pearson、bootstrap-stable、region-balanced、contiguous、mRMR 使用相同外层折、内层折、top40 容量和 ExtraTrees 候选；
2. **特征组对照**：raw、global DeltaQ、Local DeltaQ 及其七种组合，在相同外层折内重新选择 Local 特征；
3. **读出器对照**：Ridge、RBF-SVR、RandomForest、XGBoost、compact ExtraTrees 使用相同折和完整 fold-selected 表征。

历史 development-mean、XGBoost、深度适配模型和 top-k 表仍保留用于来源追踪，但不能覆盖上述严格嵌套证据。

### 7.3 横向模型比较

严格 development 外层折的代表性读出器结果为：

| 读出器 | MAE (cycles) | 解释 |
|---|---:|---|
| formal ExtraTrees 主配置 | **43.01 ± 5.43** | 嵌套主实验 |
| compact ExtraTrees | 46.24 ± 10.01 | 消融脚本中的紧凑对照 |
| RandomForest | 51.45 ± 13.06 | 次优树集成 |
| XGBoost | 53.73 ± 13.03 | 统一折内对照 |
| RBF-SVR | 103.29 | 明显较差 |
| RidgeCV | 690.20，方差极大 | 共线小样本下不稳定 |

这只能说明当前特征、样本量和搜索边界下 ExtraTrees 最稳健，不能推出其他算法在一般电池任务上无效。

### 7.4 历史 top-k 消融的正确地位

top20/40/60/80 的 secondary 结果仅作为冻结后的回顾性敏感性对照。当前 region-balanced top40 主模型由 development 内的严格嵌套比较确定，并非根据这些 secondary 数字选出。

### 7.5 区域稳定性、必要性与替代性

严格区域实验不只统计“是否反复入选”，还在相同 development 外层折中比较：全区域 Pearson、只保留 0.525–0.650 核心区、排除核心区、从已选集合移除核心区、等宽随机区域、随机 40 特征和核心区置乱。

| 区域/控制 | MAE (cycles) |
|---|---:|
| 全区域 Pearson top40 | 42.49 ± 8.25 |
| 只保留核心区 | 43.69 ± 11.70 |
| 排除核心区 | 43.82 ± 11.25 |
| 已选集合移除核心区 | 42.70 ± 9.47 |
| 随机 40 特征（多次重复均值） | 约 46.31 |

核心区置乱使 MAE 升至约 98 cycles，说明已训练模型依赖这组相关特征；但核心区只保留、排除或移除后的模型仍接近全区域结果，说明邻近区域和描述符具有强替代性。置乱是分布外依赖诊断，不是因果干预，也不能证明唯一电化学机制。

### 7.6 为什么最终采用 region-balanced top40

N150 严格嵌套结果如下：

| 选择器 | 外层 MAE | 选择分数 | 特征 Jaccard |
|---|---:|---:|---:|
| **region-balanced** | **43.01 ± 5.43** | **49.097** | 0.530 |
| bootstrap-stable | 43.29 ± 5.83 | 49.151 | 0.794 |
| contiguous band | 43.14 ± 6.08 | 49.359 | 0.603 |
| Pearson top40 | 44.08 ± 7.47 | 50.758 | 0.616 |
| mRMR | 44.27 ± 6.92 | 50.960 | 0.786 |

region-balanced 的平均 MAE、折间波动和预声明分数均最低，但它与 bootstrap-stable 的配对误差差异仍不显著。因此应写成“按预声明规则冻结”，不能写成 region-balanced 显著优于所有方法。连续带没有稳定胜出，是正式负结果。

### 7.7 随机划分为什么只作历史诊断

混合三批数据后随机划分曾得到明显更高的 \(R^2\)。该现象说明验证协议可能比结构变化更显著地影响表面结果。随机混合结果只能用于协议敏感性说明，不能替代 development 严格嵌套评价和 protocol-frozen secondary-batch 评价。

---

## 8. 评价指标及其含义

设被评价电池数为 \(n\)，真实 knee 为 \(y_i\)，预测为 \(\hat y_i\)。

### 8.1 MAE

\[
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^n|y_i-\hat y_i|.
\]

单位是 cycles，最直观地表示平均偏差多少个循环。MAE 对极端误差的敏感度低于 RMSE。

### 8.2 RMSE

\[
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-\hat y_i)^2}.
\]

平方项放大大误差。当 RMSE 明显高于 MAE 时，通常说明少数困难电池形成长尾误差。

### 8.3 MAPE

\[
\mathrm{MAPE}=\frac{100\%}{n}\sum_{i=1}^n
\left|\frac{y_i-\hat y_i}{y_i}\right|.
\]

它便于比较不同 knee 尺度，但会对很小的真实值敏感。当前 knee 均为正且不接近零，因此可正常使用。

### 8.4 决定系数 \(R^2\)

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}.
\]

- \(R^2=1\)：完美预测；
- \(R^2=0\)：与直接预测测试集均值相当；
- \(R^2<0\)：比测试集均值基准还差。

历史 constant baseline 的负 \(R^2\) 表明开发集均值无法覆盖 secondary batch 的后移分布。

---

## 9. 正式结果如何解读

### 9.1 严格 development 主结果

在 83 块 development 电池上，外层 5 折×2 的正式选择结果为：

| 选择器 | MAE (cycles) | RMSE (cycles) | q90 error | 预声明分数 |
|---|---:|---:|---:|---:|
| **region-balanced** | **43.01 ± 5.43** | **65.18** | **87.84** | **49.097** |
| bootstrap-stable | 43.29 ± 5.83 | 65.57 | 89.85 | 49.151 |
| contiguous band | 43.14 ± 6.08 | 65.83 | 86.47 | 49.359 |
| Pearson top40 | 44.08 ± 7.47 | 66.90 | 90.55 | 50.758 |
| mRMR | 44.27 ± 6.92 | 67.47 | 91.80 | 50.960 |

region-balanced 的 MAE 数值和预声明分数均最低，但它与 bootstrap-stable 的配对差异很小且置信区间跨零，因此论文仍不宣称选择器间存在显著优势。

### 9.2 protocol-frozen secondary-batch 结果

冻结 region-balanced top40 与 formal ExtraTrees 后，在 40 块 secondary 电池上执行 30 个预指定种子：

| 模型/状态 | MAE | RMSE | MAPE | \(R^2\) |
|---|---:|---:|---:|---:|
| 历史 Pearson top40，按新标签重计 | 77.31 ± 0.90 | 111.09 ± 0.94 | 9.07 ± 0.12% | 0.794 ± 0.003 |
| **冻结 region-balanced top40** | **76.59 ± 0.75** | **113.93 ± 0.91** | **9.00 ± 0.10%** | **0.783 ± 0.003** |

新模型 MAE 略低，但 RMSE 和 \(R^2\) 略差。按 30-seed 平均预测计算，新减旧的单电池绝对误差差为 \(-0.69\) cycles，95% CI 为 \([-4.09,3.31]\)，符号翻转检验 \(p=0.753\)。正确结论是“与历史锚点相当”，而不是“显著优于历史模型”。

### 9.3 特征组贡献

七组严格折内消融的 MAE 为：

| 特征组 | MAE (cycles) |
|---|---:|
| raw only | 92.67 ± 21.58 |
| global DeltaQ only | 59.46 ± 15.21 |
| Local DeltaQ only | 52.85 ± 9.83 |
| raw + global | 43.00 ± 9.46 |
| raw + local | 42.73 ± 5.69 |
| global + local | 53.65 ± 9.89 |
| full: raw + global + local | 43.17 ± 5.43 |

Local-only 明显优于 raw-only，但完整表征并没有在平均 MAE 上清晰优于 raw+global 或 raw+local。证据支持“Local DeltaQ 是一种紧凑、可替代的局部表征”，不支持“它提供了不可替代的独立增益”。

### 9.4 观测预算与数据效率

在同一 N150 合格电池队列中进行公平比较：

| 观测窗口 | 胜出选择器 | development MAE |
|---:|---|---:|
| N50 | mRMR | 59.71 ± 21.85 |
| N100 | mRMR | 44.85 ± 6.13 |
| N150 | region-balanced | 43.01 ± 5.43 |

重复 development 子采样学习曲线从 25% 到 100% 的 MAE 依次为 63.45、52.45、45.78 和 43.20 cycles。更多早期循环和更多 development 电池均持续提供价值；N50 和 N100 在匹配 5 折×2 实验中均由 mRMR 按预声明分数胜出。

### 9.5 尾部误差、置信区间和校准

30-seed 平均预测的 cell-bootstrap 95% CI 为：

- MAE：51.83–103.24 cycles；
- RMSE：75.22–148.25 cycles；
- MAPE：6.35–12.10%；
- \(R^2\)：0.601–0.887。

secondary 单电池绝对误差的中位数为 43.82 cycles，q90 为 190.61，q95 为 265.12，最大值为 376.17。名义 90% 的 development-calibrated 残差区间在 secondary 上只覆盖 77.5%，其中晚膝点组只覆盖 64.3%。因此“种子波动很小”不等于“跨批次不确定性已校准”。

### 9.6 结果支持什么、不支持什么

当前证据支持：

- 前 50/100/150 cycles 均包含 knee 预测信息，150 cycles 最稳定；
- 严格嵌套 region-balanced LDQ-ET 在 development 上达到约 42-cycle MAE；
- 冻结模型在 protocol-frozen secondary batch 上维持约 76–77-cycle MAE，与历史锚点相当；
- ExtraTrees 是当前小样本表格场景中最稳健的已测读出器；
- 局部区域信息是分布式且可替代的，单一区域并非必要或充分；
- 批次偏移主要表现为长尾误差和区间欠覆盖，而不是模型种子不稳定。

当前证据不支持：

- region-balanced 显著优于 bootstrap-stable 或历史 Pearson top40；
- Local DeltaQ 相对 raw+global 存在清晰、独立且不可替代的平均误差增益；
- 0.525–0.650 对应唯一确定的电化学机制；
- secondary 是从未触碰的外部确认集；
- 方法已泛化到其他化学体系、温度、实验室或在线 BMS；
- ExtraTrees、区间选择或 conformal 方法本身是本项目的新算法。

---

## 10. 从代码角度复现整个流程

### 10.1 目录地图

```text
Besttery/
├─ data/                  原始/处理后数据、标签和特征表
├─ scripts/               数据处理、训练、消融、统计和绘图代码
├─ results/               正式与历史实验输出
├─ manuscript_ionics/     当前论文 LaTeX、参考文献和 PDF
├─ docs/                  设计说明、审计记录和本教程
├─ paper/                 本地参考文献与阅读材料
└─ archive/               已退出主线但保留审计价值的探索代码/文档
```

### 10.2 数据与标签阶段

| 阶段 | 脚本 | 主要输出 |
|---|---|---|
| 原始 MAT 转换 | `scripts/build_severson_pkl_from_mat.py` | 标准化电池级 PKL 数据 |
| SOH 校正 | `scripts/correct_soh_curves.py` | `processed_cycles_soh_corrected_20170630.csv` |
| knee 检测 | `scripts/detect_knee_points.py` | `knee_labels_20170630.csv` |
| 标签稳定性 | `scripts/analyze_knee_label_stability.py` | `results/knee_label_stability/` |

### 10.3 特征阶段

| 阶段 | 脚本 | 主要输出 |
|---|---|---|
| 前 150 cycles 基础特征 | `scripts/extract_early_knee_features.py` | `early_features_N150_20170630.csv` |
| 全局 DeltaQ 扩展 | 对应 DeltaQ 提取脚本 | `early_features_N150_extended_20170630.csv` |
| Local DeltaQ 候选 | `scripts/extract_local_deltaq_features.py` | 365 个候选 local 特征 |
| 历史 Pearson top-k | `scripts/select_local_deltaq_features.py` | top20/40/60/80 特征表与 ranking；仅作锚点来源 |
| 严格折内选择 | `scripts/run_nested_contiguous_ldq.py` | 每折动态生成 top40，不读取预选后的 secondary 结果 |

候选池读取：

```text
data/early_features_N150_local_deltaq_20170630.csv
```

`early_features_N150_local_deltaq_top40_20170630.csv` 只用于历史 Pearson top40 精确复现；严格主实验必须从完整 365-candidate 表开始，并在训练折内选特征。

### 10.4 训练与评价阶段

核心命令：

```powershell
# 1. N150 严格嵌套开发、冻结、批次评价与离线汇总
python scripts/run_nested_contiguous_ldq.py --phase develop
python scripts/run_nested_contiguous_ldq.py --phase freeze
python scripts/run_nested_contiguous_ldq.py --phase external
python scripts/run_nested_contiguous_ldq.py --phase finalize

# 2. 特征组、模型、区域控制与学习曲线
python scripts/run_nested_ablation_learning.py

# 3. 论文图与机器验证
python scripts/prepare_enhanced_manuscript_figures.py
python scripts/validate_enhancement_outputs.py
```

主要结果目录：

```text
results/reproduction_ldq_et_N150_20260713/
results/nested_contiguous_ldq_N150_20260715_nonoverlap/
results/nested_contiguous_ldq_N50_commonN150_5x2_20260715_nonoverlap/
results/nested_contiguous_ldq_N100_commonN150_5x2_20260715_nonoverlap/
results/nested_ablation_learning_N150_20260715_nonoverlap/
results/enhancement_validation_20260715_nonoverlap/
```

### 10.5 论文编译

```powershell
Set-Location manuscript_ionics
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

论文只维护一个正文源文件 `manuscript_ionics/main.tex`，输出为 `main.pdf`。标签拟合改善分布、全局形状统计公式、完整 14 项 matrix-profile 描述符，以及22个 raw/SOH 特征的分组定义均已纳入正文，不再生成或引用独立补充材料。最终 PDF 需通过编译、日志扫描和页面渲染检查。

### 10.6 最小伪代码

```python
# 1. 标签：用完整寿命的单调 SOH 生成 knee cycle
for cell in batteries:
    soh = smooth_and_monotonize(cell.capacity_curve)
    y[cell] = best_two_segment_breakpoint(soh)

# 2. 输入：只读取前 150 cycles
for cell in batteries:
    raw = early_capacity_soh_features(cell, end_cycle=150)
    delta_q = q_curve(cell, cycle=149) - q_curve(cell, cycle=9)
    global_dq = global_statistics(delta_q)
    local_candidates = overlapping_region_statistics(delta_q)

# 3. 严格嵌套：每个外层训练折内重新插补、选 Local 特征和选模型
development = official_train + primary_test
for outer_train, outer_valid in repeated_five_fold(development, repeats=2):
    selector, params = inner_three_fold_select(
        raw_22, global_dq_8, local_candidates,
        y, cells=outer_train,
        candidates=["pearson", "bootstrap_stable", "region_balanced", "contiguous", "mrmr"],
    )
    top40 = selector.fit(outer_train).transform(outer_train, outer_valid)
    model = ExtraTrees(**params).fit(top40.X_train, y[outer_train])
    save_outer_predictions(outer_valid, model.predict(top40.X_valid))

# 4. 按预声明 development 分数冻结；不读取 secondary 指标
frozen = freeze_best_development_pipeline()

# 5. 全 development 重训，secondary 仅作协议控制批次评价
X_dev, X_secondary = frozen.fit_transform(development, secondary_test)
model = ExtraTrees(**frozen.params).fit(X_dev, y[development])
pred = model.predict(X_secondary)
report_metrics(y[secondary_test], pred)
```

---

## 11. 理论层面的理解

### 11.1 这是一个小样本、高维、分布偏移问题

开发样本只有 83，输入为 70 维，且 chronology-later secondary 批次标签分布后移。主要风险是：

- 维数相对样本数偏高；
- 特征筛选容易受个别电池影响；
- 复杂模型容易记忆开发批次；
- 随机混合验证掩盖批次偏移；
- 后续批次样本少导致指标置信区间较宽。

当前设计通过折内 top40、region cap、浅层 ExtraTrees、严格嵌套开发评价、冻结后批次压力测试、重复种子和 cell bootstrap 分别缓解这些风险。它不能消除单数据集与历史 secondary 反馈造成的证据边界。

### 11.2 Local DeltaQ 可理解为带空间先验的字典

全局统计假设“位置不重要”，而 Local DeltaQ 明确保留位置。每个重叠区域和统计量组合可视为一个弱基函数：

\[
\phi_{j,m}(\Delta Q)=g_m(\Delta Q|_{R_j}),
\]

其中 \(R_j\) 是局部区域，\(g_m\) 是 mean、range、slope 等算子。整个 Local DeltaQ 表征是：

\[
\Phi(\Delta Q)=\{\phi_{j,m}(\Delta Q)\}_{j,m}.
\]

特征筛选相当于从这个局部字典中按目标相关性、稳定性、区域覆盖或冗余约束选择 40 个基函数。ExtraTrees 再学习这些基函数及 raw/global 特征间的非线性交互。

### 11.3 偏差—方差视角

- 只用全局统计：维度低、方差小，但可能有较大表征偏差；
- 使用全部 365 local 特征：表达力强，但小样本估计方差大；
- top40：试图降低表征偏差，同时控制维度；
- 深度 6 的 ExtraTrees：保留非线性能力，同时限制单树复杂度；
- 1200 棵树平均：降低模型随机方差。

历史 top40–top60 平台、严格实验中多种选择器接近以及随机 40 特征略差，都与这一偏差—方差权衡相符，但不能据此证明某个固定 top-k 在未来数据上最优。

### 11.4 为什么仍保留 Pearson，同时引入区域平衡

绝对 Pearson 相关只衡量单变量线性关联，理论上会漏掉纯非线性或条件交互特征。但在小样本训练折上，复杂 wrapper selection 更容易过拟合。简单相关筛选的优势是：

- 参数少；
- 计算透明；
- 可重复审计；
- 易于 bootstrap；
- 将复杂的非线性交互留给 ExtraTrees。

它是小样本条件下的重要透明锚点，不是声称 Pearson 是最强选择算法。region-balanced 在 Pearson 类评分之上增加“每个区域最多 3 个描述符”的覆盖约束，降低单一相关特征簇占满 top40 的风险；严格结果显示这种约束按预声明综合分数胜出，但优势很小。

---

## 12. 常见误解与排查清单

### 误解 1：完整寿命轨迹被用作模型输入

错误。完整轨迹只生成 knee 标签；输入特征只来自前 150 cycles。

### 误解 2：每个循环是一个训练样本

错误。每块电池是一个样本，循环是形成该电池早期表征的序列观测。

### 误解 3：top40 是在全部 123 块电池上选的

错误。历史 Pearson top40 只使用 41 块 official train；当前严格实验在每个训练折内动态选择 top40，冻结配置才使用全部 83 块 development 电池。40 块 secondary 从不参与新选择。

### 误解 4：secondary batch 参与了 HPO

错误。当前严格嵌套流程的特征选择、模型比较、超参数确定和参数拟合只使用 83-cell development；secondary 不参与任何正式模型开发决策。

### 误解 5：30 seeds 相当于 30 个独立外部数据集

错误。它们是同一数据上的 30 次随机模型训练，只测算法随机性。

### 误解 6：稳定区域就是已证实的物理反应区域

错误。它是归一化放电坐标上的统计敏感带，目前缺少机理隔离证据。

### 误解 7：历史 top60 MAE 更低，所以严格主模型应改成 top60

不成立。top60 仅属于回顾性敏感性结果；当前严格比较固定 top40 容量，并按 development 内综合分数选择 region-balanced。若要重新决定 top-k，仍必须仅在 development 中嵌套选择。

### 误解 8：历史结果更高就应该替换主模型

不一定。Dual-view、MoE、factorized 等结果不属于当前严格嵌套主线。任何候选方法若要替换主模型，必须在 development 内完成选择和配置冻结，再使用 secondary 进行一次性批次评价。

### 误解 9：名义 90% 区间已经达到部署级可靠性

错误。该经验残差区间在 secondary 上只覆盖 77.5%，晚膝点组只有 64.3%。它是批次偏移诊断，不是满足交换性条件的精确 split-conformal 保证。

---

## 13. 当前方法的创新叙述应该如何表达

最稳妥的创新主张不是“发明了 ExtraTrees”“首次使用局部窗口”或“首次做区间选择”，而是一个面向特定任务的方法与审计组合：

1. **问题层面**：将早期绝对 knee-cycle 预测明确设计为 chronology-aware 的开发/后续批次问题，而不是混合随机划分；
2. **表征层面**：把全局 DeltaQ 展开为重叠局部描述符字典，为 knee 任务提供有位置感的紧凑表示；
3. **学习层面**：把插补、五类选择器、top40 和 ExtraTrees 全部放入嵌套训练折，并按预声明分数冻结 region-balanced 配置；
4. **验证层面**：同时报告 observation budget、数据效率、特征组、模型、区域替代、尾部误差和区间欠覆盖；
5. **解释层面**：把“相关、稳定、预测依赖、必要性和机制”严格区分，主动报告局部增益不清晰与连续带未胜出的负结果。

一句话版本：

> 本项目构建并严格审计了一种面向小样本早期 knee-cycle 预测的 Local DeltaQ 区域表征；在只使用前 150 cycles 时，严格嵌套 region-balanced LDQ-ET 在 development 上稳定工作，并在历史已暴露的后续批次上保持与旧锚点相当的点预测性能，但尚未证明独立局部增益、唯一机理或部署级跨批次校准。

---

## 14. 仍需补强的实验与未来路线

N50/N100/N150、25/50/75/100% 数据效率、区域只保留/排除/移除/随机控制、五类读出器和经验区间已经完成，不再列为“待做”。若目标是达到更高水平发表要求，下一阶段优先级如下。

### 第一优先级：新的真正未触碰外部数据

当前 secondary batch 已被用于多轮架构比较。最有价值的下一步不是继续在同一批数据上堆结构，而是冻结：

- cycle 9/149 DeltaQ 定义；
- 50/25 局部窗口；
- region-balanced top40 规则与每区最多 3 个描述符；
- 70 维输入；
- ExtraTrees 超参数；
- 评价指标。

然后在新的化学体系、实验室、充电协议或新批次上一次性验证。

### 第二优先级：跨批次不确定性重新校准

当前固定宽度经验残差区间明显欠覆盖。下一步应在独立校准批次上比较：

- split conformal、加权 conformal 与 group/batch-aware conformal；
- 随 knee 尺度或退化阶段变化的异方差区间；
- coverage、平均宽度、late-knee 条件覆盖和 coverage-width trade-off；
- 预先定义的拒绝预测或低置信度告警策略。

不能继续在现有 secondary 标签上调整区间后再把同一批次称为外部验证。

### 第三优先级：多标签定义下的完整重训

现有 piecewise 与 Kneedle 排序接近，但绝对位置平均相差 32.85 cycles。下一步应针对多种预声明 knee 定义重复完整嵌套训练，而不是只比较离线标签：

- piecewise-linear；
- Kneedle；
- derivative/curvature threshold；
- 相对 knee 位置或带标签不确定性的目标。

这样可以区分“模型误差”与“目标定义误差”。

### 第四优先级：从预测依赖走向机理验证

当前区域控制已经表明核心带可被邻近区域替代。若要提出物理解释，需要引入独立电化学证据，例如 DVA/ICA、阻抗、析锂诊断或拆解结果，并预先定义区域—机理对应关系。仅靠特征重要性、重入选频率或置乱误差不能完成这一步。

---

## 15. 给后续 AI 的推荐提问方式

把本文件提供给 AI 后，可按以下顺序学习。

### 第一轮：建立全局认识

```text
请只根据这份项目指南，用高中数学水平解释：
1. 项目预测什么；
2. 为什么只用前150个循环；
3. DeltaQ和Local DeltaQ分别是什么；
4. 为什么 secondary 仅在模型冻结后用于 protocol-frozen 批次评价。
不要引入文档外的新结果。
```

### 第二轮：理解算法细节

```text
请从公式开始，举一个只有10个网格点的玩具例子，演示：
DeltaQ计算、重叠窗口切分、局部统计量、top-k相关筛选、ExtraTrees预测。
明确玩具数值不是真实项目结果。
```

### 第三轮：理解实验设计

```text
请比较内层3折、外层5折×2、30个random seeds、secondary cell bootstrap、
development OOF残差区间和特征集合Jaccard。分别说明重采样对象、用途和不能回答的问题。
```

### 第四轮：审稿人式自检

```text
请作为严格审稿人，只根据本指南列出：
哪些结论有直接实验支持，哪些只是合理解释，哪些目前不能声称。
同时检查是否存在数据泄漏或测试集反馈风险。
```

### 第五轮：代码导读

```text
请按数据流为我制定代码阅读顺序，每次只讲一个脚本。
讲解时包含：输入、输出、核心函数、关键参数、潜在泄漏点、
以及它对应论文的哪一部分。先从correct_soh_curves.py开始。
```

### AI 使用约束

要求 AI 始终遵守：

- 以现行脚本和结果文件为事实来源；
- 不把历史探索模型说成当前主模型；
- 不把统计敏感区说成已证明的电化学机制；
- 不混淆种子标准差和样本 bootstrap 置信区间；
- 不用 secondary batch 的结果反向建议已“无偏验证”的新结构；
- 若文档与代码冲突，应指出冲突并优先核查代码和 metadata，而不是自行补全。

---

## 16. 最终记忆卡片

```text
任务：前150 cycles -> 预测未来绝对 knee cycle
数据：Severson/MIT-Stanford corrected 2017-06-30
样本：41 train + 42 primary = 83 development；40 protocol-frozen secondary-batch evaluation
主模型：严格嵌套 region-balanced LDQ-ET top40
输入：22 raw/SOH + 8 global DeltaQ + 40 fold-selected Local DeltaQ = 70
Local候选：365；选择、插补、配置和模型全部在训练折内拟合
回归器：1200-tree ExtraTrees，max_features=0.5，depth=6
开发结果：5-fold×2 MAE 43.01±5.43；contiguous 机制未稳定胜出
批次结果：MAE 76.59±0.75；RMSE 113.93±0.91；
          MAPE 9.00±0.10%；R² 0.783±0.003
配对结论：相对历史 top40 的 MAE 差 -0.69 cycles，CI跨零，p=0.753
区域结论：0.525–0.650 是相关特征簇，但存在强替代性，不是唯一机制
不确定性：名义90%区间只覆盖77.5%，晚膝组64.3%
最大边界：单一快充数据集；仍需真正独立的多来源外部数据
```

如果只能记住一句话：

> 严格嵌套后，区域平衡 LDQ-ET 仍能维持约76--77 cycles的批次点预测，但局部区域增量、单一机制解释和跨批次区间校准都必须保持克制。

---

## 17. 2026-07-13 严格增强实验的实现入口

### 17.1 严格嵌套选择与冻结

`scripts/run_nested_contiguous_ldq.py` 支持四个阶段：

1. `develop`：在83个开发电芯上执行外层5折×2、内层3折的选择；
2. `freeze`：只使用完整开发集再次交叉验证并写入 `frozen_config.json`；
3. `external`：按冻结的30个种子执行 secondary 批次预测；
4. `finalize`：只从已保存预测恢复统计后处理，不重新打开特征表或重训模型。

候选选择器为 Pearson、bootstrap-stable、region-balanced、contiguous 和 mRMR。每个外层训练折都重新拟合特征选择与中位数插补。主选择分数为平均 MAE、折间标准差、最差折和预先设定复杂度惩罚之和。

### 17.2 公平输入预算

N50、N100、N150 均通过 `--cohort-path data/early_features_N150_local_deltaq_20170630.csv` 限定到同一123-cell队列，避免早膝电芯随窗口变化造成样本组成混杂。预算实验只读取 development 的83个电芯，不对 N50/N100 单独运行 secondary。

### 17.3 消融与数据效率

`scripts/run_nested_ablation_learning.py` 复用冻结选择器，并输出：

- 七个特征组；
- Ridge、SVR、RandomForest、XGBoost 和 compact ExtraTrees；
- 全区域、核心区、排除核心区、已选集合移除核心区、等宽随机区间、随机特征；
- 核心区置乱依赖诊断；
- 25%、50%、75%、100% 分层重复子采样学习曲线。

所有上述结果只用 development outer-valid 评价。置乱结果属于分布外依赖诊断，不能写成因果干预。

### 17.4 图表与验证

`scripts/prepare_enhanced_manuscript_figures.py` 从正式 CSV/JSON 生成7张核心图，每张提供 SVG、PDF、600-dpi PNG 和源 CSV。`scripts/validate_enhancement_outputs.py` 对折数、OOF覆盖、冻结哈希、30×40预测、区间覆盖、公共队列、消融完整性和图表导出执行机器检查。

当前机器验证输出为：

```text
development_selected_method = region_balanced
external_mae_seed_mean      = 76.5871179140
external_mae_seed_std       = 0.7542058330
external_interval_coverage  = 0.775
```

验证报告位于 `results/enhancement_validation_20260715_nonoverlap/validation_report.md`。7 张增强图位于 `manuscript_ionics/figures/`，19 份绘图源 CSV 位于 `manuscript_ionics/figure_source_data/`，视觉检查记录位于 `manuscript_ionics/figure_qa/figure_qa_report.md`。

### 17.5 论文、LaTeX 与 PDF 交付

- 论文源文件：`manuscript_ionics/main.tex`；
- 参考文献：`manuscript_ionics/references.bib`；
- 引用核验：`manuscript_ionics/reference_verification.md`；
- 最终审稿审计：`manuscript_ionics/review_report.md`；
- 编译 PDF：`manuscript_ionics/main.pdf`；
- 稳定交付副本：`output/pdf/Besttery_LDQ_ET_strict_nested_manuscript_20260713.pdf`；
- PDF QA：`manuscript_ionics/pdf_qa_report.md`。

最终 PDF 为 20 页 A4，已通过 `latexmk`、BibTeX、日志扫描、Poppler 全页渲染、20 页 contact sheet 和关键页全分辨率检查。源 PDF 与交付副本 SHA-256 相同。

### 17.6 结果目录与证据角色

| 目录 | 角色 |
|---|---|
| `results/reproduction_ldq_et_N150_20260713/` | 历史 Pearson top40 精确复现 |
| `results/nested_contiguous_ldq_N150_20260715_nonoverlap/` | N150 非重叠标签下的严格开发、冻结配置、30-seed 批次评价 |
| `results/nested_contiguous_ldq_N50_commonN150_5x2_20260715_nonoverlap/` | 公共队列 N50 严格预算 |
| `results/nested_contiguous_ldq_N100_commonN150_5x2_20260715_nonoverlap/` | 公共队列 N100 严格预算 |
| `results/nested_ablation_learning_N150_20260715_nonoverlap/` | 特征组、模型、区域和学习曲线 |
| `results/enhancement_validation_20260715_nonoverlap/` | 自动完整性检查 |

任何新结论都应先对应到上述正式目录中的 CSV/JSON，再写入论文或汇报；历史探索目录不得覆盖正式结果。

---

## 18. 本次更新摘要

2026-07-13 的同步已经完成以下口径统一：

1. 把历史 Pearson top40 与当前严格嵌套 region-balanced top40 分开；
2. 明确 secondary 与正式模型开发完全隔离，仅用于 protocol-frozen 后续批次评价；
3. 写入 N50/N100/N150、数据效率、特征组、读出器、区域控制、尾部误差和区间欠覆盖结果；
4. 将 contiguous 未稳定胜出、Local DeltaQ 独立增益不清晰和区域强替代性明确列为负结果；
5. 同步正式脚本、结果目录、7 图/19 CSV、LaTeX 编译和 20 页 PDF QA 状态；
6. 将未来工作收敛到真正新外部数据、跨批次校准、多标签完整重训和独立机理验证。

本指南、`manuscript_ionics/main.tex`、`docs/EXPERIMENT_REGISTRY_20260713.md` 与 `docs/ENHANCED_RESULTS_REPORT_20260713.md` 现采用同一主线口径。
