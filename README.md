# Battery Knee-Cycle Prediction with Local DeltaQ

本项目研究锂离子电池容量退化拐点（capacity knee point）的早期预测。当前论文只保留一个正式主模型：**严格嵌套、区域平衡的 LDQ-ET**，即 Local DeltaQ 表征与 Extremely Randomized Trees（ExtraTrees）读出器。Dual-view、Soft-MoE、寿命辅助多任务和其他融合架构均属于探索性实验，不是稿件主模型。

## 当前论文主张

早期循环中的 knee 相关信息在重叠 Local DeltaQ 区域中具有高度相关和可替代的表征。新协议在每个训练折内重做特征选择、缺失值处理和模型选择，并在开发集证据冻结后才执行新的 secondary 批次预测。

当前证据支持以下结论：

- region-balanced 与 bootstrap-stable 在严格嵌套开发评估中几乎并列，连续带机制没有稳定胜出；
- ExtraTrees 提供了当前小样本高共线表征下最稳健的代表读出；
- 0.525--0.650 区域被模型依赖，但核心区单独、移除核心区和其他区域均可给出相近误差，说明存在大量替代信息；
- 完整 raw+global+local 与 raw+global 在开发折上的平均 MAE 基本相同，因此不能再声称 Local DeltaQ 有明确独立增益；
- 这些区域是预测关联和依赖诊断，不是唯一电化学机理的证明；
- 结论限定于 Severson/MIT-Stanford 快充数据；secondary 已被历史探索查看，新的结果称为 protocol-frozen, historically informed secondary-batch evaluation。

## 数据与任务

- 数据：Severson/MIT-Stanford corrected 2017-06-30 release；
- 原始规模：124 cells；
- 输入：每个电池前 150 cycles；
- 输出：由完整退化轨迹离线生成的绝对 knee cycle；
- 开发集：official train 41 cells + primary test 42 cells；
- 批次评价：secondary test 40 cells；
- secondary 标签不用于本轮严格嵌套特征选择、超参数选择或种子选择，但历史 top-k/架构探索已经查看过该批次。

完整寿命轨迹仅用于生成监督标签。模型推理只使用前150个循环内的信息。

## 唯一主模型：nested region-balanced LDQ-ET top40

最终输入共70个描述符：

- 22个 raw capacity/SOH 描述符；
- 8个 global DeltaQ 描述符；
- 40个从365个候选 Local DeltaQ 描述符中选出的局部描述符。

最终 ExtraTrees 配置：

- `n_estimators=1200`；
- `max_features=0.5`；
- `max_depth=6`；
- `min_samples_leaf=1`；
- `min_samples_split=2`；
- `bootstrap=False`。

严格嵌套 5-fold × 2 的开发结果：region-balanced MAE 为 `43.01 +/- 5.43` cycles，bootstrap-stable 为 `43.29 +/- 5.83`，contiguous 为 `43.14 +/- 6.08`。按预先固定的“均值 + 波动 + 最差折 + 复杂度”分数选择 region-balanced；完整开发集最终冻结 `max_per_region=3`，40个局部描述符分布于13个空间区域和1个形态组。

新的 secondary batch 30随机种子结果：

| Model | MAE (cycles) | RMSE (cycles) | MAPE | R2 | Status |
|---|---:|---:|---:|---:|---|
| 历史 LDQ-ET top40 精确复现 | 77.23 +/- 0.89 | 111.07 +/- 0.93 | 9.04 +/- 0.12% | 0.794 +/- 0.003 | historical anchor |
| **冻结 region-balanced LDQ-ET** | **76.59 +/- 0.75** | **113.93 +/- 0.91** | **9.00 +/- 0.10%** | **0.783 +/- 0.003** | protocol-frozen, historically informed |

对40个 secondary 电池的平均预测做 cell bootstrap，MAE 为76.41 cycles，95% CI 为51.83--103.24。相对历史 anchor 的逐电芯 MAE 差为 -0.69 cycles（95% CI -4.09--3.31；sign-flip p=0.753），即统计上持平而非显著提升。开发集校准的名义90%残差区间在 secondary 上仅覆盖77.5%，晚膝组覆盖64.3%。

## 区域稳定性与贡献边界

区域稳定性分析只使用41个official training cells：

- 候选 Local DeltaQ 描述符：365；
- bootstrap：1000次；
- top40集合平均Jaccard：0.599；
- top40描述符平均入选频率：0.719；
- 0.525--0.650核心区域的最高描述符入选频率：0.921--0.953。

新严格嵌套结果显示，bootstrap-stable 的特征集 Jaccard 最高（0.792），region-balanced 较低（0.545），但两者开发误差无统计可分的差异。核心区置乱会显著恶化预测，但核心区单独、排除核心区、以及从已选集合中移除核心区都保持约43--44 cycles MAE。这说明模型依赖一组相关局部特征，但不能把单一区域写成不可替代的机制。

## 正式证据与探索性结果的边界

以下结果可用于方法探索或补充讨论，但不得替代LDQ-ET主结果：

- LDQ dual-view 1:1：外部MAE约74.44 cycles；
- 开发集内权重搜索得到0.40/0.60融合：外部MAE约74.65 cycles；
- factorized-knee：外部MAE约72.00 cycles，但尾部误差和RMSE更差；
- Soft-MoE：相对LDQ-ET只有小幅MAE改善，且模式路由缺乏足够样本支撑；
- cycle-life多任务ExtraTrees：MAE未优于单任务LDQ-ET。

由于secondary batch已经被用于比较这些探索架构，它们不能被描述为在完全未触碰外部数据上确认的新主模型。

## 目录结构

```text
data/                 原始数据、处理后标签和特征表
docs/                 实验设计、复现记录和证据审查
manuscript_ionics/    Ionics/Springer LaTeX稿件、图、参考文献和审稿报告
paper/                本地文献与阅读记录
results/              正式实验和探索实验输出
scripts/              数据处理、特征提取、训练、消融和绘图脚本
archive/              已退出正式主线但保留用于审计的探索脚本和历史文档
```

## 关键文件

- 项目从零实现与理解指南：`docs/PROJECT_IMPLEMENTATION_GUIDE_ZH.md`
- 稿件：`manuscript_ionics/main.tex`
- 当前PDF：`manuscript_ionics/main.pdf`
- 模拟审稿与修订状态：`manuscript_ionics/review_report.md`
- 参考文献核验：`manuscript_ionics/reference_verification.md`
- 区域稳定性结果：`results/baseline_uncertainty_region_stability_N150/`
- 历史锚点精确复现：`results/reproduction_ldq_et_N150_20260713/`
- 严格嵌套、冻结和 batch 评价：`results/nested_contiguous_ldq_N150_20260715_nonoverlap/`
- 特征组、模型族、区域和数据效率消融：`results/nested_ablation_learning_N150_20260715_nonoverlap/`
- 新增图表源数据：`manuscript_ionics/figure_source_data/`

## 主要复现实验

在项目根目录运行：

```powershell
python scripts/run_nested_contiguous_ldq.py --phase develop
python scripts/run_nested_contiguous_ldq.py --phase freeze
python scripts/run_nested_contiguous_ldq.py --phase external
python scripts/run_nested_ablation_learning.py --frozen-config results/nested_contiguous_ldq_N150_20260715_nonoverlap/frozen_config.json
python scripts/prepare_enhanced_manuscript_figures.py
python scripts/validate_enhancement_outputs.py
```

编译论文：

```powershell
Set-Location manuscript_ionics
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## 归档说明

2026-07-13已将dual-view、Uzi、MoE、factorized-knee、OOD和混合融合等探索脚本，以及对应的过期阶段文档移入：

```text
archive/2026-07-13_exploratory_cleanup/
```

这些文件没有被删除，但不再属于当前LDQ-ET正式复现入口。历史`results/`输出保持原位，用于结果审计。

## 尚需完成

投稿前仍需完成：

1. 在真正未参与架构探索的新数据集、新化学体系或新协议上确认；
2. 用独立校准集或 shift-aware 方法重新建立可靠预测区间；
3. 用电化学或 post-mortem 证据检验区域解释，而不是继续堆叠预测相关性；
4. 补齐作者、单位、基金、作者贡献以及公开代码/数据仓库链接。

## 结论边界

目前不能声称：

- 0.525--0.650区域对应唯一确定的电化学机理；
- LDQ-ET已泛化到其他化学体系、温度、实验室或在线BMS数据；
- ExtraTrees本身是算法创新；
- dual-view、MoE或寿命辅助已经在未触碰的新外部数据上得到确认。
