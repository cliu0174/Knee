# 实验登记表（2026-07-13）

| ID | 实验 | 选择数据 | 评价数据 | 状态 | 结果目录 | 证据等级 |
|---|---|---|---|---|---|---|
| R0 | 正式 LDQ-ET top40 重复验证 | development；旧 top40 来自 official train | secondary | 完成 | `results/reproduction_ldq_et_N150_20260713/` | 正式结果复现 |
| N1 | 折内 Pearson / stable / balanced / contiguous / mRMR 严格嵌套比较 | development outer-train/inner-CV | development outer-valid | 完成；region-balanced 胜出 | `results/nested_contiguous_ldq_N150_20260713/` | 开发选择证据 |
| N2 | N50 严格嵌套输入预算（公共 N150 队列） | development only | development outer-valid | 完成；bootstrap-stable 胜出 | `results/nested_contiguous_ldq_N50_commonN150_5x2_20260713/` | 开发选择证据 |
| N3 | N100 严格嵌套输入预算（公共 N150 队列） | development only | development outer-valid | 完成；bootstrap-stable 胜出 | `results/nested_contiguous_ldq_N100_commonN150_5x2_20260713/` | 开发选择证据 |
| F1 | 冻结开发集胜出方案 | full development CV only | 无 secondary 读取 | 完成；max-per-region=3，14 regions | `results/nested_contiguous_ldq_N150_20260713/frozen_config.json` | 冻结记录 |
| E1 | 冻结方案 30-seed secondary 评价 | 冻结配置 | secondary 一次执行；失败后仅离线恢复后处理 | 完成 | `results/nested_contiguous_ldq_N150_20260713/` | secondary-informed protocol-controlled |
| A1 | 七组特征组消融 | development fold内 | development outer-valid | 完成 | `results/nested_ablation_learning_N150_20260713/` | 正式/补充 |
| A2 | 核心区域只保留/移除/随机区域/置换 | development fold内 | development outer-valid | 完成 | `results/nested_ablation_learning_N150_20260713/` | 正式贡献证据 |
| D1 | 25/50/75/100% 数据效率 | development repeated subsampling | held-out development cells | 完成 | `results/nested_ablation_learning_N150_20260713/` | 正式数据效率 |
| U1 | cross-fitted residual interval | development OOF calibration | secondary coverage/width | 完成；整体覆盖77.5% | `results/nested_contiguous_ldq_N150_20260713/` | 补充可靠性 |
| P1 | 论文图表与源数据 | 只读正式结果 CSV | 不适用 | 完成；7图、19源CSV、目视QA通过 | `manuscript_ionics/figures/` | 论文产物 |
| M1 | 正文、参考文献、README、指南同步 | 已冻结证据 | 不适用 | 完成；20页PDF编译及逐页QA通过 | 项目文档与 `manuscript_ionics/main.pdf` | 论文产物 |

## 预先固定规则

- 主选择不得读取 secondary 指标。
- 每个外层折内重新拟合特征排序、bootstrap 稳定性、连续带宽、mRMR、缺失值中位数和模型配置。
- 细胞是唯一独立样本；同一细胞不拆分。
- 模型比较使用相同外层折、内层折、特征文件、指标和随机种子。
- 新机制只有在 development selection score、RMSE、尾部误差和稳定性上形成一致证据时才进入冻结评价。
- 如果新机制没有稳定优于严格嵌套 Pearson/原 LDQ-ET，唯一正式主模型仍为 LDQ-ET top40；新机制写为负结果或补充结果。
- 由于历史 secondary 反馈，任何新的 secondary 结果均标记为 secondary-informed，不使用“untouched confirmation”。
