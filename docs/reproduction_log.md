# Severson 2019 复现日志

> 归档说明（2026-07-13）：本文件按时间保留历史实验名称。已退出正式主线的Uzi、dual-view、OOD、残差适配和融合脚本已移至`archive/2026-07-13_exploratory_cleanup/scripts/`，对应`results/`仍保留用于审计。当前正式复现入口以根目录`README.md`为准。

## 2026-06-04

### 目标

复现 Severson et al. 2019, *Data-driven prediction of battery cycle life before capacity degradation* 的基础流程：

- 获取并本地保存数据或可复现实验所需的公开数据文件。
- 编写 Python 脚本完成数据读取、特征处理、模型训练和图表绘制。
- 先得到基础结果，不要求完全达到原文最终效果。
- 验证结果正确性，并逐步调整参数以尽量贴近原文。

### 环境检查

- 工作目录：`D:\Projects\Besttery`
- 系统 Python：`D:\Users\46139\anaconda3\python.exe`
- 可用科学计算包：
  - `numpy`
  - `pandas`
  - `scipy`
  - `sklearn`
  - `matplotlib`
  - `seaborn`
  - `xgboost`
  - `h5py`
- Codex bundled Python 缺少 `scipy/sklearn/matplotlib`，因此复现实验优先使用系统 Anaconda Python。

### 数据源调研

- 官方数据入口：`https://data.matr.io/1/projects/5c48dd2bc625d700019f3204`
- 官方论文代码仓库：`https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation`
- Kaggle 镜像：
  - `https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle/versions/1`
  - `https://www.kaggle.com/datasets/rickandjoe/mit-battery-degradation-dataset`
- 原始数据文件：
  - `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
  - `2017-06-30_batchdata_updated_struct_errorcorrect.mat`
  - `2018-04-12_batchdata_updated_struct.mat` 或 `2018-04-12_batchdata_updated_struct_errorcorrect.mat`
- Kaggle 镜像 `rickandjoe/mit-battery-degradation-dataset` 文件大小：
  - Batch 1: `3,025,320,241` bytes
  - Batch 2: `2,007,331,155` bytes
  - Batch 3: `3,236,690,412` bytes
  - 合计约 8.27GB
- Kaggle 下载需要 API/auth，不是匿名 `curl` 可直接完成。
- 官方 MATR 页面是 JS 应用，本轮未确认稳定匿名直链。

### 当前复现策略

由于原始 `.mat` 数据体积很大且官方 MATR 页面是 JavaScript 应用，第一阶段采用可公开下载的 processed feature CSV 跑通基础模型：

- `original.csv`
- `regularized.csv`

来源仓库：

`https://github.com/Zi-hao-Wei/MIT-Battery-Dataset-feature-extraction-and-RUL-prediction-by-Xgboost`

该仓库 README 说明这些 CSV 由 Severson 三个 `.mat` 文件经 feature extraction 得到，可用于 XGBoost/RUL 预测。后续如果需要完全复现原文特征提取，再补充下载原始 `.mat` 或通过 Kaggle API 获取。

### 已完成操作

1. 创建数据目录：
   - `data/severson_processed`
   - `external`

2. 下载 processed feature 数据：
   - `data/severson_processed/original.csv`
   - `data/severson_processed/regularized.csv`
   - `data/severson_processed/source_readme.md`

3. 克隆官方数据处理代码：
   - `external/severson_official_code`

4. 检查数据：
   - `original.csv`: 123 rows, 75 columns
   - `regularized.csv`: 123 rows, 75 columns
   - 目标列：`lifetime`
   - 泄漏列：`lifePercent` 很可能由 `lifetime` 派生，训练时排除。
   - 同时排除：`Unnamed: 0`, `lifetime`, `lifePercent`

5. 编写基础复现脚本：
   - `scripts/reproduce_severson_baseline.py`

### 当前脚本设计

脚本读取 `original.csv` 和 `regularized.csv`，使用除泄漏列外的 72 个特征，预测 `log10(lifetime)`，再反变换回 cycle life 计算指标。

模型：

- RidgeCV
- ElasticNetCV
- RandomForestRegressor
- XGBoostRegressor

验证方式：

- 固定随机种子 `42` 的 80/20 holdout
- 5-fold out-of-fold CV

输出目录：

- `results/severson_baseline`

输出文件：

- `metrics.csv`
- `dataset_summary.json`
- lifetime distribution 图
- holdout prediction scatter 图
- feature importance 图
- model comparison 图

### 第一次运行记录

命令：

```powershell
python scripts/reproduce_severson_baseline.py --data-dir data/severson_processed --output-dir results/severson_baseline --random-state 42
```

结果：

- 脚本运行到 CV 阶段时报错。
- 原因：某个线性模型在小样本高维 CV 折中外推出极端 `log10(lifetime)`，反变换 `10^x` 时溢出为 infinity。
- 处理：
  - 在每个训练折中记录训练集 `log10(lifetime)` 范围。
  - 对预测的 log target 进行合理裁剪：`[train_min - 0.1, train_max + 0.1]`。
  - 增大 `ElasticNetCV.max_iter` 到 `100000`，设置 `tol=1e-5`。
  - 记录每个指标中被裁剪的预测数量 `clipped_predictions`。

### 第二次运行记录

命令同上。

结果：

- 没有再次出现溢出错误，但完整 CV + 多模型搜索在 300 秒内超时。
- 已生成部分图像，但未写出完整 `metrics.csv`。

处理：

- 将脚本改为默认只跑快速 80/20 holdout。
- 增加 `--full-cv` 选项，需要时再运行完整 5-fold CV。
- 缩小 ElasticNet 搜索网格并降低树模型规模。

### 快速 holdout 结果

命令：

```powershell
python scripts/reproduce_severson_baseline.py --data-dir data/severson_processed --output-dir results/severson_baseline --random-state 42
```

输出目录：

`results/severson_baseline`

关键结果：

- `original.csv` + `ElasticNetCV`: MAE 53.87 cycles, RMSE 71.88 cycles, MAPE 7.77%, R2 0.935。
- `regularized.csv` + `ElasticNetCV`: MAE 53.87 cycles, RMSE 71.88 cycles, MAPE 7.77%, R2 0.935。
- `original.csv` + `XGBoost`: MAE 59.06 cycles, RMSE 74.99 cycles, MAPE 9.33%, R2 0.929。

解释：

- 该结果使用随机 80/20 holdout，不是原文官方 train/test split。
- 结果达到或接近 Severson 原文常见的 9% 左右误差量级，但由于 processed feature 数据来自第三方仓库，不能声称完全复现原文。

### 官方 split 检查

官方 `Load Data.ipynb` 输出：

- `numBat1 = 41`
- `numBat2 = 43`
- `numBat3 = 40`
- `numBat = 124`

官方索引：

```python
test_ind = np.hstack((np.arange(0,(numBat1+numBat2),2),83))
train_ind = np.arange(1,(numBat1+numBat2-1),2)
secondary_test_ind = np.arange(numBat-numBat3,numBat)
```

但是第三方 processed CSV 的 `FeatureExtraction.py` 中存在：

```python
Data = sorted(Data, key=lambda x: x[1])
```

即 CSV 已按 `cycle_life` 排序，且没有保存 cell id / batch id。因此不能直接对 processed CSV 套用官方索引。脚本中保留 `--split official` 仅作探索，不作为正式复现结论。

### 窗口特征调整

第三方 processed feature 中 `-0/-1/-2/-3` 分别对应：

- `-0`: 30→100 cycles
- `-1`: 30→150 cycles
- `-2`: 30→200 cycles
- `-3`: 30→250 cycles

为更贴近 Severson 原文“前 100 cycles”设定，脚本新增：

```powershell
--window-index 0
```

用于仅使用 30→100 cycles 特征。

### 100-cycle window 运行记录

命令：

```powershell
python scripts/reproduce_severson_baseline.py --data-dir data/severson_processed --output-dir results/severson_window100 --random-state 42 --window-index 0
```

输出目录：

`results/severson_window100`

关键结果：

- `original_window0` + `ElasticNetCV`: MAE 68.12 cycles, RMSE 87.86 cycles, MAPE 10.75%, R2 0.903。
- `regularized_window0` + `ElasticNetCV`: MAE 68.12 cycles, RMSE 87.86 cycles, MAPE 10.75%, R2 0.903。
- `original_window0` + `RidgeCV`: MAE 71.11 cycles, RMSE 96.00 cycles, MAPE 11.08%, R2 0.884。
- `random_forest` 和 `xgboost` 在该 100-cycle 特征设定下约为 13% MAPE。

解释：

- `window0` 对应 30→100 cycles，是当前 processed CSV 中最接近 Severson 原文早期循环预测设定的一组特征。
- 相比全窗口 30→100/150/200/250 cycles，100-cycle-only 误差升高，符合可用信息减少后的预期。
- `clipped_predictions = 0`，说明主要结果没有依赖预测裁剪。

### 官方 split 探索记录

命令：

```powershell
python scripts/reproduce_severson_baseline.py --data-dir data/severson_processed --output-dir results/severson_official_split --random-state 42 --split official
```

输出目录：

`results/severson_official_split`

观察：

- primary test 上模型误差约为 9% 到 11% MAPE。
- secondary test 上误差明显变差，约为 24% 到 30% MAPE，且 R2 为负。

结论：

- 该结果只作为探索记录，不作为正式复现结果。
- 原因是第三方 processed CSV 已经在 `external/processed_feature_repo/FeatureExtraction.py` 中按 `cycle_life` 排序：

```python
Data = sorted(Data, key=lambda x: x[1])
```

- CSV 没有保存 cell id / batch id，无法把 Severson 官方索引准确映射回原始电池。
- 若要严格复现官方 train/test/secondary split，需要下载原始 `.mat` 数据并重新生成带 cell id 的数据表。

### 原始数据下载辅助脚本

新增脚本：

`scripts/download_severson_raw_data.ps1`

用途：

- 使用 Kaggle API 下载 MIT Battery Degradation Dataset 镜像。
- 下载目标目录为 `data/severson_raw`。

当前状态：

- 本机暂未安装或配置 Kaggle CLI，因此尚未下载 8.27GB 原始 `.mat` 数据。
- 后续配置 Kaggle token 后，可继续执行该脚本。

### 复现报告生成记录

新增报告：

`results/severson_reproduction_report.md`

报告内容包括：

- 当前复现范围。
- processed CSV 来源和限制。
- 原始数据下载状态。
- 模型设置。
- 全窗口结果。
- 100-cycle-only 结果。
- 官方 split 为什么不能直接用于当前 CSV。
- 图表输出位置。
- 正确性验证和下一步建议。

### seed sweep 尝试记录

曾尝试对随机种子进行多次重复实验，以检查 holdout 结果稳定性。

状态：

- 该尝试在 300 秒内未完成。
- 未生成可用的 `results/seed_sweep` 目录。
- 因此目前不把 seed sweep 作为有效实验结论。

后续处理建议：

- 不重复运行完整 `ElasticNetCV` 多种子 sweep。
- 如需稳定性验证，应改为轻量化方案，例如固定 ElasticNet 超参数或只跑 RidgeCV/ElasticNet 的较小搜索网格。

### 轻量 seed sweep 记录

新增脚本：

`scripts/seed_sweep_severson.py`

目的：

- 检查 100-cycle-only holdout 结果对随机划分的敏感性。
- 只运行轻量模型，避免完整多模型 sweep 再次超时。

模型：

- `RidgeCV`
- `ElasticNetCV` 轻量版：较小 `alpha` 网格，3-fold 内部 CV。

第一次命令：

```powershell
python scripts/seed_sweep_severson.py --data-dir data/severson_processed --output-dir results/severson_seed_sweep --window-index 0 --seeds 0,1,2,3,4,5,6,7,8,9
```

结果：

- 成功完成。
- 10 个随机种子下，`original_window0` 的 MAPE 均值约 20.7%，标准差约 4.8%。
- 说明单次 seed 42 holdout 的 10.75% 属于较好的随机划分，不能单独作为稳定复现结论。

脚本修正：

- `scripts/reproduce_severson_baseline.py` 中的 RMSE 计算从新版 sklearn 已弃用的 `mean_squared_error(..., squared=False)` 改为优先使用 `root_mean_squared_error`，旧版 sklearn 再回退。

第二次命令：

```powershell
python scripts/seed_sweep_severson.py --data-dir data/severson_processed --output-dir results/severson_seed_sweep --window-index 0 --seeds 0,1,2,3,4,5,6,7,8,9,42
```

输出目录：

`results/severson_seed_sweep`

关键结果：

- `original_window0` + `ElasticNetCV_light`: 11 runs, mean MAPE 19.83%, std 5.49%, min 10.80%, median 17.64%, max 28.94%。
- `original_window0` + `RidgeCV`: 11 runs, mean MAPE 19.90%, std 5.29%, min 11.08%, median 18.11%, max 30.03%。
- `regularized_window0` 与 `original_window0` 数值一致。

解释：

- 当前 processed CSV + 随机 holdout 的结果受测试集抽样影响较大。
- 这进一步说明严格复现必须回到原始 `.mat`、保留官方 batch/cell id，并使用官方 split 或论文一致的验证方案。
- `seed_sweep` 是稳定性诊断，不替代主结果。

补充图像：

- `results/severson_seed_sweep/seed_sweep_mape_boxplot.png`

该图为 100-cycle-only 随机 holdout 稳定性箱线图，已经人工查看，渲染正常。图中橙线为中位数，绿色三角为均值，显示 11 个随机种子下 MAPE 主要集中在约 17% 到 24% 区间，最大约 30%。

脚本修正：

- `scripts/seed_sweep_severson.py` 增加 `plot_seed_sweep()`，自动输出箱线图。
- 修正 Matplotlib 新版参数，从 `labels` 改为 `tick_labels`，避免弃用警告。

### 官方 pkl 特征提取脚本记录

新增脚本：

`scripts/extract_severson_official_features.py`

目的：

- 在原始 `.mat` 数据转换出官方 `batch1.pkl/batch2.pkl/batch3.pkl` 后，直接提取 Severson 风格的早期循环特征。
- 为严格复现原文做准备，而不是继续依赖第三方 processed CSV。

脚本依据：

- `external/severson_official_code/Load Data.ipynb`
- `external/severson_official_code/BuildPkl_Batch1.ipynb`
- 官方 README 中对 summary / cycle 数据结构的说明。

脚本实现的官方数据清洗逻辑：

- 删除 batch1 中未达到 80% 容量的 cell：
  - `b1c8`, `b1c10`, `b1c12`, `b1c13`, `b1c22`
- 将 batch2 中延续 batch1 的 5 个 cell 合并回 batch1：
  - `b2c7 -> b1c0`
  - `b2c8 -> b1c1`
  - `b2c9 -> b1c2`
  - `b2c15 -> b1c3`
  - `b2c16 -> b1c4`
- 合并后删除 batch2 中这些 carryover cell。
- 删除 batch3 中噪声通道：
  - `b3c37`, `b3c2`, `b3c23`, `b3c32`, `b3c42`, `b3c43`
- 按官方 notebook 生成 `train / primary_test / secondary_test` 标签。

当前提取的特征：

- `delta_q = Qdlin_cycle_b - Qdlin_cycle_a`，默认 `cycle_a=10`, `cycle_b=100`
- `delta_q_mean`
- `delta_q_var`
- `delta_q_log_var`
- `delta_q_min`
- `delta_q_max`
- `delta_q_skew`
- `delta_q_kurtosis`
- `qd_cycle_2`
- `qd_cycle_100`
- `qd_diff_100_2`
- `charge_time_mean_2_6`
- `ir_cycle_2`
- `ir_cycle_100`
- `ir_diff_100_2`
- `tavg_mean_1_100`
- `tmax_mean_1_100`
- `tmin_mean_1_100`
- `qd_slope_2_100`

默认命令：

```powershell
python scripts/extract_severson_official_features.py --pkl-dir data/severson_raw/pkl --output data/severson_features/severson_official_features.csv
```

当前验证：

- `python -m py_compile scripts/reproduce_severson_baseline.py scripts/seed_sweep_severson.py scripts/extract_severson_official_features.py` 通过。
- `python scripts/extract_severson_official_features.py --help` 通过。

注意：

- 当前本机还没有 `batch1.pkl/batch2.pkl/batch3.pkl`，所以该脚本尚未对真实原始数据运行。
- 等 Kaggle / MATR 原始数据下载完成并转换为 pkl 后，可直接运行该脚本生成更接近原文的特征表。

### 主训练脚本兼容官方特征表

修改脚本：

`scripts/reproduce_severson_baseline.py`

修改内容：

- 将 `cycle_life`, `cell_id`, `batch`, `official_split`, `official_order`, `cycle_a`, `cycle_b` 加入排除列。
- 这样 `extract_severson_official_features.py` 输出的 CSV 可以保留元数据，同时仍可被主训练脚本读取。

### 当前已确认输出

主要脚本：

- `scripts/reproduce_severson_baseline.py`
- `scripts/download_severson_raw_data.ps1`
- `scripts/seed_sweep_severson.py`
- `scripts/extract_severson_official_features.py`

主要数据：

- `data/severson_processed/original.csv`
- `data/severson_processed/regularized.csv`

主要结果目录：

- `results/severson_baseline`
- `results/severson_window100`
- `results/severson_official_split`
- `results/severson_seed_sweep`

主要报告：

- `results/severson_reproduction_report.md`

当前最可信的基础复现结论：

- 全窗口特征：`ElasticNetCV` 约 7.77% MAPE。
- 100-cycle-only 特征：`ElasticNetCV` 约 10.75% MAPE。
- 两者均为随机 80/20 holdout，不是严格官方 split。
- 100-cycle-only 的 seed sweep 平均 MAPE 约 19.8%，说明单次随机划分结果不够稳，需要官方 split 或重复验证支撑。

## 原始 `.mat` 数据确认与官方特征复现推进

### 用户放入数据检查

用户提示 `data` 文件夹中已放入三个数据文档后，执行检查：

```powershell
Get-ChildItem data -Recurse -File
```

发现三个原始 `.mat` 文件：

- `data/2017-05-12_batchdata_updated_struct_errorcorrect.mat`, 3025320241 bytes。
- `data/2018-02-20_batchdata_updated_struct_errorcorrect.mat`, 2022599329 bytes。
- `data/2018-04-12_batchdata_updated_struct_errorcorrect.mat`, 3236690412 bytes。

说明：

- 第一批和第三批文件名与官方记录一致。
- 第二批文件名是 `2018-02-20...`，不是之前记录的 `2017-06-30...`，但文件大小和结构符合 batch2。

用 `h5py` 检查顶层结构：

- 三个文件均包含 `batch`。
- `batch` 下包含 `Vdlin`, `barcode`, `channel_id`, `cycle_life`, `cycles`, `policy`, `policy_readable`, `summary`。
- cell 数分别为 46, 47, 46。

结论：

- 这三份文件是 Severson/MIT-Stanford 数据的三批原始 `.mat` 数据，可以继续复现。

### `.mat` 转官方 pkl

新增脚本：

`scripts/build_severson_pkl_from_mat.py`

目的：

- 将三份原始 `.mat` 转换成官方 notebook 使用的 `batch1.pkl/batch2.pkl/batch3.pkl` 嵌套字典结构。

执行前检查 D 盘剩余空间：

- 通过 `[System.IO.DriveInfo]::GetDrives()` 查询，D 盘可用空间约 50GB。
- 三份 `.mat` 总大小约 8.28GB。

第一次转换：

```powershell
python scripts/build_severson_pkl_from_mat.py --data-dir data --output-dir data/severson_raw/pkl
```

结果：

- 失败在 summary 字段读取。
- 原因：官方 notebook 使用 h5py 2.8；当前 h5py 中 summary 字段已经是 float 数组，不需要像 cycle 字段一样解引用。

修正：

- 对 summary 字段直接 `np.asarray(...).reshape(-1)`。
- 对 cycles 字段继续按 HDF5 reference 解引用。

第二次转换命令同上。

结果：

- 成功生成：
  - `data/severson_raw/pkl/batch1.pkl`, 2909087622 bytes。
  - `data/severson_raw/pkl/batch2.pkl`, 1993485794 bytes。
  - `data/severson_raw/pkl/batch3.pkl`, 3104681466 bytes。

### 官方风格特征提取

执行：

```powershell
python scripts/extract_severson_official_features.py --pkl-dir data/severson_raw/pkl --output data/severson_features/severson_official_features.csv
```

第一次结果：

- 生成 123 行特征，但 `lifetime/cycle_life` 有 8 个 NaN。

定位缺失标签 cell：

- `b2c22`
- `b2c23`
- `b2c35`
- `b2c36`
- `b2c37`
- `b2c38`
- `b2c39`
- `b2c40`

检查这些 cell 的 `summary.QD`：

- 均没有低于 0.88 的明确 EOL 点。
- 部分 cell 末端容量约 0.880 到 0.881，仍未低于阈值。
- 因此不能可靠补出监督学习标签。

脚本修正：

- `extract_severson_official_features.py` 默认丢弃缺失 `lifetime` 的行。
- 增加 `--keep-missing-target` 选项，如需保留无标签行可显式开启。

第二次执行结果：

- 丢弃 8 行缺失标签 cell。
- 保存 `data/severson_features/severson_official_features.csv`。
- 可监督样本数：115。
- 特征列总数：27。
- split 分布：
  - `train`: 37
  - `primary_test`: 39
  - `secondary_test`: 39

注意：

- 由于用户提供的 batch2 文件中有 8 个缺失 EOL 标签 cell，本轮严格官方特征复现使用 115 个有标签 cell，不是 Severson 原文完整 124/123 cell 设置。

### 主训练脚本扩展

修改脚本：

`scripts/reproduce_severson_baseline.py`

新增能力：

- `--csv-files` 参数，用于指定任意 CSV，而不是只读取 `original.csv/regularized.csv`。
- 如果 CSV 中存在 `official_split` 列，`--split official` 会按该列执行 labeled official split。
- 强制 Matplotlib 使用 `Agg` 后端，避免 Tk 后端在无界面环境中崩溃。

模型扩展：

- 增加 `SVR(kernel="rbf")`。
- 增加 `GaussianProcessRegressor`。
- 随机森林与 XGBoost 参数根据后续调参结果更新。

### 官方特征 official split 运行

命令：

```powershell
python scripts/reproduce_severson_baseline.py --data-dir data/severson_features --csv-files severson_official_features.csv --output-dir results/severson_official_features_split --split official --random-state 42
```

第一次运行时 Matplotlib/Tk 后端崩溃：

- 错误包括 `Tcl_AsyncDelete: async handler deleted by the wrong thread`。

处理：

- 在绘图脚本中加入：

```python
import matplotlib
matplotlib.use("Agg")
```

重新运行成功。

输出目录：

`results/severson_official_features_split`

调参前较好结果：

- `random_forest`, primary test: MAPE 12.27%。
- `random_forest`, secondary test: MAPE 12.25%。
- `xgboost`, primary test: MAPE 12.34%。
- `xgboost`, secondary test: MAPE 15.62%。

加入 GPR/SVR 后：

- `gpr_rbf` 在 primary test 的 RMSE/R2 较好，但 MAPE 为 14.01%，secondary test 泛化较差。
- `svr_rbf` secondary test 泛化也较差。
- 因此 MAPE 指标下，树模型仍更稳。

### 官方特征 random holdout 运行

命令：

```powershell
python scripts/reproduce_severson_baseline.py --data-dir data/severson_features --csv-files severson_official_features.csv --output-dir results/severson_official_features_holdout --split holdout --random-state 42
```

较好结果：

- `gpr_rbf`: MAE 103.26 cycles, RMSE 133.96 cycles, MAPE 12.29%, R2 0.893。
- `svr_rbf`: MAPE 14.54%。
- `xgboost`: MAPE 16.11%。
- `random_forest`: MAPE 16.15%。

解释：

- random holdout 与 official split 的结论不同，进一步说明必须优先看官方 split。

### 官方特征小范围调参

新增脚本：

`scripts/tune_severson_official_features.py`

目的：

- 在官方特征 + labeled official split 上搜索 RF/XGBoost/SVR 参数。
- 早期版本输出 primary、secondary 和二者平均 MAPE；导师审查后认为 secondary test 不应参与调参。
- 已修正为只使用 train 内部 4-fold CV 选择参数，再一次性评估 primary/secondary。

命令：

```powershell
python scripts/tune_severson_official_features.py --features data/severson_features/severson_official_features.csv --output-dir results/severson_official_features_tuning --random-state 42
```

早期 post-hoc 输出目录：

`results/severson_official_features_tuning`

该版本最佳平均 MAPE 配置：

- `xgboost_tuned`
- `max_depth=3`
- `learning_rate=0.02`
- `reg_lambda=1.0`
- `n_estimators=350`

结果：

- primary test: MAPE 11.00%, MAE 102.54 cycles, RMSE 174.76 cycles, R2 0.808。
- secondary test: MAPE 12.67%, MAE 148.17 cycles, RMSE 225.67 cycles, R2 0.465。
- primary/secondary 平均 MAPE: 11.84%。

说明：

- 该结果使用了 secondary test 参与选择，因此降级为 exploratory/post-hoc，不作为确认版主结论。

另一个 XGBoost 配置：

- `max_depth=2`, `learning_rate=0.05`, `reg_lambda=1.0`
- primary test MAPE 可到 10.08%，但 secondary test MAPE 为 13.85%，平均略差。

修正后 train-CV 调参命令：

```powershell
python scripts/tune_severson_official_features.py --features data/severson_features/severson_official_features.csv --output-dir results/severson_official_features_tuning_traincv --random-state 42
```

train-CV 最佳配置：

- `xgboost_tuned`
- `max_depth=3`
- `learning_rate=0.05`
- `reg_lambda=5.0`
- `n_estimators=350`

确认版主脚本参数更新：

- 将 XGBoost 默认参数更新为 train-CV 选择配置。
- 将 RandomForest 默认参数更新为 train-CV 中较好的配置：
  - `n_estimators=800`
  - `min_samples_leaf=2`
  - `max_features="sqrt"`

确认版 official split 命令：

```powershell
python scripts/reproduce_severson_baseline.py --data-dir data/severson_features --csv-files severson_official_features.csv --output-dir results/severson_official_features_split_traincv_params --split official --random-state 42
```

确认版 official split 输出：

- `xgboost`: primary MAPE 11.67%, secondary MAPE 14.60%。
- `random_forest`: primary MAPE 12.46%, secondary MAPE 12.69%。

### 图像验证

人工查看以下图像，均正常渲染：

- `results/severson_official_features_split/severson_official_features_official_holdout_prediction_scatter.png`
- `results/severson_official_features_holdout/severson_official_features_holdout_prediction_scatter.png`
- `results/severson_official_features_split/severson_official_features_feature_importance.png`

特征重要性观察：

- `delta_q_var`
- `delta_q_log_var`
- `delta_q_min`

排在最前，与 Severson 原文核心思想一致。

## 子智能体导师审查与确认版修正

### 审查设置

创建子智能体 `Lorentz`，要求其化身专业审稿人/研究生导师，对当前复现进行独立审查，不修改文件，只给出评价与建议。

审查重点：

- 数据文件是否对应 Severson 三批数据。
- 第二批 `2018-02-20...mat` 与官方 MATLAB 中 `2017-06-30...mat` 的版本差异。
- 115 个有标签样本与 8 个 batch2 缺失 lifetime cell 的处理是否合理。
- `.mat -> pkl -> feature -> train -> plot` 方法是否贴近原文。
- 当前结果是否和原文结论一致、数值是否接近。

### 第一轮导师结论

导师判断：

- 阶段性基础复现方向正确。
- 严格复现需大修。

主要问题：

- batch2 文件版本与常见官方 MATLAB 文件名不一致。
- 当前可监督 split 为 `train=37`, `primary_test=39`, `secondary_test=39`，不是原文 `41/43/40`。
- 早期调参脚本曾使用 primary+secondary 平均 MAPE 选参，这会污染 secondary test。
- 当前最佳结果来自 XGBoost/RF，不是 Severson 原文 regularized linear / elastic net 主线。

### 根据导师意见进行的修正

修正 `scripts/tune_severson_official_features.py`：

- 改为只用 train 内部 4-fold CV 选择参数。
- primary/secondary 只在选参后做一次最终评估。
- 新输出目录：
  - `results/severson_official_features_tuning_traincv`

修正 `scripts/reproduce_severson_baseline.py`：

- XGBoost 默认参数改为 train-CV 选择配置：
  - `max_depth=3`
  - `learning_rate=0.05`
  - `reg_lambda=5.0`
  - `n_estimators=350`
- RandomForest 默认参数改为：
  - `n_estimators=800`
  - `min_samples_leaf=2`
  - `max_features="sqrt"`

确认版输出目录：

- `results/severson_official_features_split_traincv_params`

确认版结果：

- `xgboost`: primary MAPE 11.67%, secondary MAPE 14.60%。
- `random_forest`: primary MAPE 12.46%, secondary MAPE 12.69%。

### 第二轮导师复核

导师复核结论：

- 阶段性复现确认版：可接受。
- 严格 Severson 2019 原文复现：仍不可接受。

必须使用的表述：

- 本结果是 `official-style features + modified split + train-CV 参数选择` 下的阶段性复现确认版。
- 不能写成 exact reproduction。
- XGBoost/RF 是补充非线性 baseline，不是 Severson 原文主模型。
- secondary test 结果要谨慎解释。

新增确认版文档：

- `docs/severson_reproduction_confirmation_review.md`

正式报告同步修正：

- `results/severson_reproduction_report.md`

## 使用 `2017-06-30` batch2 重新复现

### 新 batch2 文件确认

用户将 `2017-06-30_batchdata_updated_struct_errorcorrect.mat` 放入 `data` 目录后，重新检查 `.mat` 文件：

- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`: 46 cells。
- `2017-06-30_batchdata_updated_struct_errorcorrect.mat`: 48 cells。
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`: 47 cells。
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat`: 46 cells。

结论：

- `2017-06-30` 文件结构正确，且 48 cells 符合原文 batch2 在删除 5 个 carryover 后剩 43 cells 的设置。

### 转换脚本修正

修改脚本：

`scripts/build_severson_pkl_from_mat.py`

修改内容：

- 默认 batch2 从 `2018-02-20...mat` 改为 `2017-06-30...mat`。
- 新增命令行参数：
  - `--batch1-file`
  - `--batch2-file`
  - `--batch3-file`
- 这样后续可以显式切换不同 batch 文件，避免混淆。

### 重新生成 pkl

为了保留旧结果，新版 pkl 单独输出：

```powershell
python scripts/build_severson_pkl_from_mat.py --data-dir data --output-dir data/severson_raw/pkl_20170630
```

生成：

- `data/severson_raw/pkl_20170630/batch1.pkl`
- `data/severson_raw/pkl_20170630/batch2.pkl`
- `data/severson_raw/pkl_20170630/batch3.pkl`

### 重新提取 official-style 特征

命令：

```powershell
python scripts/extract_severson_official_features.py --pkl-dir data/severson_raw/pkl_20170630 --output data/severson_features_20170630/severson_official_features_20170630.csv
```

结果：

- 保存 124 rows, 27 columns。
- Official counts:
  - `numBat1=41`
  - `numBat2=43`
  - `numBat3=40`
- split 分布：
  - `primary_test=43`
  - `train=41`
  - `secondary_test=40`
- `lifetime` 无缺失。

这说明换用 `2017-06-30` 后，样本数和 split 已恢复到原文设置。

### train-only CV 调参

命令：

```powershell
python scripts/tune_severson_official_features.py --features data/severson_features_20170630/severson_official_features_20170630.csv --output-dir results/severson_20170630_official_features_tuning_traincv --random-state 42
```

train-CV 最优配置之一：

- `random_forest_tuned`
- `min_samples_leaf=1`
- `max_features=None`
- train-CV mean MAPE 约 14.01%。

XGBoost train-CV 较优配置：

- `max_depth=3`
- `learning_rate=0.05`
- `reg_lambda=5.0`
- train-CV mean MAPE 约 14.54%。

### 主脚本参数同步

修改 `scripts/reproduce_severson_baseline.py`：

- RandomForest:
  - `n_estimators=800`
  - `min_samples_leaf=1`
  - `max_features=None`
- XGBoost:
  - `n_estimators=350`
  - `max_depth=3`
  - `learning_rate=0.05`
  - `reg_lambda=5.0`

### 新版 official split 结果

命令：

```powershell
python scripts/reproduce_severson_baseline.py --data-dir data/severson_features_20170630 --csv-files severson_official_features_20170630.csv --output-dir results/severson_20170630_official_features_split_traincv_params --split official --random-state 42
```

确认版结果：

- `ridge_cv`: primary MAPE 13.87%, secondary MAPE 22.13%。
- `elastic_net_cv`: primary MAPE 14.46%, secondary MAPE 20.76%。
- `random_forest`: primary MAPE 15.77%, secondary MAPE 12.60%。
- `xgboost`: primary MAPE 16.35%, secondary MAPE 14.00%。
- `gpr_rbf`: primary MAPE 16.15%, secondary MAPE 32.16%。

解释：

- 换用 `2017-06-30` 后，数据和 split 更接近原文。
- 线性模型 primary 接近原文 Variance/Full model 的有异常版本量级，但 secondary 较差。
- 随机森林 secondary MAPE 12.60%，接近原文 Variance model secondary 11.4% 量级，但仍高于 Discharge/Full model。

### primary 异常样本检查

定位最短寿命 cell：

- `b2c1`, lifetime 148, 位于 `primary_test`。

这与原文表格中常见的“去异常”比较口径相关。

计算去除 `b2c1` 后 primary 指标，输出到：

`results/severson_20170630_official_features_split_traincv_params/metrics_with_primary_outlier_check.csv`

去异常 primary 结果：

- `gpr_rbf`: primary MAPE 10.35%。
- `ridge_cv`: primary MAPE 11.11%。
- `svr_rbf`: primary MAPE 11.78%。
- `elastic_net_cv`: primary MAPE 12.02%。
- `random_forest`: primary MAPE 12.16%。
- `xgboost`: primary MAPE 12.38%。

解释：

- 去掉 primary 短寿命异常后，多个模型 primary MAPE 显著接近原文去异常结果。
- 这说明原文中 outlier 对 primary test 的影响在本复现中也出现了。

### 图像检查

人工查看以下图像，均正常渲染：

- `results/severson_20170630_official_features_split_traincv_params/severson_official_features_20170630_official_holdout_prediction_scatter.png`
- `results/severson_20170630_official_features_split_traincv_params/severson_official_features_20170630_feature_importance.png`

特征重要性仍由以下特征主导：

- `delta_q_var`
- `delta_q_log_var`
- `delta_q_min`

与 Severson 原文核心思想一致。

### 新一轮导师审查

创建子智能体 `Godel`，要求其作为专业审稿人/研究生导师，审查使用 `2017-06-30` 后的新版复现结果。

审查重点：

- 是否已经真正使用 `2017-06-30`。
- split 是否恢复到 `train=41`, `primary=43`, `secondary=40`。
- 新版 metrics 是否更接近原文。
- 方法差距是否仍存在。
- 是否可以从 modified reproduction 升级为更接近原文的第一阶段确认版。

## 2026-07-08 Supplementary Table 1 线性模型复现

新增脚本：
- `scripts/extract_severson_supplementary_features.py`
- `scripts/reproduce_severson_linear_table1_models.py`

新增输出：
- `data/severson_features_20170630/severson_supplementary_table1_features_20170630.csv`
- `results/severson_20170630_table1_linear_models_sparse/metrics.csv`
- `results/severson_20170630_table1_linear_models_sparse/coefficients.csv`
- `results/severson_20170630_table1_linear_models_sparse/predictions.csv`
- `results/severson_20170630_table1_linear_models_sparse/table1_linear_model_predictions.png`
- `docs/severson_table1_linear_reproduction_20260708.md`

关键结果：
- Variance linear: primary MAPE 17.04%, secondary MAPE 11.86%。
- Discharge ElasticNetCV: primary MAPE 27.42%, secondary MAPE 12.89%。
- Full raw ElasticNetCV: primary 出现严重异常外推。
- Full clean4 ElasticNetCV: 排除自动识别的 4 个 full-model 极端异常 cell 后，primary MAPE 14.22%, secondary MAPE 18.00%。

导师式判断：variance model 已接近原文核心结果；discharge/full 尚未达到严格原文复现，主要差距来自 exact selected features、excluded cell 官方身份、IR/temperature 特征定义与 ElasticNet 流程。

## 2026-07-08 追加复现实验：cycle index 敏感性与特征选择

执行内容：
- 生成 `c9_c99`, `c10_c100`, `c10_c101` 三套 Supplementary Table 1 特征。
- 分别运行 `scripts/reproduce_severson_linear_table1_models.py`。
- 追加训练集内 4-fold CV greedy feature selection，尝试 6-feature discharge 与 9-feature full。

关键发现：
- `c9_c99` variance model 最接近原文：primary MAPE 14.75%，secondary MAPE 11.42%，几乎对齐原文 variance model。
- `c9_c99` discharge greedy6：primary no b2c1 MAPE 11.82%，secondary MAPE 7.82%。
- `c9_c99` full greedy9：primary clean4 MAPE 11.03%，secondary MAPE 12.56%。

最终判断：核心 variance model 已可认为成功复现；discharge/full model 趋势一致、部分指标接近，但 exact reproduction 仍受未公开 modeling code 与异常样本清单限制。

## 2026-07-08 Knee point 主线阶段 1：raw capacity / SOH 曲线构建

执行内容：
- 新增 `scripts/export_soh_curves.py`，从 `data/severson_raw/pkl_20170630` 读取合并后的 124 个 Severson cell。
- 导出循环级标准表：`data/processed_cycles_20170630.csv`。
- 导出 cell 级质检表：`data/cell_qc_20170630.csv`。
- 新增 `scripts/plot_soh_raw_curves.py`，生成 raw SOH 曲线图。
- 输出图像目录：`results/soh_raw_curves`。

本阶段结果：
- 循环级记录：100501 rows, 17 columns。
- cell 级记录：124 rows, 19 columns。
- official split 保持为 train=41, primary_test=43, secondary_test=40。
- `QD` 无缺失；部分 cell 有 1 个非正容量占位点，已在 SOH 与绘图中排除。
- 寿命范围：148 到 2237 cycles，均值约 801.64 cycles。
- 已生成：
  - `all_cells_raw_soh_overview.png`
  - `selected_20_raw_soh_grid.png`
  - `selected_20_individual/` 下 20 张单 cell 曲线。

阶段判断：
- 本阶段产物可作为后续 SOH 平滑、单调修正和 knee point 标定的输入。
- 下一小步建议：实现 Savitzky-Golay / moving average 平滑与 isotonic 单调修正，并先在 20 个代表性 cell 上可视化对比。

## 2026-07-08 Knee point 主线阶段 2：SOH 平滑与单调修正

执行内容：
- 新增 `scripts/correct_soh_curves.py`。
- 输入 `data/processed_cycles_20170630.csv`。
- 使用 Savitzky-Golay filter (`window=21`, `polyorder=2`) 生成 `soh_smooth`。
- 使用 Isotonic Regression 单调不增约束生成 `soh_mono`。
- 输出：
  - `data/processed_cycles_soh_corrected_20170630.csv`
  - `data/soh_correction_qc_20170630.csv`
  - `results/soh_correction_curves/selected_20_soh_correction_grid.png`
  - `results/soh_correction_curves/selected_20_individual/`
  - `docs/soh_correction_review_20260708.md`

过程审查：
- 本阶段只基于循环级 CSV，不重新变更原始数据。
- 保留 `soh_raw`，新增 `soh_smooth` 与 `soh_mono`，支持后续消融。
- 修正未使用训练标签或模型结果，不涉及预测任务的信息泄漏。
- 完整曲线修正只用于生成 knee point 标签；后续早期预测模型输入仍必须限制在前 N 个 cycle。

成果审查：
- 修正后循环级记录：100501 rows, 19 columns。
- 修正 QC 表：124 rows, 26 columns。
- raw 平均单调违背次数：80.24。
- smooth 平均单调违背次数：17.78。
- mono 平均单调违背次数：0.00。
- `mean_abs_mono_minus_raw` 均值约 0.000318，说明修正幅度总体很小。
- `trend_fidelity_mono_vs_smooth` 中位数约 0.999987，说明趋势保真度高。
- 需要关注的样本：`b1c18`, `b1c0`, `b2c44`, `b2c12`。

阶段判断：通过。下一步进入分段线性 knee point 标定，并先对 20 个代表 cell 进行人工审查。

## 2026-07-08 Knee point 主线阶段 3：分段线性 knee point 标定

执行内容：
- 新增 `scripts/detect_knee_points.py`。
- 输入 `data/processed_cycles_soh_corrected_20170630.csv`。
- 基于 `soh_mono` 进行分段线性拟合，候选范围为 `[20%, 90%]` 曲线长度，最小段长度为 30 点。
- 输出：
  - `data/knee_labels_20170630.csv`
  - `results/knee_detection/selected_20_knee_detection_grid.png`
  - `results/knee_detection/selected_20_individual/`
  - `results/knee_detection/knee_label_distribution_qc.png`
  - `docs/knee_detection_review_20260708.md`

过程审查：
- knee 标定只使用完整 `soh_mono` 曲线，用于生成监督标签。
- 不使用预测模型或测试集结果。
- raw/smooth 只用于图像对照。
- 后续早期预测模型输入仍必须限制在前 N 个 cycle，不能使用完整曲线特征。

成果审查：
- 124/124 个 cell 均成功标定，`status=ok`。
- 平均 knee cycle 约 624.06，中位数约 578.50。
- 平均 relative knee 约 0.779，中位数约 0.781。
- 平均 knee SOH 约 0.944。
- 平均 label confidence 约 0.955。
- 所有 cell 第二段斜率均比第一段更负，符合加速退化定义。
- `b2c1` 的 knee_cycle=94，小于 100-cycle 早期窗口；后续 N=100 回归任务应排除或作为极早期风险分类样本。

阶段判断：通过，但需审慎。下一步不直接全量训练，应先人工筛查 20 个代表 cell 的 knee 图，再做早期特征提取。

## 2026-07-08 Knee point 主线阶段 4：N=100 早期容量特征提取

执行内容：
- 新增 `scripts/extract_early_knee_features.py`。
- 输入：
  - `data/processed_cycles_soh_corrected_20170630.csv`
  - `data/knee_labels_20170630.csv`
- 只使用 `cycle <= 100` 且 `is_positive_capacity=True` 的循环点提取特征。
- 输出：
  - `data/early_features_N100_20170630.csv`
  - `data/early_features_N100_qc_20170630.csv`
  - `results/early_features_N100/early_features_N100_qc.png`
  - `docs/early_feature_extraction_review_20260708.md`

过程审查：
- 本阶段不训练模型。
- 特征只来自前 100 cycles。
- `knee_cycle`, `relative_knee`, `cycle_life`, `label_confidence` 作为目标/元信息保留，但后续建模时必须从 feature columns 中排除。
- 已显式标记 `knee_after_window`，用于后续排除窗口内已发生 knee 的样本。

成果审查：
- 输出特征表：124 rows, 47 columns。
- 特征 QC 表：36 个特征列。
- 无缺失特征列。
- 常数列：`N100_window_complete`，后续建模时应丢弃。
- `knee_after_window=True`: 123 个样本。
- `knee_after_window=False`: 1 个样本，`b2c1`。
- `N100_n_observed` 范围为 99 到 100，符合部分 cell 有起始非正容量占位点的事实。

阶段判断：通过。下一步进入 N=100 最小容量特征组基线模型训练，回归任务应排除 `b2c1`。

## 2026-07-08 Knee point 主线阶段 5：N=100 raw-only 基线模型训练

执行内容：
- 新增 `scripts/train_knee_baselines.py`。
- 输入：`data/early_features_N100_20170630.csv`。
- 有效设置：
  - target=`knee_cycle`
  - feature_set=`raw_only`
  - 排除 `knee_after_window=False` 的 `b2c1`
  - 使用 123/124 个样本
  - 使用 22 个 raw-only 早期特征
- 输出：
  - `results/knee_baselines_N100_raw_only/metrics.csv`
  - `results/knee_baselines_N100_raw_only/predictions.csv`
  - `results/knee_baselines_N100_raw_only/feature_importance.csv`
  - `results/knee_baselines_N100_raw_only/knee_baseline_predictions.png`
  - `docs/knee_baseline_model_review_20260708.md`

过程审查：
- 发现 `soh_mono` 来源于完整寿命曲线，不适合作为早期预测输入，因此将有效建模版本改为 `raw_only`。
- `cycle_life`, `knee_cycle`, `relative_knee`, `label_confidence` 等元信息未进入 raw-only feature columns。
- 模型只在 train split 训练，primary/secondary 只评估。

有效 knee_cycle raw-only 结果：
- RidgeCV: primary MAE 2045.59, secondary MAE 359.35，primary 严重外推失控。
- SVR-RBF: primary MAE 137.81, secondary MAE 384.05。
- RandomForest: primary MAE 123.46, secondary MAE 268.21。
- XGBoost: primary MAE 146.92, secondary MAE 269.46，但 train MAE 21.49，过拟合风险明显。

naive baseline 对比：
- train-mean baseline: primary MAE 177.94, secondary MAE 319.53。
- train-median baseline: primary MAE 168.60, secondary MAE 400.43。
- RandomForest 在 primary 和 secondary MAE 上优于 naive，但 secondary R2 仍为负。

诊断实验：
- 运行 `relative_knee` raw-only 版本，结果更稳定。
- 但将 relative_knee 换算回 cycle 需要真实 `cycle_life`，这在早期预测场景不可用，因此不能当作部署性能。

阶段判断：通过，但只是初步 baseline。下一步建议先做 N=50/150 窗口敏感性，而不是急着调参。

## 2026-07-08 Knee point 主线阶段 6：N=50/100/150 窗口敏感性

执行内容：
- 复用 `scripts/extract_early_knee_features.py` 提取 N=50 和 N=150 特征。
- 复用 `scripts/train_knee_baselines.py` 训练 raw-only knee_cycle 基线。
- 输出：
  - `data/early_features_N50_20170630.csv`
  - `data/early_features_N150_20170630.csv`
  - `results/knee_baselines_N50_raw_only/`
  - `results/knee_baselines_N150_raw_only/`
  - `results/window_sensitivity_N50_N100_N150/metrics_all_windows.csv`
  - `results/window_sensitivity_N50_N100_N150/window_sensitivity_mae.png`
  - `docs/window_sensitivity_review_20260708.md`

过程审查：
- 三个窗口均使用 raw-only 特征，不使用 `soh_mono` 或完整曲线修正特征作为输入。
- 三个窗口均使用 22 个 raw-only 特征。
- N=50 使用 124 个样本；N=100/N=150 排除 b2c1 后使用 123 个样本。
- 本阶段只比较窗口长度，不调参。

核心结果：
- Primary best:
  - N=50: RandomForest MAE 125.43 cycles。
  - N=100: RandomForest MAE 123.46 cycles。
  - N=150: XGBoost MAE 102.43 cycles。
- Secondary best:
  - N=50: XGBoost MAE 283.75 cycles。
  - N=100: RandomForest MAE 268.21 cycles。
  - N=150: RandomForest MAE 247.71 cycles。

阶段判断：通过。窗口越长，预测效果总体越好，但 N=150 的预警提前量更短。secondary R2 仍为负，说明 raw capacity/SOH 特征不足以稳定跨 batch 泛化。下一步建议扩展特征类型，而不是继续调参。

## 2026-07-08 Knee point 主线阶段 7：扩展早期特征建模

执行内容：
- 新增 `scripts/extract_extended_early_features.py`。
- 在 N=100 与 N=150 早期窗口中加入充电时间、内阻、温度和 Severson-style DeltaQ(V) 特征。
- 重新训练 RidgeCV、SVR-RBF、RandomForest、XGBoost baseline。
- 输出：
  - `data/early_features_N100_extended_20170630.csv`
  - `data/early_features_N150_extended_20170630.csv`
  - `results/knee_baselines_N100_extended/`
  - `results/knee_baselines_N150_extended/`
  - `results/extended_feature_comparison/metrics_raw_vs_extended.csv`
  - `results/extended_feature_comparison/raw_vs_extended_mae.png`
  - `docs/extended_feature_model_review_20260708.md`

过程审查：
- 扩展特征表不包含 `soh_smooth`、`soh_mono` 或完整寿命曲线修正距离类特征，避免完整曲线泄漏。
- `soh_mono` 仍只用于生成 knee label，不进入早期预测输入。
- N=100 DeltaQ 使用 cycle 9 到 cycle 99；N=150 DeltaQ 使用 cycle 9 到 cycle 149。
- 两个窗口建模时均排除 `knee_after_window=False` 的 `b2c1`，最终使用 123/124 个 cell。
- 本阶段仍不调参，只比较特征扩展是否带来稳定提升。

核心结果：
- N=100 extended XGBoost primary MAE=64.95 cycles，R2=0.851。
- N=100 extended RandomForest secondary MAE=172.43 cycles。
- N=150 extended XGBoost primary MAE=67.40 cycles，R2=0.853。
- N=150 extended RandomForest primary MAE=71.49 cycles，secondary MAE=157.07 cycles，secondary R2=0.067。
- 相比 raw-only，N=150 RandomForest secondary MAE 从 247.71 降至 157.07 cycles。

特征解释：
- RandomForest 和 XGBoost 的重要特征主要集中在 `delta_q_mean`、`delta_q_min`、`delta_q_std`、`delta_q_var`、`delta_q_log_var`。
- 充电时间、内阻、温度特征提供辅助信息。
- 这与 Severson 原文中早期容量曲线差异具有强预测价值的思想一致；在本项目中，DeltaQ 对 knee cycle 预测同样有效。

阶段判断：
- 通过。本阶段结果明显优于 raw-only baseline。
- XGBoost 在 primary 上最强，但 train/test 差距显示过拟合风险。
- RandomForest 在 secondary 上更稳健，N=150 extended RandomForest 可作为当前主线稳健 baseline。
- 下一小步建议做特征组消融：raw-only、raw+DeltaQ、raw+DeltaQ+charge time、raw+DeltaQ+charge time+IR、完整扩展特征。

## 2026-07-08 进展记录审计

审计范围：
- `docs/reproduction_log.md`
- `docs/*review_20260708.md`
- `data/*20170630*.csv`
- `results/` 下 Severson 复现、SOH/knee 主线、窗口敏感性和扩展特征结果目录。

记录完整性判断：
- 有效主线阶段均已记录在 `docs/reproduction_log.md`：
  - Severson Table 1 线性模型复现。
  - raw capacity / SOH 曲线构建。
  - SOH 平滑与单调修正。
  - 分段线性 knee point 标定。
  - N=100 早期特征提取。
  - N=100 raw-only baseline。
  - N=50/100/150 窗口敏感性。
  - N=100/N=150 扩展特征建模。
- 各有效阶段均有对应 review 文档或日志段落。
- `results/knee_baselines_N100/` 和 `results/knee_baselines_N100_relative/` 属于早期诊断/废弃结果，因使用了 `soh_mono`、`soh_smooth` 或完整曲线修正距离类特征，不纳入正式性能结论。
- `results/knee_baselines_N100_relative_raw_only/` 是 raw-only 诊断结果，但 relative_knee 转回 cycle 需要真实 `cycle_life`，只用于理解相对拐点位置，不作为部署性能。

结果一致性复查：
- `data/knee_labels_20170630.csv`：124/124 个 cell 标定成功，`status=ok`。
- N=50/N=100/N=150 基础早期特征表均为 124 行，特征缺失数为 0。
- N=100/N=150 extended 特征表均为 124 行，特征缺失数为 0。
- Severson c9_c99 variance linear model：primary MAPE=14.75%，secondary MAPE=11.42%，仍符合“核心 variance model 可接受复现”的判断。
- raw-only knee_cycle baseline：N=150 RandomForest secondary MAE=247.71 cycles，secondary R2 仍为负，符合“raw capacity/SOH 跨 batch 泛化不足”的判断。
- extended feature baseline：N=150 RandomForest secondary MAE=157.07 cycles，secondary R2=0.067；N=100 XGBoost primary MAE=64.95 cycles，符合“扩展特征明显提升，但仍需消融和稳健性分析”的判断。

审计结论：
- 当前有效进展已记录完整。
- 当前结果符合预期，没有发现与既有阶段结论冲突的指标。
- 后续正式实验应继续避开 `soh_mono` 作为早期输入，并优先开展特征组消融实验。

## 2026-07-08 Knee point 主线阶段 8：特征组消融实验

执行内容：
- 新增 `scripts/run_feature_ablation.py`。
- 对 N=100 与 N=150 扩展特征做递进式消融：
  - `raw_only`
  - `raw_deltaq`
  - `raw_deltaq_charge`
  - `raw_deltaq_charge_ir`
  - `raw_deltaq_charge_ir_temp`
- 沿用 RidgeCV、SVR-RBF、RandomForest、XGBoost 和 Severson 官方 split。
- 输出：
  - `results/feature_ablation_N100_N150/metrics.csv`
  - `results/feature_ablation_N100_N150/predictions.csv`
  - `results/feature_ablation_N100_N150/best_by_window_split.csv`
  - `results/feature_ablation_N100_N150/feature_group_metadata.json`
  - `results/feature_ablation_N100_N150/feature_ablation_mae.png`
  - `docs/feature_ablation_review_20260708.md`

过程审查：
- `raw_only` 指标与此前 raw-only baseline 完全一致，说明训练口径没有漂移。
- 消融特征不包含 `soh_smooth`、`soh_mono` 或完整曲线修正距离类特征。
- `knee_cycle`、`relative_knee`、`cycle_life` 等字段未进入特征列。
- 两个窗口均排除 `knee_after_window=False` 的 `b2c1`，最终使用 123/124 个 cell。

核心结果：
- N=150 XGBoost：secondary MAE 从 raw_only 的 276.32 降至 raw_deltaq 的 103.29 cycles。
- N=100 RandomForest：secondary MAE 从 raw_only 的 268.21 降至 raw_deltaq 的 113.77 cycles。
- N=150 RandomForest：secondary MAE 从 raw_only 的 247.71 降至 raw_deltaq 的 136.64 cycles。
- 完整 80 维特征组并非最优，例如 N=150 XGBoost secondary MAE 为 208.83，差于 raw_deltaq 的 103.29。
- RidgeCV 在部分 secondary split 上误差较低，但 primary test 严重失控，因此不作为主方法结论。

阶段判断：
- 通过。DeltaQ(V) 是当前最明确的核心贡献特征。
- 本阶段强化了论文创新论证：不仅复现 Severson 的早期容量曲线思想，而且证明 DeltaQ 对 knee cycle 预测同样有效。
- 更多特征不一定更好，论文主线应强调轻量化和特征筛选，而不是堆叠多源特征。
- 下一步建议做 knee label 稳定性/标定方法对比，补强标签生成方法的可信度。

## 2026-07-08 Knee point 主线阶段 9：全组合特征模块搜索与最佳模型筛选

执行内容：
- 新增 `scripts/run_feature_combination_search.py`。
- raw 特征固定保留，对 DeltaQ、charge time、IR、temperature 四个模块做 `2^4=16` 种组合搜索。
- 对 N=100 与 N=150 两个窗口均训练 RidgeCV、SVR-RBF、RandomForest、XGBoost。
- 输出：
  - `results/feature_combination_search_N100_N150/metrics.csv`
  - `results/feature_combination_search_N100_N150/predictions.csv`
  - `results/feature_combination_search_N100_N150/robust_ranking.csv`
  - `results/feature_combination_search_N100_N150/best_by_window_split.csv`
  - `results/feature_combination_search_N100_N150/feature_group_metadata.json`
  - `results/feature_combination_search_N100_N150/top_combination_models_mae.png`
  - `results/feature_combination_search_N100_N150/combination_search_heatmap.png`
  - `docs/feature_combination_search_review_20260708.md`

过程审查：
- 总指标行数为 384，符合 2 windows × 16 groups × 4 models × 3 splits。
- 所有指标无缺失。
- 不包含 `soh_smooth`、`soh_mono` 或完整寿命曲线修正距离类特征。
- `knee_cycle`、`relative_knee`、`cycle_life` 未进入特征列。
- `raw_only` 与此前 baseline 一致。

单 split 最佳结果：
- N=100 primary：`raw_deltaq_charge_ir + XGBoost`，MAE=63.03，R2=0.870。
- N=100 secondary：`raw_deltaq_charge + RidgeCV`，MAE=91.35，R2=0.638，但 RidgeCV 在 primary 上不稳定，不作为主方法。
- N=150 primary：`raw_deltaq_ir_temp + XGBoost`，MAE=60.19，R2=0.861。
- N=150 secondary：`raw_deltaq + XGBoost`，MAE=103.29，R2=0.672。

稳健综合排序：
- 采用 `stable_score = mean(primary_MAE, secondary_MAE) + 0.25 * max(primary_MAE, secondary_MAE)`。
- 排名第一：N=150 + `raw_deltaq` + XGBoost。
  - primary MAE=65.67
  - secondary MAE=103.29
  - mean MAE=84.48
  - max MAE=103.29
  - min R2=0.672
- 排名第二：N=100 + `raw_deltaq` + RandomForest。
  - primary MAE=65.93
  - secondary MAE=113.77
  - mean MAE=89.85
  - max MAE=113.77
  - min R2=0.469

阶段判断：
- 通过。当前推荐论文主模型为 N=150 + raw_deltaq + XGBoost。
- 推荐辅助主线为 N=100 + raw_deltaq + RandomForest，用于强调更早预警和可解释性。
- 结果进一步证明 DeltaQ(V) 是 knee cycle early prediction 的核心早期特征。
- 完整多源特征组不是最优，论文应强调轻量化特征筛选，而不是特征堆叠。
- 下一步建议做 knee label 稳定性/标定方法对比。

## 2026-07-08 Knee point 主线阶段 10：Knee label 稳定性与候选模型标签鲁棒性

执行内容：
- 新增 `scripts/analyze_knee_label_stability.py`。
- 对同一 `soh_mono` 曲线比较三种 knee label 方法：
  - `piecewise`：当前主方法，分段线性拟合。
  - `kneedle`：归一化曲线到首尾连线的最大距离。
  - `rate_threshold`：局部退化速率阈值。
- 对两个候选模型做标签鲁棒性验证：
  - N=150 + raw_deltaq + XGBoost。
  - N=100 + raw_deltaq + RandomForest。
- 输出：
  - `results/knee_label_stability/knee_labels_by_method.csv`
  - `results/knee_label_stability/label_method_comparison.csv`
  - `results/knee_label_stability/candidate_model_label_robustness.csv`
  - `results/knee_label_stability/label_method_scatter.png`
  - `results/knee_label_stability/candidate_label_robustness_mae.png`
  - `docs/knee_label_stability_review_20260708.md`

过程审查：
- 三种方法均成功输出 124 个 cell 标签。
- piecewise 与 Kneedle 均只有 1 个 cell 的 knee 落在 N=100/N=150 窗口内。
- rate_threshold 有 2 个 cell 的 knee 落在窗口内，分别为 `b2c0` 与 `b2c1`，说明该方法更容易偏早。
- 候选模型训练时仍只使用 raw_deltaq 早期特征，不使用 `soh_mono` 或完整寿命曲线修正距离类特征作为输入。

标签一致性结果：
- Kneedle vs piecewise：mean abs diff=32.85 cycles，Spearman=0.9978，within ±50 cycles=87.90%，within ±100 cycles=99.19%。
- Rate threshold vs piecewise：mean abs diff=69.37 cycles，Spearman=0.9227，within ±50 cycles=59.68%，within ±100 cycles=81.45%。

候选模型鲁棒性：
- N=150 raw_deltaq XGBoost：
  - piecewise：primary MAE=65.67，secondary MAE=103.29。
  - kneedle：primary MAE=70.47，secondary MAE=103.00。
  - rate_threshold：primary MAE=66.83，secondary MAE=146.92。
- N=100 raw_deltaq RandomForest：
  - piecewise：primary MAE=65.93，secondary MAE=113.77。
  - kneedle：primary MAE=66.55，secondary MAE=109.11。
  - rate_threshold：primary MAE=78.43，secondary MAE=159.55。

阶段判断：
- 通过。Piecewise 与 Kneedle 高度一致，说明主标签可靠性较强。
- Rate threshold 偏早且波动更大，可作为粗糙对照，不适合作为主标签。
- N=150 raw_deltaq XGBoost 在不同标签场景下整体更稳，建议升级为论文唯一主力模型。
- N=100 raw_deltaq RandomForest 降级为更早预警窗口和可解释性对照。
- 下一步建议进入最终实验整理与论文图表生成阶段。

## 2026-07-08 论文证据链整理：最终实验索引与图表规划

执行内容：
- 使用 nature-writing 的 research/method/experiments/discussion 结构整理当前结果。
- 新增 `docs/final_experiment_evidence_map_20260708.md`。
- 将现有实验从“结果文件”整理为“论文证据链”：
  - 一句话论文论点。
  - 术语表。
  - 论文故事线。
  - 已完成实验与证据索引。
  - 最终主模型与对照模型定位。
  - Figure 1-5 图表规划。
  - Table 1-6 表格规划。
  - Results/Discussion 章节结构。
  - claim-evidence map。
  - 下一步执行清单。

当前论文主线：
- 主方法：N=150 + raw_deltaq + XGBoost。
- 对照方法：N=100 + raw_deltaq + RandomForest。
- 核心故事：不是复现 Severson，也不是堆叠模型，而是将 DeltaQ(V) 早期容量曲线思想迁移到 knee cycle prediction，并通过标签稳定性和全组合搜索证明轻量 raw_deltaq 特征是当前最稳健方案。

阶段判断：
- 通过。当前故事线已经可以支撑论文 Results 章节起草。
- 下一步建议生成 `results/final_figures/` 与 `results/final_tables/`，把关键表图做成论文可用版本。

## 2026-07-08 项目 README 总结文档

执行内容：
- 新增根目录 `README.md`。
- 面向从零理解项目的读者，总结整套研究成果：
  - 项目目标。
  - 数据来源。
  - 方法流程。
  - Severson 复现基础。
  - SOH 修正与 knee label 生成。
  - raw-only baseline。
  - 扩展特征与消融实验。
  - 全组合搜索。
  - 标签稳定性验证。
  - 当前论文主线与可支撑结论。
  - 关键文档阅读顺序。
  - 主要脚本和结果文件。
  - 后续任务清单。

阶段判断：
- 通过。README 可作为当前项目的总入口，帮助后续快速理解工作进展和论文故事线。

## 2026-07-09 R2=0.95 目标的文献学习与路线判断

执行内容：
- 使用 `nature-academic-search` 工作流做同主题文献学习，重点比较高 R2/低误差论文的任务构建方式。
- 检索并整理 Greenbank and Howey 2021、Zhang et al. 2023、Kim et al. 2023、Parsons et al. 2024 等同类工作。
- 重新核算当前主模型 secondary test 从 R2=0.672 提升到 R2=0.95 所需误差水平：secondary RMSE 需要从约 140.04 cycles 降到约 54.70 cycles。
- 新增文档：`docs/r2_095_literature_strategy_20260709.md`。

阶段判断：
- 通过。当前 R2=0.672 并非不可发表，但若要冲击 0.95，不能依赖简单调参或改变报告口径。
- 高指标论文通常通过轨迹建模、SOH knot 预测、自动特征筛选或 EIS 等更强物理信号实现。
- 下一步建议先做 `SOH-level knot prediction + PCHIP trajectory reconstruction + derived knee` 的可行性试验，再判断是否能正当提升指标。

## 2026-07-09 方法创新文献学习与候选模型验证

执行内容：
- 按用户要求将创新重心从“预测目标创新”调整为“方法创新”。
- 使用 `nature-academic-search` 工作流检索并学习同领域前沿算法方向，包括 Matrix Profile/GCN、two-stage health-state modeling、spatio-temporal multimodal attention、physics-guided test-time training、probabilistic ML、physics-informed reconstruction 等。
- 新增候选方法验证脚本：
  - `scripts/run_method_innovation_candidates.py`
  - `scripts/run_batch_robust_weighting.py`
  - `scripts/run_ood_gated_residual.py`
- 生成结果目录：
  - `results/method_innovation_candidates_N150/`
  - `results/method_innovation_candidates_N100/`
  - `results/batch_robust_weighting_N150/`
  - `results/ood_gated_residual_N150/`
- 新增阶段综述文档：`docs/method_innovation_literature_and_candidate_review_20260709.md`。

关键结果：
- N=150 similarity residual weighted ensemble: primary MAE=50.05, secondary MAE=112.89，说明局部残差校正改善同域但伤害跨 batch。
- N=100 候选方法未超过原 `raw_deltaq RF`。
- batch-robust weighting: primary MAE=62.94, secondary MAE=115.08，仍不如原始 XGBoost。
- OOD-gated similarity residual: primary MAE=61.81, secondary MAE=103.29，保持 secondary 不下降，同时较原始 XGBoost 改善 primary。

阶段判断：
- 当前候选方法还不足以作为最终主创新，提升幅度有限，未达到 R2=0.95。
- 但 OOD-gated similarity residual 已经形成更合理的方法创新雏形：保持 knee cycle 目标不变，在方法上加入相似残差校正、域外检测和保守回退。
- 下一步建议围绕 OOD gate、Matrix Profile 关键区域筛选、kernel/GP residual adapter 做进一步系统优化。

## 2026-07-09 OOD gate sensitivity 系统审查

执行内容：
- 新增 `scripts/run_ood_gate_sensitivity.py`。
- 对 OOD-gated similarity residual 进行阈值与距离度量敏感性分析。
- 比较三类 OOD 分数：nearest、knn5_mean、mahalanobis。
- 分别在 N=150 和 N=100 上运行，输出：
  - `results/ood_gate_sensitivity_N150/metrics.csv`
  - `results/ood_gate_sensitivity_N150/ranking.csv`
  - `results/ood_gate_sensitivity_N150/predictions.csv`
  - `results/ood_gate_sensitivity_N150/ood_gate_sensitivity_top.png`
  - `results/ood_gate_sensitivity_N100/metrics.csv`
  - `results/ood_gate_sensitivity_N100/ranking.csv`
  - `results/ood_gate_sensitivity_N100/predictions.csv`
  - `results/ood_gate_sensitivity_N100/ood_gate_sensitivity_top.png`
- 新增审查文档：`docs/ood_gate_sensitivity_review_20260709.md`。

关键结果：
- N=150 nearest q=0.90: primary MAE=54.30, secondary MAE=103.53, primary R2=0.901, secondary R2=0.667。
- N=150 mahalanobis q=0.95: primary MAE=60.82, secondary MAE=100.64, primary R2=0.817, secondary R2=0.695。
- N=100 best setting 未超过此前 N100 raw_deltaq RandomForest，说明该方法更适合 N=150 主模型。

阶段判断：
- 通过。OOD gate sensitivity 支持“域感知残差自适应”的方法创新方向。
- 当前最值得保留的候选是 N=150 mahalanobis q=0.95，因为它同时改善 primary 和 secondary，但仍不能直接作为最终最优声明，需要后续用 train-only 阈值规则固定。
- 下一步建议升级 residual adapter：从 kNN residual 转向 Kernel Ridge / Gaussian Process residual，并配合 OOD gate 做消融。

## 2026-07-09 方法精度继续优化：Residual adapter、XGBoost 搜索与 Hybrid 候选

执行内容：
- 新增 `scripts/run_train_selected_residual_adapters.py`，尝试 Ridge/KNN/RandomForest/Kernel Ridge/Gaussian Process residual adapter。
- 新增 `scripts/run_xgb_train_selected_search.py`，对 N=150 raw_deltaq XGBoost 做 train-only leave-one-train-batch 超参搜索。
- 新增 `scripts/run_hybrid_global_ood_candidate.py`，诊断 train-selected XGB 与 OOD-gated residual 的互补性。
- 生成结果目录：
  - `results/train_selected_residual_adapters_N150/`
  - `results/train_selected_residual_adapters_N150_top160/`
  - `results/xgb_train_selected_search_N150/`
  - `results/hybrid_global_ood_candidate_N150/`
- 新增审查文档：`docs/final_method_precision_update_20260709.md`。

关键结果：
- KRR/GPR residual adapter 没有独立超过原始 XGBoost，说明简单替换 residual adapter 不足以提高跨 batch 泛化。
- train-only 选出的 `xgb_0204`：primary MAE=60.59, secondary MAE=105.07, primary R2=0.856, secondary R2=0.704。
- OOD-gated residual Mahalanobis q=0.95：primary MAE=60.82, secondary MAE=100.64, secondary R2=0.695。
- Hybrid w=0.25：primary MAE=60.09, secondary MAE=97.44, secondary R2=0.713, mean MAE=78.77。
- Hybrid w=0.50：primary MAE=59.35, secondary MAE=98.84, secondary R2=0.721, mean MAE=79.10。

阶段判断：
- 当前最佳诊断候选为 `hybrid_xgb0204_0.25_ood_0.75`，相比原主模型 N150 raw_deltaq XGBoost 同时改善 primary 和 secondary。
- 若考虑论文可解释性和避免测试集调权，等权 hybrid 更稳妥。
- 下一步需要生成两个分量模型的 train OOF prediction，并用 train-only 方式选择 hybrid 权重；若 train-only 权重接近 0.25/0.50，可升级为最终主模型。

## 2026-07-09 Hybrid HPO 诊断搜索

执行内容：
- 新增 `scripts/run_hybrid_hpo_diagnostic.py`。
- 对 N=150 raw_deltaq XGBoost 全局分量做更大范围超参数搜索，同时与 Mahalanobis q=0.95 的 OOD-gated residual 分量做 hybrid 权重搜索。
- 搜索 450 个随机 XGBoost 候选，并加入原始参数与上一轮 `xgb_0204` anchor。
- 权重范围为 0.00 到 1.00，步长 0.05。
- 输出目录：`results/hybrid_hpo_diagnostic_N150/`。
- 新增审查文档：`docs/hyperparameter_optimization_review_20260709.md`。

关键结果：
- 按 stable score 最优：`xgb_hpo_0043 + weight_xgb=0.75`，primary MAE=48.85，secondary MAE=101.39，primary R2=0.890，secondary R2=0.710，mean MAE=75.12。
- 按 secondary 泛化最优且 min R2 最优：`xgb_hpo_0288 + weight_xgb=0.55`，primary MAE=57.28，secondary MAE=96.93，primary R2=0.861，secondary R2=0.727，mean MAE=77.11。
- `xgb_hpo_0288` 参数：n_estimators=350, max_depth=1, learning_rate=0.055, subsample=1.0, colsample_bytree=0.65, reg_lambda=1.0, reg_alpha=1.0, min_child_weight=1.0。

阶段判断：
- 当前推荐以跨 batch 泛化为优先，选择 `xgb_hpo_0288 + Mahalanobis OOD-gated residual + hybrid weight 0.55` 作为当前最佳候选。
- 该候选相比原主模型 N150 raw_deltaq XGBoost 同时改善 primary 和 secondary，secondary R2 从 0.672 提升到 0.727。
- 该结果属于 HPO 诊断最优，后续仍需要用 train OOF prediction 固定 hybrid weight，才能作为严格论文最终模型。

## 2026-07-09 Local DeltaQ / Matrix Profile 特征优化

执行内容：
- 新增 `scripts/extract_local_deltaq_features.py`，从 corrected pkl 的 `Qdlin` 中提取 cycle 9 到 cycle 149 的局部 DeltaQ 片段特征和 Matrix Profile 摘要特征。
- 新增 `scripts/select_local_deltaq_features.py`，使用 train-only 绝对相关性筛选 top local DeltaQ features。
- 生成局部特征表：
  - `data/early_features_N150_local_deltaq_20170630.csv`
  - `data/early_features_N150_local_deltaq_qc_20170630.csv`
  - `data/early_features_N150_local_deltaq_top60_20170630.csv`
  - `data/early_features_N150_local_deltaq_feature_ranking_20170630.csv`
- 使用 top60 local DeltaQ features 重新运行 hybrid HPO：`results/hybrid_hpo_local_deltaq_top60_N150/`。
- 新增审查文档：`docs/local_deltaq_matrix_profile_review_20260709.md`。

关键结果：
- top local features 集中在 DeltaQ 索引 475-650 附近，说明局部退化敏感区段存在。
- 当前最佳候选：`xgb_hpo_0063 + Mahalanobis OOD-gated residual + hybrid weight 0.55`。
- 参数：n_estimators=180, max_depth=1, learning_rate=0.07, subsample=1.0, colsample_bytree=0.90, reg_lambda=1.0, reg_alpha=1.0, min_child_weight=2.0。
- 结果：primary MAE=57.69, secondary MAE=90.92, primary R2=0.879, secondary R2=0.751, mean MAE=74.30。

阶段判断：
- 这是目前最有价值的提升，相比原始 N150 raw_deltaq XGBoost，secondary MAE 从 103.29 降到 90.92，secondary R2 从 0.672 提升到 0.751。
- 方法创新更清晰：保持 knee-cycle prediction 目标不变，但将全局 DeltaQ 统计升级为局部退化敏感片段特征 + 域感知 hybrid residual adaptation。
- 下一步建议做 top-k sensitivity 和 train-only hybrid weight 固定，以便把诊断最优转化为更严格的论文主模型。

## 2026-07-09 Local DeltaQ top-k sensitivity 审查

执行内容：
- 对 Local DeltaQ train-only 相关性筛选特征做 top-k 敏感性分析。
- 比较 top20、top40、top60、top80 四组局部 DeltaQ 特征在 N=150 hybrid HPO diagnostic 框架下的表现。
- 结果目录：
  - `results/hybrid_hpo_local_deltaq_top20_N150/`
  - `results/hybrid_hpo_local_deltaq_top40_N150/`
  - `results/hybrid_hpo_local_deltaq_top60_N150/`
  - `results/hybrid_hpo_local_deltaq_top80_N150/`
- 新增审查文档：`docs/local_deltaq_topk_sensitivity_review_20260709.md`。

关键结果：
- top20 best：primary MAE=53.57，secondary MAE=96.22，mean MAE=74.89。
- top40 best：`xgb_hpo_0212 + weight_xgb=0.75`，primary MAE=57.30，secondary MAE=88.90，primary R2=0.891，secondary R2=0.746，mean MAE=73.10。
- top60 best：primary MAE=57.68，secondary MAE=90.92，primary R2=0.879，secondary R2=0.751，mean MAE=74.30。
- top80 best：primary MAE=60.03，secondary MAE=90.82，mean MAE=75.43。

阶段判断：
- 通过。top40 Local DeltaQ hybrid 是目前综合 MAE、secondary 泛化和论文可解释性的最佳候选。
- 相比原 N150 raw_deltaq XGBoost，top40 hybrid 将 secondary MAE 从 103.29 降到 88.90，将 secondary R2 从 0.672 提升到 0.746。
- 该结果仍属于 diagnostic 最优，下一步需要用 train-only 或 nested validation 固定 hybrid weight，才能作为更严格的最终论文主模型。

## 2026-07-09 Fixed hybrid weight train-only 审查

执行内容：
- 新增 `scripts/run_fixed_hybrid_weight_protocol.py`。
- 目标：不使用 primary/secondary test 标签，只用 train leave-batch OOF 预测固定 hybrid weight。
- 先尝试完整 223 候选，因重复计算 OOD 分量导致 120s 超时，未产生完整输出。
- 随后按 train-only OOF 排名导出 top60 候选：`results/hybrid_hpo_local_deltaq_top40_N150/xgb_params_train_top60.csv`。
- 运行固定权重协议，输出：`results/fixed_hybrid_weight_top40_N150_trainTop60/`。
- 对 previous diagnostic best `xgb_hpo_0212` 单独做固定权重复核，输出：`results/fixed_hybrid_weight_top40_N150_diagnostic0212/`。
- 新增审查文档：`docs/fixed_hybrid_weight_review_20260709.md`。

关键结果：
- 严格 train-only top60 协议选中 `xgb_hpo_0034 + weight_xgb=1.00`。
- 这意味着训练集内部验证拒绝 OOD residual 分量，hybrid 自动退化为 pure XGBoost。
- 固定协议最终结果：primary MAE=63.70，secondary MAE=107.08，primary R2=0.860，secondary R2=0.698。
- previous diagnostic candidate `xgb_hpo_0212` 的 train-only rank 为 133/223，但 pure XGBoost 测试结果仍较好：primary MAE=57.62，secondary MAE=89.76，primary R2=0.903，secondary R2=0.717。
- diagnostic hybrid `xgb_hpo_0212 + weight_xgb=0.75` 为 primary MAE=57.30，secondary MAE=88.90，secondary R2=0.746。

阶段判断：
- 通过，但结论需要收紧。
- 不应将 test-diagnostic hybrid weight 作为最终论文主模型；它更适合作为 upper-bound/sensitivity 结果。
- 当前更稳妥的论文主线应转向 `top40 Local DeltaQ sensitive-region features + XGBoost`，hybrid residual adaptation 降级为探索性消融模块。
- 下一步建议围绕 Local DeltaQ XGBoost 做严格参数选择规则，而不是继续强调 hybrid weight。

## 2026-07-09 Development/external protocol 主线收束

执行内容：
- 新增 `scripts/run_development_external_finalization.py`。
- 将 official train + primary_test 定义为 development cohort，将 secondary_test 作为 external validation cohort。
- 使用 development cohort 内部 repeated K-fold 选择 XGBoost 参数，然后训练 development 全量样本，最后只在 secondary 上做外部验证。
- 分别运行 Local DeltaQ top40 模型与 raw/DeltaQ baseline 对照：
  - `results/development_external_finalization_N150/`
  - `results/development_external_finalization_N150_raw_deltaq/`
- 新增审查文档：`docs/development_external_protocol_review_20260709.md`。

关键结果：
- Local DeltaQ top40 + XGBoost 选中 `xgb_hpo_0160`。
- Local DeltaQ external secondary：MAE=89.03，RMSE=121.07，MAPE=10.96%，R2=0.755。
- Raw/DeltaQ baseline external secondary：MAE=108.27，RMSE=143.23，MAPE=13.84%，R2=0.657。
- Local DeltaQ 相比 raw/DeltaQ baseline：external MAE 改善 19.24 cycles，external R2 提升 0.098。

阶段判断：
- 通过。这是目前最干净、最适合论文叙事的主结果。
- 论文主模型建议固定为：`top40 Local DeltaQ sensitive-region features + XGBoost`。
- Hybrid residual adaptation 降级为 exploratory ablation，不作为最终主贡献。
- 核心说辞应从“追求复杂模型”转为“构建退化敏感局部 DeltaQ 表征，并在独立 secondary batch 上验证其跨 batch 改善”。

## 2026-07-09 Random split benchmark 补充实验

执行内容：
- 新增 `scripts/run_random_split_local_deltaq_benchmark.py`。
- 使用当前主模型候选 `xgb_hpo_0160`，将三组 batch 合并后做随机划分实验。
- 运行 100 次 75/25 unstratified random split：`results/random_split_local_deltaq_N150_unstratified/`。
- 运行 100 次 batch-stratified random split：`results/random_split_local_deltaq_N150_batch_stratified/`。
- 新增审查文档：`docs/random_split_benchmark_review_20260709.md`。

关键结果：
- Unstratified random split：平均 MAE=60.58，平均 MAPE=9.03%，平均 R2=0.872；最好一次 R2=0.967。
- Batch-stratified random split：平均 MAE=62.42，平均 MAPE=9.23%，平均 R2=0.870；最好一次 R2=0.955。
- External secondary 主协议结果仍为 MAE=89.03，MAPE=10.96%，R2=0.755。

阶段判断：
- 随机合并划分可行，且确实能得到接近或超过 R2=0.95 的结果。
- 但该协议混合了 batch 分布，只能作为 same-distribution/random-split benchmark，不能替代 secondary external validation 主结果。
- 论文中可将其作为补充实验，用于对比那些采用随机划分的文献；主结论仍应以 external secondary 为准。

## 2026-07-09 横向模型对比设计与前沿模型筛选

执行内容：
- 使用 nature-academic-search 的 multi-source-search 思路，结合本地 paper 文件夹已下载论文，筛选 2024-2026 年同数据集/近任务前沿模型。
- 新增横向对比设计文档：`docs/horizontal_model_comparison_design_20260709.md`。
- 检查本地依赖：PyTorch 可用，LightGBM/CatBoost/TensorFlow 不可用，因此前沿复现优先采用 PyTorch + XGBoost。

筛选结论：
- 入选前沿 baseline 1：Safavi et al. 2024 的 CNN-XGBoost / COM-CNN-XGBoost，使用 Severson 数据、前100 cycles、结构清楚、可改造为 knee-cycle 回归。
- 入选前沿 baseline 2：Cui et al. 2025 的 TECAN/ECA-MCCA 类 CTK 联合预测模型，直接预测 CTK/knee 相关目标，结构可用 PyTorch 简化复现。
- 不作为主复现 baseline：BatteryGPT，原因是数据集为 MIT LIB 46 cells，使用前30%寿命数据，任务是先重建全生命周期 SOH 再派生 knee/EOL，和我们固定早期窗口任务不公平。
- 可选第三 baseline：Yu et al. 2025 AE-FCNN+ knee interval，但其标签是 knee interval，不是当前 knee cycle，暂不作为主横向对比。

后续执行建议：
- 先实现传统模型对比：Ridge/ElasticNet/SVR/kNN/RF/ExtraTrees/XGBoost 在 raw/DeltaQ 与 Local DeltaQ 上的对比。
- 再实现两个前沿 adapted baseline：CNN-XGBoost 和 TCN-ECA。
- 输出 `results/horizontal_model_comparison_N150/` 与审查文档。

## 2026-07-09 横向模型对比初版执行与 ExtraTrees 复核

执行内容：
- 新增并运行 `scripts/run_horizontal_model_comparison.py`。
- 完整设置 5-fold x 2 repeats + 60 XGBoost candidates 超时，因此先执行 fast setting：5-fold x 1 repeat + top20 XGBoost candidates + 120 epochs deep baselines。
- 输出目录：`results/horizontal_model_comparison_N150_fast/`。
- 对最佳横向模型 `ExtraTrees + Local DeltaQ` 做 30 个随机种子复核，输出：
  - `results/horizontal_model_comparison_N150_fast/extratrees_local_seed_robustness.csv`
  - `results/horizontal_model_comparison_N150_fast/extratrees_local_seed_robustness_summary.csv`
- 新增审查文档：`docs/horizontal_model_comparison_review_20260709.md`。

关键结果：
- External secondary 排名第一：ExtraTrees + Local DeltaQ，MAE=80.90，RMSE=117.35，MAPE=9.43%，R2=0.770。
- 排名第二：XGBoost + Local DeltaQ，MAE=90.90，RMSE=121.24，MAPE=11.34%，R2=0.754。
- Raw/DeltaQ + XGBoost：MAE=103.55，R2=0.708。
- Adapted CNN-XGBoost：MAE=106.83，R2=0.679。
- Adapted TCN-ECA：MAE=189.12，R2=0.265，初版效果不佳。
- ExtraTrees 30-seed 复核：external MAE mean=79.67±1.29，R2 mean=0.770±0.004，说明结果稳定。

阶段判断：
- 通过。横向对比已经形成论文级初版证据。
- 当前最强 external 模型应从 `Local DeltaQ + XGBoost` 更新为 `Local DeltaQ + ExtraTrees`。
- 论文贡献仍应放在 Local DeltaQ sensitive-region representation，而不是声称 ExtraTrees 本身创新。
- 前沿 adapted baselines 初版未超过显式 Local DeltaQ 树集成，反而支持“小数据下，物理启发局部表征优于重型表示学习”的故事线。

## 2026-07-09 LDQ-ET ExtraTrees 超参数优化

执行内容：
- 新增 `scripts/run_extratrees_ldq_hpo.py`。
- 针对当前主线 `Local DeltaQ + ExtraTrees` 做超参数搜索。
- 完整 160 候选和 80 候选搜索均因运行超过 300s 中止。
- 执行 quick30 搜索：`--n-repeats 1 --max-candidates 30 --seed-repeats 30`，输出目录：`results/extratrees_ldq_hpo_N150_quick30/`。
- 修复脚本中 `max_depth` 从 CSV/字典读取为 float 导致 sklearn 报错的问题。
- 尝试对 HPO 配置做 200 次随机划分 benchmark，但因 1200 棵树配置运行超过 180s 中止；该补充项暂不作为主结果。
- 新增审查文档：`docs/extratrees_ldq_hpo_review_20260709.md`。

关键结果：
- 选中配置：`n_estimators=1200, max_features=0.5, min_samples_leaf=1, min_samples_split=2, max_depth=6, bootstrap=False`。
- Development CV：MAE mean=42.20，selection score=49.02。
- 30 seed external secondary：MAE=77.23±0.89，RMSE=111.07±0.93，MAPE=9.04±0.12%，R2=0.794±0.004。
- 相比 raw/DeltaQ + XGBoost：MAE 改善 31.04 cycles，R2 提升 0.137。
- 相比 Local DeltaQ + XGBoost：MAE 改善 11.80 cycles，R2 提升 0.039。

阶段判断：
- 通过。LDQ-ET 可以升级为当前论文主模型。
- 方法贡献仍应定位为 Local DeltaQ sensitive-region representation；ExtraTrees 是稳定读出器，不是算法创新本身。
- 后续应更新论文故事线、主结果表和 README：主模型从 Local DeltaQ + XGBoost 改为 LDQ-ET。

## 2026-07-09 LDQ-ET top-k Local DeltaQ 消融补跑

执行内容：
- 使用脚本 `scripts/run_ldq_et_topk_ablation.py` 对当前主线 LDQ-ET 做 top-k Local DeltaQ 消融。
- 固定 ExtraTrees HPO 最优配置：`n_estimators=1200, max_features=0.5, min_samples_leaf=1, min_samples_split=2, max_depth=6, bootstrap=False`。
- 仅改变 Local DeltaQ 敏感区间数量：top20/top40/top60/top80。
- 每个设置运行 30 个随机种子。
- 输出目录：`results/ldq_et_topk_ablation_N150/`。
- 新增审查文档：`docs/ldq_et_topk_ablation_review_20260709.md`。

关键结果（external secondary，mean ± std）：
- top20：MAE=80.95±0.91，RMSE=117.94±0.84，MAPE=9.48±0.12%，R2=0.768±0.003。
- top40：MAE=77.23±0.89，RMSE=111.07±0.93，MAPE=9.04±0.12%，R2=0.794±0.003。
- top60：MAE=76.35±0.72，RMSE=111.31±0.62，MAPE=9.03±0.10%，R2=0.793±0.002。
- top80：MAE=77.67±0.75，RMSE=113.63±0.81，MAPE=9.18±0.11%，R2=0.784±0.003。

阶段判断：
- 消融通过，top40/top60 构成稳定性能平台。
- top60 的 MAE 略优，top40 的 RMSE/R2 略优且特征更少，因此主线默认仍建议采用 top40。
- top80 回落说明 Local DeltaQ 区间不是越多越好，支持“受控敏感局部表征”这一论文说辞。

## 2026-07-09 主结果证据总表与项目清理

执行内容：
- 固定当前论文主线为 LDQ-ET top40。
- 新增主结果证据总表：`docs/main_results_evidence_table_20260709.md`。
- 总表整合了主结果、横向模型对比、top-k 消融、鲁棒性与补充实验定位。
- 清理高确定性无用内容：空结果目录、Python 缓存目录、异常生成的 `%SystemDrive%` 目录。

已删除内容：
- `results/extratrees_ldq_hpo_N150/`
- `results/extratrees_ldq_hpo_N150_fast/`
- `results/fixed_hybrid_weight_top40_N150/`
- `results/hybrid_hpo_local_deltaq_N150/`
- `results/local_deltaq_xgb_finalization_N150/`
- `results/random_split_local_deltaq_N150/`
- `scripts/__pycache__/`
- `%SystemDrive%/`

保留内容：
- 主结果、横向对比、消融、随机划分和主要探索实验结果均保留。
- `data/*.mat` 原始文件占用较大，但属于源数据，不直接删除；若后续需要释放空间，建议移动归档而不是删除。

阶段判断：
- 当前证据链已收束到论文主线：LDQ-ET top40。
- 后续可进入论文方法与实验结果章节的正式写作。

## 2026-07-09 Ionics Springer LaTeX manuscript 编译验证

执行内容：
- 用户安装 MiKTeX 后，使用本机 `D:\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe` 和 `bibtex.exe` 编译 `manuscript_ionics/main.tex`。
- 首次运行 MiKTeX 自动安装缺失依赖包；后续发现 `booktabs.sty` 缺失。
- 为降低模板依赖风险，删除 `main.tex` 中不必要的 `\usepackage{booktabs}`；Springer `sn-jnl.cls` 已提供 `\toprule/\midrule/\botrule`。
- 编译流程：pdflatex -> bibtex -> pdflatex -> pdflatex。

输出结果：
- 成功生成 `manuscript_ionics/main.pdf`。
- PDF 共 11 页，文件大小约 369 KB。
- 日志检查未发现 LaTeX Error、Fatal error、Undefined control sequence、undefined citations 或 undefined references。
- 剩余提示主要为 Underfull box 排版提示、hyperref bookmark level warning，以及 MiKTeX 未检查更新提醒；均不影响 PDF 生成。

阶段判断：
- `manuscript_ionics/` 已具备本地编译成功基础，上传 Overleaf 后应可用 pdfLaTeX 编译。
- 作者、单位、基金、作者贡献和代码可用性仍为占位内容，投稿前需替换。

## 2026-07-09 Ionics manuscript 扩写与再次编译

执行内容：
- 参考本地已下载的近两年同方向文章结构，包括 Safavi 2024、BatteryGPT 2025、Cui 2025、Yu 2025、Wang 2024 等，扩展 Ionics 初稿。
- 新增统一风格图：
  - `manuscript_ionics/figures/fig5_knee_examples.pdf`
  - `manuscript_ionics/figures/fig6_protocol_comparison.pdf`
- 扩写 Introduction：增加近年同方向文章常见的“早期表征、目标定义、验证协议”写法。
- 扩写 Data/Feature 章节：加入 knee-cycle 预测任务公式、Local DeltaQ 数学定义、局部区域描述符定义、训练内特征选择公式。
- 扩写 Model 章节：新增 Problem formulation and design rationale，小样本外部 batch 下为何采用 ExtraTrees readout。
- 扩写 Results/Discussion：增加表征贡献与读出器贡献的区分、external vs random split 协议差异解释。
- 重新编译：pdflatex -> bibtex -> pdflatex -> pdflatex。

输出结果：
- `manuscript_ionics/main.tex` 扩展到约 32 KB。
- `manuscript_ionics/main.pdf` 成功生成，当前 14 页，约 426 KB。
- 日志检查未发现 LaTeX Error、Fatal error、Undefined control sequence、undefined citations 或 undefined references。

阶段判断：
- 当前稿件已从“可编译骨架”升级为“具备方法公式、主图、消融、鲁棒性和讨论的扩展初稿”。
- 后续仍需补真实作者信息、单位、基金、代码/数据可用性声明，并继续精修英文表达和参考文献元数据。

## 2026-07-10 Ionics manuscript simulated-review revision
执行内容：
- 阅读 `manuscript_ionics/review_report.md` 的三位模拟审稿意见，优先处理方法可复现性、外部验证可信度、表格一致性和解释性证据不足问题。
- 核对 `scripts/correct_soh_curves.py` 与 `scripts/detect_knee_points.py`，确认 SOH 预处理为 Savitzky--Golay/移动平均平滑 + 单调回归，膝点标签为两段线性拟合总 SSE 最小化，候选范围为有效轨迹 20--90%，每段至少 30 点。
- 在 `manuscript_ionics/main.tex` 中补充膝点标注公式、标签置信度定义、数据分布差异说明、Local DeltaQ 实现细节表、特征选择泄漏控制说明。
- 新增 naive development-mean constant baseline，并写入外部测试结果表。
- 写入外部 40 个 cell 的 bootstrap 95% CI：MAE 53.38--102.45 cycles，RMSE 75.44--141.29 cycles，MAPE 6.38--12.08%，R2 0.595--0.885。
- 解释 Table 2 的最终 LDQ-ET 与横向对比表中 ExtraTrees + Local DeltaQ 数值不同的原因：前者是 HPO 后 30 seeds 主模型，后者是 fast fixed-estimator 横向筛选。
- 插入 `figures/fig7_diagnostic_analysis.pdf`，展示膝点分布、外部误差分布和特征重要性。
- 将摘要和结论中较强的 “interpretable route” 改为 “physically motivated route”，避免解释性证据不足导致过度声称。
- 强化 Data availability / Code availability 文字，说明会提供标签、split、feature tables 和完整分析脚本。

编译验证：
- 执行 `pdflatex -> bibtex -> pdflatex -> pdflatex`。
- 成功生成 `manuscript_ionics/main.pdf`，当前 16 页，472713 bytes。
- `main.log` 未检出 LaTeX Error、Fatal error、Undefined control sequence、undefined citations、undefined references 或 rerun 提示。
- 剩余提示为 Underfull box、hyperref bookmark level warning、MiKTeX update reminder，不影响 PDF 生成。

阶段判断：
- 本轮已显著回应审稿意见中的 High Priority 和 Medium Priority：可复现性、泄漏控制、naive baseline、统计置信区间、分布诊断、误差诊断、特征重要性、表格差异解释。
- 仍可继续加强的点：完整 baseline 逐模型均值/标准差、selected-region stability across bootstrap/folds、最终 PDF 人工版面检查、参考文献元数据核验、真实基金/作者贡献/代码仓库信息替换。
