# Retinal Movie Decoding — Project README

Baseline 5 作业实现进度：**线性解码阶段完成**

数据来源：蝾螈视网膜神经群体电影响应数据集  
论文参考：Kim et al. 2021, *Neural Computation* — "Nonlinear Decoding of Natural Images From Large-Scale Primate Retinal Ganglion Recordings"  
数据 DOI：https://doi.org/10.5061/dryad.4qrfj6qm8

---

## 文件结构

```
project/
├── dataset/                       原始数据（从 Dryad 下载）
│   ├── movieBinnedSpiking.mat
│   ├── binaryCheckerboard.mat
│   ├── MultipleMoviesStim_1_tree.avi
│   ├── MultipleMoviesStim_2_water.avi
│   ├── MultipleMoviesStim_3_grasses.avi
│   ├── MultipleMoviesStim_4_fish.avi
│   └── MultipleMoviesStim_5_opticflow.avi
│
├── dataset_npz/                 构建好的 .npz 数据集（build_dataset.py 输出）
│   ├── dataset_train.npz          movie 0,1,2 → 训练集
│   ├── dataset_val.npz            movie 3 (Fish) → 验证集
│   ├── dataset_test.npz           movie 4 (Self Motion) → 测试集
│   └── dataset_meta.npz           元数据
│
├── linear_results/              线性模型输出（linear_models.py 输出）
│   ├── metrics_table.csv
│   ├── reconstruction.png
│   ├── metrics_bar.png
│   └── weights_heatmap.png
│
├── explore_salamander_dataset.py  数据探查脚本（已运行，留作参考）
├── build_dataset.py               数据集构建脚本
└── linear_models.py               线性解码模型脚本
```

---

## 环境依赖

```bash
pip install numpy scipy scikit-learn scikit-image matplotlib opencv-python
```

PyTorch 目前仅加载数据时用到，线性模型阶段不强依赖，后续非线性阶段需要安装。

---

## 运行步骤

### Step 1：构建数据集

修改 `build_dataset.py` 顶部的路径，然后运行：

```bash
python build_dataset.py
```

**关键参数：**

| 参数 | 值 | 说明 |
|---|---|---|
| `DATA_DIR` | `D:/BMI/pj/dataset` | 原始数据目录 |
| `OUT_DIR` | `D:/BMI/pj/dataset_npz` | 输出目录 |
| `IMG_SIZE` | 64 | 帧降采样分辨率 |
| `DELAY` | 4 | spike 时间延迟（bin，对应 66.7 ms） |
| `WINDOW` | 1 | 单 bin spike 向量（直接取延迟后那一个 bin） |

**时间对齐逻辑（重要）：**  
STA 标定显示蝾螈视网膜响应延迟 ≈ 66.7 ms（棋盘格实验 30 Hz 下 delay=2 bin）。  
换算到电影数据（60 Hz）→ delay = 4 bin。  
解码第 `t` 帧时，输入为 `spike[t+4]` 单个 bin 的 93 维向量。

**数据集规格：**

| 集合 | 电影 | 样本数 | X 形状 | Y 形状 |
|---|---|---|---|---|
| train | Tree / Water / Grasses | ~3500 | `[N, 93]` | `[N, 64, 64]` |
| val | Fish | ~1190 | `[N, 93]` | `[N, 64, 64]` |
| test | Self Motion | ~1190 | `[N, 93]` | `[N, 64, 64]` |

> movie 0 (Tree) 的视频只有 600 帧，实验中循环播放两次。`build_dataset.py` 自动处理（复制拼接）。

---

### Step 2：运行线性模型

修改 `linear_models.py` 顶部路径，然后运行：

```bash
python linear_models.py
```

**关键参数：**

| 参数 | 值 | 说明 |
|---|---|---|
| `NPZ_DIR` | `D:/BMI/pj/dataset_npz` | 数据集目录 |
| `OUT_DIR` | `D:/BMI/pj/linear_results` | 结果输出目录 |
| `LP_SIGMA` | 3 | 低频目标的高斯模糊 σ（Baseline 5 Stage 1 用） |
| `PCA_K` | 50 | PCA+Ridge 的输出主成分数 |
| `RUN_SLOW` | False | True 时额外运行 MultiTaskLasso（约 2~5 分钟） |

**运行的模型：**

| 模型 | 说明 | 对应论文位置 |
|---|---|---|
| OLS | 无正则化线性回归，作为对照下界 | — |
| Ridge | 岭回归（L2），自动交叉验证选 α | 论文线性 baseline |
| Ridge-LP | 以高斯模糊帧为目标的岭回归 | **Baseline 5 Stage 1** |
| PCA+Ridge | 输出空间 PCA 降维后回归 | — |
| MultiTaskLasso | 群稀疏 L1（`RUN_SLOW=True` 开启） | 论文 LASSO 方法 |

**输出文件：**

- `metrics_table.csv` — 所有模型在测试集上的 MSE / PSNR / SSIM / Pearson 相关系数
- `reconstruction.png` — Ground Truth vs 各模型重建对比图（5 列 = 5 帧，5 行 = GT + 4 模型）
- `metrics_bar.png` — 指标柱状对比图
- `weights_heatmap.png` — Ridge 权重矩阵空间分布（每个神经元对图像各位置的贡献）

---

## 当前结果（测试集）

| 模型 | MSE ↓ | PSNR ↑ | SSIM ↑ | Pearson ↑ |
|---|---|---|---|---|
| OLS | 0.0222 | 17.73 dB | 0.193 | 0.056 |
| Ridge | 0.0195 | 18.20 dB | 0.211 | 0.063 |
| Ridge-LP | 0.0168 | 18.82 dB | 0.267 | 0.030 |
| PCA+Ridge | 0.0168 | 18.79 dB | 0.245 | 0.086 |

**Ridge-LP 额外评测（对比低频 GT）：** SSIM = 0.68，PSNR = 21.45 dB  
→ 说明线性方法解码低频（全局轮廓）质量良好，与论文结论一致。

**关键观察：** 所有线性模型的重建结果视觉上都很模糊，SSIM 在 0.2 左右。这是线性解码的上限，不是 bug——MSE 最小化会预测条件均值（平均图像），高频细节互相抵消。PCA+Ridge 的预测在不同帧之间几乎没有变化，直观验证了这一点。

---

## 下一步计划（Baseline 5 Stage 2 & 3）

```
Stage 1（已完成）  Ridge-LP → 低频重建
                        ↓
Stage 2（待做）    CNN / MLP → 高频残差预测
                   残差目标 = GT 帧 − Ridge-LP 预测
                        ↓
Stage 1 + Stage 2  Combined 重建
                        ↓
Stage 3（待做）    去模糊网络（U-Net）→ 最终重建
```

Stage 2 输入：X（`[N, 93]` spike 向量）  
Stage 2 目标：`Y_highpass = Y_gt − Y_lp_pred`（值域约 `[−0.5, 0.5]`）  
评测时将 Stage 1 输出 + Stage 2 输出相加，再用 MSE / PSNR / SSIM 与 GT 比较。
