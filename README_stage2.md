# Retinal Movie Decoding — Project README

Baseline 5 作业实现进度：**非线性解码（CNN） + Combine 阶段完成**，去模糊网络待扩展。

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
├── nonlinear_hp_results/ 非线性高频解码输出
│ ├── metrics_table.csv
│ ├── reconstruction_hp.png
│ ├── residual_visualization.png
│ ├── loss_curve.png
│ ├── best_nonlinear_hp.pt
│ ├── preds_test_lp.npy
│ ├── preds_test_hp.npy
│ └── preds_test_combined.npy
│
├── explore_salamander_dataset.py  数据探查脚本（已运行，留作参考）
├── build_dataset.py               数据集构建脚本
├── linear_models.py               线性解码模型脚本
└── nonlinear_models.py          非线性高频解码脚本（CNN / MLP）
```

---

## 环境依赖

```bash
pip install numpy scipy scikit-learn scikit-image matplotlib opencv-python
```

PyTorch 用于非线性模型训练（支持 CUDA 加速）。

---

## 运行步骤

### Step 3：运行非线性高频解码

修改 `Non-Linear-HaoSong.py` 顶部的路径，然后运行：

```bash
python nonlinear_models.py
```

**关键参数：**

| 参数 | 值 | 说明 |
|---|---|---|
| `MODEL_TYPE` | `"CNN"` | 推荐 CNN（MLP 效果差） |
| `WINDOW` | 3 | 通过 `build_dataset.py` 重新生成 |
| `LP_SIGMA` | 3 | 与线性阶段一致 |
| `BATCH_SIZE` | 64 | 可调 |
| `LR` | 5e-5 | 初始学习率 |
| `WEIGHT_DECAY` | 5e-5 | L2 正则化 |
| `PATIENCE` | 8 | 早停耐心值 |
| `LAMBDA_GRAD` | 0.5 | 梯度损失权重 |
| `Dropout`(FC) | 0.3 | FC 层 dropout |
| `Dropout2d` (decoder) | 0.2 | 卷积层 dropout2d |

**输出文件（`nonlinear_hp_results/`）：**  

`metrics_table.csv` — Ridge-LP、非线性残差、组合重建的测试集指标
`reconstruction_hp.png` — 原图、真低频、Ridge-LP 预测、组合预测、误差图对比
`residual_visualization.png` — 真实高频残差、预测高频残差、残差误差（红蓝图）
`loss_curve.png` — 训练/验证损失曲线
`best_nonlinear_hp.pt` — 最佳模型权重
`preds_test_lp.npy` / `preds_test_hp.npy` / `preds_test_combined.npy` — 测试集预测数组

**线性基线结果（WINDOW=1）：**

| 模型 | MSE ↓ | PSNR ↑ | SSIM ↑ | Pearson ↑ |
|---|---|---|---|---|
| OLS | 0.0222 | 17.73 dB | 0.193 | 0.056 |
| Ridge | 0.0195 | 18.20 dB | 0.211 | 0.063 |
| Ridge-LP | 0.0168 | 18.82 dB | 0.267 | 0.030 |
| PCA+Ridge | 0.0168 | 18.79 dB | 0.245 | 0.086 |

> Ridge-LP 对真实低频图评估：SSIM = 0.68，PSNR = 21.45 dB，证明线性解码低频有效。

**非线性基线结果（WINDOW=3）：**

| 模型 | MSE ↓ | PSNR ↑ | SSIM ↑ | 残差Pearson ↑ |
|---|---|---|---|---|
| Ridge-LP (baseline) | 0.01632 | 18.96 dB | 0.2696 | — |
| CNN-HP (WINDOW=3) | 0.01596 | 19.13 dB | 0.2860 | 0.253 |
| MLP-HP (WINDOW=3) | 0.01635 | 18.99 dB | 0.2707 | 0.212 |

> 结论：CNN 非线性高频解码使 SSIM 相对提升 6.1%，残差预测 Pearson 达 0.25，证明模型有效捕捉了边缘纹理。MLP 因缺乏空间归纳偏置，几乎没有提升。

**时间窗口对比（CNN 模型）：**

| WINDOW | 输入维度 | Combined SSIM | 残差Pearson | 说明 |
|---|---|---|---|---|
| 1 | 93 | 0.2846 | 0.258 | 基线良好 |
| 3 | 279 | 0.2860 | 0.253 | 最优 |
| 5 | 465 | 0.2832 | 0.256 | 略降，冗余增加 |

> 生理意义：3 个 bin（约 50 ms）覆盖了 RGC 对刺激响应的主要窗口，与延迟 66.7 ms 基本匹配；过长时间窗口引入噪声。

---

## 下一步计划（Baseline 5 Stage 3）

```
Stage 1（已完成）  Ridge-LP → 低频重建
                        ↓
Stage 2（已完成）    CNN → 高频残差预测 → 组合得到 combined （已得到`preds_test_combined.npy`）
                        ↓
Stage 3（待做）    去模糊网络（U-Net）→ 最终重建
```