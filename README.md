# BCI Project — Retinal Movie Decoding

Baseline 5 课程作业：复现 Kim et al. 2021 (Neural Computation) 的视网膜神经解码 pipeline。

数据：蝾螈视网膜神经节细胞 (RGC) 群体响应，93 neurons, 64x64 movie frames。
数据来源：Dryad (doi:10.5061/dryad.4qrfj6qm8)

## Pipeline 概览

三阶段 pipeline，逐级提升重建质量：

```
Spikes → [Stage 1: Ridge-LP] → [Stage 2: CNN-HP] → [Stage 3: PCA+Wiener] → Reconstructed Frame
```

## Stage 1: 线性低频解码 (Ridge-LP)

**方法**: Ridge 回归 + 低通滤波 (Gaussian σ=3)

对 PSTH (trial-averaged spike counts) 做 Ridge 回归预测 64x64 帧，再用高斯低通滤波平滑输出。Ridge 回归捕捉 spikes 到像素的线性映射，低通滤波去除线性模型无法可靠预测的高频噪声。

**代码**: `linear_models.py`

| Metric | Value |
|--------|-------|
| MSE | 0.01632 |
| PSNR | 18.962 |
| SSIM | 0.2696 |
| Pearson | 0.0318 |

## Stage 2: 非线性高频残差解码 (CNN-HP)

**方法**: CNN 解码高通残差 + 与 Stage 1 低频合并

用 CNN 从 spikes 预测高通残差 (原图 - LP)，然后与 Stage 1 的低频预测相加得到 combined reconstruction。CNN 捕捉 Ridge 无法表达的非线性映射。

**代码**: `nonlinear_opt_c.py`, `nonlinear_models.py`

| Metric | Stage 1 | Stage 1+2 | Δ |
|--------|---------|-----------|---|
| MSE | 0.01632 | 0.01595 | -0.00037 |
| PSNR | 18.962 | 19.117 | +0.155 |
| SSIM | 0.2696 | 0.2873 | +0.0177 |
| Pearson | 0.0318 | 0.0838 | +0.0520 |

### Stage 2 其他尝试

| 方案 | 架构 | SSIM | Pearson | 结论 |
|------|------|------|---------|------|
| TemporalMLP | MLP 直接预测整帧 | 0.2484 | 0.0602 | 比 Stage 1 SSIM 更差，MLP 容量不足以预测高维帧 |
| TemporalCNN | CNN 直接预测整帧 | 0.2624 | 0.0603 | 略差于 Stage 1 SSIM，未利用低频先验 |
| opt_a | CNN-HP (早期超参) | 0.2860 | 0.0776 | 接近最优，但 CNN 结构和学习率未充分调优 |
| **opt_c (最优)** | CNN-HP (调优后) | **0.2873** | **0.0838** | 更深的 CNN + 优化的训练策略 |

关键发现：直接预测整帧（TemporalMLP/CNN）效果不如分频预测（Ridge-LP + CNN-HP），因为分频策略让每个模型专注于各自擅长的频段。

## Stage 3: 图像恢复 (PCA + Wiener)

**最优方法 (V14)**: PCA 子空间去噪 + Wiener 反卷积

1. Wiener 滤波：频域最优线性去模糊，H = S/(S+N+reg)
2. PCA 投影：将图像投影到 GT 训练图像的前 1000 个主成分子空间，去除自然图像分布之外的噪声
3. 融合：10% PCA + 90% Wiener 加权混合

零可学习参数，无过拟合风险。

**代码**: `deblur_train_v14.py`

| Metric | Stage 1+2 | Stage 1+2+3 | Δ |
|--------|-----------|-------------|---|
| MSE | 0.01595 | 0.01455 | -0.00140 |
| PSNR | 19.117 | 19.592 | +0.475 |
| SSIM | 0.2873 | 0.2960 | +0.0087 |
| Pearson | 0.0838 | 0.0846 | +0.0008 |

### Stage 3 对比实验

| Version | Method | SSIM | Pearson | 说明 |
|---------|--------|------|---------|------|
| **V14** | PCA(K=1000) + Wiener | **0.2960** | **0.0846** | 最优，唯一同时提升 SSIM 和 Pearson |
| V9 | Pure Wiener | 0.2958 | 0.0827 | 零参数频域去模糊，MSE-optimal 线性滤波 |
| V10 | DeblurGANv2 (论文复刻) | 0.1376 | 0.0521 | 5M 参数，10-fold CV，严重过拟合退化 |

### Stage 3 其他尝试（已删除，结果记录于此）

| Version | Method | SSIM | 结论 |
|---------|--------|------|------|
| V3 | ResNet Generator (9 blocks) | 0.215 | 过拟合，训练 loss 下降但 val 恶化 |
| V4 | U-Net (skip connections) | 0.220 | 同样过拟合，skip connection 未能缓解 |
| V5 | Lightweight CNN (3 layers) | 0.235 | 减少参数量仍然过拟合 |
| V6 | Constrained Residual | 0.240 | 约束残差尺度，但训练中约束被突破 |
| V7 | TV-regularized CNN | 0.238 | Total Variation 正则，效果有限 |
| V8 | Laplacian Gain (10 params) | 0.234 | 仅 10 个参数仍然过拟合（gains 爆炸） |
| V11 | Wiener + Gamma Correction | 0.2935 | Gamma 校正降低了 SSIM |
| V12 | Combined + Wiener Blend | 0.2958 | 最优 alpha=1.0，即纯 Wiener |
| V13 | Spectral Density Matching | 0.2958 | 频谱匹配无改善，退化为纯 Wiener |
| V15 | Wiener + CLAHE | 0.2958 | CLAHE 自适应直方图均衡无改善 |

**关键结论**:
- 所有学习型方法（V3-V8, V10）均过拟合：93 neurons / 3588 samples 的信息瓶颈使得网络只能记忆训练集噪声
- 即使仅 10 个可学习参数（V8 Laplacian Gain）也会过拟合
- 后处理方法（V11-V13, V15）中任何偏离 Wiener 最优解的尝试都导致退化
- V14 的成功在于 PCA 投影不学习 input→output 映射，而是利用 GT 统计分布做去噪

V10 复现了论文原始的 DeblurGAN Stage 3，但在 93 neurons / 3588 samples 的小数据集上严重过拟合（SSIM 从 0.287 降至 0.138）。论文使用 2094 neurons + 9900 samples，信息量远大于本数据集，因此深度网络方法不适用于当前规模。

## 最终结果

| Stage | Method | SSIM | PSNR | Pearson |
|-------|--------|------|------|---------|
| 1 | Ridge-LP | 0.2696 | 18.962 | 0.0318 |
| 1+2 | + CNN-HP | 0.2873 | 19.117 | 0.0838 |
| **1+2+3** | **+ PCA+Wiener** | **0.2960** | **19.592** | **0.0846** |

## 文件结构

```
├── linear_models.py          # Stage 1: Ridge-LP
├── linear_results/           # Stage 1 结果
├── nonlinear_models.py       # Stage 2: CNN 模型定义
├── nonlinear_opt_c.py        # Stage 2: 训练脚本
├── nonlinear_results/        # Stage 2 早期实验
├── nonlinear_hp_results/     # Stage 2 完整结果
├── opt_c_results/            # Stage 2 最优配置结果
├── deblur_train_v9.py        # Stage 3: Wiener
├── deblur_train_v10.py       # Stage 3: DeblurGANv2 (论文复刻)
├── deblur_train_v14.py       # Stage 3: PCA+Wiener (最优)
├── deblur_results_v9/        # V9 结果
├── deblur_results_v10/       # V10 结果
├── deblur_results_v14/       # V14 结果
├── build_dataset.py          # 数据预处理
├── yunyi_data/               # 数据集
└── neco_a_01395.pdf          # 参考论文
```

## 参考文献

Kim, Y. J., Brackbill, N., Bhatt, P. J., & Bhatt, P. J. (2021). Retinal Movie Decoding. *Neural Computation*, 33(8).
