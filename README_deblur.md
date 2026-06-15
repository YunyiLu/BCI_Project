# Stage 2 & Stage 3 — 使用说明

Baseline 5 非线性视网膜解码：三阶段管线中的非线性高频解码 + 去模糊网络。

论文：Kim et al. 2021, *Neural Computation* — "Nonlinear Decoding of Natural Images From Large-Scale Primate Retinal Ganglion Recordings"

---

## 文件结构

```
nonlinear_models.py       Stage 2: CNN/MLP 非线性高频残差解码 (训练全流程)
deblur_models.py          Stage 3: ResNet Generator / VGG 感知损失 / U-Net 备选
deblur_train.py           Stage 3: 去模糊网络训练脚本
generate_stage2_preds.py  Stage 2 辅助: 从 checkpoint 生成全量预测 (不需要重新训练)
```

---

## 完整管线

```
dataset_npz/
  dataset_train.npz
  dataset_val.npz
  dataset_test.npz
      │
      ├──► Stage 2: python nonlinear_models.py
      │      ├─ RidgeCV 训练 Ridge-LP
      │      ├─ 训练 CNN 高频残差解码器 (4.8M params)
      │      └─ 输出 nonlinear_hp_results/
      │           preds_train_combined.npy  ──┐
      │           preds_val_combined.npy    ──┤
      │           preds_test_combined.npy   ──┤
      │           best_nonlinear_hp.pt         │
      │           metrics_table.csv            │
      │           reconstruction_hp.png        │
      │                                        │
      └──► Stage 3: python deblur_train.py ◄──┘
             ├─ 加载 Stage 2 combined 预测 (自动检测)
             ├─ 训练 ResNet Generator (7.8M params)
             └─ 输出 deblur_results/
                    best_deblur.pt
                    preds_test_deblurred.npy
                    metrics_table.csv
                    reconstruction_deblur.png
```

---

## 环境

```bash
pip install torch torchvision numpy scipy scikit-learn scikit-image matplotlib --break-system-packages
```

支持 CUDA（自动检测），CPU 也可运行。

---

## Stage 2: 非线性高频解码

### 运行

```bash
# 完整训练 (CNN, 默认)
python nonlinear_models.py

# 快速验证流程正确 (3 epoch)
python nonlinear_models.py --dry_run

# 使用 MLP 对照
python nonlinear_models.py --model MLP

# 调整超参
python nonlinear_models.py --lr 5e-5 --weight_decay 1e-4 --epochs 300 --lambda_grad 0.2
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | `CNN` | CNN 或 MLP |
| `--npz_dir` | `dataset_npz` | 数据集目录 |
| `--out_dir` | `nonlinear_hp_results` | 输出目录 |
| `--lp_sigma` | `3` | 高斯模糊 sigma |
| `--epochs` | `200` | 最大训练轮数 |
| `--lr` | `5e-5` | 学习率 |
| `--weight_decay` | `5e-5` | AdamW 权重衰减 |
| `--warmup` | `0` | 前 N 个 epoch 线性 warmup (0=禁用) |
| `--patience` | `8` | 早停容忍度 |
| `--lambda_grad` | `0.5` | 梯度损失权重 |
| `--batch_size` | `64` | 批大小 |
| `--dry_run` | — | 3 epoch 快速验证 |

### CNN 架构

```
Input [B, 279]  ← WINDOW=3: 93 neurons × 3 time bins
  │
  FC Encoder:
    Linear(279→512) → LayerNorm → GELU → Dropout(0.3)
    Linear(512→8192) → GELU → Reshape [B, 128, 8, 8]
  │
  CNN Decoder (3× upsample):
    ConvTranspose(128→96, s=2) → GroupNorm(8) → GELU     [16×16]
    Conv(96→96) → GroupNorm(8) → GELU → Dropout2d(0.2)
    ConvTranspose(96→64, s=2) → GroupNorm(8) → GELU      [32×32]
    Conv(64→64) → GroupNorm(8) → GELU → Dropout2d(0.2)
    ConvTranspose(64→32, s=2) → GroupNorm(8) → GELU      [64×64]
    Conv(32→32) → GELU
    Conv(32→1)                                            [64×64]
  │
Output: 高频残差 [B, 64, 64]
```

参数量: 4,804,961

### 损失函数

```
L_total = MSE(pred_hp, true_hp) + λ_grad * GradL1(pred_hp, true_hp)
```

其中 GradL1 是一阶横向/纵向梯度 L1 损失，帮助恢复高频边缘细节。`λ_grad` 默认 0.1。

### 输出文件

| 文件 | 说明 |
|---|---|
| `best_nonlinear_hp.pt` | 训练好的模型 |
| `preds_{split}_lp.npy` | Ridge-LP 低频预测 |
| `preds_{split}_hp.npy` | CNN/MLP 高频残差预测 |
| `preds_{split}_combined.npy` | Ridge-LP + CNN-HP (Stage 3 输入) |
| `true_hp_{split}.npy` / `true_lp_{split}.npy` | 真实残差/低频 |
| `metrics_table.csv` | 三组指标 (Ridge-LP / HP / Combined) |
| `reconstruction_hp.png` | 6 帧重建对比 (GT / Low-pass / Combined / Error) |
| `residual_visualization.png` | 6 帧残差对比 (True HP / Pred HP / Error) |
| `loss_curve.png` | 训练曲线 |

---

## Stage 2 辅助: 从 checkpoint 生成预测

如果已有训练好的 checkpoint，不需重新训练：

```bash
python C:\Users\njwjx\Documents\BaiduSyncdisk\course_大四\脑机接口\BCI_Project\generate_stage2_preds.py --ckpt nonlinear_hp_results/best_nonlinear_hp.pt --npz_dir dataset_npz --out_dir nonlinear_hp_results
```

与完整训练脚本输出相同格式的预测文件。

---

## Stage 3: 去模糊网络

### 自动对接

Stage 3 **自动检测** `nonlinear_hp_results/` 下的 combined 预测文件。如果找不到，自动回退为仅使用 Ridge-LP 预测。

```bash
# 自动模式 (第一优先级: 命令行指定 > 第二优先级: 自动检测 nonlinear_hp_results/ > 第三优先级: Ridge-LP)
python deblur_train.py

# 手动指定 Stage 2 输出
python deblur_train.py \
    --combined_train nonlinear_hp_results/preds_train_combined.npy \
    --combined_val   nonlinear_hp_results/preds_val_combined.npy \
    --combined_test  nonlinear_hp_results/preds_test_combined.npy

# 指定不同 Stage 2 输出目录
python deblur_train.py --combined_dir my_stage2_output
```

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--npz_dir` | `dataset_npz` | 数据集目录 |
| `--out_dir` | `deblur_results` | 输出目录 |
| `--model` | `resnet` | resnet (论文) 或 unet (备选) |
| `--n_blocks` | `6` | ResNet block 数量 |
| `--epochs` | `32` | 训练轮数 |
| `--lr` | `1e-5` | Adam 学习率 |
| `--batch_size` | `32` | 批大小 |
| `--lambda_perceptual` | `0.5` | VGG 感知损失权重 |
| `--lp_sigma` | `3` | 高斯模糊 sigma |
| `--no_residual` | — | 禁用残差学习 (不推荐) |
| `--combined_train/val/test` | 自动检测 | Stage 2 combined 预测路径 |
| `--combined_dir` | `nonlinear_hp_results` | Stage 2 输出目录 (自动搜索) |

### ResNet Generator 架构

```
Input: combined pred [B, 1, 64, 64]
  → ReflectionPad2d(3) → Conv7×7(1→64) → IN → ReLU
  → Down×2: Conv3×3 stride=2 (64→128→256, 64→32→16)
  → 6× ResnetBlock (256 ch, 16×16)
  → Up×2: ConvTranspose2d stride=2 (256→128→64, 16→32→64)
  → ReflectionPad2d(3) → Conv7×7(64→1)
  → output = input + model(input)  ← 残差学习
Output: deblurred [B, 1, 64, 64]
```

参数量: 7,825,153

### 损失函数

```
L_total = L1(pred, GT) + λ * VGG_perceptual(pred, GT)
```

VGG 感知损失使用预训练 VGG-19 的 `relu3_4` 层（论文 Section 4.5 原文："features from the third convolutional layer of VGG-19 before pooling"）。

### 输出文件

| 文件 | 说明 |
|---|---|
| `best_deblur.pt` | 最佳模型权重 |
| `preds_test_deblurred.npy` | 测试集去模糊预测 |
| `metrics_table.csv` | Combined vs Deblurred 指标对比 |
| `reconstruction_deblur.png` | 6 帧 × 4 列 (GT / Combined / Deblurred / Error) |
| `metrics_comparison.png` | 三阶段指标柱状图 |
| `loss_curve.png` | 训练 Loss / SSIM / LR 曲线 |

---

## 论文对应关系

### Stage 2 (Section 4.4 "Nonlinear Decoder")

| 论文 | 代码 |
|---|---|
| 非线性解码高频残差 (GT - low-pass) | `CNNHighpassDecoder` 以 `Y_hp` 为目标 |
| 低频/高频分解 | Ridge-LP (low-pass) + CNN (high-pass residual) |
| 空间受限逐像素 MLP (2000+ neuron 专用) | CNN (93 neuron, 引入空间归纳偏置) |

### Stage 3 (Section 4.5 "Deblurring Network")

| 论文 | 代码 |
|---|---|
| DeblurGANv2 ResNet Generator, 6 blocks | `ResnetGenerator(n_blocks=6)` |
| Adam lr=1e-5, 每 8 epoch 减半 | `StepLR(step=8, γ=0.5)` |
| L1 + VGG perceptual loss (conv3) | `L1Loss + VGGPerceptualLoss('relu3_4')` |
| 残差学习 | `learn_residual=True` |
| 无对抗损失 | 无 discriminator |

---

## 一键运行

```bash
# 1. Stage 2: 训练 CNN 高频解码器 (约 15-30 min, GPU)
python nonlinear_models.py

# 2. Stage 3: 训练去模糊网络 (约 10-20 min, GPU)
#    自动检测 Stage 2 输出, 无需指定路径
python deblur_train.py

# 结果在:
#   nonlinear_hp_results/  ← Stage 2 输出
#   deblur_results/        ← Stage 3 输出
```