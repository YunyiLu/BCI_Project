"""
Stage 3: Deblurring Network — 训练与评测脚本
基于 Kim et al. 2021, Neural Computation — Baseline 5, Section 4.5

论文对应参数:
  - 模型: DeblurGANv2 ResNet Generator, 6 个 ResNet block
  - 优化器: Adam, lr=1e-5, 每 8 epoch 减半
  - 训练轮数: 32 epochs
  - 损失: L1 像素损失 + VGG-19 感知损失 (conv3_4)
  - 无对抗损失 (论文明确排除)
  - 残差学习: output = input + model(input)

数据流:
  Input (combined_pred) = Ridge-LP 预测 + CNN-HP 残差预测
  Target = 真实帧 (GT)
  Output (deblurred) = DeblurNet(combined_pred)

用法:
  python deblur_train.py
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import RidgeCV
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from deblur_models import ResnetGenerator, VGGPerceptualLoss, DeblurLoss, UNetDeblur, count_parameters

# ============================================================
# 配置参数
# ============================================================

# --- 路径 ---
NPZ_DIR = 'dataset_npz'                          # .npz 数据集目录
OUT_DIR = 'deblur_results'                        # 输出目录
PREDS_DIR = None                                  # 预计算预测目录 (None = 自动生成 Ridge-LP)

# 预计算预测文件 (由 Stage 2 nonlinear_models.py 生成)
# 自动检测 nonlinear_hp_results/preds_{split}_combined.npy
# 也可通过命令行 --combined_{train,val,test} 覆盖
COMBINED_TRAIN_PATH = None   # 自动检测 nonlinear_hp_results/
COMBINED_VAL_PATH = None     # 自动检测 nonlinear_hp_results/
COMBINED_TEST_PATH = None    # 自动检测 nonlinear_hp_results/
COMBINED_DIR = 'nonlinear_hp_results'  # Stage 2 输出目录 (自动搜索)

# --- 数据 ---
LP_SIGMA = 3                    # 高斯模糊 sigma (与 Stage 1 一致)
IMG_SIZE = 64                   # 图像尺寸

# --- 模型 ---
MODEL_TYPE = 'resnet'           # 'resnet' (论文) 或 'unet' (备选)
N_BLOCKS = 6                    # ResNet block 数量 (论文 grid search 后选 6)
NGF = 64                        # 基础通道数
LEARN_RESIDUAL = True           # 残差学习 (论文默认 True)

# --- 训练 ---
BATCH_SIZE = 32
LR = 1e-5                       # 论文: 初始学习率 1e-5
LR_STEP = 8                     # 论文: 每 8 epoch 减半
LR_GAMMA = 0.5                  # 论文: 减半因子
N_EPOCHS = 32                   # 论文: 32 epochs
LAMBDA_PERCEPTUAL = 0.5         # 感知损失权重 (论文未明确, 推荐 0.1~1.0)
WEIGHT_DECAY = 0                # 论文未使用 weight decay

# --- 硬件 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4 if torch.cuda.is_available() else 0


# ============================================================
# 数据加载
# ============================================================

def load_npz_data(npz_dir):
    """加载 train/val/test .npz 数据集"""
    data = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(npz_dir, f'dataset_{split}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 {path}，请先运行 build_dataset.py")
        arr = np.load(path)
        data[split] = {
            'X': arr['X'].astype(np.float32),
            'Y': arr['Y'].astype(np.float32),  # [N, 64, 64]
            'movie_idx': arr['movie_idx'],
            't_idx': arr['t_idx'],
        }
        print(f"[数据] {split}: X {arr['X'].shape}, Y {arr['Y'].shape}")
    return data


def generate_ridge_lp_predictions(data, lp_sigma=3.0):
    """
    生成 Ridge-LP 预测
    复现 Stage 1 的 Ridge 回归 (以高斯模糊帧为目标)
    论文 Section 4.3: Ridge regression with λ via 3-fold CV
    """
    X_train = data['train']['X']
    Y_train = data['train']['Y']

    # 高斯模糊
    Y_train_lp = np.array([gaussian_filter(f, sigma=lp_sigma) for f in Y_train])
    Y_train_lp_flat = Y_train_lp.reshape(len(Y_train_lp), -1)

    # RidgeCV 自动选 α
    print(f"[Ridge-LP] 训练中... X: {X_train.shape}, Y_lp: {Y_train_lp_flat.shape}")
    ridge = RidgeCV(alphas=np.logspace(-2, 5, 20), fit_intercept=False)
    ridge.fit(X_train, Y_train_lp_flat)
    print(f"[Ridge-LP] 最佳 alpha: {ridge.alpha_:.1f}")

    preds = {}
    for split in ['train', 'val', 'test']:
        X = data[split]['X']
        pred_flat = ridge.predict(X)
        preds[split] = pred_flat.reshape(-1, IMG_SIZE, IMG_SIZE).astype(np.float32)
        print(f"[Ridge-LP] {split}: {preds[split].shape}")

    return preds


def load_or_generate_combined_predictions(data, ridge_lp_preds):
    """
    加载或生成 combined 预测 (Ridge-LP + CNN-HP)

    自动检测逻辑:
      1. 优先使用命令行指定的路径
      2. 其次自动搜索 COMBINED_DIR 下 preds_{split}_combined.npy
      3. 回退到 Ridge-LP only
    """
    combined = {}

    for split in ['train', 'val', 'test']:
        path_var = {'train': COMBINED_TRAIN_PATH,
                     'val': COMBINED_VAL_PATH,
                     'test': COMBINED_TEST_PATH}[split]

        # 如果未指定路径，自动搜索 Stage 2 输出目录
        if path_var is None:
            auto_path = os.path.join(COMBINED_DIR, f'preds_{split}_combined.npy')
            if os.path.exists(auto_path):
                path_var = auto_path

        if path_var and os.path.exists(path_var):
            combined[split] = np.load(path_var)
            print(f"[Combined] 加载 {split}: {path_var} → {combined[split].shape}")
        else:
            if path_var:
                print(f"[Combined] 未找到 {path_var}，回退到 Ridge-LP only")
            else:
                print(f"[Combined] {split}: 未指定路径，回退到 Ridge-LP only")
            combined[split] = ridge_lp_preds[split].copy()

    return combined


def build_dataloaders(data, combined_preds, batch_size=32):
    """构建 PyTorch DataLoader"""
    loaders = {}
    for split in ['train', 'val', 'test']:
        # 输入: combined 预测 → 添加 channel 维度 [N, 1, 64, 64]
        X_tensor = torch.from_numpy(combined_preds[split]).unsqueeze(1)
        # 目标: 真实帧
        Y_tensor = torch.from_numpy(data[split]['Y']).unsqueeze(1)

        dataset = TensorDataset(X_tensor, Y_tensor)
        shuffle = (split == 'train')
        loaders[split] = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=shuffle, num_workers=NUM_WORKERS,
                                     pin_memory=(DEVICE.type == 'cuda'))
    return loaders


# ============================================================
# 评测指标
# ============================================================

def compute_metrics(pred, target):
    """
    计算 MSE, PSNR, SSIM
    pred, target: numpy arrays [N, 64, 64] 或 [64, 64]
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]

    mse_list, psnr_list, ssim_list = [], [], []

    for i in range(len(pred)):
        p = np.clip(pred[i], 0, 1)
        t = np.clip(target[i], 0, 1)

        mse = np.mean((p - t) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse + 1e-10))

        # SSIM: data_range=1.0, channel_axis=None for grayscale
        s = ssim(p, t, data_range=1.0)

        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(s)

    return {
        'MSE': np.mean(mse_list),
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list),
    }


# ============================================================
# 训练
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    """验证/测试"""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(1)     # [N, 64, 64]
    all_targets = np.concatenate(all_targets, axis=0).squeeze(1)  # [N, 64, 64]

    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(loader.dataset)

    return metrics, all_preds


def train_deblur(model, train_loader, val_loader, test_loader,
                 n_epochs, lr, lr_step, lr_gamma, device, out_dir):
    """
    训练去模糊网络
    论文: Adam(lr=1e-5), StepLR(step=8, gamma=0.5), 32 epochs
    """
    # 损失函数: L1 + λ * VGG perceptual
    l1_loss = nn.L1Loss()

    # VGG 感知损失
    print("[VGG] 加载预训练 VGG-19...")
    vgg_loss = VGGPerceptualLoss().to(device)

    criterion = lambda pred, target: (
        l1_loss(pred, target) + LAMBDA_PERCEPTUAL * vgg_loss(pred, target)
    )

    # 优化器: Adam (论文)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # 学习率调度: 每 step 个 epoch 减半 (论文)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # 早停
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 12
    no_improve = 0

    history = defaultdict(list)

    print(f"\n{'='*60}")
    print(f"开始训练 (论文配置: Adam lr={lr}, StepLR step={lr_step}, γ={lr_gamma}, {n_epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, n_epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_metrics, _ = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        # 学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ssim'].append(val_metrics['SSIM'])
        history['lr'].append(current_lr)

        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val SSIM: {val_metrics['SSIM']:.4f} | "
              f"LR: {current_lr:.2e}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_deblur.pt'))
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n早停: {patience} epochs 未改善")
            break

    print(f"\n最佳模型: Epoch {best_epoch}, Val Loss: {best_val_loss:.6f}")

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(out_dir, 'best_deblur.pt')))

    return model, history


# ============================================================
# 可视化
# ============================================================

def visualize_results(data, combined_preds, deblurred_preds, out_dir):
    """生成可视化对比图"""
    test_Y = data['test']['Y']

    # --- 1. 重建对比图 (6 帧 × 4 列) ---
    fig, axes = plt.subplots(6, 4, figsize=(12, 18))
    indices = np.linspace(0, len(test_Y) - 1, 6, dtype=int)

    titles = ['Ground Truth', 'Combined\n(Stage 1+2 Input)',
              'Deblurred\n(Stage 3 Output)', 'Error\n(|Deblurred - GT|)']

    for i, idx in enumerate(indices):
        gt = test_Y[idx]
        combined = combined_preds['test'][idx]
        deblurred = deblurred_preds[idx]
        error_map = np.abs(deblurred - gt)

        images = [gt, combined, deblurred, error_map]
        vmins = [0, 0, 0, 0]
        vmaxs = [1, 1, 1, max(0.3, error_map.max())]
        cmaps = ['gray', 'gray', 'gray', 'hot']

        for j, (img, vm, vx, cm) in enumerate(zip(images, vmins, vmaxs, cmaps)):
            axes[i, j].imshow(img, cmap=cm, vmin=vm, vmax=vx)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'reconstruction_deblur.png'), dpi=150)
    plt.close()
    print(f"[可视化] reconstruction_deblur.png 已保存")

    # --- 2. 指标对比柱状图 ---
    # 计算各类预测的指标
    categories = ['Ridge-LP\n(Stage 1)', 'Combined\n(Stage 1+2)', 'Deblurred\n(Stage 3)']
    mse_vals, psnr_vals, ssim_vals = [], [], []

    for preds in [combined_preds['test'], combined_preds['test'], deblurred_preds]:
        # Note: Ridge-LP = combined if no HP preds available; 回调时调整
        m = compute_metrics(preds, test_Y)
        mse_vals.append(m['MSE'])
        psnr_vals.append(m['PSNR'])
        ssim_vals.append(m['SSIM'])

    # Override first: 用 Ridge-LP 单独计算
    ridge_m = compute_metrics(combined_preds['test'], test_Y)
    mse_vals[0] = ridge_m['MSE']
    psnr_vals[0] = ridge_m['PSNR']
    ssim_vals[0] = ridge_m['SSIM']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for ax, vals, title in zip(axes,
                                [mse_vals, psnr_vals, ssim_vals],
                                ['MSE ↓', 'PSNR (dB) ↑', 'SSIM ↑']):
        bars = ax.bar(categories, vals, color=colors, width=0.5)
        ax.set_title(title, fontsize=12)
        # 在柱上标注数值
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'metrics_comparison.png'), dpi=150)
    plt.close()
    print(f"[可视化] metrics_comparison.png 已保存")

    # --- 3. 损失曲线 ---
    # (history 在 train_deblur 中返回, 此处调用 train_deblur 后单独保存)
    # 移至 main 函数


def save_loss_curve(history, out_dir):
    """保存训练损失曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SSIM
    axes[1].plot(history['val_ssim'], color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('Validation SSIM')
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(history['lr'], color='red', marker='o', markersize=3)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=150)
    plt.close()
    print(f"[可视化] loss_curve.png 已保存")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Stage 3: Deblurring Network')
    parser.add_argument('--npz_dir', type=str, default=NPZ_DIR)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    parser.add_argument('--model', type=str, default=MODEL_TYPE, choices=['resnet', 'unet'])
    parser.add_argument('--n_blocks', type=int, default=N_BLOCKS)
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lambda_perceptual', type=float, default=LAMBDA_PERCEPTUAL)
    parser.add_argument('--lp_sigma', type=float, default=LP_SIGMA)
    parser.add_argument('--combined_train', type=str, default=COMBINED_TRAIN_PATH)
    parser.add_argument('--combined_val', type=str, default=COMBINED_VAL_PATH)
    parser.add_argument('--combined_test', type=str, default=COMBINED_TEST_PATH)
    parser.add_argument('--combined_dir', type=str, default=COMBINED_DIR,
                        help='Stage 2 输出目录 (自动搜索 combined 预测)')
    parser.add_argument('--no_residual', action='store_true',
                        help='禁用残差学习')
    args = parser.parse_args()

    # 更新全局变量
    global COMBINED_TRAIN_PATH, COMBINED_VAL_PATH, COMBINED_TEST_PATH, COMBINED_DIR
    COMBINED_TRAIN_PATH = args.combined_train
    COMBINED_VAL_PATH = args.combined_val
    COMBINED_TEST_PATH = args.combined_test
    COMBINED_DIR = args.combined_dir

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 3: Deblurring Network (Baseline 5)")
    print(f"论文: Kim et al. 2021, Neural Computation")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据集...")
    data = load_npz_data(args.npz_dir)

    # 2. 生成 Ridge-LP 预测
    print("\n[2/6] 生成 Ridge-LP 预测 (Stage 1)...")
    ridge_lp_preds = generate_ridge_lp_predictions(data, lp_sigma=args.lp_sigma)

    # 3. 加载/生成 combined 预测
    print("\n[3/6] 加载 Combined 预测 (Stage 1+2)...")
    combined_preds = load_or_generate_combined_predictions(data, ridge_lp_preds)

    # 提示: 如果只有 Ridge-LP
    any_found = any(
        os.path.exists(os.path.join(COMBINED_DIR, f'preds_{s}_combined.npy'))
        for s in ['train', 'val', 'test']
    )
    if not any_found:
        print(f"\n  + 注意: 未找到 Stage 2 combined 预测文件")
        print(f"    -> 将使用 Ridge-LP 预测作为去模糊网络输入")
        print(f"    -> 要使用 CNN-HP combined，先运行: python nonlinear_models.py")
        print(f"    -> 或手动指定: --combined_dir <stage2输出目录>")

    # 4. 构建 DataLoader
    print("\n[4/6] 构建 DataLoader...")
    loaders = build_dataloaders(data, combined_preds, batch_size=args.batch_size)

    # 5. 创建模型
    print("\n[5/6] 创建去模糊网络...")
    learn_residual = not args.no_residual

    if args.model == 'resnet':
        model = ResnetGenerator(
            input_nc=1, output_nc=1, ngf=NGF,
            n_blocks=args.n_blocks, learn_residual=learn_residual
        )
        print(f"  模型: ResNet Generator (DeblurGANv2)")
        print(f"  ResNet blocks: {args.n_blocks}")
    else:
        model = UNetDeblur(in_channels=1, out_channels=1)
        model.learn_residual = learn_residual
        print(f"  模型: U-Net (备选)")

    model = model.to(DEVICE)
    print(f"  参数量: {count_parameters(model):,}")
    print(f"  残差学习: {learn_residual}")

    # 6. 训练
    print("\n[6/6] 训练去模糊网络...")
    model, history = train_deblur(
        model, loaders['train'], loaders['val'], loaders['test'],
        n_epochs=args.epochs, lr=args.lr,
        lr_step=LR_STEP, lr_gamma=LR_GAMMA,
        device=DEVICE, out_dir=args.out_dir,
    )

    # --- 测试集评测 ---
    print("\n" + "=" * 60)
    print("测试集最终评测")
    print("=" * 60)

    l1_loss = nn.L1Loss()
    vgg_loss = VGGPerceptualLoss().to(DEVICE)
    test_criterion = lambda p, t: l1_loss(p, t) + args.lambda_perceptual * vgg_loss(p, t)

    test_metrics, deblurred_preds = evaluate_model(
        model, loaders['test'], test_criterion, DEVICE
    )

    # 基础指标: Combined (Stage 1+2) 输入
    combined_metrics = compute_metrics(combined_preds['test'], data['test']['Y'])

    print(f"\n{'指标':<15} {'Combined (Stage 1+2)':<25} {'Deblurred (Stage 3)':<25} {'提升':<15}")
    print("-" * 80)
    for key in ['MSE', 'PSNR', 'SSIM']:
        before = combined_metrics[key]
        after = test_metrics[key]
        if key == 'MSE':
            improvement = (before - after) / before * 100
            direction = '↓'
        else:
            improvement = (after - before) / before * 100
            direction = '↑'
        print(f"{key:<15} {before:<25.6f} {after:<25.6f} {improvement:+.2f}% {direction}")

    # 保存测试预测
    np.save(os.path.join(args.out_dir, 'preds_test_deblurred.npy'), deblurred_preds)

    # 保存指标表
    metrics_csv = os.path.join(args.out_dir, 'metrics_table.csv')
    with open(metrics_csv, 'w') as f:
        f.write("Model,Target,MSE,PSNR,SSIM,Note\n")
        # Combined
        f.write(f"Combined (Stage 1+2),whole frame,"
                f"{combined_metrics['MSE']:.6f},{combined_metrics['PSNR']:.2f},"
                f"{combined_metrics['SSIM']:.4f},input to deblurring network\n")
        # Deblurred
        f.write(f"Deblurred (Stage 3),whole frame,"
                f"{test_metrics['MSE']:.6f},{test_metrics['PSNR']:.2f},"
                f"{test_metrics['SSIM']:.4f},ResNet-{args.n_blocks} deblurred output\n")

    print(f"\n[保存] {metrics_csv}")

    # --- 可视化 ---
    print("\n生成可视化...")
    visualize_results(data, combined_preds, deblurred_preds, args.out_dir)
    save_loss_curve(history, args.out_dir)

    print(f"\n{'='*60}")
    print(f"Stage 3 完成! 所有输出保存在: {args.out_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()