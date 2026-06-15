"""
nonlinear_highpass_decoder.py

非线性高频残差解码脚本
用于复刻 Kim et al. 2021:
Nonlinear Decoding of Natural Images From Large-Scale Primate Retinal Ganglion Recordings

输入:
  dataset_npz/
    dataset_train.npz
    dataset_val.npz
    dataset_test.npz
    dataset_meta.npz

输出:
  nonlinear_hp_results/
    metrics_table.csv
    reconstruction_hp.png
    residual_visualization.png
    loss_curve.png
    preds_test_lp.npy
    preds_test_hp.npy
    preds_test_combined.npy
    best_nonlinear_hp.pt

核心思路:
  1. Ridge-LP 解码低频图像:
       X -> Y_lowpass

  2. 神经网络解码高频残差:
       X -> Y_highpass = Y - GaussianBlur(Y)

  3. 合成:
       Y_combined = RidgeLP(X) + NN_HP(X)

运行:
  python nonlinear_highpass_decoder.py

依赖:
  pip install numpy scipy scikit-learn scikit-image matplotlib torch
"""

import os
import csv
import time
import random
import numpy as np
from collections import OrderedDict
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────────────────────────

NPZ_DIR = "dataset_npz"
OUT_DIR = "nonlinear_hp_results"

# 与 linear_models.py 保持一致
LP_SIGMA = 3

# 模型选择: "MLP" 或 "CNN"
MODEL_TYPE = "CNN"
# MODEL_TYPE = "MLP"

# 训练超参
SEED = 42
BATCH_SIZE = 64
EPOCHS = 200
# LR = 1e-4           # CNN调参
LR = 5e-5
# WEIGHT_DECAY = 1e-5 # CNN调参
WEIGHT_DECAY = 5e-5
# PATIENCE = 30       # CNN调参
PATIENCE = 8

# 损失函数权重
LAMBDA_MSE = 1.0
# LAMBDA_GRAD = 0.1   # CNN调参
LAMBDA_GRAD = 0.5

# 是否保存预测数组
SAVE_PREDS = True

# Ridge 参数
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

os.makedirs(OUT_DIR, exist_ok=True)

# 统一使用全局变量定义键名
CKPT_KEY_MODEL_STATE = "model_state"
CKPT_KEY_CONFIG = "model_config"
CKPT_KEY_MEAN = "x_mean"
CKPT_KEY_STD = "x_std"

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def load_split(npz_dir, split):
    d = np.load(os.path.join(npz_dir, f"dataset_{split}.npz"))
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    movie_idx = d["movie_idx"]
    t_idx = d["t_idx"]
    return X, Y, movie_idx, t_idx


def make_lowpass(Y_3d, sigma):
    """
    Y_3d: [N, H, W]
    return: Gaussian low-pass [N, H, W]
    """
    return np.stack(
        [gaussian_filter(y, sigma=sigma) for y in Y_3d],
        axis=0
    ).astype(np.float32)


def gradient_loss(pred, target):
    """
    pred, target: torch.Tensor [B, 1, H, W]
    高频残差对边缘敏感，加入简单一阶梯度损失。
    """
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]

    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = (dx_pred - dx_tgt).abs().mean()
    loss_y = (dy_pred - dy_tgt).abs().mean()
    return loss_x + loss_y


def clip01(x):
    return np.clip(x, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────
# 模型定义 (可被外部 import)
# ─────────────────────────────────────────────────────────────

import torch
import torch.nn as nn


class MLPHighpassDecoder(nn.Module):
    """
    简单 MLP:
      X [B,n_input] -> high-pass image [B,1,H,W]

    优点:
      - 和经典 feedforward neural decoder 接近
      - 小数据集下比较稳

    缺点:
      - 没有显式利用图像局部空间结构
    """
    def __init__(self, n_input, img_size):
        super().__init__()
        self.img_size = img_size

        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(2048, img_size * img_size),
        )

    def forward(self, x):
        y = self.net(x)
        y = y.view(-1, 1, self.img_size, self.img_size)
        return y


class CNNHighpassDecoder(nn.Module):
    """
    Spike-to-image CNN decoder:

      X [B,n_input]
        ↓ FC
      feature map [B,128,8,8]
        ↓ ConvTranspose / Conv
      high-pass image [B,1,64,64]

    这种结构比纯 MLP 更强地引入图像空间先验。
    对当前 64x64 帧比较合适。
    """
    def __init__(self, n_input, img_size):
        super().__init__()

        assert img_size == 64, "当前 CNNHighpassDecoder 默认输出 64x64。若 IMG_SIZE 不是 64，需要改上采样结构。"

        self.fc = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            # nn.Dropout(0.15), # CNN调参
            nn.Dropout(0.3),

            nn.Linear(512, 128 * 8 * 8),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Dropout2d(0.2),   # CNN调参添加

            # 16 -> 32
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout2d(0.2),   # CNN调参添加

            # 32 -> 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.fc(x)
        z = z.view(-1, 128, 8, 8)
        y = self.decoder(z)
        return y


# ─────────────────────────────────────────────────────────────
# 以下为训练流程，仅在直接运行时执行
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────
    # Step 0 : 随机种子
    # ─────────────────────────────────────────────────────────────

    set_seed(SEED)

    # ─────────────────────────────────────────────────────────────
    # Step 1 : 加载数据
    # ─────────────────────────────────────────────────────────────

    section("Step 1 : 加载数据集")

    X_train, Y_train, mov_train, t_train = load_split(NPZ_DIR, "train")
    X_val,   Y_val,   mov_val,   t_val   = load_split(NPZ_DIR, "val")
    X_test,  Y_test,  mov_test,  t_test  = load_split(NPZ_DIR, "test")

    meta = np.load(os.path.join(NPZ_DIR, "dataset_meta.npz"), allow_pickle=True)
    IMG_SIZE = int(meta["img_size"])
    NCELL = int(meta["ncell"])
    WINDOW = int(meta["window"])
    N_INPUT = X_train.shape[1]
    N_PIXEL = IMG_SIZE * IMG_SIZE

    for name, X, Y in [
        ("train", X_train, Y_train),
        ("val", X_val, Y_val),
        ("test", X_test, Y_test),
    ]:
        print(f"  {name:5s}  X={X.shape}  Y={Y.shape}  "
              f"X∈[{X.min():.3f},{X.max():.3f}]  "
              f"Y∈[{Y.min():.3f},{Y.max():.3f}]")

    print(f"\n  IMG_SIZE={IMG_SIZE}, NCELL={NCELL}, WINDOW={WINDOW}, N_INPUT={N_INPUT}")

    # ─────────────────────────────────────────────────────────────
    # Step 2 : 构造低频和高频残差目标
    # ─────────────────────────────────────────────────────────────

    section("Step 2 : 构造 low-pass / high-pass 目标")

    print(f"  使用 Gaussian σ={LP_SIGMA} 构造低频目标")

    Y_train_lp = make_lowpass(Y_train, LP_SIGMA)
    Y_val_lp   = make_lowpass(Y_val,   LP_SIGMA)
    Y_test_lp  = make_lowpass(Y_test,  LP_SIGMA)

    Y_train_hp = (Y_train - Y_train_lp).astype(np.float32)
    Y_val_hp   = (Y_val   - Y_val_lp).astype(np.float32)
    Y_test_hp  = (Y_test  - Y_test_lp).astype(np.float32)

    print(f"  Y_train_lp 范围: [{Y_train_lp.min():.3f}, {Y_train_lp.max():.3f}]")
    print(f"  Y_train_hp 范围: [{Y_train_hp.min():.3f}, {Y_train_hp.max():.3f}]")
    print(f"  high-pass mean={Y_train_hp.mean():.5f}, std={Y_train_hp.std():.5f}")

    # ─────────────────────────────────────────────────────────────
    # Step 3 : 训练 Ridge-LP，获得低频预测
    # ─────────────────────────────────────────────────────────────

    section("Step 3 : 训练 Ridge-LP 低频解码器")

    from sklearn.linear_model import RidgeCV

    Y_train_lp_flat = Y_train_lp.reshape(len(Y_train_lp), -1)

    ridge_lp = RidgeCV(alphas=ALPHAS, cv=5)

    t0 = time.time()
    ridge_lp.fit(X_train, Y_train_lp_flat)
    t_ridge = time.time() - t0

    print(f"  Ridge-LP 训练完成，用时 {t_ridge:.1f}s，最优 alpha={ridge_lp.alpha_:.4g}")

    def pred_ridge_lp(X):
        pred = ridge_lp.predict(X).reshape(-1, IMG_SIZE, IMG_SIZE)
        return clip01(pred.astype(np.float32))

    LP_train_pred = pred_ridge_lp(X_train)
    LP_val_pred   = pred_ridge_lp(X_val)
    LP_test_pred  = pred_ridge_lp(X_test)

    print(f"  LP_train_pred: {LP_train_pred.shape}, range=[{LP_train_pred.min():.3f},{LP_train_pred.max():.3f}]")

    # ─────────────────────────────────────────────────────────────
    # Step 4 : 评测函数
    # ─────────────────────────────────────────────────────────────

    section("Step 4 : 定义评测函数")

    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    def evaluate(y_pred_3d, y_true_3d, tag=""):
        """
        y_pred_3d, y_true_3d : numpy [N, H, W]
        返回 dict: MSE, PSNR, SSIM, Pearson
        """
        y_pred_3d = clip01(y_pred_3d)
        y_true_3d = clip01(y_true_3d)

        mse = float(np.mean((y_pred_3d - y_true_3d) ** 2))

        psnr = float(np.mean([
            psnr_fn(y_true_3d[i], y_pred_3d[i], data_range=1.0)
            for i in range(len(y_pred_3d))
        ]))

        ssim = float(np.mean([
            ssim_fn(y_true_3d[i], y_pred_3d[i], data_range=1.0)
            for i in range(len(y_pred_3d))
        ]))

        p = y_pred_3d.ravel()
        q = y_true_3d.ravel()

        if np.std(p) < 1e-8 or np.std(q) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(p, q)[0, 1])

        label = f" [{tag}]" if tag else ""
        print(f"    MSE={mse:.5f}  PSNR={psnr:.2f} dB  "
              f"SSIM={ssim:.4f}  Pearson={corr:.4f}{label}")

        return {
            "MSE": mse,
            "PSNR": psnr,
            "SSIM": ssim,
            "Pearson": corr,
        }


    def evaluate_residual(y_pred_hp, y_true_hp, tag=""):
        """
        高频残差不在 [0,1]，不能直接用 SSIM/PSNR。
        这里主要评估 MSE 和 Pearson。
        """
        mse = float(np.mean((y_pred_hp - y_true_hp) ** 2))

        p = y_pred_hp.ravel()
        q = y_true_hp.ravel()

        if np.std(p) < 1e-8 or np.std(q) < 1e-8:
            corr = 0.0
        else:
            corr = float(np.corrcoef(p, q)[0, 1])

        label = f" [{tag}]" if tag else ""
        print(f"    HP-MSE={mse:.6f}  HP-Pearson={corr:.4f}{label}")

        return {
            "HP_MSE": mse,
            "HP_Pearson": corr,
        }


    print("  评测函数就绪")

    print("\n  Ridge-LP baseline:")
    print("  val :", end="")
    ridge_val_metrics = evaluate(LP_val_pred, Y_val, tag="Ridge-LP vs whole")
    print("  test:", end="")
    ridge_test_metrics = evaluate(LP_test_pred, Y_test, tag="Ridge-LP vs whole")

    # ─────────────────────────────────────────────────────────────
    # Step 5 : PyTorch 数据集
    # ─────────────────────────────────────────────────────────────

    section("Step 5 : 构建 PyTorch Dataset / DataLoader")

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {DEVICE}")

    # 标准化 X，只使用 train 统计量
    X_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    X_std = X_train.std(axis=0, keepdims=True).astype(np.float32) + 1e-6

    X_train_z = (X_train - X_mean) / X_std
    X_val_z   = (X_val   - X_mean) / X_std
    X_test_z  = (X_test  - X_mean) / X_std

    class HighpassDataset(Dataset):
        """
        输入:
          X: 神经元响应 [N, n_input]

        目标:
          Y_hp: 高频残差 [N, H, W]
        """
        def __init__(self, X, Y_hp):
            self.X = X.astype(np.float32)
            self.Y_hp = Y_hp.astype(np.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = torch.from_numpy(self.X[idx])                  # [n_input]
            y = torch.from_numpy(self.Y_hp[idx][None, :, :])   # [1,H,W]
            return x, y


    train_set = HighpassDataset(X_train_z, Y_train_hp)
    val_set   = HighpassDataset(X_val_z,   Y_val_hp)
    test_set  = HighpassDataset(X_test_z,  Y_test_hp)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"  train batches={len(train_loader)}, val batches={len(val_loader)}")

    # ─────────────────────────────────────────────────────────────
    # Step 6 : 实例化非线性高频解码模型
    # ─────────────────────────────────────────────────────────────

    section("Step 6 : 实例化非线性高频解码模型")

    if MODEL_TYPE.upper() == "MLP":
        model = MLPHighpassDecoder(N_INPUT, IMG_SIZE)
    elif MODEL_TYPE.upper() == "CNN":
        model = CNNHighpassDecoder(N_INPUT, IMG_SIZE)
    else:
        raise ValueError(f"未知 MODEL_TYPE={MODEL_TYPE}，请选择 'MLP' 或 'CNN'")

    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  MODEL_TYPE={MODEL_TYPE}")
    print(f"  可训练参数量: {n_params / 1e6:.3f} M")
    print(model)

    # ─────────────────────────────────────────────────────────────
    # Step 7 : 训练非线性高频解码器
    # ─────────────────────────────────────────────────────────────

    section("Step 7 : 训练非线性高频残差解码器")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    mse_loss_fn = nn.MSELoss()

    def run_one_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_grad = 0.0
        n_total = 0

        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                pred = model(x)

                loss_mse = mse_loss_fn(pred, y)
                loss_grad = gradient_loss(pred, y)
                loss = LAMBDA_MSE * loss_mse + LAMBDA_GRAD * loss_grad

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            bs = x.shape[0]
            total_loss += float(loss.item()) * bs
            total_mse += float(loss_mse.item()) * bs
            total_grad += float(loss_grad.item()) * bs
            n_total += bs

        return {
            "loss": total_loss / n_total,
            "mse": total_mse / n_total,
            "grad": total_grad / n_total,
        }


    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
    }

    best_path = os.path.join(OUT_DIR, "best_nonlinear_hp.pt")

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_log = run_one_epoch(train_loader, train=True)
        val_log = run_one_epoch(val_loader, train=False)

        scheduler.step(val_log["loss"])

        history["train_loss"].append(train_log["loss"])
        history["val_loss"].append(val_log["loss"])
        history["train_mse"].append(train_log["mse"])
        history["val_mse"].append(val_log["mse"])

        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:03d}/{EPOCHS}  "
            f"train_loss={train_log['loss']:.6f}  "
            f"val_loss={val_log['loss']:.6f}  "
            f"val_mse={val_log['mse']:.6f}  "
            f"lr={lr_now:.2e}"
        )

        if val_log["loss"] < best_val - 1e-7:
            best_val = val_log["loss"]
            best_epoch = epoch
            bad_epochs = 0

            ckpt = {
                CKPT_KEY_MODEL_STATE: model.state_dict(),
                CKPT_KEY_CONFIG: {
                    "model_type": MODEL_TYPE,
                    "img_size": IMG_SIZE,
                    "n_input": N_INPUT,
                    "lp_sigma": LP_SIGMA,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                },
                CKPT_KEY_MEAN: X_mean,
                CKPT_KEY_STD: X_std,
            }
            torch.save(ckpt, best_path)
            print(f"    ★ 保存最佳模型: {best_path}")
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            print(f"\n  Early stopping: 连续 {PATIENCE} 个 epoch 未提升")
            break

    t_train = time.time() - t0
    print(f"\n  训练完成，用时 {t_train / 60:.1f} min")
    print(f"  最佳 epoch={best_epoch}, best_val_loss={best_val:.6f}")

    # ─────────────────────────────────────────────────────────────
    # Step 8 : 加载最佳模型并预测高频残差
    # ─────────────────────────────────────────────────────────────

    section("Step 8 : 加载最佳模型并生成 high-pass 预测")

    import torch
    import numpy as np

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)

    if CKPT_KEY_MODEL_STATE in ckpt:
        model.load_state_dict(ckpt[CKPT_KEY_MODEL_STATE])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # 如果整个文件就是 state_dict
        model.load_state_dict(ckpt)

    # 如果 checkpoint 中包含配置，可以打印出来（可选）
    if CKPT_KEY_CONFIG in ckpt:
        cfg = ckpt[CKPT_KEY_CONFIG]
        print(f"  加载模型配置: {cfg}")
    # 如果 checkpoint 中包含标准化参数，可以加载（用于可能的推理）
    if CKPT_KEY_MEAN in ckpt and CKPT_KEY_STD in ckpt:
        X_mean_loaded = ckpt[CKPT_KEY_MEAN]
        X_std_loaded = ckpt[CKPT_KEY_STD]
        print(f"  加载标准化参数: mean shape {X_mean_loaded.shape}, std shape {X_std_loaded.shape}")

    model.eval()

    @torch.no_grad()
    def predict_hp(loader):
        preds = []

        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            pred = model(x)                     # [B,1,H,W]
            pred = pred.squeeze(1).cpu().numpy()
            preds.append(pred.astype(np.float32))

        return np.concatenate(preds, axis=0)

    HP_train_pred = predict_hp(train_loader)
    HP_val_pred   = predict_hp(val_loader)
    HP_test_pred  = predict_hp(test_loader)

    print(f"  HP_test_pred: {HP_test_pred.shape}, "
          f"range=[{HP_test_pred.min():.3f},{HP_test_pred.max():.3f}], "
          f"mean={HP_test_pred.mean():.5f}, std={HP_test_pred.std():.5f}")

    # 合成最终图像
    COMB_val_pred  = clip01(LP_val_pred  + HP_val_pred)
    COMB_test_pred = clip01(LP_test_pred + HP_test_pred)

    # ─────────────────────────────────────────────────────────────
    # Step 9 : 评估
    # ─────────────────────────────────────────────────────────────

    section("Step 9 : 评估 Ridge-LP / Nonlinear-HP / Combined")

    results = OrderedDict()

    print("\n  [1] Ridge-LP only")
    print("  val :", end="")
    results["Ridge-LP_val"] = evaluate(LP_val_pred, Y_val, tag="low-pass only")
    print("  test:", end="")
    results["Ridge-LP_test"] = evaluate(LP_test_pred, Y_test, tag="low-pass only")

    print("\n  [2] Nonlinear high-pass residual")
    print("  val :", end="")
    hp_val_metrics = evaluate_residual(HP_val_pred, Y_val_hp, tag="HP residual")
    print("  test:", end="")
    hp_test_metrics = evaluate_residual(HP_test_pred, Y_test_hp, tag="HP residual")

    print("\n  [3] Ridge-LP + Nonlinear-HP")
    print("  val :", end="")
    results["Combined_val"] = evaluate(COMB_val_pred, Y_val, tag="LP + HP")
    print("  test:", end="")
    results["Combined_test"] = evaluate(COMB_test_pred, Y_test, tag="LP + HP")

    # 额外评估：和真实低频/高频对齐情况
    print("\n  [4] 辅助评估")
    print("  LP test vs true LP:", end="")
    lp_vs_true_lp = evaluate(LP_test_pred, Y_test_lp, tag="LP pred vs true LP")
    print("  HP test residual :", end="")
    hp_vs_true_hp = evaluate_residual(HP_test_pred, Y_test_hp, tag="HP pred vs true HP")

    # ─────────────────────────────────────────────────────────────
    # Step 10 : 保存指标
    # ─────────────────────────────────────────────────────────────

    section("Step 10 : 保存指标表")

    metrics_path = os.path.join(OUT_DIR, "metrics_table.csv")

    rows = []

    rows.append({
        "Model": "Ridge-LP",
        "Target": "whole frame",
        **results["Ridge-LP_test"],
        "HP_MSE": "",
        "HP_Pearson": "",
        "Note": f"LP_SIGMA={LP_SIGMA}, alpha={ridge_lp.alpha_}",
    })

    rows.append({
        "Model": f"{MODEL_TYPE}-HP",
        "Target": "high-pass residual",
        "MSE": "",
        "PSNR": "",
        "SSIM": "",
        "Pearson": "",
        **hp_test_metrics,
        "Note": "residual-only evaluation",
    })

    rows.append({
        "Model": f"Ridge-LP + {MODEL_TYPE}-HP",
        "Target": "whole frame",
        **results["Combined_test"],
        "HP_MSE": "",
        "HP_Pearson": "",
        "Note": "combined reconstruction",
    })

    fieldnames = [
        "Model",
        "Target",
        "MSE",
        "PSNR",
        "SSIM",
        "Pearson",
        "HP_MSE",
        "HP_Pearson",
        "Note",
    ]

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  → 已保存: {metrics_path}")

    print("\n  测试集指标汇总:")
    for row in rows:
        print(f"  {row['Model']:<24s}  target={row['Target']:<20s}  "
              f"MSE={row.get('MSE','')}  SSIM={row.get('SSIM','')}  "
              f"Pearson={row.get('Pearson','')}  "
              f"HP_Pearson={row.get('HP_Pearson','')}")

    # ─────────────────────────────────────────────────────────────
    # Step 11 : 可视化
    # ─────────────────────────────────────────────────────────────

    section("Step 11 : 可视化")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 11a. loss curve ──────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{MODEL_TYPE} high-pass decoder loss")
    ax.legend()
    ax.grid(alpha=0.3)

    loss_path = os.path.join(OUT_DIR, "loss_curve.png")
    fig.savefig(loss_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  → 已保存: {loss_path}")

    # ── 11b. 最终重建对比图 ─────────────────────────────────────
    N_SHOW = 6
    idxs = np.linspace(0, len(Y_test) - 1, N_SHOW, dtype=int)

    row_items = [
        ("Ground Truth", Y_test),
        ("True Low-pass", Y_test_lp),
        ("Ridge-LP pred", LP_test_pred),
        ("Combined pred", COMB_test_pred),
        ("Abs Error LP", np.abs(LP_test_pred - Y_test)),
        ("Abs Error Combined", np.abs(COMB_test_pred - Y_test)),
    ]

    fig, axes = plt.subplots(
        len(row_items),
        N_SHOW,
        figsize=(N_SHOW * 2.1, len(row_items) * 2.0)
    )

    for r, (label, imgs) in enumerate(row_items):
        for c, idx in enumerate(idxs):
            ax = axes[r, c]
            img = imgs[idx]

            if "Error" in label:
                ax.imshow(img, cmap="magma", vmin=0, vmax=0.5)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)

            ax.axis("off")

            if c == 0:
                ax.set_ylabel(label, fontsize=8)

            if r == 0:
                ax.set_title(f"test idx={idx}", fontsize=8)

    plt.suptitle(f"Reconstruction: Ridge-LP + {MODEL_TYPE}-HP", fontsize=11)
    plt.tight_layout()

    recon_path = os.path.join(OUT_DIR, "reconstruction_hp.png")
    fig.savefig(recon_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  → 已保存: {recon_path}")

    # ── 11c. 高频残差可视化 ─────────────────────────────────────
    # high-pass residual 有正有负，所以用 RdBu_r
    hp_abs_max = max(
        np.percentile(np.abs(Y_test_hp), 99),
        np.percentile(np.abs(HP_test_pred), 99),
        1e-6
    )

    row_items_hp = [
        ("True HP", Y_test_hp),
        ("Pred HP", HP_test_pred),
        ("HP Error", HP_test_pred - Y_test_hp),
    ]

    fig, axes = plt.subplots(
        len(row_items_hp),
        N_SHOW,
        figsize=(N_SHOW * 2.1, len(row_items_hp) * 2.0)
    )

    for r, (label, imgs) in enumerate(row_items_hp):
        for c, idx in enumerate(idxs):
            ax = axes[r, c]
            img = imgs[idx]

            ax.imshow(img, cmap="RdBu_r", vmin=-hp_abs_max, vmax=hp_abs_max)
            ax.axis("off")

            if c == 0:
                ax.set_ylabel(label, fontsize=8)

            if r == 0:
                ax.set_title(f"test idx={idx}", fontsize=8)

    plt.suptitle(f"High-pass residual decoding: {MODEL_TYPE}", fontsize=11)
    plt.tight_layout()

    hp_vis_path = os.path.join(OUT_DIR, "residual_visualization.png")
    fig.savefig(hp_vis_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  → 已保存: {hp_vis_path}")

    # ─────────────────────────────────────────────────────────────
    # Step 12 : 保存预测数组
    # ─────────────────────────────────────────────────────────────

    section("Step 12 : 保存预测数组")

    if SAVE_PREDS:
        np.save(os.path.join(OUT_DIR, "preds_test_lp.npy"), LP_test_pred)
        np.save(os.path.join(OUT_DIR, "preds_test_hp.npy"), HP_test_pred)
        np.save(os.path.join(OUT_DIR, "preds_test_combined.npy"), COMB_test_pred)

        np.save(os.path.join(OUT_DIR, "true_test_hp.npy"), Y_test_hp)
        np.save(os.path.join(OUT_DIR, "true_test_lp.npy"), Y_test_lp)

        print(f"  → 已保存预测数组至: {OUT_DIR}")
    else:
        print("  SAVE_PREDS=False，跳过预测数组保存")

    # ─────────────────────────────────────────────────────────────
    # 完成
    # ─────────────────────────────────────────────────────────────

    section("完成！")

    print(f"""
      输出目录:
        {OUT_DIR}/

      主要文件:
        metrics_table.csv
        loss_curve.png
        reconstruction_hp.png
        residual_visualization.png
        best_nonlinear_hp.pt
        preds_test_lp.npy
        preds_test_hp.npy
        preds_test_combined.npy

      当前设置:
        MODEL_TYPE = {MODEL_TYPE}
        LP_SIGMA   = {LP_SIGMA}
        BATCH_SIZE = {BATCH_SIZE}
        EPOCHS     = {EPOCHS}
        LR         = {LR}

      如何判断非线性高频解码是否有效:

        1. 看 residual_visualization.png:
           Pred HP 是否能恢复边缘、纹理、局部明暗变化。

        2. 看 metrics_table.csv:
           Combined 的 SSIM / Pearson / PSNR 是否高于 Ridge-LP。

        3. 如果 Combined 没提升:
           - 说明高频预测过弱或噪声过大；
           - 可尝试:
               MODEL_TYPE = "MLP"
               增大 WINDOW
               降低 LR
               增大训练集
               调小 LP_SIGMA
               调整 LAMBDA_GRAD
    """)
