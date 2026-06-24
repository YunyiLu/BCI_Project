"""
full_pipeline.py — 完整三阶段神经图像解码 pipeline (修订版)

Stage 1: Ridge-LP 线性低频解码器
Stage 2: CNN 非线性高频残差解码器
Stage 2.5: 文献式 10-fold 数据增强 → 生成 test-quality combined 输出
Stage 3: 在增强数据集上拟合 PCA + Wiener 参数，对真实测试集去模糊

参考: Kim et al., "Nonlinear Decoding of Natural Images From
Large-Scale Primate Retinal Ganglion Recordings", §4.5
"""

import os, csv, time, random, copy
import numpy as np
from collections import OrderedDict
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────
NPZ_DIR = "D:/MyFiles/4下 智能脑机接口/GroupProject/dataset_npz"
OUT_DIR = "D:/MyFiles/4下 智能脑机接口/GroupProject/full_pipeline_results_v2"

# Stage 1
LP_SIGMA = 3
ALPHAS   = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

# Stage 2
SEED         = 42
BATCH_SIZE   = 64
EPOCHS       = 200
LR           = 2e-4
WEIGHT_DECAY = 5e-5
PATIENCE     = 20
LAMBDA_MSE   = 1.0
LAMBDA_GRAD  = 1.0

# Stage 2.5 数据增强 (文献 §4.5)
N_FOLDS = 10                  # 文献：10 折
FOLD_EPOCHS = 60              # 每折 CNN 训练 epoch (比主训练少以节省时间)
FOLD_PATIENCE = 12

# Stage 3
PCA_K_LIST = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
PCA_MAX    = 1000
WIENER_REG = 1e-5

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 66); print(title); print("=" * 66)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError: pass

def clip01(x): return np.clip(x, 0.0, 1.0)

def make_lowpass(Y_3d, sigma):
    return np.stack([gaussian_filter(y, sigma=sigma) for y in Y_3d]).astype(np.float32)

def load_split(npz_dir, split):
    d = np.load(os.path.join(npz_dir, f"dataset_{split}.npz"))
    return d["X"].astype(np.float32), d["Y"].astype(np.float32)

def gradient_loss(pred, target):
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


# ─────────────────────────────────────────────────────────────
# Step 1: 加载数据
# ─────────────────────────────────────────────────────────────
set_seed(SEED)
section("Step 1 : 加载数据集")

X_train, Y_train = load_split(NPZ_DIR, "train")
X_val,   Y_val   = load_split(NPZ_DIR, "val")
X_test,  Y_test  = load_split(NPZ_DIR, "test")

meta = np.load(os.path.join(NPZ_DIR, "dataset_meta.npz"), allow_pickle=True)
IMG_SIZE = int(meta["img_size"])
N_INPUT  = X_train.shape[1]

for n, X, Y in [("train",X_train,Y_train),("val",X_val,Y_val),("test",X_test,Y_test)]:
    print(f"  {n:5s}  X={X.shape}  Y={Y.shape}")

# Low/High-pass 目标
Y_train_lp = make_lowpass(Y_train, LP_SIGMA)
Y_val_lp   = make_lowpass(Y_val,   LP_SIGMA)
Y_test_lp  = make_lowpass(Y_test,  LP_SIGMA)
Y_train_hp = (Y_train - Y_train_lp).astype(np.float32)
Y_val_hp   = (Y_val   - Y_val_lp).astype(np.float32)
Y_test_hp  = (Y_test  - Y_test_lp).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 评测函数
# ─────────────────────────────────────────────────────────────
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

def evaluate(y_pred, y_true, tag=""):
    y_pred = clip01(y_pred); y_true = clip01(y_true)
    mse  = float(np.mean((y_pred - y_true) ** 2))
    psnr = float(np.mean([psnr_fn(y_true[i], y_pred[i], data_range=1.0) for i in range(len(y_pred))]))
    ssim = float(np.mean([ssim_fn(y_true[i], y_pred[i], data_range=1.0) for i in range(len(y_pred))]))
    p, q = y_pred.ravel(), y_true.ravel()
    corr = float(np.corrcoef(p, q)[0, 1]) if np.std(p) > 1e-8 and np.std(q) > 1e-8 else 0.0
    print(f"    MSE={mse:.5f}  PSNR={psnr:.2f}  SSIM={ssim:.4f}  Pearson={corr:.4f}  [{tag}]")
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "Pearson": corr}


# ─────────────────────────────────────────────────────────────
# PyTorch 基础设施
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import RidgeCV

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  device = {DEVICE}")


class HPDataset(Dataset):
    def __init__(self, X, Yhp): self.X = X.astype(np.float32); self.Y = Yhp.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i][None,:,:])


class CNNHighpassDecoder(nn.Module):
    def __init__(self, n_input, img_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_input, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 128*8*8), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(), nn.Dropout2d(0.1),
            nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(), nn.Dropout2d(0.1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        self.output_scale = nn.Parameter(torch.tensor(3.0))
    def forward(self, x):
        z = self.fc(x).view(-1, 128, 8, 8)
        return self.decoder(z) * self.output_scale


def train_cnn_decoder(X_tr, Yhp_tr, X_va, Yhp_va,
                     epochs, patience, lr=LR, wd=WEIGHT_DECAY, verbose=True):
    """训练一个 CNN 高频解码器，返回最佳模型 state_dict。"""
    X_mean = X_tr.mean(0, keepdims=True).astype(np.float32)
    X_std  = X_tr.std(0, keepdims=True).astype(np.float32) + 1e-6
    X_tr_z = (X_tr - X_mean) / X_std
    X_va_z = (X_va - X_mean) / X_std

    tr_loader = DataLoader(HPDataset(X_tr_z, Yhp_tr), batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    va_loader = DataLoader(HPDataset(X_va_z, Yhp_va), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = CNNHighpassDecoder(X_tr.shape[1], IMG_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    mse_fn = nn.MSELoss()

    def run(loader, train):
        model.train() if train else model.eval()
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            if train: optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                pred = model(x)
                loss = LAMBDA_MSE * mse_fn(pred, y) + LAMBDA_GRAD * gradient_loss(pred, y)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            total += float(loss.item()) * x.shape[0]; n += x.shape[0]
        return total / n

    best_val, bad, best_state = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        tr = run(tr_loader, True); va = run(va_loader, False); scheduler.step(va)
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"    ep{ep:03d}  train={tr:.5f}  val={va:.5f}")
        if va < best_val - 1e-7:
            best_val, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if bad >= patience:
            if verbose: print(f"    early stop @ ep{ep}")
            break

    model.load_state_dict(best_state)
    return model, X_mean, X_std, best_val


@torch.no_grad()
def cnn_predict(model, X, X_mean, X_std):
    Xz = (X - X_mean) / X_std
    ds = HPDataset(Xz, np.zeros((len(X), IMG_SIZE, IMG_SIZE), dtype=np.float32))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    out = []
    model.eval()
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        out.append(model(x).squeeze(1).cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


# ═════════════════════════════════════════════════════════════
# Stage 1: Ridge-LP (使用全部训练集)
# ═════════════════════════════════════════════════════════════
section("Stage 1 : Ridge-LP 线性低频解码器 (全训练集)")
ridge_full = RidgeCV(alphas=ALPHAS, cv=5)
t0 = time.time()
ridge_full.fit(X_train, Y_train_lp.reshape(len(Y_train_lp), -1))
print(f"  Ridge-LP 完成 {time.time()-t0:.1f}s, alpha={ridge_full.alpha_:.4g}")

def lp_predict(ridge, X):
    return clip01(ridge.predict(X).reshape(-1, IMG_SIZE, IMG_SIZE).astype(np.float32))

LP_test_pred = lp_predict(ridge_full, X_test)
LP_val_pred  = lp_predict(ridge_full, X_val)
print("\n  Ridge-LP test:")
m_ridge = evaluate(LP_test_pred, Y_test, tag="Stage1")


# ═════════════════════════════════════════════════════════════
# Stage 2: CNN-HP (使用全部训练集，用于真实测试集)
# ═════════════════════════════════════════════════════════════
section("Stage 2 : CNN 非线性高频残差解码器 (全训练集)")
t0 = time.time()
print(f"  在全部训练集上训练主 CNN (epochs<={EPOCHS}, patience={PATIENCE})...")
main_model, main_mean, main_std, _ = train_cnn_decoder(
    X_train, Y_train_hp, X_val, Y_val_hp,
    epochs=EPOCHS, patience=PATIENCE, verbose=True
)
torch.save(main_model.state_dict(), os.path.join(OUT_DIR, "main_cnn.pt"))
print(f"  主 CNN 训练完成 {(time.time()-t0)/60:.1f} min")

HP_test_pred = cnn_predict(main_model, X_test, main_mean, main_std)
HP_val_pred  = cnn_predict(main_model, X_val,  main_mean, main_std)

COMB_test = clip01(LP_test_pred + HP_test_pred)
COMB_val  = clip01(LP_val_pred  + HP_val_pred)

print("\n  Combined (LP+HP) test:")
m_comb = evaluate(COMB_test, Y_test, tag="Stage2 Combined")


# ═════════════════════════════════════════════════════════════
# Stage 2.5: 文献式 10-fold 数据增强
#   将 train 集分为 N_FOLDS 份，每次留 1 份，用其余 N_FOLDS-1 份
#   重新训练 Ridge & CNN，对留出折预测 → 拼回所有训练样本的
#   "test-quality" combined 输出（无数据泄漏）
# ═════════════════════════════════════════════════════════════
section(f"Stage 2.5 : {N_FOLDS}-fold 数据增强 (文献 §4.5)")

N_train = len(X_train)
rng = np.random.RandomState(SEED)
perm = rng.permutation(N_train)
folds = np.array_split(perm, N_FOLDS)
print(f"  训练集 {N_train} 样本 → {N_FOLDS} 折, 每折≈{len(folds[0])} 样本")

# 用于存放每个训练样本被"留出"时的 combined 预测
COMB_train_aug = np.zeros((N_train, IMG_SIZE, IMG_SIZE), dtype=np.float32)
GT_train_aug   = Y_train.copy()  # 对应的 ground truth (顺序保持原样)

fold_t0 = time.time()
for fi, val_idx in enumerate(folds):
    print(f"\n  --- Fold {fi+1}/{N_FOLDS}  (held-out N={len(val_idx)}) ---")
    tr_idx = np.setdiff1d(perm, val_idx, assume_unique=False)

    X_tr_f, Y_tr_f       = X_train[tr_idx], Y_train[tr_idx]
    X_ho_f               = X_train[val_idx]
    Y_tr_f_lp, Y_tr_f_hp = Y_train_lp[tr_idx], Y_train_hp[tr_idx]

    # —— (a) Fold 内 Ridge-LP ——
    ridge_f = RidgeCV(alphas=ALPHAS, cv=3)
    ridge_f.fit(X_tr_f, Y_tr_f_lp.reshape(len(Y_tr_f_lp), -1))
    LP_ho = lp_predict(ridge_f, X_ho_f)

    # —— (b) Fold 内 CNN-HP ——
    # 使用同一个 val 集做早停 (无泄漏，因为 val 与 train 已分离)
    model_f, mean_f, std_f, bv = train_cnn_decoder(
        X_tr_f, Y_tr_f_hp, X_val, Y_val_hp,
        epochs=FOLD_EPOCHS, patience=FOLD_PATIENCE, verbose=False
    )
    HP_ho = cnn_predict(model_f, X_ho_f, mean_f, std_f)

    # —— (c) 拼接 combined ——
    COMB_ho = clip01(LP_ho + HP_ho)
    COMB_train_aug[val_idx] = COMB_ho

    # 记录该折在留出集上的质量
    ho_mse = float(np.mean((COMB_ho - Y_train[val_idx]) ** 2))
    print(f"    fold val-loss={bv:.5f}  held-out MSE={ho_mse:.5f}")

    # 释放显存
    del model_f
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

print(f"\n  10-fold 增强完成，总用时 {(time.time()-fold_t0)/60:.1f} min")
print(f"  增强数据集: COMB_train_aug={COMB_train_aug.shape}, GT_train_aug={GT_train_aug.shape}")
print(f"  (相比直接评估测试集 N={len(Y_test)}, 用于 Stage3 拟合的样本量提升 ×{N_train/len(Y_test):.1f})")

# 保存增强数据集
np.savez(os.path.join(OUT_DIR, "augmented_dataset.npz"),
         combined=COMB_train_aug, gt=GT_train_aug)
print(f"  → 已保存增强数据集到 augmented_dataset.npz")

# 与真训练集的 fold-out combined 质量
print("\n  增强训练集 vs GT 质量统计:")
m_aug = evaluate(COMB_train_aug, GT_train_aug, tag="augmented train pool")


# ═════════════════════════════════════════════════════════════
# Stage 3: 在增强数据集上拟合 Wiener + PCA, 应用到真实测试集
# ═════════════════════════════════════════════════════════════
section("Stage 3 : PCA + Wiener 去模糊 (在增强数据集上拟合参数)")
from sklearn.decomposition import PCA

# —— 3.1 Wiener: 使用 COMB_train_aug 与 GT_train_aug 估计噪声/信号 PSD ——
def compute_wiener(combined, gt, reg=WIENER_REG):
    noise = combined - gt
    N_psd = np.mean(np.abs(np.fft.fft2(noise)) ** 2, axis=0)
    S_psd = np.mean(np.abs(np.fft.fft2(gt)) ** 2, axis=0)
    return S_psd / (S_psd + N_psd + reg)

def apply_wiener(images, H):
    out = np.zeros_like(images)
    for i in range(len(images)):
        out[i] = np.real(np.fft.ifft2(np.fft.fft2(images[i]) * H))
    return out

print("  在增强数据集上估计 Wiener 滤波器 H(u,v)...")
H = compute_wiener(COMB_train_aug, GT_train_aug, reg=WIENER_REG)
wiener_train_aug = clip01(apply_wiener(COMB_train_aug, H))
wiener_val       = clip01(apply_wiener(COMB_val,  H))
wiener_test      = clip01(apply_wiener(COMB_test, H))

print("\n  Wiener test:")
m_wiener = evaluate(wiener_test, Y_test, tag="Stage3 Wiener")

# —— 3.2 PCA: 在 GT 训练集上拟合自然图像低维子空间 ——
print("\n  在 GT 训练集上拟合 PCA 基底...")
gt_flat = Y_train.reshape(len(Y_train), -1)
n_max = min(PCA_MAX, len(Y_train))
pca = PCA(n_components=n_max)
pca.fit(gt_flat)
cum = np.cumsum(pca.explained_variance_ratio_)
print(f"  90% var → K={np.searchsorted(cum, 0.9)+1}")
print(f"  95% var → K={np.searchsorted(cum, 0.95)+1}")
print(f"  99% var → K={np.searchsorted(cum, 0.99)+1}")

# —— 3.3 在增强数据集上搜索最优 K (而不是仅在小 val 集上) ——
mean_vec = pca.mean_
wiener_aug_flat = wiener_train_aug.reshape(len(wiener_train_aug), -1)
print(f"\n  在增强数据集 (N={len(wiener_train_aug)}) 上搜索最优 K ...")
best_ssim, best_k = -1, n_max
for k in PCA_K_LIST:
    if k > n_max: continue
    C = pca.components_[:k]
    proj = (wiener_aug_flat - mean_vec) @ C.T @ C + mean_vec
    proj_img = clip01(proj.reshape(wiener_train_aug.shape))
    m = evaluate(proj_img, GT_train_aug, tag=f"K={k}")
    if m["SSIM"] > best_ssim:
        best_ssim, best_k = m["SSIM"], k
print(f"\n  Best K = {best_k}  (augmented SSIM={best_ssim:.6f})")

# —— 3.4 在增强数据集上搜索最优 blending α ——
C = pca.components_[:best_k]
proj_aug = clip01(((wiener_aug_flat - mean_vec) @ C.T @ C + mean_vec).reshape(wiener_train_aug.shape))

print(f"\n  在增强数据集上搜索最优 blending α ...")
best_alpha, best_blend_ssim = 1.0, -1
for alpha in np.arange(0.0, 1.01, 0.05):
    blended = clip01(alpha * proj_aug + (1 - alpha) * wiener_train_aug)
    s = float(np.mean([ssim_fn(GT_train_aug[i], blended[i], data_range=1.0)
                       for i in range(len(GT_train_aug))]))
    if s > best_blend_ssim:
        best_blend_ssim, best_alpha = s, alpha
print(f"  Best α = {best_alpha:.2f}  (augmented SSIM={best_blend_ssim:.6f})")

# —— 3.5 应用到真实测试集 ——
wiener_test_flat = wiener_test.reshape(len(wiener_test), -1)
proj_test = clip01(((wiener_test_flat - mean_vec) @ C.T @ C + mean_vec).reshape(wiener_test.shape))

if best_alpha < 1.0:
    final = clip01(best_alpha * proj_test + (1 - best_alpha) * wiener_test)
    final_name = f"PCA(K={best_k}) + Wiener (α={best_alpha:.2f})"
else:
    final = proj_test
    final_name = f"PCA(K={best_k}) only"

print(f"\n  Final pipeline = {final_name}")
print("\n  Stage3 Final test:")
m_final = evaluate(final, Y_test, tag="Stage3 Final")


# ═════════════════════════════════════════════════════════════
# 汇总
# ═════════════════════════════════════════════════════════════
section("汇总：各阶段测试集指标")

summary = OrderedDict([
    ("Stage1 Ridge-LP",            m_ridge),
    ("Stage2 Combined (LP+HP)",    m_comb),
    ("Stage3 Wiener (aug-fit)",    m_wiener),
    ("Stage3 Final (PCA+Wiener)",  m_final),
])

print(f"\n  {'Model':<32s}  {'MSE':<10s}  {'PSNR':<8s}  {'SSIM':<8s}  {'Pearson':<8s}")
print("  " + "-" * 75)
for name, m in summary.items():
    print(f"  {name:<32s}  {m['MSE']:<10.5f}  {m['PSNR']:<8.2f}  {m['SSIM']:<8.4f}  {m['Pearson']:<8.4f}")

# 保存指标
csv_path = os.path.join(OUT_DIR, "metrics_table.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["Model", "MSE", "PSNR", "SSIM", "Pearson"])
    w.writeheader()
    for name, m in summary.items():
        w.writerow({"Model": name, **m})

# 保存预测
np.save(os.path.join(OUT_DIR, "preds_test_stage1_ridge_lp.npy"), LP_test_pred)
np.save(os.path.join(OUT_DIR, "preds_test_stage2_combined.npy"), COMB_test)
np.save(os.path.join(OUT_DIR, "preds_test_stage3_final.npy"),    final)

# 保存 Stage3 参数
np.savez(os.path.join(OUT_DIR, "stage3_params.npz"),
         wiener_H=H, pca_components=pca.components_[:best_k],
         pca_mean=mean_vec, best_K=best_k, best_alpha=best_alpha)

# ─────────────────────────────────────────────────────────────
# 可视化：各 Stage 重建结果对比图
# ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SHOW = 8   # 想展示多少张测试样本，可改
idxs = np.linspace(0, len(Y_test) - 1, N_SHOW, dtype=int)

rows = [
    ("Ground Truth",      Y_test,        None),
    ("Stage1\nRidge-LP",  LP_test_pred,  m_ridge),
    ("Stage2\nLP + CNN-HP", COMB_test,   m_comb),
    ("Stage3\nWiener",    wiener_test,   m_wiener),
    ("Stage3 Final\nPCA+Wiener", final,  m_final),
]

fig, axes = plt.subplots(len(rows), N_SHOW,
                         figsize=(N_SHOW * 1.8, len(rows) * 1.9))

for r, (label, imgs, metric) in enumerate(rows):
    for c, idx in enumerate(idxs):
        ax = axes[r, c]
        ax.imshow(clip01(imgs[idx]), cmap="gray", vmin=0, vmax=1)
        # 关闭刻度但保留坐标轴框，使 ylabel 可显示
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # 左侧加行标签
        if c == 0:
            ax.set_ylabel(label, fontsize=10, rotation=0,
                          ha="right", va="center", labelpad=40)
        # 顶部加列标签 (样本索引)
        if r == 0:
            ax.set_title(f"#{idx}", fontsize=9)
        # 每张子图下方写单样本 PSNR / SSIM (除 GT 外)
        if metric is not None:
            gt = clip01(Y_test[idx]); pr = clip01(imgs[idx])
            psnr_i = psnr_fn(gt, pr, data_range=1.0)
            ssim_i = ssim_fn(gt, pr, data_range=1.0)
            ax.set_xlabel(f"P={psnr_i:.1f}\nS={ssim_i:.3f}", fontsize=7)

plt.suptitle(
    "Reconstruction across pipeline stages\n"
    "Ridge-LP → CNN-HP → 10-fold augmented PCA+Wiener",
    fontsize=12, y=1.00
)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "pipeline_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ 重建对比图已保存: {out_path}")

# 另存一张「全局平均指标条形图」便于汇报
fig2, ax2 = plt.subplots(1, 3, figsize=(12, 3.2))
names   = [n for n, *_ in [(k,) for k in summary.keys()]]
mse_v   = [summary[n]["MSE"]  for n in names]
psnr_v  = [summary[n]["PSNR"] for n in names]
ssim_v  = [summary[n]["SSIM"] for n in names]
colors  = ["#9aa0a6", "#4285f4", "#fbbc04", "#34a853"]

for ax, vals, title in zip(ax2, [mse_v, psnr_v, ssim_v],
                           ["MSE (↓)", "PSNR / dB (↑)", "SSIM (↑)"]):
    bars = ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names],
                       fontsize=8, rotation=0)
    ax.set_title(title, fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "metrics_barplot.png")
fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  ✓ 指标条形图已保存:  {bar_path}")

np.save(os.path.join(OUT_DIR, "preds_test_stage3_wiener.npy"), wiener_test)