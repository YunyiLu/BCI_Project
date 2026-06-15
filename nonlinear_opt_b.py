"""
nonlinear_opt_b.py

Optimized variant B of nonlinear high-pass residual decoder.

Changes from nonlinear_models_yunyi.py:
  1. Residual blocks in decoder (after each ConvTranspose2d: Conv3x3->GN->GELU->Conv3x3->GN + skip)
  2. CosineAnnealingWarmRestarts scheduler (T_0=30, T_mult=2) instead of ReduceLROnPlateau
  3. EPOCHS=300, PATIENCE=25
  4. lr=5e-5, gradient loss weight=0.5 (unchanged)

Output: ./opt_b_results/
"""

import os
import csv
import time
import random
import numpy as np
from collections import OrderedDict
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------

NPZ_DIR = "./yunyi_data/dataset_npz"
OUT_DIR = "./opt_b_results"

LP_SIGMA = 3
MODEL_TYPE = "CNN"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 300
LR = 5e-5
WEIGHT_DECAY = 5e-5
PATIENCE = 25

LAMBDA_MSE = 1.0
LAMBDA_GRAD = 0.5

SAVE_PREDS = True
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

os.makedirs(OUT_DIR, exist_ok=True)

CKPT_KEY_MODEL_STATE = "model_state"
CKPT_KEY_CONFIG = "model_config"
CKPT_KEY_MEAN = "x_mean"
CKPT_KEY_STD = "x_std"

# -----------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------

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
    return np.stack(
        [gaussian_filter(y, sigma=sigma) for y in Y_3d], axis=0
    ).astype(np.float32)


def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = (dx_pred - dx_tgt).abs().mean()
    loss_y = (dy_pred - dy_tgt).abs().mean()
    return loss_x + loss_y


def clip01(x):
    return np.clip(x, 0.0, 1.0)

# -----------------------------------------------------------------
# Step 0: Seed
# -----------------------------------------------------------------

set_seed(SEED)

# -----------------------------------------------------------------
# Step 1: Load data
# -----------------------------------------------------------------

section("Step 1 : Load dataset")

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
          f"X in [{X.min():.3f},{X.max():.3f}]  "
          f"Y in [{Y.min():.3f},{Y.max():.3f}]")

print(f"\n  IMG_SIZE={IMG_SIZE}, NCELL={NCELL}, WINDOW={WINDOW}, N_INPUT={N_INPUT}")

# -----------------------------------------------------------------
# Step 2: Low-pass / high-pass targets
# -----------------------------------------------------------------

section("Step 2 : Construct low-pass / high-pass targets")

print(f"  Gaussian sigma={LP_SIGMA}")

Y_train_lp = make_lowpass(Y_train, LP_SIGMA)
Y_val_lp   = make_lowpass(Y_val,   LP_SIGMA)
Y_test_lp  = make_lowpass(Y_test,  LP_SIGMA)

Y_train_hp = (Y_train - Y_train_lp).astype(np.float32)
Y_val_hp   = (Y_val   - Y_val_lp).astype(np.float32)
Y_test_hp  = (Y_test  - Y_test_lp).astype(np.float32)

print(f"  Y_train_lp range: [{Y_train_lp.min():.3f}, {Y_train_lp.max():.3f}]")
print(f"  Y_train_hp range: [{Y_train_hp.min():.3f}, {Y_train_hp.max():.3f}]")
print(f"  high-pass mean={Y_train_hp.mean():.5f}, std={Y_train_hp.std():.5f}")

# -----------------------------------------------------------------
# Step 3: Ridge-LP
# -----------------------------------------------------------------

section("Step 3 : Train Ridge-LP")

from sklearn.linear_model import RidgeCV

Y_train_lp_flat = Y_train_lp.reshape(len(Y_train_lp), -1)

ridge_lp = RidgeCV(alphas=ALPHAS, cv=5)

t0 = time.time()
ridge_lp.fit(X_train, Y_train_lp_flat)
t_ridge = time.time() - t0

print(f"  Ridge-LP done in {t_ridge:.1f}s, best alpha={ridge_lp.alpha_:.4g}")

def pred_ridge_lp(X):
    pred = ridge_lp.predict(X).reshape(-1, IMG_SIZE, IMG_SIZE)
    return clip01(pred.astype(np.float32))

LP_train_pred = pred_ridge_lp(X_train)
LP_val_pred   = pred_ridge_lp(X_val)
LP_test_pred  = pred_ridge_lp(X_test)

print(f"  LP_train_pred: {LP_train_pred.shape}, range=[{LP_train_pred.min():.3f},{LP_train_pred.max():.3f}]")

# -----------------------------------------------------------------
# Step 4: Evaluation functions
# -----------------------------------------------------------------

section("Step 4 : Evaluation functions")

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

def evaluate(y_pred_3d, y_true_3d, tag=""):
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
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "Pearson": corr}


def evaluate_residual(y_pred_hp, y_true_hp, tag=""):
    mse = float(np.mean((y_pred_hp - y_true_hp) ** 2))
    p = y_pred_hp.ravel()
    q = y_true_hp.ravel()
    if np.std(p) < 1e-8 or np.std(q) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(p, q)[0, 1])
    label = f" [{tag}]" if tag else ""
    print(f"    HP-MSE={mse:.6f}  HP-Pearson={corr:.4f}{label}")
    return {"HP_MSE": mse, "HP_Pearson": corr}


print("  Evaluation functions ready")
print("\n  Ridge-LP baseline:")
print("  val :", end="")
ridge_val_metrics = evaluate(LP_val_pred, Y_val, tag="Ridge-LP vs whole")
print("  test:", end="")
ridge_test_metrics = evaluate(LP_test_pred, Y_test, tag="Ridge-LP vs whole")

# -----------------------------------------------------------------
# Step 5: PyTorch Dataset
# -----------------------------------------------------------------

section("Step 5 : PyTorch Dataset / DataLoader")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {DEVICE}")

X_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
X_std = X_train.std(axis=0, keepdims=True).astype(np.float32) + 1e-6

X_train_z = (X_train - X_mean) / X_std
X_val_z   = (X_val   - X_mean) / X_std
X_test_z  = (X_test  - X_mean) / X_std

class HighpassDataset(Dataset):
    def __init__(self, X, Y_hp):
        self.X = X.astype(np.float32)
        self.Y_hp = Y_hp.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y_hp[idx][None, :, :])
        return x, y


train_set = HighpassDataset(X_train_z, Y_train_hp)
val_set   = HighpassDataset(X_val_z,   Y_val_hp)
test_set  = HighpassDataset(X_test_z,  Y_test_hp)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

print(f"  train batches={len(train_loader)}, val batches={len(val_loader)}")

# -----------------------------------------------------------------
# Step 6: Model with Residual Blocks
# -----------------------------------------------------------------

section("Step 6 : CNN Decoder with Residual Blocks")


class ResBlock(nn.Module):
    """Residual block: Conv3x3 -> GN -> GELU -> Conv3x3 -> GN + skip"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class CNNHighpassDecoderResB(nn.Module):
    """
    CNN decoder with residual blocks after each ConvTranspose2d stage.

    X [B, n_input]
      -> FC -> [B, 128, 8, 8]
      -> ConvTranspose2d 8->16, ResBlock
      -> ConvTranspose2d 16->32, ResBlock
      -> ConvTranspose2d 32->64, ResBlock
      -> Conv2d -> [B, 1, 64, 64]
    """
    def __init__(self, n_input, img_size):
        super().__init__()
        assert img_size == 64

        self.fc = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128 * 8 * 8),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            ResBlock(96),
            nn.Dropout2d(0.2),

            # 16 -> 32
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock(64),
            nn.Dropout2d(0.2),

            # 32 -> 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResBlock(32),

            # final conv
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.fc(x)
        z = z.view(-1, 128, 8, 8)
        y = self.decoder(z)
        return y


model = CNNHighpassDecoderResB(N_INPUT, IMG_SIZE).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  MODEL_TYPE={MODEL_TYPE} + ResBlocks")
print(f"  Trainable params: {n_params / 1e6:.3f} M")
print(model)

# -----------------------------------------------------------------
# Step 7: Training with CosineAnnealingWarmRestarts
# -----------------------------------------------------------------

section("Step 7 : Train with CosineAnnealingWarmRestarts")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2
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
    "train_loss": [], "val_loss": [],
    "train_mse": [], "val_mse": [],
}

best_path = os.path.join(OUT_DIR, "best_nonlinear_hp.pt")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    train_log = run_one_epoch(train_loader, train=True)
    val_log = run_one_epoch(val_loader, train=False)

    scheduler.step()

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
        print(f"    * Best model saved: {best_path}")
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print(f"\n  Early stopping: {PATIENCE} epochs without improvement")
        break

t_train = time.time() - t0
print(f"\n  Training done in {t_train / 60:.1f} min")
print(f"  Best epoch={best_epoch}, best_val_loss={best_val:.6f}")

# -----------------------------------------------------------------
# Step 8: Load best model and predict
# -----------------------------------------------------------------

section("Step 8 : Load best model and predict high-pass")

ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)

if CKPT_KEY_MODEL_STATE in ckpt:
    model.load_state_dict(ckpt[CKPT_KEY_MODEL_STATE])
elif "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

if CKPT_KEY_CONFIG in ckpt:
    cfg = ckpt[CKPT_KEY_CONFIG]
    print(f"  Loaded config: {cfg}")

model.eval()

@torch.no_grad()
def predict_hp(loader):
    preds = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        pred = model(x)
        pred = pred.squeeze(1).cpu().numpy()
        preds.append(pred.astype(np.float32))
    return np.concatenate(preds, axis=0)

HP_train_pred = predict_hp(train_loader)
HP_val_pred   = predict_hp(val_loader)
HP_test_pred  = predict_hp(test_loader)

print(f"  HP_test_pred: {HP_test_pred.shape}, "
      f"range=[{HP_test_pred.min():.3f},{HP_test_pred.max():.3f}], "
      f"mean={HP_test_pred.mean():.5f}, std={HP_test_pred.std():.5f}")

# Combined
COMB_val_pred  = clip01(LP_val_pred  + HP_val_pred)
COMB_test_pred = clip01(LP_test_pred + HP_test_pred)

# -----------------------------------------------------------------
# Step 9: Evaluation
# -----------------------------------------------------------------

section("Step 9 : Evaluate Ridge-LP / Nonlinear-HP / Combined")

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

print("\n  [3] Ridge-LP + Nonlinear-HP (Combined)")
print("  val :", end="")
results["Combined_val"] = evaluate(COMB_val_pred, Y_val, tag="LP + HP")
print("  test:", end="")
results["Combined_test"] = evaluate(COMB_test_pred, Y_test, tag="LP + HP")

print("\n  [4] Auxiliary evaluation")
print("  LP test vs true LP:", end="")
lp_vs_true_lp = evaluate(LP_test_pred, Y_test_lp, tag="LP pred vs true LP")
print("  HP test residual :", end="")
hp_vs_true_hp = evaluate_residual(HP_test_pred, Y_test_hp, tag="HP pred vs true HP")

# -----------------------------------------------------------------
# Step 10: Save metrics
# -----------------------------------------------------------------

section("Step 10 : Save metrics")

metrics_path = os.path.join(OUT_DIR, "metrics_table.csv")

rows = []
rows.append({
    "Model": "Ridge-LP",
    "Target": "whole frame",
    **results["Ridge-LP_test"],
    "HP_MSE": "", "HP_Pearson": "",
    "Note": f"LP_SIGMA={LP_SIGMA}, alpha={ridge_lp.alpha_}",
})
rows.append({
    "Model": f"{MODEL_TYPE}-HP-ResB",
    "Target": "high-pass residual",
    "MSE": "", "PSNR": "", "SSIM": "", "Pearson": "",
    **hp_test_metrics,
    "Note": "residual-only evaluation",
})
rows.append({
    "Model": f"Ridge-LP + {MODEL_TYPE}-HP-ResB",
    "Target": "whole frame",
    **results["Combined_test"],
    "HP_MSE": "", "HP_Pearson": "",
    "Note": "combined reconstruction",
})

fieldnames = ["Model", "Target", "MSE", "PSNR", "SSIM", "Pearson",
              "HP_MSE", "HP_Pearson", "Note"]

with open(metrics_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"  Saved: {metrics_path}")

print("\n  Test metrics summary:")
for row in rows:
    print(f"  {row['Model']:<30s}  target={row['Target']:<20s}  "
          f"MSE={row.get('MSE','')}  SSIM={row.get('SSIM','')}  "
          f"Pearson={row.get('Pearson','')}  "
          f"HP_Pearson={row.get('HP_Pearson','')}")

# -----------------------------------------------------------------
# Step 11: Visualization
# -----------------------------------------------------------------

section("Step 11 : Visualization")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Loss curve
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(history["train_loss"], label="train loss")
ax.plot(history["val_loss"], label="val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("CNN-ResB high-pass decoder loss (opt_b)")
ax.legend()
ax.grid(alpha=0.3)

loss_path = os.path.join(OUT_DIR, "loss_curve.png")
fig.savefig(loss_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {loss_path}")

# Reconstruction comparison
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

fig, axes = plt.subplots(len(row_items), N_SHOW,
                         figsize=(N_SHOW * 2.1, len(row_items) * 2.0))

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

plt.suptitle("Reconstruction: Ridge-LP + CNN-HP-ResB (opt_b)", fontsize=11)
plt.tight_layout()

recon_path = os.path.join(OUT_DIR, "reconstruction_hp.png")
fig.savefig(recon_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {recon_path}")

# Residual visualization
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

fig, axes = plt.subplots(len(row_items_hp), N_SHOW,
                         figsize=(N_SHOW * 2.1, len(row_items_hp) * 2.0))

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

plt.suptitle("High-pass residual decoding: CNN-ResB (opt_b)", fontsize=11)
plt.tight_layout()

hp_vis_path = os.path.join(OUT_DIR, "residual_visualization.png")
fig.savefig(hp_vis_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {hp_vis_path}")

# -----------------------------------------------------------------
# Step 12: Save predictions
# -----------------------------------------------------------------

section("Step 12 : Save predictions")

if SAVE_PREDS:
    np.save(os.path.join(OUT_DIR, "preds_test_lp.npy"), LP_test_pred)
    np.save(os.path.join(OUT_DIR, "preds_test_hp.npy"), HP_test_pred)
    np.save(os.path.join(OUT_DIR, "preds_test_combined.npy"), COMB_test_pred)
    np.save(os.path.join(OUT_DIR, "true_test_hp.npy"), Y_test_hp)
    np.save(os.path.join(OUT_DIR, "true_test_lp.npy"), Y_test_lp)
    print(f"  Saved predictions to: {OUT_DIR}")
else:
    print("  SAVE_PREDS=False, skipped")

# -----------------------------------------------------------------
# Done
# -----------------------------------------------------------------

section("Done!")

print(f"""
  Output directory: {OUT_DIR}/

  Key changes in opt_b:
    - ResBlocks after each ConvTranspose2d
    - CosineAnnealingWarmRestarts (T_0=30, T_mult=2)
    - EPOCHS={EPOCHS}, PATIENCE={PATIENCE}
    - LR={LR}, LAMBDA_GRAD={LAMBDA_GRAD}
""")
