"""
deblur_train_v10.py — Stage 3 v10: 完全复刻论文方法

论文 Section 4.5 的精确复现：
- DeblurGANv2 ResNet Generator, 6 blocks
- Adam optimizer, LR=1e-5, 每 8 epoch 减半
- 32 epochs
- L1 loss + VGG perceptual loss (论文排除了 adversarial loss)
- 10-fold CV 生成 test-quality 训练数据

这是论文方法的 1:1 复刻，用来展示在 93 neurons 小数据上的效果。
"""

import os, csv, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

COMBINED_DIR = "./opt_c_results"
NPZ_DIR = "./yunyi_data/dataset_npz"
OUT_DIR = "./deblur_results_v10"
SEED = 42

# 论文参数（Section 4.5）
N_BLOCKS = 6
LR = 1e-5
EPOCHS = 32
BATCH_SIZE = 16
N_FOLDS = 10  # 论文用 10-fold

# Stage 2 CNN config (same as opt_c for K-fold)
LP_SIGMA = 3
ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
CNN_EPOCHS = 80
CNN_LR = 2e-4
CNN_BATCH = 64
CNN_PATIENCE = 15
CNN_LAMBDA_GRAD = 1.0

os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def clip01(x): return np.clip(x, 0.0, 1.0)

def make_lowpass(Y, sigma):
    return np.stack([gaussian_filter(y, sigma=sigma) for y in Y], axis=0).astype(np.float32)


# -----------------------------------------------------------------
# DeblurGANv2 ResNet Generator (论文: 6 blocks)
# -----------------------------------------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, n_blocks=6, ngf=64):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
        mult = 4
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        for i in range(2):
            mult = 2 ** (2 - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                   3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True),
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, 7)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)


# -----------------------------------------------------------------
# VGG Perceptual Loss (论文用的)
# -----------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:17])
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_rgb = pred.expand(-1, 3, -1, -1)
        target_rgb = target.expand(-1, 3, -1, -1)
        pred_norm = (pred_rgb - self.mean) / self.std
        target_norm = (target_rgb - self.mean) / self.std
        return F.l1_loss(self.feature_extractor(pred_norm), self.feature_extractor(target_norm))


# -----------------------------------------------------------------
# Stage 2 CNN (for K-fold)
# -----------------------------------------------------------------

class CNNHighpassDecoder(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 128 * 8 * 8),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1),
            nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.GroupNorm(8, 96), nn.GELU(), nn.Dropout2d(0.1),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(), nn.Dropout2d(0.1),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )
        self.output_scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        z = self.fc(x)
        z = z.view(-1, 128, 8, 8)
        return self.decoder(z) * self.output_scale


def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (dx_pred - dx_tgt).abs().mean() + (dy_pred - dy_tgt).abs().mean()


def train_cnn_hp_fold(X_tr, Y_hp_tr, X_ho, n_input, device):
    X_mean = X_tr.mean(axis=0, keepdims=True)
    X_std = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr_z = ((X_tr - X_mean) / X_std).astype(np.float32)
    X_ho_z = ((X_ho - X_mean) / X_std).astype(np.float32)

    class HPDataset(Dataset):
        def __init__(self, X, Y):
            self.X = torch.from_numpy(X)
            self.Y = torch.from_numpy(Y[:, None, :, :])
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.Y[i]

    tr_loader = DataLoader(HPDataset(X_tr_z, Y_hp_tr), batch_size=CNN_BATCH,
                           shuffle=True, num_workers=0, pin_memory=True)
    model = CNNHighpassDecoder(n_input).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)
    mse_fn = nn.MSELoss()

    best_loss = float("inf"); bad = 0; best_state = None
    for epoch in range(1, CNN_EPOCHS + 1):
        model.train(); total, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = mse_fn(pred, y) + CNN_LAMBDA_GRAD * gradient_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * x.size(0); n += x.size(0)
        train_loss = total / n
        scheduler.step(train_loss)
        if train_loss < best_loss - 1e-7:
            best_loss = train_loss; bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= CNN_PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    ho_tensor = torch.from_numpy(X_ho_z)
    ho_loader = DataLoader(ho_tensor, batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in ho_loader:
            preds.append(model(batch.to(device)).squeeze(1).cpu().numpy())
    return np.concatenate(preds, axis=0)


# -----------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------

class DeblurDataset(Dataset):
    def __init__(self, combined, gt):
        self.combined = combined.astype(np.float32)
        self.gt = gt.astype(np.float32)
    def __len__(self): return len(self.combined)
    def __getitem__(self, idx):
        return torch.from_numpy(self.combined[idx][None]), torch.from_numpy(self.gt[idx][None])


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    set_seed(SEED)
    device = torch.device("cuda:0")
    print(f"V10 论文完全复刻 | Device: {device}")
    print(f"  DeblurGANv2 ResNet Generator, {N_BLOCKS} blocks, ngf=64")
    print(f"  LR={LR}, Epochs={EPOCHS}, {N_FOLDS}-fold CV")

    # Load data
    def load_split(split):
        d = np.load(os.path.join(NPZ_DIR, f"dataset_{split}.npz"))
        return d["X"].astype(np.float32), d["Y"].astype(np.float32)

    X_train, Y_train = load_split("train")
    X_val, Y_val = load_split("val")
    X_test, Y_test = load_split("test")
    N_INPUT = X_train.shape[1]
    IMG_SIZE = Y_train.shape[1]
    print(f"  Data: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

    # Step 1: 10-fold CV 生成 test-quality training predictions (论文方法)
    print(f"\n  Step 1: {N_FOLDS}-fold CV for test-quality training predictions...")
    Y_train_lp = make_lowpass(Y_train, LP_SIGMA)
    Y_train_hp = (Y_train - Y_train_lp).astype(np.float32)

    combined_train = np.zeros_like(Y_train)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold_i, (train_idx, hold_idx) in enumerate(kf.split(X_train)):
        print(f"    Fold {fold_i+1}/{N_FOLDS}: train={len(train_idx)}, hold-out={len(hold_idx)}")
        X_tr, X_ho = X_train[train_idx], X_train[hold_idx]
        Y_tr_lp = Y_train_lp[train_idx]
        Y_tr_lp_flat = Y_tr_lp.reshape(len(Y_tr_lp), -1)

        ridge = RidgeCV(alphas=ALPHAS, cv=3)
        ridge.fit(X_tr, Y_tr_lp_flat)
        lp_ho = clip01(ridge.predict(X_ho).reshape(-1, IMG_SIZE, IMG_SIZE).astype(np.float32))

        Y_hp_tr = Y_train_hp[train_idx]
        hp_ho = train_cnn_hp_fold(X_tr, Y_hp_tr, X_ho, N_INPUT, device)
        combined_train[hold_idx] = clip01(lp_ho + hp_ho)

    # Step 2: val/test predictions using full model
    print(f"\n  Step 2: Full-model val/test predictions...")
    Y_train_lp_flat = Y_train_lp.reshape(len(Y_train_lp), -1)
    ridge_full = RidgeCV(alphas=ALPHAS, cv=5)
    ridge_full.fit(X_train, Y_train_lp_flat)

    lp_val = clip01(ridge_full.predict(X_val).reshape(-1, IMG_SIZE, IMG_SIZE).astype(np.float32))
    lp_test = clip01(ridge_full.predict(X_test).reshape(-1, IMG_SIZE, IMG_SIZE).astype(np.float32))

    hp_val = train_cnn_hp_fold(X_train, Y_train_hp, X_val, N_INPUT, device)
    hp_test = train_cnn_hp_fold(X_train, Y_train_hp, X_test, N_INPUT, device)

    combined_val = clip01(lp_val + hp_val)
    combined_test = clip01(lp_test + hp_test)

    # Step 3: Train DeblurGANv2 (论文精确参数)
    print(f"\n  Step 3: Train DeblurGANv2...")
    train_loader = DataLoader(DeblurDataset(combined_train, Y_train),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(DeblurDataset(combined_val, Y_val),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(DeblurDataset(combined_test, Y_test),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = ResnetGenerator(n_blocks=N_BLOCKS, ngf=64).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params/1e6:.2f}M")

    l1_loss = nn.L1Loss()
    vgg_loss = VGGPerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # 论文: LR 每 8 epoch 减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_val = float("inf"); best_epoch = -1
    best_path = os.path.join(OUT_DIR, "best_deblur.pt")

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train(); t_total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = l1_loss(pred, y) + vgg_loss(pred, y)
            loss.backward()
            optimizer.step()
            t_total += loss.item() * x.size(0); n += x.size(0)
        train_loss = t_total / n

        model.eval(); v_total, nv = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = l1_loss(pred, y) + vgg_loss(pred, y)
                v_total += loss.item() * x.size(0); nv += x.size(0)
        val_loss = v_total / nv

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:02d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}  lr={lr_now:.1e}")

        if val_loss < best_val:
            best_val = val_loss; best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    print(f"\n  Done in {(time.time()-t0)/60:.1f} min, best epoch={best_epoch}")

    # Evaluate
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn

    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            all_preds.append(model(x.to(device)).squeeze(1).cpu().numpy())
    deblurred = clip01(np.concatenate(all_preds, axis=0))

    def eval_np(pred, gt):
        pred, gt = clip01(pred), clip01(gt)
        mse = float(np.mean((pred-gt)**2))
        psnr = float(np.mean([psnr_fn(gt[i], pred[i], data_range=1.0) for i in range(len(gt))]))
        ssim_val = float(np.mean([ssim_fn(gt[i], pred[i], data_range=1.0) for i in range(len(gt))]))
        corr = float(np.corrcoef(pred.ravel(), gt.ravel())[0,1])
        return {"MSE": mse, "PSNR": psnr, "SSIM": ssim_val, "Pearson": corr}

    m_c = eval_np(combined_test, Y_test)
    m_d = eval_np(deblurred, Y_test)

    print(f"\n  {'Metric':<10} {'Combined':<18} {'V10 DeblurGAN':<18} {'Delta':<10}")
    print(f"  {'-'*56}")
    for k in ["MSE","PSNR","SSIM","Pearson"]:
        print(f"  {k:<10} {m_c[k]:<18.6f} {m_d[k]:<18.6f} {m_d[k]-m_c[k]:+.6f}")

    np.save(os.path.join(OUT_DIR, "preds_test_deblurred.npy"), deblurred)
    with open(os.path.join(OUT_DIR, "metrics_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model","MSE","PSNR","SSIM","Pearson"])
        w.writeheader()
        w.writerow({"Model": "Combined (10-fold CV)", **m_c})
        w.writerow({"Model": "V10 DeblurGANv2 (paper replica)", **m_d})
    print(f"\n  Saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
