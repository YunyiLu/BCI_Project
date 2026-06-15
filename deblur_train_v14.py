"""
deblur_train_v14.py — Stage 3 v14: PCA subspace projection

Denoise by projecting onto the principal component subspace of GT images.
Components beyond a threshold capture noise, not signal.

1. Compute PCA of GT training images
2. Apply Wiener to test images
3. Project Wiener output onto top-K PCA components
4. Tune K on val set

This removes any component of the prediction that doesn't appear in the
natural image subspace — a form of structural denoising.
"""

import os, csv
import numpy as np
from sklearn.decomposition import PCA

COMBINED_DIR = "./opt_c_results"
NPZ_DIR = "./yunyi_data/dataset_npz"
OUT_DIR = "./deblur_results_v14"
os.makedirs(OUT_DIR, exist_ok=True)

def clip01(x): return np.clip(x, 0.0, 1.0)

def compute_wiener(combined_train, gt_train, reg=1e-5):
    noise = combined_train - gt_train
    N_psd = np.mean(np.abs(np.fft.fft2(noise)) ** 2, axis=0)
    S_psd = np.mean(np.abs(np.fft.fft2(gt_train)) ** 2, axis=0)
    return S_psd / (S_psd + N_psd + reg)

def apply_wiener(images, H):
    result = np.zeros_like(images)
    for i in range(len(images)):
        result[i] = np.real(np.fft.ifft2(np.fft.fft2(images[i]) * H))
    return result


def main():
    print("V14 PCA Subspace Projection")

    combined_train = np.load(os.path.join(COMBINED_DIR, "preds_train_combined.npy"))
    combined_val = np.load(os.path.join(COMBINED_DIR, "preds_val_combined.npy"))
    combined_test = np.load(os.path.join(COMBINED_DIR, "preds_test_combined.npy"))
    gt_train = np.load(os.path.join(NPZ_DIR, "dataset_train.npz"))["Y"].astype(np.float32)
    gt_val = np.load(os.path.join(NPZ_DIR, "dataset_val.npz"))["Y"].astype(np.float32)
    gt_test = np.load(os.path.join(NPZ_DIR, "dataset_test.npz"))["Y"].astype(np.float32)

    from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn

    def eval_np(pred, gt):
        pred, gt = clip01(pred), clip01(gt)
        mse = float(np.mean((pred-gt)**2))
        psnr = float(np.mean([psnr_fn(gt[i], pred[i], data_range=1.0) for i in range(len(gt))]))
        ssim_val = float(np.mean([ssim_fn(gt[i], pred[i], data_range=1.0) for i in range(len(gt))]))
        corr = float(np.corrcoef(pred.ravel(), gt.ravel())[0,1])
        return {"MSE": mse, "PSNR": psnr, "SSIM": ssim_val, "Pearson": corr}

    m_c = eval_np(combined_test, gt_test)
    print(f"  Combined: SSIM={m_c['SSIM']:.6f}")

    # Wiener first
    H = compute_wiener(combined_train, gt_train, reg=1e-5)
    wiener_val = clip01(apply_wiener(combined_val, H))
    wiener_test = clip01(apply_wiener(combined_test, H))
    m_w = eval_np(wiener_test, gt_test)
    print(f"  Wiener: SSIM={m_w['SSIM']:.6f}")

    # Fit PCA on GT training images
    print("\n  Fitting PCA on GT training images...")
    gt_flat = gt_train.reshape(len(gt_train), -1)  # [3588, 4096]
    n_max = min(1000, len(gt_train))
    pca = PCA(n_components=n_max)
    pca.fit(gt_flat)
    print(f"  Explained variance (top 10): {np.cumsum(pca.explained_variance_ratio_[:10])}")
    print(f"  Components needed for 90% var: {np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9)+1}")
    print(f"  Components needed for 95% var: {np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95)+1}")
    print(f"  Components needed for 99% var: {np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.99)+1}")

    # Search K on val (both on combined and wiener input)
    print("\n  Searching K on val (Wiener input)...")
    wiener_val_flat = wiener_val.reshape(len(wiener_val), -1)
    mean = pca.mean_

    best_ssim = -1
    best_k = n_max
    for k in [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        if k > n_max:
            continue
        # Project: reconstruct with top-K components
        components = pca.components_[:k]
        projected = (wiener_val_flat - mean) @ components.T @ components + mean
        projected_imgs = clip01(projected.reshape(wiener_val.shape))
        m = eval_np(projected_imgs, gt_val)
        print(f"    K={k:4d}  val_SSIM={m['SSIM']:.6f}  val_Pearson={m['Pearson']:.6f}")
        if m['SSIM'] > best_ssim:
            best_ssim = m['SSIM']
            best_k = k

    print(f"\n  Best K={best_k}")

    # Also try blending: α * projected + (1-α) * wiener
    print("  Searching blend α (projected + wiener)...")
    components = pca.components_[:best_k]
    wiener_test_flat = wiener_test.reshape(len(wiener_test), -1)
    projected_test = clip01(((wiener_test_flat - mean) @ components.T @ components + mean).reshape(wiener_test.shape))

    best_blend_ssim = -1
    best_alpha = 1.0
    proj_val = clip01(((wiener_val_flat - mean) @ components.T @ components + mean).reshape(wiener_val.shape))
    for alpha in np.arange(0.0, 1.01, 0.05):
        blended = clip01(alpha * proj_val + (1-alpha) * wiener_val)
        m = eval_np(blended, gt_val)
        if m['SSIM'] > best_blend_ssim:
            best_blend_ssim = m['SSIM']
            best_alpha = alpha

    print(f"  Best blend alpha={best_alpha:.2f} (val SSIM={best_blend_ssim:.6f})")

    # Final results
    if best_alpha < 1.0:
        result = clip01(best_alpha * projected_test + (1-best_alpha) * wiener_test)
        name = f"V14 PCA(K={best_k})+Wiener(a={best_alpha:.2f})"
    else:
        result = projected_test
        name = f"V14 PCA(K={best_k})"

    m_d = eval_np(result, gt_test)
    print(f"\n  {'Metric':<10} {'Combined':<18} {name[:18]:<18} {'Delta':<10}")
    for k in ["MSE","PSNR","SSIM","Pearson"]:
        print(f"  {k:<10} {m_c[k]:<18.6f} {m_d[k]:<18.6f} {m_d[k]-m_c[k]:+.6f}")

    np.save(os.path.join(OUT_DIR, "preds_test_deblurred.npy"), result)
    with open(os.path.join(OUT_DIR, "metrics_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model","MSE","PSNR","SSIM","Pearson"])
        w.writeheader()
        w.writerow({"Model": "Combined (Stage 1+2)", **m_c})
        w.writerow({"Model": "V9 Wiener", **m_w})
        w.writerow({"Model": name, **m_d})
    print(f"  Saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
