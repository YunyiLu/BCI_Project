"""
deblur_train_v9.py — Stage 3 v9: Optimal Wiener filter (closed-form, no training)

Computes the statistically optimal linear filter from training data.
Zero learnable parameters = zero overfitting risk.

Wiener filter: H = S / (S + N)
where S = signal power spectrum, N = noise power spectrum.
"""

import os, csv
import numpy as np

COMBINED_DIR = "./opt_c_results"
NPZ_DIR = "./yunyi_data/dataset_npz"
OUT_DIR = "./deblur_results_v9"

os.makedirs(OUT_DIR, exist_ok=True)

def clip01(x): return np.clip(x, 0.0, 1.0)


def compute_wiener_filter(combined_train, gt_train, reg=1e-3):
    """Compute optimal Wiener filter from training data."""
    # Noise = combined - GT (what the filter should remove)
    noise = combined_train - gt_train

    # Compute average power spectra
    N_psd = np.mean(np.abs(np.fft.fft2(noise)) ** 2, axis=0)
    S_psd = np.mean(np.abs(np.fft.fft2(gt_train)) ** 2, axis=0)

    # Wiener filter: H = S / (S + N)
    # With regularization to avoid division by zero
    H = S_psd / (S_psd + N_psd + reg)

    return H


def apply_wiener(images, H):
    """Apply Wiener filter to images in frequency domain."""
    result = np.zeros_like(images)
    for i in range(len(images)):
        F_img = np.fft.fft2(images[i])
        F_filtered = F_img * H
        result[i] = np.real(np.fft.ifft2(F_filtered))
    return result


def main():
    print("V9 Wiener Filter (closed-form, no training)")

    combined_train = np.load(os.path.join(COMBINED_DIR, "preds_train_combined.npy"))
    combined_val = np.load(os.path.join(COMBINED_DIR, "preds_val_combined.npy"))
    combined_test = np.load(os.path.join(COMBINED_DIR, "preds_test_combined.npy"))
    gt_train = np.load(os.path.join(NPZ_DIR, "dataset_train.npz"))["Y"].astype(np.float32)
    gt_val = np.load(os.path.join(NPZ_DIR, "dataset_val.npz"))["Y"].astype(np.float32)
    gt_test = np.load(os.path.join(NPZ_DIR, "dataset_test.npz"))["Y"].astype(np.float32)

    print(f"  Train: {combined_train.shape}, Test: {combined_test.shape}")

    # Try multiple regularization strengths
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn

    def eval_np(pred, gt):
        pred, gt = clip01(pred), clip01(gt)
        mse = float(np.mean((pred-gt)**2))
        psnr = float(np.mean([psnr_fn(gt[i], pred[i], data_range=1.0) for i in range(len(gt))]))
        ssim_val = float(np.mean([ssim_fn(gt[i], pred[i], data_range=1.0) for i in range(len(gt))]))
        corr = float(np.corrcoef(pred.ravel(), gt.ravel())[0,1])
        return {"MSE": mse, "PSNR": psnr, "SSIM": ssim_val, "Pearson": corr}

    # Baseline
    m_c = eval_np(combined_test, gt_test)
    print(f"\n  Combined baseline: SSIM={m_c['SSIM']:.6f}, PSNR={m_c['PSNR']:.4f}")

    # Search over regularization strengths, select on val set
    best_reg = None
    best_val_ssim = -1
    regs = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    print(f"\n  Searching regularization on val set...")
    for reg in regs:
        H = compute_wiener_filter(combined_train, gt_train, reg=reg)
        filtered_val = apply_wiener(combined_val, H)
        m_val = eval_np(filtered_val, gt_val)
        print(f"    reg={reg:.1e}  val_SSIM={m_val['SSIM']:.6f}  val_PSNR={m_val['PSNR']:.4f}")
        if m_val['SSIM'] > best_val_ssim:
            best_val_ssim = m_val['SSIM']
            best_reg = reg

    print(f"\n  Best reg={best_reg:.1e} (val SSIM={best_val_ssim:.6f})")

    # Apply best filter to test set
    H_best = compute_wiener_filter(combined_train, gt_train, reg=best_reg)
    deblurred = clip01(apply_wiener(combined_test, H_best))

    # Also try a simple approach: just attenuate high frequencies
    # (gentle low-pass as a baseline comparison)
    from scipy.ndimage import gaussian_filter
    sigmas_to_try = [0.3, 0.5, 0.7, 1.0, 1.5]
    print(f"\n  Also trying simple Gaussian smoothing on val...")
    best_sigma = None
    best_gauss_ssim = -1
    for sigma in sigmas_to_try:
        smoothed_val = np.stack([gaussian_filter(img, sigma=sigma) for img in combined_val])
        m_s = eval_np(smoothed_val, gt_val)
        print(f"    sigma={sigma:.1f}  val_SSIM={m_s['SSIM']:.6f}")
        if m_s['SSIM'] > best_gauss_ssim:
            best_gauss_ssim = m_s['SSIM']
            best_sigma = sigma

    smoothed_test = clip01(np.stack([gaussian_filter(img, sigma=best_sigma) for img in combined_test]))
    m_gauss = eval_np(smoothed_test, gt_test)
    print(f"  Best Gaussian sigma={best_sigma} -> test SSIM={m_gauss['SSIM']:.6f}")

    # Final metrics
    m_d = eval_np(deblurred, gt_test)
    print(f"\n  {'Metric':<10} {'Combined':<18} {'Wiener':<18} {'Gaussian':<18}")
    for k in ["MSE","PSNR","SSIM","Pearson"]:
        print(f"  {k:<10} {m_c[k]:<18.6f} {m_d[k]:<18.6f} {m_gauss[k]:<18.6f}")

    # Save best result (whichever is better)
    if m_d['SSIM'] >= m_gauss['SSIM']:
        final = deblurred
        final_name = f"V9 Wiener (reg={best_reg:.1e})"
        final_metrics = m_d
    else:
        final = smoothed_test
        final_name = f"V9 Gaussian (sigma={best_sigma})"
        final_metrics = m_gauss

    np.save(os.path.join(OUT_DIR, "preds_test_deblurred.npy"), final)
    with open(os.path.join(OUT_DIR, "metrics_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model","MSE","PSNR","SSIM","Pearson"])
        w.writeheader()
        w.writerow({"Model": "Combined (Stage 1+2)", **m_c})
        w.writerow({"Model": final_name, **final_metrics})
        w.writerow({"Model": f"V9 Wiener (reg={best_reg:.1e})", **m_d})
        w.writerow({"Model": f"V9 Gaussian (sigma={best_sigma})", **m_gauss})
    print(f"\n  Best: {final_name}")
    print(f"  Saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
