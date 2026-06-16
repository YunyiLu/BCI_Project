"""Generate final visualizations for Baseline 5 (Kim et al. 2021) reconstruction pipeline."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 130

# ---- load data ----
gt = np.load('yunyi_data/dataset_npz/dataset_test.npz')['Y']          # (N,64,64)
lp = np.load('opt_c_results/preds_test_lp.npy')                       # Stage 1
comb = np.load('opt_c_results/preds_test_combined.npy')              # Stage 1+2
v14 = np.load('deblur_results_v14/preds_test_deblurred.npy')         # Stage 1+2+3 (best)
v10 = np.load('deblur_results_v10/preds_test_deblurred.npy')         # DeblurGANv2 (overfit)

N = gt.shape[0]

def ssim_single(a, b):
    """Single-image SSIM (global, no windowing) for ranking sample selection."""
    a = a.ravel(); b = b.ravel()
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    return ((2*mu_a*mu_b + c1)*(2*cov + c2)) / ((mu_a**2 + mu_b**2 + c1)*(va + vb + c2))

# pick representative samples: best, median, by final-stage SSIM
scores = np.array([ssim_single(gt[i], v14[i]) for i in range(N)])
order = np.argsort(scores)[::-1]
picks = [order[0], order[N//2], order[int(N*0.85)]]  # good / median / weaker

# =================================================================
# FIG 1: Stage-by-stage reconstruction grid
# =================================================================
stages = [
    ('Ground Truth', gt, 'GT'),
    ('Stage 1\nRidge-LP', lp, 'S1'),
    ('Stage 1+2\n+ CNN-HP', comb, 'S2'),
    ('Stage 1+2+3\n+ PCA+Wiener', v14, 'S3'),
]
n_rows = len(picks)
n_cols = len(stages)
fig = plt.figure(figsize=(2.4*n_cols, 2.4*n_rows + 0.5))
gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.06, hspace=0.06)

for r, idx in enumerate(picks):
    for c, (title, arr, _) in enumerate(stages):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(arr[idx], cmap='gray', vmin=0, vmax=arr[idx].max())
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=11, fontweight='bold')
        if c == 0:
            label = ['Best','Median','Weaker'][r]
            ax.set_ylabel(label, fontsize=11, fontweight='bold')

fig.suptitle('Baseline 5 — Stage-by-Stage Reconstruction (Salamander RGC, 64x64)',
             fontsize=13, fontweight='bold', y=0.99)
fig.savefig('viz_stage_reconstruction.png', bbox_inches='tight')
plt.close(fig)
print('saved viz_stage_reconstruction.png')

# =================================================================
# FIG 2: Metrics progression across stages
# =================================================================
labels = ['Stage 1\nRidge-LP', 'Stage 1+2\n+CNN-HP', 'Stage 1+2+3\n+PCA+Wiener']
mse  = [0.01632, 0.01595, 0.01455]
psnr = [18.962, 19.117, 19.592]
ssim = [0.2696, 0.2873, 0.2960]
pear = [0.0318, 0.0838, 0.0846]

metrics = [('MSE ↓', mse, '#d62728'),
           ('PSNR (dB) ↑', psnr, '#1f77b4'),
           ('SSIM ↑', ssim, '#2ca02c'),
           ('Pearson r ↑', pear, '#9467bd')]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
x = np.arange(len(labels))
for ax, (name, vals, col) in zip(axes, metrics):
    bars = ax.bar(x, vals, color=col, alpha=0.85, width=0.6)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:.4g}',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.margins(y=0.18)
    ax.spines[['top','right']].set_visible(False)

fig.suptitle('Metric Progression Across Pipeline Stages', fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig('viz_metrics_progression.png', bbox_inches='tight')
plt.close(fig)
print('saved viz_metrics_progression.png')

# =================================================================
# FIG 3: Best method vs DeblurGANv2 (overfitting evidence)
# =================================================================
fig, axes = plt.subplots(3, 4, figsize=(10, 7.6))
methods = [('Ground Truth', gt), ('Stage 1+2', comb),
           ('V14 PCA+Wiener\n(best)', v14), ('V10 DeblurGANv2\n(overfit)', v10)]
for r, idx in enumerate(picks):
    for c, (title, arr) in enumerate(methods):
        ax = axes[r, c]
        ax.imshow(arr[idx], cmap='gray', vmin=0, vmax=arr[idx].max())
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=11, fontweight='bold')
fig.suptitle('Stage 3 Comparison: Zero-Parameter (V14) beats Deep Net (V10) on Small Data',
             fontsize=12, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig('viz_stage3_comparison.png', bbox_inches='tight')
plt.close(fig)
print('saved viz_stage3_comparison.png')
print('done')
