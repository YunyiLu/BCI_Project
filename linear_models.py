"""
linear_models.py

多种线性解码模型集成评测脚本
输入  : dataset_npz/  (由 build_dataset.py 生成)
输出  : linear_results/
          metrics_table.csv         — 所有模型 × 所有指标
          reconstruction.png        — GT vs 各模型重建对比图
          weights_heatmap.png       — Ridge 权重矩阵可视化
          predictions_test_{model}.npy (可选存)

运行  : python linear_models.py
依赖  : pip install numpy scipy scikit-learn scikit-image matplotlib
"""

import os
import time
import numpy as np
from collections import OrderedDict
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

NPZ_DIR    = "D:/BMI/pj/dataset_npz"  # build_dataset.py 的输出目录
OUT_DIR    = "D:/BMI/pj/linear_results"

LP_SIGMA   = 3       # 低频目标的高斯模糊 σ（像素），64×64 图上 σ=3 效果合理
PCA_K      = 50      # PCA+Ridge 的输出主成分数
RUN_SLOW   = False   # True → 额外运行 MultiTaskLasso（较慢，约 2~5 分钟）
SAVE_PREDS = False   # True → 把测试集预测保存为 .npy，供后续阶段拼接

os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Step 1 : 加载数据
# ─────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)

section("Step 1 : 加载数据集")

def load_split(npz_dir, split):
    d = np.load(os.path.join(npz_dir, f"dataset_{split}.npz"))
    return d["X"].astype(np.float32), d["Y"].astype(np.float32)

X_train, Y_train = load_split(NPZ_DIR, "train")
X_val,   Y_val   = load_split(NPZ_DIR, "val")
X_test,  Y_test  = load_split(NPZ_DIR, "test")

meta     = np.load(os.path.join(NPZ_DIR, "dataset_meta.npz"), allow_pickle=True)
IMG_SIZE = int(meta["img_size"])     # 64
N_PIXEL  = IMG_SIZE * IMG_SIZE       # 
NCELL  = int(meta["ncell"])    # 新增
WINDOW = int(meta["window"])   # 新增


for name, X, Y in [("train", X_train, Y_train),
                    ("val",   X_val,   Y_val),
                    ("test",  X_test,  Y_test)]:
    print(f"  {name:5s}  X={X.shape}  Y={Y.shape}")

# 展平 Y 为 [N, 4096]，方便 sklearn 使用
Y_train_flat = Y_train.reshape(len(Y_train), -1)
Y_val_flat   = Y_val.reshape(len(Y_val),   -1)
Y_test_flat  = Y_test.reshape(len(Y_test),  -1)

# 低频（low-pass）目标：对训练/验证帧做高斯模糊
def make_lowpass(Y_3d, sigma):
    """Y_3d: [N, H, W] → [N, H, W] 高斯模糊版"""
    return np.stack([gaussian_filter(y, sigma=sigma) for y in Y_3d])

print(f"\n  生成低频目标 (Gaussian σ={LP_SIGMA})...")
Y_train_lp      = make_lowpass(Y_train, LP_SIGMA)
Y_val_lp        = make_lowpass(Y_val,   LP_SIGMA)
Y_test_lp       = make_lowpass(Y_test,  LP_SIGMA)
Y_train_lp_flat = Y_train_lp.reshape(len(Y_train), -1)
print(f"  低频帧范围: [{Y_train_lp.min():.3f}, {Y_train_lp.max():.3f}]")

# ─────────────────────────────────────────────────────────────
# Step 2 : 评测函数
# ─────────────────────────────────────────────────────────────
section("Step 2 : 定义评测函数")

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity   as ssim_fn

def evaluate(y_pred_3d, y_true_3d, tag=""):
    """
    y_pred_3d, y_true_3d : numpy [N, H, W]，值域 [0, 1]
    返回 dict: MSE, PSNR, SSIM, Pearson
    """
    y_pred_3d = np.clip(y_pred_3d, 0.0, 1.0)
    mse    = float(np.mean((y_pred_3d - y_true_3d) ** 2))
    psnr   = float(np.mean([psnr_fn(y_true_3d[i], y_pred_3d[i], data_range=1.0)
                             for i in range(len(y_pred_3d))]))
    ssim   = float(np.mean([ssim_fn(y_true_3d[i], y_pred_3d[i], data_range=1.0)
                             for i in range(len(y_pred_3d))]))
    # Pearson 相关（展平后计算全局像素相关系数）
    p, q   = y_pred_3d.ravel(), y_true_3d.ravel()
    corr   = float(np.corrcoef(p, q)[0, 1])
    label  = f" [{tag}]" if tag else ""
    print(f"    MSE={mse:.5f}  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}  "
          f"Pearson={corr:.4f}{label}")
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "Pearson": corr}

print("  评测函数就绪（MSE ↓ / PSNR ↑ / SSIM ↑ / Pearson ↑）")

# ─────────────────────────────────────────────────────────────
# Step 3 : 定义模型 & 训练
# ─────────────────────────────────────────────────────────────
section("Step 3 : 训练各线性模型")

from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.decomposition import PCA

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
# results[model_name] = {"train_time": float, ...   ← 
#                        "test_pred": array, ...     ← 

results = OrderedDict()

# ── 辅助函数 ─────────────────────────────────────────────────
def fit_and_eval(name, model, X_tr, Y_tr_flat, target_note="whole frame"):
    """拟合模型并在 val / test 上评测，返回 results 条目"""
    print(f"\n  [{name}]  target={target_note}")
    t0 = time.time()
    model.fit(X_tr, Y_tr_flat)
    t_train = time.time() - t0
    print(f"    训练完成  {t_train:.1f}s", end="")
    if hasattr(model, "alpha_"):
        print(f"  最优 α = {model.alpha_:.4g}", end="")
    print()

    def pred(X):
        p = model.predict(X).reshape(-1, IMG_SIZE, IMG_SIZE)
        return np.clip(p, 0.0, 1.0)

    print("    val  :", end="")
    val_metrics  = evaluate(pred(X_val),  Y_val,  tag="vs whole")
    print("    test :", end="")
    test_metrics = evaluate(pred(X_test), Y_test, tag="vs whole")

    return {
        "train_time": t_train,
        "val":  val_metrics,
        "test": test_metrics,
        "test_pred": pred(X_test),
        "best_alpha": getattr(model, "alpha_", None),
    }

# ── 1. OLS（无正则化）────────────────────────────────────────
print("\n  [1/4] OLS (LinearRegression)")
ols = LinearRegression(n_jobs=-1)
results["OLS"] = fit_and_eval("OLS", ols, X_train, Y_train_flat)

# ── 2. Ridge（全帧目标）─────────────────────────────────────
print("\n  [2/4] Ridge（全帧目标，自动选 α）")
ridge = RidgeCV(alphas=ALPHAS, cv=5)
results["Ridge"] = fit_and_eval("Ridge", ridge, X_train, Y_train_flat)

# ── 3. Ridge-LP（低频目标 → Baseline 5 Stage 1）─────────────
print("\n  [3/4] Ridge-LP（低频目标，Baseline 5 Stage 1）")
ridge_lp = RidgeCV(alphas=ALPHAS, cv=5)
ridge_lp.fit(X_train, Y_train_lp_flat)
t_lp = 0.0  # 与上面复用结构

def _pred_lp(X):
    return np.clip(ridge_lp.predict(X).reshape(-1, IMG_SIZE, IMG_SIZE), 0.0, 1.0)

print(f"    训练完成  最优 α = {ridge_lp.alpha_:.4g}")
print("    val  :", end="")
val_lp_whole  = evaluate(_pred_lp(X_val),  Y_val,  tag="vs whole")
print("    val  :", end="")
val_lp_lp     = evaluate(_pred_lp(X_val),  Y_val_lp,  tag="vs low-pass")
print("    test :", end="")
test_lp_whole = evaluate(_pred_lp(X_test), Y_test, tag="vs whole")
print("    test :", end="")
test_lp_lp    = evaluate(_pred_lp(X_test), Y_test_lp, tag="vs low-pass ★")

results["Ridge-LP"] = {
    "train_time":    0.0,
    "val":           val_lp_whole,
    "test":          test_lp_whole,
    "test_lp_eval":  test_lp_lp,     # 额外：对比低频 GT
    "test_pred":     _pred_lp(X_test),
    "best_alpha":    ridge_lp.alpha_,
}

# ── 4. PCA + Ridge（输出 PCA 降维后回归）────────────────────
print(f"\n  [4/4] PCA(k={PCA_K}) + Ridge")
pca = PCA(n_components=PCA_K)
Y_train_pca = pca.fit_transform(Y_train_flat)    # [N_train, 50]
print(f"    PCA 拟合完成  解释方差 = {pca.explained_variance_ratio_.sum():.3f}")

ridge_pca = RidgeCV(alphas=ALPHAS, cv=5)
ridge_pca.fit(X_train, Y_train_pca)
print(f"    Ridge 拟合完成  最优 α = {ridge_pca.alpha_:.4g}")

def _pred_pca(X):
    pc = ridge_pca.predict(X)
    flat = pca.inverse_transform(pc)
    return np.clip(flat.reshape(-1, IMG_SIZE, IMG_SIZE), 0.0, 1.0)

print("    val  :", end="")
val_pca  = evaluate(_pred_pca(X_val),  Y_val)
print("    test :", end="")
test_pca = evaluate(_pred_pca(X_test), Y_test)

results["PCA+Ridge"] = {
    "train_time": 0.0,
    "val":  val_pca,
    "test": test_pca,
    "test_pred": _pred_pca(X_test),
    "best_alpha": ridge_pca.alpha_,
}

# ── 5. MultiTaskLasso（可选，较慢）───────────────────────────
if RUN_SLOW:
    print("\n  [5/5] MultiTaskLasso（RUN_SLOW=True）")
    from sklearn.linear_model import MultiTaskLassoCV
    mtl = MultiTaskLassoCV(alphas=None, cv=3, max_iter=1000, n_jobs=-1)
    results["MultiTaskLasso"] = fit_and_eval(
        "MultiTaskLasso", mtl, X_train, Y_train_flat)

# ─────────────────────────────────────────────────────────────
# Step 4 : 汇总结果表
# ─────────────────────────────────────────────────────────────
section("Step 4 : 汇总评测结果（测试集）")

import csv

METRIC_KEYS = ["MSE", "PSNR", "SSIM", "Pearson"]

# 打印表格
col_w = [18, 10, 10, 8, 10, 12]
header = ["Model", "MSE↓", "PSNR↑(dB)", "SSIM↑", "Pearson↑", "Note"]
sep    = "  ".join(f"{h:<{w}}" for h, w in zip(header, col_w))
print("\n  " + sep)
print("  " + "-" * len(sep))

csv_rows = []
for name, res in results.items():
    m = res["test"]
    note = ""
    if name == "Ridge-LP":
        lp = res["test_lp_eval"]
        note = f"vs LP: SSIM={lp['SSIM']:.4f}"
    row = [name,
           f"{m['MSE']:.5f}",
           f"{m['PSNR']:.2f}",
           f"{m['SSIM']:.4f}",
           f"{m['Pearson']:.4f}",
           note]
    print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(row, col_w)))
    csv_rows.append({"Model": name, **{k: m[k] for k in METRIC_KEYS}, "Note": note})

# 保存 CSV
csv_path = os.path.join(OUT_DIR, "metrics_table.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Model"] + METRIC_KEYS + ["Note"])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\n  → 已保存: {csv_path}")

# ─────────────────────────────────────────────────────────────
# Step 5 : 可视化
# ─────────────────────────────────────────────────────────────
section("Step 5 : 可视化")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 5a. 重建对比图 ───────────────────────────────────────────
N_SHOW   = 5
model_names = list(results.keys())
n_models = len(model_names)
n_rows   = 1 + n_models   # GT 行 + 各模型行

# 挑 N_SHOW 个均匀分布的测试样本
idxs = np.linspace(0, len(Y_test) - 1, N_SHOW, dtype=int)

fig, axes = plt.subplots(n_rows, N_SHOW,
                         figsize=(N_SHOW * 2.2, n_rows * 2.2))
axes = np.atleast_2d(axes)

row_labels = ["Ground Truth"] + model_names

for row, label in enumerate(row_labels):
    for col, idx in enumerate(idxs):
        ax = axes[row, col]
        if row == 0:
            img = Y_test[idx]
        else:
            img = results[label]["test_pred"][idx]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(label, fontsize=8, rotation=90,
                          labelpad=4, va="center")

# 添加列标题（帧索引）
for col, idx in enumerate(idxs):
    axes[0, col].set_title(f"frame {idx}", fontsize=8)

plt.suptitle("Reconstruction comparison — linear models (test set)",
             fontsize=10, y=1.01)
plt.tight_layout()
recon_path = os.path.join(OUT_DIR, "reconstruction.png")
fig.savefig(recon_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  → 已保存: {recon_path}")

# ── 5b. 指标对比柱状图 ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

metrics_to_plot = [
    ("PSNR",    "PSNR (dB) ↑",  False),
    ("SSIM",    "SSIM ↑",        False),
    ("MSE",     "MSE ↓",         True),   # True = lower is better → invert axis
]

colors = ["#B4B2A9", "#AFA9EC", "#9FE1CB", "#FAC775",
          "#F5C4B3", "#B5D4F4"]  # one per model

for ax, (key, ylabel, invert) in zip(axes, metrics_to_plot):
    vals = [results[n]["test"][key] for n in model_names]
    bars = ax.bar(model_names, vals,
                  color=colors[:len(model_names)], width=0.55,
                  edgecolor="none")
    ax.set_title(ylabel, fontsize=10)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    # 数值标注
    for bar, v in zip(bars, vals):
        fmt = f"{v:.4f}" if key in ("MSE", "SSIM", "Pearson") else f"{v:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                fmt, ha="center", va="bottom", fontsize=7)
    if invert:
        ax.invert_yaxis()

plt.suptitle("Metrics comparison (test set)", fontsize=11)
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "metrics_bar.png")
fig.savefig(bar_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  → 已保存: {bar_path}")

# ── 5c. Ridge 权重矩阵热图（每神经元对每像素的影响）────────
print("  生成 Ridge 权重热图...")
W = ridge.coef_           # [4096, 93] → 每个像素行是一个 93-dim 向量
# 每个神经元的权重展开成 64×64 图，显示前 12 个
W_T = W.T              # [93, 4096] → [93, H*W]
if WINDOW == 1:
    # 直接展示每个神经元的空间权重图
    W_neuron = W_T                          # [93, 4096]
    title_suffix = "(single-bin weights)"
else:
    # 每个神经元有 WINDOW 个 bin 的权重，取 L2 范数得到强度图
    W_reshaped = W_T.reshape(NCELL, WINDOW, N_PIXEL)   # [93, W, 4096]
    W_neuron   = np.linalg.norm(W_reshaped, axis=1)    # [93, 4096]
    title_suffix = f"(L2 norm over {WINDOW} time bins)"
n_show_neurons = min(12, W_neuron.shape[0])
ncols = 6
nrows_w = (n_show_neurons + ncols - 1) // ncols

fig, axes = plt.subplots(nrows_w, ncols,
                         figsize=(ncols * 1.8, nrows_w * 1.8))
axes = np.atleast_2d(axes)

for idx in range(n_show_neurons):
    r, c = divmod(idx, ncols)
    ax = axes[r, c]
    wmap = W_neuron[idx].reshape(IMG_SIZE, IMG_SIZE)
    lim  = np.abs(wmap).max() + 1e-9
    ax.imshow(wmap, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(f"n{idx}", fontsize=8)
    ax.axis("off")

# 关掉多余子图
for idx in range(n_show_neurons, nrows_w * ncols):
    r, c = divmod(idx, ncols)
    axes[r, c].axis("off")

plt.suptitle("Ridge weight maps — spatial tuning of each neuron",
             fontsize=10)
plt.tight_layout()
weight_path = os.path.join(OUT_DIR, "weights_heatmap.png")
fig.savefig(weight_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  → 已保存: {weight_path}")

# ── 5d. 可选：保存预测数组 ──────────────────────────────────
if SAVE_PREDS:
    for name, res in results.items():
        p = os.path.join(OUT_DIR, f"preds_test_{name.replace('+','_')}.npy")
        np.save(p, res["test_pred"])
    print(f"  → 预测数组已保存至 {OUT_DIR}")

# ─────────────────────────────────────────────────────────────
# 完成
# ─────────────────────────────────────────────────────────────
section("完成！")
print(f"""
  输出文件:
    {OUT_DIR}/
    ├── metrics_table.csv          → 所有模型指标（可直接贴入报告）
    ├── reconstruction.png         → GT vs 各模型重建对比图
    ├── metrics_bar.png            → 指标柱状图
    └── weights_heatmap.png        → Ridge 各神经元权重空间分布

  关键结论看这里:
    Ridge-LP 的 "vs LP: SSIM=..." 反映 Baseline 5 Stage 1 解码低频的能力
    Ridge vs Ridge-LP 的 SSIM/PSNR 差值说明低频化对全帧指标的代价
    下一步: 用 Ridge-LP 的输出作为低频分量，训练非线性 CNN 解码高频残差
""")
