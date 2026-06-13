"""
build_dataset.py

构建蝾螈视网膜电影解码数据集，保存为 .npz 文件
供后续线性 baseline 和神经网络模型直接加载使用

数据集结构
----------
样本对  (X, Y):
  X[i]  PSTH[t + DELAY, :]      shape [93]         神经元发放概率
  Y[i]  frame[t]                 shape [64, 64]     归一化灰度帧

时间对齐逻辑（重要）:
  STA 标定显示蝾螈视网膜延迟 ≈ 66.7 ms
    checkerboard 采样率 30 Hz → 最佳 delay = 2 bin = 66.7 ms
    电影数据     采样率 60 Hz → 对应 delay = 4 bin
  即：神经元在看到第 t 帧后约 4 个 bin 才发放
  因此：解码第 t 帧，用 spike[t + 4] 作为输入

输出文件
--------
  dataset_train.npz   movie 0, 1, 2
  dataset_val.npz     movie 3
  dataset_test.npz    movie 4
  dataset_meta.npz    元数据（fs, delay, 分辨率等）

依赖
----
  pip install numpy scipy opencv-python matplotlib
"""

import os
import sys
import numpy as np
import scipy.io as sio

# ─────────────────────────────────────────────────────────────
# 配置区（只需修改这里）
# ─────────────────────────────────────────────────────────────

DATA_DIR   = "D:/BMI/pj/dataset"       # .mat 和 .avi 所在目录
OUT_DIR    = "D:/BMI/pj/dataset_npz"   # 输出目录

MOVIE_FILE = "movieBinnedSpiking.mat"

IMG_SIZE   = 64    # 帧降采样分辨率（建议 64 起步，内存足够可改为 128）
DELAY      = 4     # spike 相对帧的时间延迟（bins, 60 Hz 下 4 bins ≈ 66.7 ms）
WINDOW     = 1     # 时间窗大小（1 = 单 bin；>1 则把连续多个 bin 拼为输入）

TRAIN_MOVS = [0, 1, 2]   # movie: Tree, Water, Grasses
VAL_MOVS   = [3]          # movie: Fish
TEST_MOVS  = [4]          # movie: Self Motion

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def getvar(mat, name):
    """大小写不敏感地从 loadmat 结果里取变量"""
    if name in mat:
        return mat[name]
    low = name.lower()
    for k in mat:
        if not k.startswith("__") and k.lower() == low:
            return mat[k]
    available = [k for k in mat if not k.startswith("__")]
    raise KeyError(f"找不到变量 '{name}'，现有变量: {available}")


def scalar(x):
    arr = np.asarray(x).squeeze()
    return arr.item() if arr.ndim == 0 else arr


def to_str_list(x):
    arr = np.asarray(x).squeeze()
    out = []
    for el in np.atleast_1d(arr):
        if isinstance(el, np.ndarray):
            el = el.squeeze()
            out.append(str(el.item()) if el.ndim == 0 else "".join(map(str, el.ravel())))
        else:
            out.append(str(el))
    return out


# ─────────────────────────────────────────────────────────────
# Step 1 : 加载 spike 数据 → 计算 PSTH
# ─────────────────────────────────────────────────────────────
section("Step 1 : 加载 spike 数据，计算 PSTH")

mat_path = os.path.join(DATA_DIR, MOVIE_FILE)
if not os.path.exists(mat_path):
    sys.exit(f"[错误] 找不到 {mat_path}，请确认 DATA_DIR 路径")

mat      = sio.loadmat(mat_path)
binned   = np.asarray(getvar(mat, "binned")).astype(np.float32)   # [91,1203,93,5]
nreps    = np.asarray(scalar(getvar(mat, "nreps"))).astype(int).ravel()
fs       = float(scalar(getvar(mat, "samplingfreq")))
movnames = to_str_list(getvar(mat, "movnames"))
ncell    = int(scalar(getvar(mat, "ncell")))
nmov     = int(scalar(getvar(mat, "nmov")))
T_bins   = binned.shape[1]                                         # 1203

print(f"  binned   : {binned.shape}  (reps × time_bins × neurons × movies)")
print(f"  ncell={ncell}, nmov={nmov}, fs={fs} Hz, T_bins={T_bins}")
print(f"  nreps    : {nreps.tolist()}")
print(f"  movnames : {movnames}")

# PSTH：对每部电影，沿有效 reps 轴求平均 → [T_bins, ncell]
psth_list = []
for i in range(nmov):
    r = nreps[i]
    psth = binned[:r, :, :, i].mean(axis=0).astype(np.float32)  # [1203, 93]
    psth_list.append(psth)
    mean_rate = psth.mean() * fs  # 平均发放率 (spikes/s)
    print(f"  movie{i} [{movnames[i]:>12}]  reps={r:2d}  "
          f"PSTH {psth.shape}  "
          f"mean firing rate = {mean_rate:.2f} Hz")

# ─────────────────────────────────────────────────────────────
# Step 2 : 加载视频帧
# ─────────────────────────────────────────────────────────────
section("Step 2 : 加载视频帧")

try:
    import cv2
except ImportError:
    sys.exit("[错误] 未安装 opencv-python，请运行: pip install opencv-python")


def load_video_frames(avi_path, target_size):
    """
    读取 .avi 的全部帧，转灰度并降采样到 [target_size × target_size]。
    返回 numpy 数组 [N, H, W]，像素值归一化到 [0, 1]（float32）。
    """
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {avi_path}")

    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (target_size, target_size),
                             interpolation=cv2.INTER_AREA)
        frames.append(resized.astype(np.float32) / 255.0)
    cap.release()

    if not frames:
        raise ValueError(f"视频为空: {avi_path}")
    return np.stack(frames, axis=0)   # [N, H, W]


frame_list = []
for i in range(nmov):
    # 文件名形如 MultipleMoviesStim_1_tree.avi
    cands = sorted([f for f in os.listdir(DATA_DIR)
                    if f.lower().endswith(".avi") and f"_{i + 1}_" in f])
    if not cands:
        sys.exit(f"[错误] 找不到 movie{i} 对应的 .avi 文件（在 {DATA_DIR} 中搜索 '_{i+1}_'）")

    avi_path = os.path.join(DATA_DIR, cands[0])
    frames   = load_video_frames(avi_path, IMG_SIZE)
    n_raw    = frames.shape[0]

    # movie 0 (Tree) 视频只有 600 帧，实验中循环播放 2 次 → 首尾拼接得 1200 帧
    if n_raw == 600:
        frames = np.concatenate([frames, frames], axis=0)
        note   = f"600 帧 × 2（循环）→ {frames.shape[0]} 帧"
    else:
        note   = f"{n_raw} 帧"

    frame_list.append(frames)   # [1200, IMG_SIZE, IMG_SIZE]
    print(f"  movie{i} [{movnames[i]:>12}]  {cands[0]}  {note}  {IMG_SIZE}×{IMG_SIZE}  "
          f"像素范围 [{frames.min():.3f}, {frames.max():.3f}]")

# ─────────────────────────────────────────────────────────────
# Step 3 : 构建样本对 (X, Y)
# ─────────────────────────────────────────────────────────────
section("Step 3 : 构建样本对 (X, Y)")

print(f"  时间延迟 DELAY = {DELAY} bins  ({DELAY / fs * 1000:.1f} ms)")
print(f"  时间窗   WINDOW = {WINDOW}  →  X 维度 = {ncell * WINDOW}")
print(f"  目标帧分辨率 = {IMG_SIZE} × {IMG_SIZE}")
print()


def build_samples(movie_indices):
    """
    对指定电影列表批量构建所有有效 (X, Y) 样本对（向量化，速度快）。

    返回
    ----
    X         float32  [N, ncell * WINDOW]   输入：神经元发放概率（时间窗拼接）
    Y         float32  [N, IMG_SIZE, IMG_SIZE] 目标：对应视频帧
    movie_idx int8     [N]                   样本来自哪部电影（0~4）
    t_idx     int16    [N]                   帧时间索引 t（方便调试和可视化）
    """
    X_segs, Y_segs, mov_segs, t_segs = [], [], [], []

    for i in movie_indices:
        psth   = psth_list[i]    # [1203, ncell]
        frames = frame_list[i]   # [1200, IMG_SIZE, IMG_SIZE]

        # 对齐：统一以帧数（1200）为基准，丢弃多余 3 个 spike bin
        T_use = min(psth.shape[0], frames.shape[0])   # 1200

        # 有效帧索引范围（frame[t] 需配对 spike[t + DELAY]，
        # 时间窗最左端 spike[t + DELAY - WINDOW + 1] 也需 >= 0）
        t_start = max(0, WINDOW - 1 - DELAY)           # 通常为 0
        t_end   = T_use - 1 - DELAY                     # 最大有效 t
        if t_end < t_start:
            print(f"    [警告] movie{i} 有效样本数为 0，请检查 DELAY/WINDOW 配置")
            continue

        t_arr = np.arange(t_start, t_end + 1, dtype=np.int32)   # [N_mov]
        N_mov = len(t_arr)

        # ---- 向量化抽取 spike（时间窗）----
        if WINDOW == 1:
            # 最常用情形：单 bin，直接索引
            X_mov = psth[t_arr + DELAY, :]                       # [N_mov, ncell]
        else:
            # 多 bin 窗口：spike 索引矩阵 [N_mov, WINDOW]
            win_offsets = np.arange(WINDOW, dtype=np.int32)       # [0, 1, ..., W-1]
            s_idx = (t_arr[:, None] + DELAY - WINDOW + 1 +
                     win_offsets[None, :])                         # [N_mov, WINDOW]
            # psth[s_idx] → [N_mov, WINDOW, ncell]，reshape → [N_mov, ncell*WINDOW]
            X_mov = psth[s_idx, :].reshape(N_mov, -1)

        # ---- 抽取目标帧 ----
        Y_mov = frames[t_arr]                                      # [N_mov, H, W]

        X_segs.append(X_mov.astype(np.float32))
        Y_segs.append(Y_mov.astype(np.float32))
        mov_segs.append(np.full(N_mov, i, dtype=np.int8))
        t_segs.append(t_arr.astype(np.int16))

        print(f"    movie{i} [{movnames[i]:>12}]  t=[{t_start},{t_end}]  "
              f"共 {N_mov} 个样本")

    X_out   = np.concatenate(X_segs,   axis=0)
    Y_out   = np.concatenate(Y_segs,   axis=0)
    mov_out = np.concatenate(mov_segs, axis=0)
    t_out   = np.concatenate(t_segs,   axis=0)
    return X_out, Y_out, mov_out, t_out


print("  训练集 (train):")
X_train, Y_train, mov_train, t_train = build_samples(TRAIN_MOVS)

print("  验证集 (val):")
X_val, Y_val, mov_val, t_val = build_samples(VAL_MOVS)

print("  测试集 (test):")
X_test, Y_test, mov_test, t_test = build_samples(TEST_MOVS)

print()
for name, X, Y in [("train", X_train, Y_train),
                    ("val",   X_val,   Y_val),
                    ("test",  X_test,  Y_test)]:
    mem_mb = (X.nbytes + Y.nbytes) / 1e6
    print(f"  {name:5s}  X={X.shape}  Y={Y.shape}  "
          f"X∈[{X.min():.3f},{X.max():.3f}]  "
          f"Y∈[{Y.min():.3f},{Y.max():.3f}]  "
          f"内存≈{mem_mb:.1f} MB")

# ─────────────────────────────────────────────────────────────
# Step 4 : 保存为 .npz
# ─────────────────────────────────────────────────────────────
section("Step 4 : 保存 .npz 文件")

os.makedirs(OUT_DIR, exist_ok=True)

splits = {
    "train": (X_train, Y_train, mov_train, t_train),
    "val":   (X_val,   Y_val,   mov_val,   t_val),
    "test":  (X_test,  Y_test,  mov_test,  t_test),
}

for split_name, (X, Y, mov, t) in splits.items():
    out_path = os.path.join(OUT_DIR, f"dataset_{split_name}.npz")
    np.savez_compressed(out_path, X=X, Y=Y, movie_idx=mov, t_idx=t)
    size_mb  = os.path.getsize(out_path) / 1e6
    print(f"  {out_path}  →  {size_mb:.1f} MB")

# 元数据：保存所有超参，后续模型直接读取，无需硬编码
meta_dict = dict(
    ncell        = np.int32(ncell),
    nmov         = np.int32(nmov),
    fs           = np.float32(fs),
    img_size     = np.int32(IMG_SIZE),
    delay        = np.int32(DELAY),
    window       = np.int32(WINDOW),
    delay_ms     = np.float32(DELAY / fs * 1000),
    nreps        = nreps.astype(np.int32),
    movnames     = np.array(movnames, dtype=object),
    train_movies = np.array(TRAIN_MOVS, dtype=np.int32),
    val_movies   = np.array(VAL_MOVS,   dtype=np.int32),
    test_movies  = np.array(TEST_MOVS,  dtype=np.int32),
)
meta_path = os.path.join(OUT_DIR, "dataset_meta.npz")
np.savez(meta_path, **meta_dict)
print(f"  {meta_path}  （元数据）")

# ─────────────────────────────────────────────────────────────
# Step 5 : 验证加载 + 可视化
# ─────────────────────────────────────────────────────────────
section("Step 5 : 验证加载 + 可视化样本对")

# 重新从磁盘加载，确认保存无误
d_train = np.load(os.path.join(OUT_DIR, "dataset_train.npz"))
d_meta  = np.load(os.path.join(OUT_DIR, "dataset_meta.npz"), allow_pickle=True)

assert d_train["X"].shape[1] == ncell * WINDOW, \
    f"X 列数应为 {ncell * WINDOW}，实际 {d_train['X'].shape[1]}"
assert d_train["Y"].shape[1] == IMG_SIZE, \
    f"Y 高度应为 {IMG_SIZE}，实际 {d_train['Y'].shape[1]}"
assert d_train["Y"].shape[2] == IMG_SIZE, \
    f"Y 宽度应为 {IMG_SIZE}，实际 {d_train['Y'].shape[2]}"

print(f"  重新加载 train : X={d_train['X'].shape}  Y={d_train['Y'].shape}")
print(f"  元数据 delay={d_meta['delay']} bins  delay_ms={float(d_meta['delay_ms']):.1f} ms")
print("  数据完整性校验通过 ✓")

# 可视化
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = 6
    fig, axes = plt.subplots(2, n_show, figsize=(14, 5))
    fig.suptitle(
        f"sample visualization  delay={DELAY} bins ({DELAY/fs*1000:.0f} ms)  "
        f"IMG_SIZE={IMG_SIZE}  WINDOW={WINDOW}",
        fontsize=11
    )

    X_v, Y_v = d_train["X"], d_train["Y"]
    step = max(1, len(X_v) // n_show)

    for col in range(n_show):
        idx = col * step

        # 上排：神经元 spike 向量（条形图）
        ax = axes[0, col]
        ax.bar(range(ncell), X_v[idx, :ncell], color="#534AB7", width=1.0, linewidth=0)
        ax.set_ylim(0, max(X_v[idx, :ncell].max() * 1.2, 0.05))
        ax.set_title(
            f"mov={mov_train[col * step]}  t={t_train[col * step]}",
            fontsize=8
        )
        if col == 0:
            ax.set_ylabel("P(spike)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xticks([])

        # 下排：目标帧
        ax = axes[1, col]
        ax.imshow(Y_v[idx], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"frame {t_train[col * step]}", fontsize=8)

    plt.tight_layout()
    vis_path = os.path.join(OUT_DIR, "sample_pairs_visualization.png")
    fig.savefig(vis_path, dpi=130, bbox_inches="tight")
    print(f"  样本对可视化  →  {vis_path}")

except ImportError:
    print("  [跳过可视化] pip install matplotlib")

# ─────────────────────────────────────────────────────────────
# 最终小抄：后续代码如何加载
# ─────────────────────────────────────────────────────────────
section("完成！后续加载方式")

import numpy as np, torch

# ── 加载数据 ──────────────────────────────────────────
d      = np.load("./dataset_npz/dataset_train.npz")
X, Y   = d["X"], d["Y"]            # X:[N,{ncell}]  Y:[N,{IMG_SIZE},{IMG_SIZE}]
meta   = np.load("./dataset_npz/dataset_meta.npz", allow_pickle=True)
DELAY  = int(meta["delay"])         # {DELAY}
FS     = float(meta["fs"])          # {fs}

# ── 转为 PyTorch Tensor ──────────────────────────────
X_t = torch.from_numpy(X)          # float32 [N, {ncell}]
Y_t = torch.from_numpy(Y)          # float32 [N, {IMG_SIZE}, {IMG_SIZE}]

# ── DataLoader ───────────────────────────────────────
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_t, Y_t)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)

# 迭代一个 batch
for x_batch, y_batch in loader:
    print(x_batch.shape, y_batch.shape)   # [64, {ncell}]  [64, {IMG_SIZE}, {IMG_SIZE}]
    break

