# Salamander Retinal Movie Decoding — Dataset README

## Background

Larval salamander retinal ganglion cell (RGC) recordings while viewing 5 natural movies (data from Dryad: `doi:10.5061/dryad.4qrfj6qm8`).

**Task:** Given the spike activity of 93 neurons at time `t`, reconstruct the video frame the retina was viewing at time `t`.

This is **Baseline 5** from the assignment, implementing the multi-stage decoding pipeline in Kim et al. 2021 (*Neural Computation*) adapted for movie (continuous frame-by-frame) decoding.

---

## Files

```
dataset_npz/
├── dataset_train.npz     # movies: Tree, Water, Grasses  → 3588 samples
├── dataset_val.npz       # movie:  Fish                  → 1196 samples
├── dataset_test.npz      # movie:  Self Motion           → 1196 samples
└── dataset_meta.npz      # recording parameters (read once, keep handy)
```

> **Split rule:** entire movies are held out — never frame-level random shuffle. Adjacent frames are highly correlated; random splitting would leak information and inflate metrics.

---

## Data Format

Each `.npz` file contains four arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `X` | `[N, 93]` | float32 | Input — PSTH spike vector (firing probability per neuron) |
| `Y` | `[N, 64, 64]` | float32 | Target — grayscale video frame, pixels in `[0, 1]` |
| `movie_idx` | `[N]` | int8 | Which movie this sample came from (0–4) |
| `t_idx` | `[N]` | int16 | Frame index `t` within that movie |

**Key numbers:**
- 93 neurons, 64×64 px frames, 60 Hz recording, ~20 s per movie
- `X` is the PSTH (average over 80–91 repeated trials) — values in `[0, 1]`
- `Y` pixels are normalized to `[0, 1]` from 8-bit grayscale

---

## Time Alignment — Critical Detail

Neural responses are **not instantaneous**. STA (spike-triggered average) analysis on a checkerboard stimulus showed ~66.7 ms latency (2 bins @ 30 Hz calibration = 4 bins @ 60 Hz movie rate).

```
Frame t  ──── 66.7 ms (4 bins) ────►  Spike[t+4]
```

So each sample pair is:

```
X[i]  =  PSTH[ t + 4, : ]   ← spike fired 4 bins AFTER seeing the frame
Y[i]  =  frame[ t ]          ← the frame we want to reconstruct
```

This means the model learns: "given what the neurons fired, infer what they were looking at 66.7 ms ago."

---

## Loading in PyTorch

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_split(path):
    d = np.load(path)
    X = torch.from_numpy(d["X"])          # float32 [N, 93]
    Y = torch.from_numpy(d["Y"])          # float32 [N, 64, 64]
    return X, Y

X_train, Y_train = load_split("dataset_npz/dataset_train.npz")
X_val,   Y_val   = load_split("dataset_npz/dataset_val.npz")
X_test,  Y_test  = load_split("dataset_npz/dataset_test.npz")

train_loader = DataLoader(TensorDataset(X_train, Y_train),
                          batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, Y_val),
                          batch_size=64, shuffle=False)

# One batch: x → [64, 93], y → [64, 64, 64]
for x, y in train_loader:
    print(x.shape, y.shape)
    break
```

**Reading metadata:**
```python
meta = np.load("dataset_npz/dataset_meta.npz", allow_pickle=True)
print(meta["delay"],    # 4 bins
      meta["delay_ms"], # 66.7 ms
      meta["fs"],       # 60.0 Hz
      meta["img_size"]) # 64
```

---

## What to Build

The target pipeline has three stages. Build them in order — each stage is a standalone model.

### Stage 1 — Linear baseline (ridge regression)
- Input: `X` `[N, 93]`
- Output: predicted low-pass (Gaussian-blurred) frame `[N, 64, 64]`
- Use `sklearn.linear_model.Ridge` or the closed-form solution
- Prepares the **low-frequency component** for later stages

### Stage 2 — Nonlinear high-pass decoder (CNN / MLP)
- Input: `X` `[N, 93]`
- Output: high-pass residual `[N, 64, 64]` (GT frame minus blurred frame)
- The paper shows only nonlinear methods can recover fine details
- Add this to Stage 1 output → **combined reconstruction**

### Stage 3 — Deblurring network
- Input: combined reconstruction from Stage 1+2
- Output: sharpened final frame
- U-Net or the DeblurGAN-v2 generator (paper reference: Kim et al. 2021, §4.5)

### Evaluation (use the same metrics across all stages)
```python
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def evaluate(y_pred, y_true):
    """y_pred, y_true: numpy arrays [N, H, W] in [0, 1]"""
    mse  = np.mean((y_pred - y_true) ** 2)
    psnr_scores = [psnr(y_true[i], y_pred[i], data_range=1.0) for i in range(len(y_pred))]
    ssim_scores = [ssim(y_true[i], y_pred[i], data_range=1.0) for i in range(len(y_pred))]
    return {"MSE": mse, "PSNR": np.mean(psnr_scores), "SSIM": np.mean(ssim_scores)}
```

Run `evaluate()` on the test split only. Report all three numbers per stage to show the incremental gain of each component.

---

## Generating the Dataset (if you need to rebuild)

```bash
pip install numpy scipy opencv-python matplotlib
python build_dataset.py
```

Edit the two paths at the top of `build_dataset.py`:
```python
DATA_DIR = "D:/BMI/pj/dataset"      # folder with .mat and .avi files
OUT_DIR  = "D:/BMI/pj/dataset_npz"  # output folder
```

Movies used:

| `movie_idx` | Name | Split | Reps |
|---|---|---|---|
| 0 | Tree | train | 83 |
| 1 | Water | train | 80 |
| 2 | Grasses | train | 84 |
| 3 | Fish | **val** | 91 |
| 4 | Self Motion | **test** | 85 |

> **Note on Tree (movie 0):** the `.avi` is only 600 frames (10 s) but the recording is 1200 bins (20 s) because the video looped twice. `build_dataset.py` handles this automatically by duplicating the frames.
