"""
Stage 2 辅助: 从已有 checkpoint 生成全量预测 (train/val/test)
用于已训练好模型后补充 train/val 预测，不需重新训练

用法:
  python generate_stage2_preds.py
  python generate_stage2_preds.py --ckpt nonlinear_hp_results/best_nonlinear_hp.pt
"""

import os, argparse, numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RidgeCV

from nonlinear_models import CNNHighpassDecoder, MLPHighpassDecoder, clip01, load_split, make_lowpass


def main():
    parser = argparse.ArgumentParser(description="Stage 2: 从 checkpoint 生成全量预测")
    parser.add_argument("--npz_dir", type=str, default="dataset_npz")
    parser.add_argument("--ckpt", type=str, default="nonlinear_hp_results/best_nonlinear_hp.pt")
    parser.add_argument("--out_dir", type=str, default="nonlinear_hp_results")
    parser.add_argument("--lp_sigma", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1. 加载 checkpoint
    print(f"\n[1/4] 加载 checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    print(f"  模型: {cfg['model_type']}, n_input={cfg['n_input']}, img_size={cfg['img_size']}")
    print(f"  最佳 epoch={cfg['best_epoch']}, val_loss={cfg['best_val']:.6f}")

    # 2. 重建模型
    print("\n[2/4] 重建模型...")
    if cfg["model_type"] == "CNN":
        model = CNNHighpassDecoder(cfg["n_input"], cfg["img_size"])
    else:
        model = MLPHighpassDecoder(cfg["n_input"], cfg["img_size"])
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)
    model.eval()

    # 标准化参数
    x_mean = ckpt["x_mean"]; x_std = ckpt["x_std"]
    if hasattr(x_mean, "numpy"): x_mean = x_mean.numpy()
    if hasattr(x_std, "numpy"): x_std = x_std.numpy()
    x_mean = x_mean.reshape(1, -1); x_std = x_std.reshape(1, -1)

    # 3. 加载数据
    print("\n[3/4] 加载数据集...")
    data = {}
    for split in ["train", "val", "test"]:
        X, Y, _, _ = load_split(args.npz_dir, split)
        data[split] = {"X": X, "Y": Y}
        print(f"  {split}: X {X.shape}, Y {Y.shape}")

    # Ridge-LP
    print(f"\n[3/4] 生成 Ridge-LP 预测 (sigma={args.lp_sigma})...")
    Y_train = data["train"]["Y"]
    Y_train_lp = make_lowpass(Y_train, args.lp_sigma)
    ridge = RidgeCV(alphas=[0.01,0.1,1,10,100,1000,10000,100000], cv=5, fit_intercept=False)
    ridge.fit(data["train"]["X"], Y_train_lp.reshape(len(Y_train_lp), -1))
    print(f"  alpha={ridge.alpha_:.4g}")

    # 4. 预测
    print("\n[4/4] CNN/MLP 前向传播...")
    for split in ["train", "val", "test"]:
        X = data[split]["X"]
        Y = data[split]["Y"]
        X_norm = (X - x_mean) / (x_std + 1e-8)

        # Ridge-LP
        lp = clip01(ridge.predict(X).reshape(-1, cfg["img_size"], cfg["img_size"]))

        # CNN-HP
        hp_list = []
        with torch.no_grad():
            for i in range(0, len(X_norm), args.batch_size):
                batch = torch.from_numpy(X_norm[i:i+args.batch_size]).to(device)
                hp_list.append(model(batch).squeeze(1).cpu().numpy())
        hp = np.concatenate(hp_list, axis=0).astype(np.float32)

        combined = clip01(lp + hp)

        for name, arr in [("lp", lp), ("hp", hp), ("combined", combined)]:
            path = os.path.join(args.out_dir, f"preds_{split}_{name}.npy")
            np.save(path, arr)
            print(f"  [保存] preds_{split}_{name}.npy  {arr.shape}")

        # 真实高频
        Y_lp = make_lowpass(Y, args.lp_sigma)
        np.save(os.path.join(args.out_dir, f"true_hp_{split}.npy"), Y - Y_lp)
        np.save(os.path.join(args.out_dir, f"true_lp_{split}.npy"), Y_lp)

    print(f"\n完成! Stage 3 对接:")
    print(f"  python deblur_train.py")


if __name__ == "__main__":
    main()