"""P3：定性分割对比图（好/中/差病例各 n 张）。

python src/visualize_predictions.py --data_root data/CARE \
    --ckpt sup=runs/sup_100/best.pt --ckpt semi=runs/semi_50/best.pt \
    --out_dir outputs/qualitative --n_per_tier 3

按第一个 ckpt 的肿瘤 Dice 在 test 上排序，取最好/居中/最差三档切片，
每张飞图：image | GT overlay | 模型A 预测 | 模型B 预测。
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_model, resolve_device
from dataset import list_slices
from evaluate import slice_metrics
from tensorstore import TensorStore


def overlay(image, mask):
    base = np.stack([image] * 3, axis=-1).astype(np.float32)
    base /= max(base.max(), 1e-6)
    out = base.copy()
    out[mask == 1] = [0, 1, 0]   # 绿=正常直肠
    out[mask == 2] = [1, 0, 0]   # 红=肿瘤
    return out


def disp(a):
    """CARE 切片存储方向相对常规轴位显示转了 90°，展示时顺时针转回
    （检查床弧线在画面下方=标准轴位）。仅影响可视化，不影响指标。"""
    return np.rot90(a, -1)


@torch.no_grad()
def predict_all(ckpt, files, device, batch_size=32, cache_dir=None,
                stage1_ckpt=None, crop_size=None):
    from cascade import cascade_predict
    model = load_model(ckpt, device)
    model1 = load_model(stage1_ckpt, device) if stage1_ckpt else None
    cascade = model1 is not None and crop_size
    loader = TensorStore.from_files(files, device, cache_dir=cache_dir) \
        .loader(batch_size, shuffle=False)
    preds, gts, names, images = [], [], [], []
    for batch in loader:
        if cascade:
            p, _ = cascade_predict(model1, model, batch["image"].to(device),
                                   crop_size, batch_size)
        else:
            p = model(batch["image"].to(device)).argmax(1)
        preds.extend(p.cpu().numpy())
        gts.extend(batch["label"].cpu().numpy())
        names.extend(batch["name"])
        images.extend(batch["image"].cpu().numpy()[:, 0])
    return names, images, gts, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="形如 name=path，可重复（最多 3 个）")
    ap.add_argument("--out_dir", default="outputs/qualitative")
    ap.add_argument("--n_per_tier", type=int, default=3, help="好/中/差各几张")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--stage1_ckpt", default=None, help="级联 stage-1 ckpt（所有模型共享）")
    ap.add_argument("--crop_size", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    files = list_slices(args.data_root, "test")
    specs = []
    for item in args.ckpt:
        name, path = item.split("=", 1)
        specs.append((name, path))
    assert 1 <= len(specs) <= 3

    # 用第一个模型的肿瘤 Dice 分档
    names, images, gts, preds0 = predict_all(specs[0][1], files, device,
                                             cache_dir=args.cache_dir,
                                             stage1_ckpt=args.stage1_ckpt,
                                             crop_size=args.crop_size)
    tumor_dice = np.array([slice_metrics(p, g, 2.0)[2]["dice"] for p, g in zip(preds0, gts)])
    valid = np.where(~np.isnan(tumor_dice))[0]  # 只看 GT 含肿瘤的切片
    order = valid[np.argsort(tumor_dice[valid])]
    tiers = {
        "bad": order[:args.n_per_tier],
        "mid": order[len(order) // 2 - args.n_per_tier // 2:
                     len(order) // 2 - args.n_per_tier // 2 + args.n_per_tier],
        "good": order[-args.n_per_tier:],
    }

    other_preds = [predict_all(p, files, device, cache_dir=args.cache_dir,
                               stage1_ckpt=args.stage1_ckpt,
                               crop_size=args.crop_size)[3]
                   for _, p in specs[1:]]
    all_preds = [preds0] + other_preds

    os.makedirs(args.out_dir, exist_ok=True)
    for tier, idxs in tiers.items():
        for j, i in enumerate(idxs):
            n_col = 2 + len(specs)
            fig, axes = plt.subplots(1, n_col, figsize=(4 * n_col, 4))
            axes[0].imshow(disp(images[i]), cmap="gray"); axes[0].set_title("image")
            axes[1].imshow(disp(overlay(images[i], gts[i]))); axes[1].set_title("GT")
            for k, (name, _) in enumerate(specs):
                axes[2 + k].imshow(disp(overlay(images[i], all_preds[k][i])))
                d = slice_metrics(all_preds[k][i], gts[i], 2.0)[2]["dice"]
                axes[2 + k].set_title(f"{name} (tumor dice={d:.2f})")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{tier} | {names[i]}")
            path = os.path.join(args.out_dir, f"{tier}_{j}_{os.path.splitext(names[i])[0]}.png")
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"已保存 {path}")


if __name__ == "__main__":
    main()
