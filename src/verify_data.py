"""P0：真实数据校验。核对病例数/切片数/标签值域，验证 parse_case_id 假设，出 4 张可视化图。

python src/verify_data.py --data_root data/CARE
"""

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import list_slices, parse_case_id, read_bbox_csv

# DataV6 实际数字（P0 核实 2026-09-15）；U-SAM 论文报告 train 317 例/26563 切片，
# 与网盘 DataV6 略有出入（多 1 例 93 切片），以实际文件为准。
EXPECTED = {"train": (318, 26656), "test": (81, 6461)}


def verify_split(data_root, split):
    files = list_slices(data_root, split)
    print(f"\n=== {split} ===")
    print(f"npz 文件数: {len(files)}（官方: {EXPECTED[split][1]}）")

    case_ids = [parse_case_id(f) for f in files]
    uniq = sorted(set(case_ids))
    print(f"病例数（parse_case_id）: {len(uniq)}（官方: {EXPECTED[split][0]}）")
    print(f"文件名样例: {[os.path.basename(f) for f in files[:5]]}")
    counts = collections.Counter(case_ids)
    print(f"每例切片数 min/max: {min(counts.values())}/{max(counts.values())}")
    # ⚠️ 若病例数与官方数字相差悬殊，说明 parse_case_id 假设不成立，须先修正再继续
    if len(uniq) != EXPECTED[split][0]:
        print("⚠️⚠️ 病例数与官方不符 —— parse_case_id 假设可能不成立，"
              "请人工核对文件名规律并修改 dataset.parse_case_id！")

    bbox = read_bbox_csv(data_root, split)
    print(f"bbox csv 行数: {len(bbox)}")

    label_values, img_dtypes, img_shapes = set(), set(), set()
    img_min, img_max = np.inf, -np.inf
    for i, p in enumerate(files):
        with np.load(p) as z:
            label_values |= set(np.unique(z["label"]).tolist())
            img_dtypes.add(str(z["image"].dtype))
            img_shapes.add(tuple(z["image"].shape))
            img_min = min(img_min, float(z["image"].min()))
            img_max = max(img_max, float(z["image"].max()))
        if i >= 199:  # 抽查 200 个足够确认值域与形状
            break
    print(f"标签值域（抽查）: {sorted(label_values)}"
          "（原始期望 [0, 1, 2, 3]，加载时 3→2，见 dataset.remap_label）")
    print(f"image dtype: {img_dtypes} shape: {img_shapes} 值域: [{img_min:.3f}, {img_max:.3f}]")
    return files


def visualize(files, out_dir, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    picks = np.linspace(0, len(files) - 1, n).astype(int)
    for j, i in enumerate(picks):
        with np.load(files[i]) as z:
            image, label = z["image"], z["label"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray"); axes[0].set_title("image")
        axes[1].imshow(label, cmap="jet", vmin=0, vmax=2); axes[1].set_title("label")
        overlay = np.stack([image] * 3, axis=-1)
        overlay = overlay / max(overlay.max(), 1e-6)
        overlay[label == 1] = [0, 1, 0]
        overlay[label == 2] = [1, 0, 0]
        axes[2].imshow(overlay); axes[2].set_title("overlay (绿=直肠 红=肿瘤)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(os.path.basename(files[i]))
        path = os.path.join(out_dir, f"sample_{j}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="outputs/verify")
    args = ap.parse_args()
    files_tr = verify_split(args.data_root, "train")
    files_te = verify_split(args.data_root, "test")
    visualize(files_tr, args.out_dir, n=4)
    print("\n请肉眼确认 outputs/verify/ 下 4 张图：图像方向正确、标签与器官吻合。")


if __name__ == "__main__":
    main()
