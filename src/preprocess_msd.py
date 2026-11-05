"""MSD Task10 Colon 3D nii.gz -> CARE 兼容 2D npz 切片（外部验证数据集）。

预处理策略：
- 窗宽窗位：HU clip [-100, 400] 后线性映射到 [0,1]（软组织窗，MSD 结直肠
  肿瘤分割文献常用窗位；与 CARE 的 [0,1] float 输入对齐）；
- 标签：二值 {0=背景, 1=肿瘤}（MSD Task10 原始定义），对应 config
  "num_classes": 2，评估只报 tumor 类；remap_label（>2 并入 2）对其是恒等操作；
- 病人级划分（防泄漏）：126 例按 seed=42 shuffle 后 85%/15% 分为
  train 目录（107 例）/ test 目录（19 例）；train 目录内部再由
  dataset.split_train_val(val_ratio=0.15) 划验证集，总体约 72/13/15；
- 空（无肿瘤）切片：结肠肿瘤切片稀疏（train 9.4% / test 10.4%），train 与 test
  目录空切片**全保留**（--empty_keep 默认 1.0）。2026-09-28 修订：此前默认 0.15
  会把 train 肿瘤片先验抬到 ~41%（test 仅 ~10%），模型在空切片上假阳性率飙升，
  而这些切片按评估协议计 0 分，均值被严重稀释；全保留使训练/测试先验对齐。
  评估协议与 CARE 一致（全部切片逐片计指标，空 GT 且空预测的切片不计入平均）。

输出目录结构与 CARE 相同：
    {out}/train/train_npz/colon001_slice0000.npz   (image float32, label uint8)
    {out}/test/test_npz/...

用法：
    python src/preprocess_msd.py --src data/MSD/Task10_Colon --out data/MSD/colon_npz
"""

import argparse
import json
import os
import random

import numpy as np


def hu_normalize(vol, hu_min=-100.0, hu_max=400.0):
    """HU clip [hu_min, hu_max] -> [0,1] float32。"""
    vol = np.clip(vol.astype(np.float32), hu_min, hu_max)
    return (vol - hu_min) / (hu_max - hu_min)


def case_id_from(fname):
    """colon_001.nii.gz -> colon001（无下划线，parse_case_id 的 rsplit('_',1) 安全）。"""
    return os.path.basename(fname).replace(".nii.gz", "").replace("_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/MSD/Task10_Colon",
                    help="含 imagesTr/labelsTr 的解压目录")
    ap.add_argument("--out", default="data/MSD/colon_npz")
    ap.add_argument("--empty_keep", type=float, default=1.0,
                    help="train 目录空切片保留比例（默认 1.0 全保留，与 test 先验对齐；"
                         "test 恒为全保留）")
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hu_min", type=float, default=-100.0)
    ap.add_argument("--hu_max", type=float, default=400.0)
    args = ap.parse_args()

    import nibabel as nib

    img_dir = os.path.join(args.src, "imagesTr")
    lab_dir = os.path.join(args.src, "labelsTr")
    cases = sorted(f for f in os.listdir(img_dir)
                   if f.endswith(".nii.gz") and not f.startswith("._"))
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    n_test = max(1, int(round(len(cases) * args.test_ratio)))
    test_cases = set(cases[:n_test])
    print(f"共 {len(cases)} 例 -> train 目录 {len(cases) - n_test} 例 / "
          f"test 目录 {n_test} 例（seed={args.seed}）")

    stats = {"n_cases": len(cases), "n_train_cases": len(cases) - n_test,
             "n_test_cases": n_test, "empty_keep_train": args.empty_keep,
             "hu_window": [args.hu_min, args.hu_max], "splits": {}}
    slice_rng = np.random.default_rng(args.seed)
    for split in ("train", "test"):
        npz_dir = os.path.join(args.out, split, f"{split}_npz")
        os.makedirs(npz_dir, exist_ok=True)
        split_cases = [c for c in cases if (c in test_cases) == (split == "test")]
        n_total = n_tumor = n_kept = 0
        label_values = set()
        for ci, fname in enumerate(split_cases):
            cid = case_id_from(fname)
            img = nib.load(os.path.join(img_dir, fname))
            lab = nib.load(os.path.join(lab_dir, fname))
            vol = hu_normalize(np.asanyarray(img.dataobj), args.hu_min, args.hu_max)
            msk = np.asanyarray(lab.dataobj).astype(np.uint8)
            assert vol.shape == msk.shape, f"{fname}: image/label 形状不一致"
            label_values.update(np.unique(msk).tolist())
            for z in range(vol.shape[2]):
                n_total += 1
                has_tumor = msk[:, :, z].any()
                n_tumor += bool(has_tumor)
                # train 目录空切片下采样；test 全保留
                if not has_tumor and split == "train" \
                        and slice_rng.random() >= args.empty_keep:
                    continue
                np.savez(os.path.join(npz_dir, f"{cid}_slice{z:04d}.npz"),
                         image=np.ascontiguousarray(vol[:, :, z]),
                         label=np.ascontiguousarray(msk[:, :, z]))
                n_kept += 1
            if (ci + 1) % 20 == 0:
                print(f"  [{split}] {ci + 1}/{len(split_cases)} 例")
        stats["splits"][split] = {
            "n_cases": len(split_cases), "n_slices_total": n_total,
            "n_tumor_slices": n_tumor,
            "tumor_slice_ratio": round(n_tumor / max(n_total, 1), 4),
            "n_slices_kept": n_kept,
            "label_values": sorted(label_values)}
        print(f"[{split}] {len(split_cases)} 例：总切片 {n_total}，"
              f"肿瘤切片 {n_tumor} ({n_tumor / max(n_total, 1):.1%})，"
              f"落盘 {n_kept}，标签值域 {sorted(label_values)}")

    with open(os.path.join(args.out, "preprocess_report.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"完成。报告已存 {args.out}/preprocess_report.json")


if __name__ == "__main__":
    main()
