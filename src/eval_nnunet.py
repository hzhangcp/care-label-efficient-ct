"""nnU-Net 预测结果评估适配器：用与 evaluate.py 完全相同的指标定义打分。

nnU-Net 的预测不是 LightUNet ckpt，无法走 evaluate.py --ckpt 路径；本脚本读取
nnUNetv2_predict 输出的病例级 nii.gz（(z,y,x)，与 prepare_nnunet_care.py 写入的
imagesTs 几何一致），按病例展开回逐切片，用 evaluate.slice_metrics（Dice/IoU/NSD，
GT 空且预测空 -> nan 等约定完全不变）和 bootstrap_ci 打分，输出与 evaluate.py
相同 schema 的 per_slice CSV + summary JSON（兼容 evaluate.py --compare 与
aggregate_results.py）。

用法：
    python src/eval_nnunet.py --pred_dir data/nnunet/predictions/Dataset501_CARE100 \
        --data_root data/CARE --out results_v2/nnunet_sup_100 \
        [--save_pred_npz data/nnunet/predictions_per_slice/Dataset501_CARE100]
"""

import argparse
import csv as _csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset
from dataset import list_slices, remap_label
from evaluate import bootstrap_ci, slice_metrics
from prepare_nnunet_care import group_by_case


def run_eval_nnunet(pred_dir, data_root, out, tau=2.0, save_pred_npz=None):
    import SimpleITK as sitk
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if save_pred_npz:
        os.makedirs(save_pred_npz, exist_ok=True)
    # CARE 3 类（本脚本只服务 CARE；FG_CLASSES 默认即 (1,2)，显式设定以防外部改动）
    dataset.set_num_classes(3)
    names = dataset.fg_class_names()

    test_groups = group_by_case(list_slices(data_root, "test"))
    rows = []
    for case_id in sorted(test_groups):
        files = test_groups[case_id]
        pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"缺预测文件: {pred_path}")
        pred_vol = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        assert pred_vol.shape[0] == len(files), \
            f"{case_id}: 预测 {pred_vol.shape[0]} 层 vs 源 {len(files)} 层"
        for i, p in enumerate(files):
            with np.load(p) as z:
                gt = remap_label(z["label"])
            pred = pred_vol[i]
            assert pred.shape == gt.shape, f"{case_id}[{i}]: {pred.shape} vs {gt.shape}"
            m = slice_metrics(pred, gt, tau)
            row = {"slice": os.path.basename(p)}
            for c in dataset.FG_CLASSES:
                for k in ("dice", "iou", "nsd"):
                    row[f"{k}_{names[c]}"] = m[c][k]
            rows.append(row)
            if save_pred_npz:
                np.savez_compressed(os.path.join(save_pred_npz, os.path.basename(p)),
                                    pred=pred.astype(np.uint8))

    csv_path = f"{out}_per_slice.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"ckpt": os.path.abspath(pred_dir), "source": "nnunet",
               "data_root": os.path.abspath(data_root), "split": "test",
               "n_slices": len(rows), "nsd_tau_px": tau,
               "stage1_ckpt": None, "crop_size": None,
               "per_slice_csv": os.path.abspath(csv_path), "metrics": {}}
    for c in dataset.FG_CLASSES:
        cname = names[c]
        summary["metrics"][cname] = {}
        for k in ("dice", "iou", "nsd"):
            mean, lo, hi, n = bootstrap_ci([r[f"{k}_{cname}"] for r in rows])
            summary["metrics"][cname][k] = {"mean": mean, "ci95": [lo, hi], "n": n}
    json_path = f"{out}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["metrics"], indent=2))
    print(f"已保存: {json_path} / {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="nnUNetv2_predict 输出目录（病例级 nii.gz）")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True, help="结果前缀，如 results_v2/nnunet_sup_100")
    ap.add_argument("--tau", type=float, default=2.0, help="NSD 容差（像素），与 evaluate.py 一致")
    ap.add_argument("--save_pred_npz", default=None,
                    help="可选：把逐切片预测另存为 npz（键 pred）的目录")
    args = ap.parse_args()
    run_eval_nnunet(args.pred_dir, args.data_root, args.out, args.tau,
                    args.save_pred_npz)


if __name__ == "__main__":
    main()
