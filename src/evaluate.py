"""在 CARE 官方 test 集上评估：Dice / IoU / NSD + bootstrap 95%CI + 配对 Wilcoxon。

用法：
    python src/evaluate.py --ckpt runs/sup_100/best.pt --data_root data/CARE --out results/sup_100
    python src/evaluate.py --compare results/sup_10_summary.json results/semi_10_summary.json

指标约定（与 AGENTS.md 一致，不可随意更改）：
- 逐切片、逐前景类（1=正常直肠, 2=肿瘤）计算；
- 某类 GT 为空时：预测也为空 -> 该切片该类不计入平均；预测非空 -> 该类记 0 分；
- NSD 为 2D 简化版：边界像素在容差 tau（默认 2 px ≈ 2mm，假设 1mm/px）内的比例，
  双向对称。CARE 为 224x224 裁剪图且无 spacing 元数据，这是论文中须声明的近似。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset
from common import load_model, resolve_device
from dataset import list_slices
from tensorstore import TensorStore

CLASS_NAMES = {1: "rectum", 2: "tumor"}


def _class_names():
    """按当前类别数给前景类命名（CARE 保持 {1:rectum, 2:tumor} 不变）。"""
    return dataset.fg_class_names()


def _boundary(mask):
    if not mask.any():
        return np.zeros_like(mask)
    er = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return mask & ~er


def surface_dice(pred, gt, tau):
    """单类 2D NSD（双向边界在 tau 内的比例）。"""
    pb, gb = _boundary(pred), _boundary(gt)
    if pb.sum() == 0 and gb.sum() == 0:
        return np.nan  # 两者皆空：由调用方按约定处理
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    a = (dt_g[pb] <= tau).sum()
    b = (dt_p[gb] <= tau).sum()
    return (a + b) / (pb.sum() + gb.sum())


def slice_metrics(pred, gt, tau):
    """逐类 dice/iou/nsd；返回 {class: {'dice','iou','nsd'}}，GT 空且预测空 -> nan。"""
    out = {}
    for c in dataset.FG_CLASSES:
        p, g = pred == c, gt == c
        if not g.any():
            score = 0.0 if p.any() else np.nan
            out[c] = {"dice": score, "iou": score, "nsd": score}
            continue
        inter = (p & g).sum()
        union = (p | g).sum()
        dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else np.nan
        iou = inter / union if union > 0 else np.nan
        nsd = surface_dice(p, g, tau)
        out[c] = {"dice": dice, "iou": iou, "nsd": nsd}
    return out


def bootstrap_ci(values, n_boot=2000, seed=42):
    """均值 bootstrap 95%CI。values 可含 nan（已剔除后传入）。"""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5)), int(len(arr))


@torch.no_grad()
def run_eval(ckpt, data_root, out, device, batch_size=32, tau=2.0, cache_dir=None,
             stage1_ckpt=None, crop_size=None):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    files = list_slices(data_root, "test")
    loader = TensorStore.from_files(files, device, cache_dir=cache_dir) \
        .loader(batch_size, shuffle=False)
    model = load_model(ckpt, device)
    model1 = load_model(stage1_ckpt, device) if stage1_ckpt else None
    cascade = model1 is not None and crop_size
    names = _class_names()  # load_model 已按 ckpt 设置类别数

    rows = []  # per-slice 结果
    for batch in loader:
        if cascade:
            from cascade import cascade_predict
            p, _ = cascade_predict(model1, model, batch["image"].to(device),
                                   crop_size, batch_size)
            preds = p.cpu().numpy()
        else:
            preds = model(batch["image"].to(device)).argmax(1).cpu().numpy()
        gts = batch["label"].cpu().numpy()
        for name, pred, gt in zip(batch["name"], preds, gts):
            m = slice_metrics(pred, gt, tau)
            row = {"slice": name}
            for c in dataset.FG_CLASSES:
                for k in ("dice", "iou", "nsd"):
                    row[f"{k}_{names[c]}"] = m[c][k]
            rows.append(row)

    csv_path = f"{out}_per_slice.csv"
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"ckpt": os.path.abspath(ckpt), "data_root": os.path.abspath(data_root),
               "split": "test", "n_slices": len(rows), "nsd_tau_px": tau,
               "stage1_ckpt": stage1_ckpt, "crop_size": crop_size,
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


def run_compare(json_a, json_b):
    """配对 Wilcoxon：A vs B 的逐切片 Dice/IoU/NSD 差值。nan 行成对剔除。"""
    import pandas as pd
    with open(json_a) as f:
        sa = json.load(f)
    with open(json_b) as f:
        sb = json.load(f)
    da = pd.read_csv(sa["per_slice_csv"]).set_index("slice")
    db = pd.read_csv(sb["per_slice_csv"]).set_index("slice")
    df = da.join(db, lsuffix="_a", rsuffix="_b", how="inner")
    print(f"配对切片数: {len(df)}（A={os.path.basename(json_a)} vs B={os.path.basename(json_b)}）")
    result = {"a": json_a, "b": json_b, "n_pairs": len(df), "tests": {}}
    for cname in sa["metrics"]:  # 类别名从 summary 推断（CARE: rectum/tumor；MSD: tumor）
        for k in ("dice", "iou", "nsd"):
            col = f"{k}_{cname}"
            pair = df[[f"{col}_a", f"{col}_b"]].dropna()
            diff = pair[f"{col}_b"] - pair[f"{col}_a"]
            if len(pair) < 5 or (diff == 0).all():
                stat = {"n": int(len(pair)), "mean_a": float(pair[f"{col}_a"].mean()),
                        "mean_b": float(pair[f"{col}_b"].mean()),
                        "p": None, "note": "差值全零或样本过少，无法检验"}
            else:
                p = float(wilcoxon(pair[f"{col}_a"], pair[f"{col}_b"]).pvalue)
                stat = {"n": int(len(pair)), "mean_a": float(pair[f"{col}_a"].mean()),
                        "mean_b": float(pair[f"{col}_b"].mean()),
                        "mean_diff": float(diff.mean()), "p": p,
                        "significant_0.05": bool(p < 0.05)}
            result["tests"][col] = stat
            print(f"{col}: A={stat['mean_a']:.4f} B={stat['mean_b']:.4f} "
                  f"p={stat.get('p')} n={stat['n']}")
    out_path = os.path.join(os.path.dirname(json_b) or ".",
                            f"compare_{os.path.basename(json_a)[:-13]}_vs_"
                            f"{os.path.basename(json_b)[:-13]}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"已保存: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--data_root")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A_SUMMARY", "B_SUMMARY"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--tau", type=float, default=2.0, help="NSD 容差（像素）")
    ap.add_argument("--cache_dir", default=None, help="preprocess_cache 产物目录")
    ap.add_argument("--stage1_ckpt", default=None, help="级联 stage-1 ckpt")
    ap.add_argument("--crop_size", type=int, default=None, help="级联裁剪边长")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1])
        return
    if not (args.ckpt and args.data_root and args.out):
        ap.error("评估模式需要 --ckpt --data_root --out")
    run_eval(args.ckpt, args.data_root, args.out, resolve_device(args.device),
             args.batch_size, args.tau, args.cache_dir,
             args.stage1_ckpt, args.crop_size)


if __name__ == "__main__":
    main()
