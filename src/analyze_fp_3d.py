"""审稿人驱动的两项分析（共用一次级联推理）：

Task A  无瘤切片上的 FP 肿瘤像素负担：逐切片统计 FP 像素数与连通域个数/大小，
        用于论证"~80% 无瘤切片有肿瘤预测"只是微小散点而非大块假阳性。
Task B  3D 病例级参考指标：按病例把切片堆成 3D 体数据，逐类（直肠/肿瘤）计算
        3D Dice 与 HD95（scipy distance_transform_edt，对称表面距离 95 分位）。

单位说明：CARE npz 无 spacing 元数据，z 轴即切片序号，故 HD95 以"像素/切片"
混合单位报告（体数据对角线作为单侧为空时的最坏值），论文中须显式注明。

用法：
    python src/analyze_fp_3d.py --tasks AB            # 默认跑两项
    python src/analyze_fp_3d.py --tasks A             # 只跑 FP 负担
    # results/masks/{tag}_test_pred.npz 已存在时直接加载，不重复推理
"""

import argparse
import json
import os
import re
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODELS = [
    ("sup_100", "runs/sup_100/best.pt", "runs/s1_100/best.pt"),
    ("sup_50", "runs/sup_50/best.pt", "runs/s1_50/best.pt"),
    ("ours_50", "runs/ours_50/best.pt", "runs/s1_50/best.pt"),
]

CLASS_NAMES = {1: "rectum", 2: "tumor"}


def parse_model_spec(item):
    """name=stage2_ckpt:stage1_ckpt -> (name, stage2, stage1)"""
    name, rest = item.split("=", 1)
    ck2, ck1 = rest.rsplit(":", 1)
    return name, ck2, ck1


def load_gt(data_root, cache_dir):
    """GT 标签 {slice_name: (H,W) uint8}。优先 cache，缺失时逐 npz 读取。"""
    from dataset import list_slices, remap_label
    lab_p = os.path.join(cache_dir, "test_labels.npy") if cache_dir else None
    names_p = os.path.join(cache_dir, "test_names.json") if cache_dir else None
    if lab_p and os.path.exists(lab_p) and os.path.exists(names_p):
        labels = np.load(lab_p, mmap_mode="r")
        with open(names_p) as f:
            names = json.load(f)
        return {n: np.asarray(labels[i]) for i, n in enumerate(names)}
    gt = {}
    for p in list_slices(data_root, "test"):
        with np.load(p) as z:
            gt[os.path.basename(p)] = remap_label(z["label"]).astype(np.uint8)
    return gt


def run_inference(tag, ckpt2, ckpt1, data_root, cache_dir, crop_size,
                  batch_size, device, mask_dir):
    """级联推理全部 test 切片 -> {name: pred uint8}，并落盘 npz（可复用）。"""
    import torch  # noqa: F401
    from cascade import cascade_predict
    from common import load_model, resolve_device
    from dataset import list_slices
    from tensorstore import TensorStore

    device = resolve_device(device)
    model2 = load_model(ckpt2, device)
    model1 = load_model(ckpt1, device)
    files = list_slices(data_root, "test")
    store = TensorStore.from_files(files, device, cache_dir=cache_dir or None)
    loader = store.loader(batch_size, shuffle=False)
    preds = {}
    for bi, batch in enumerate(loader):
        p, _ = cascade_predict(model1, model2, batch["image"].to(device),
                               crop_size, batch_size)
        arr = p.cpu().numpy()
        for i, n in enumerate(batch["name"]):
            preds[n] = arr[i]
        if (bi + 1) % 20 == 0:
            print(f"  [{tag}] {len(preds)}/{len(store)} 切片", flush=True)
    os.makedirs(mask_dir, exist_ok=True)
    path = os.path.join(mask_dir, f"{tag}_test_pred.npz")
    np.savez(path, **preds)
    print(f"  [{tag}] 预测已保存 -> {path} ({len(preds)} 切片)")
    return preds


def load_or_run_masks(specs, args):
    """masks 已存在则加载，否则推理并保存。返回 {tag: {name: pred}}。"""
    masks = {}
    for tag, ck2, ck1 in specs:
        path = os.path.join(args.mask_dir, f"{tag}_test_pred.npz")
        if os.path.exists(path):
            print(f"[{tag}] 加载已保存预测 {path}")
            with np.load(path) as z:
                masks[tag] = {k: z[k] for k in z.files}
        else:
            print(f"[{tag}] 运行级联推理 (stage2={ck2}, stage1={ck1})")
            masks[tag] = run_inference(tag, ck2, ck1, args.data_root,
                                       args.cache_dir, args.crop_size,
                                       args.batch_size, args.device,
                                       args.mask_dir)
    return masks


# ---------------- Task A: FP burden ----------------

def task_a(masks, gt):
    """无瘤（GT 无类别 2）切片上的 FP 肿瘤像素负担。"""
    out = {}
    for tag, preds in masks.items():
        fp_counts, comp_per_slice, comp_sizes = [], [], []
        n_tf = 0
        for name, g in gt.items():
            if (g == 2).any():
                continue
            n_tf += 1
            p = preds[name]
            fp = int((p == 2).sum())
            if fp == 0:
                continue
            fp_counts.append(fp)
            lab, nc = ndimage.label(p == 2)
            comp_per_slice.append(nc)
            comp_sizes.extend(ndimage.sum(np.ones_like(lab), lab,
                                          np.arange(1, nc + 1)).tolist())
        fc = np.asarray(fp_counts, dtype=float)
        n_fp = len(fc)
        out[tag] = {
            "n_tumor_free_slices": n_tf,
            "n_fp_slices": n_fp,
            "frac_fp_slices": n_fp / max(n_tf, 1),
            "fp_pixels_median": float(np.median(fc)) if n_fp else 0.0,
            "fp_pixels_p90": float(np.percentile(fc, 90)) if n_fp else 0.0,
            "fp_pixels_p99": float(np.percentile(fc, 99)) if n_fp else 0.0,
            "fp_pixels_max": float(fc.max()) if n_fp else 0.0,
            "frac_fp_slices_lt10": float((fc < 10).mean()) if n_fp else 0.0,
            "frac_fp_slices_lt50": float((fc < 50).mean()) if n_fp else 0.0,
            "frac_fp_slices_lt200": float((fc < 200).mean()) if n_fp else 0.0,
            "components_per_fp_slice_median":
                float(np.median(comp_per_slice)) if n_fp else 0.0,
            "components_per_fp_slice_mean":
                float(np.mean(comp_per_slice)) if n_fp else 0.0,
            "component_size_median":
                float(np.median(comp_sizes)) if comp_sizes else 0.0,
            "component_size_p90":
                float(np.percentile(comp_sizes, 90)) if comp_sizes else 0.0,
        }
        print(f"[Task A][{tag}] 无瘤切片 {n_tf}，含 FP {n_fp} "
              f"({out[tag]['frac_fp_slices']:.1%})，FP 像素中位数 "
              f"{out[tag]['fp_pixels_median']:.0f}")
    return out


# ---------------- Task B: 3D Dice / HD95 ----------------

def slice_index(name):
    """切片序号：CARE `case{id}_slice{nnn}.npz` 与假数据 `case{id}_{nnnn}.npz`
    两种命名都支持（取最后一个下划线后的数字）。"""
    base = name[:-4] if name.endswith(".npz") else name
    tail = base.rsplit("_", 1)[-1]
    m = re.search(r"(\d+)$", tail)
    if not m:
        raise ValueError(f"无法从文件名解析切片序号: {name}")
    return int(m.group(1))


def case_id(name):
    base = name[:-4] if name.endswith(".npz") else name
    return base.rsplit("_", 1)[0]


def surface(mask):
    """二值体数据的表面体素（3x3x3 腐蚀的补集，边界视为背景）。"""
    er = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)),
                                border_value=0)
    return mask & ~er


def hd95(pred, gt):
    """对称 HD95（体素单位）。返回 (value, status)；status:
    'ok' / 'worst'（单侧为空，取体数据对角线）/ 'exclude'（双侧为空）。"""
    pe, ge = bool(pred.any()), bool(gt.any())
    if not pe and not ge:
        return None, "exclude"
    diag = float(np.sqrt(sum(s * s for s in pred.shape)))
    if not pe or not ge:
        return diag, "worst"
    sp, sg = surface(pred), surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg)
    dt_p = ndimage.distance_transform_edt(~sp)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95)), "ok"


def dice3d(pred, gt):
    """3D Dice。双侧为空 -> None（剔除）；单侧为空 -> 0。"""
    ps, gs = int(pred.sum()), int(gt.sum())
    if ps == 0 and gs == 0:
        return None
    return 2.0 * int((pred & gt).sum()) / (ps + gs)


def bootstrap_ci(values, n_boot, seed):
    """病例级（cluster）bootstrap：有放回重抽病例，取均值分布的 2.5/97.5 分位。"""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return {"mean": float(arr.mean()),
            "ci95": [float(np.percentile(means, 2.5)),
                     float(np.percentile(means, 97.5))],
            "n": int(len(arr))}


def task_b(masks, gt, n_boot, seed):
    """3D 病例级 Dice / HD95（像素/切片单位，见模块 docstring 单位说明）。"""
    # 病例 -> 排序后的切片名
    case_names = {}
    for n in gt:
        case_names.setdefault(case_id(n), []).append(n)
    for v in case_names.values():
        v.sort(key=slice_index)

    out = {"note": "CARE 无 spacing 元数据：z 轴为切片序号，HD95 以像素/切片"
                   "混合单位报告；单侧为空时 HD95 取体数据对角线为最坏值；"
                   "双侧为空的 (case, class) 从均值中剔除。",
           "n_cases": len(case_names)}
    for tag, preds in masks.items():
        per_class = {c: {"dice": [], "hd95": [], "n_worst": 0,
                         "n_excluded": 0} for c in CLASS_NAMES}
        for cid, names in sorted(case_names.items()):
            gvol = np.stack([gt[n] for n in names])
            pvol = np.stack([preds[n] for n in names])
            for c in CLASS_NAMES:
                pb, gb = pvol == c, gvol == c
                d = dice3d(pb, gb)
                h, st = hd95(pb, gb)
                rec = per_class[c]
                if d is not None:
                    rec["dice"].append(d)
                if h is not None:
                    rec["hd95"].append(h)
                if st == "worst":
                    rec["n_worst"] += 1
                elif st == "exclude":
                    rec["n_excluded"] += 1
        out[tag] = {}
        for c, cname in CLASS_NAMES.items():
            rec = per_class[c]
            out[tag][cname] = {
                "dice": bootstrap_ci(rec["dice"], n_boot, seed),
                "hd95": bootstrap_ci(rec["hd95"], n_boot, seed),
                "n_hd95_worst_case": rec["n_worst"],
                "n_excluded_both_empty": rec["n_excluded"],
            }
            d, h = out[tag][cname]["dice"], out[tag][cname]["hd95"]
            print(f"[Task B][{tag}][{cname}] Dice {d['mean']:.4f} "
                  f"[{d['ci95'][0]:.4f},{d['ci95'][1]:.4f}] "
                  f"HD95 {h['mean']:.2f} [{h['ci95'][0]:.2f},{h['ci95'][1]:.2f}] "
                  f"(n={h['n']}, worst={rec['n_worst']}, "
                  f"excl={rec['n_excluded']})")
    return out


def print_tables(res_a, res_b):
    if res_a:
        print("\n=== Task A: 无瘤切片 FP 肿瘤像素负担 ===")
        hdr = ("model", "n_tf", "n_fp", "frac_fp", "med", "p90", "p99",
               "max", "<10", "<50", "<200", "cc/slice", "cc_med_sz")
        print("{:<9} {:>5} {:>5} {:>7} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}"
              " {:>6} {:>9} {:>10}".format(*hdr))
        for tag, s in res_a.items():
            print(f"{tag:<9} {s['n_tumor_free_slices']:>5} "
                  f"{s['n_fp_slices']:>5} {s['frac_fp_slices']:>7.1%} "
                  f"{s['fp_pixels_median']:>6.0f} {s['fp_pixels_p90']:>6.0f} "
                  f"{s['fp_pixels_p99']:>6.0f} {s['fp_pixels_max']:>6.0f} "
                  f"{s['frac_fp_slices_lt10']:>6.1%} "
                  f"{s['frac_fp_slices_lt50']:>6.1%} "
                  f"{s['frac_fp_slices_lt200']:>6.1%} "
                  f"{s['components_per_fp_slice_median']:>9.1f} "
                  f"{s['component_size_median']:>10.0f}")
    if res_b:
        print("\n=== Task B: 3D 病例级 Dice / HD95（像素/切片单位，见 note） ===")
        print(f"{'model':<9} {'class':<8} {'Dice mean [95%CI]':<28} "
              f"{'HD95 mean [95%CI]':<28} {'n':>4} {'worst':>6} {'excl':>5}")
        for tag in [t for t in res_b if t not in ("note", "n_cases")]:
            for cname in ("rectum", "tumor"):
                r = res_b[tag][cname]
                d, h = r["dice"], r["hd95"]
                print(f"{tag:<9} {cname:<8} "
                      f"{d['mean']:.4f} [{d['ci95'][0]:.4f},{d['ci95'][1]:.4f}]"
                      f"   {h['mean']:>7.2f} [{h['ci95'][0]:.2f},"
                      f"{h['ci95'][1]:.2f}] {h['n']:>7} "
                      f"{r['n_hd95_worst_case']:>6} "
                      f"{r['n_excluded_both_empty']:>5}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="AB", choices=["A", "B", "AB"])
    ap.add_argument("--data_root", default="data/CARE")
    ap.add_argument("--cache_dir", default="data/cache",
                    help="空字符串表示不用 cache（逐 npz 读 GT）")
    ap.add_argument("--mask_dir", default="results/masks")
    ap.add_argument("--out_dir", default="results/v2")
    ap.add_argument("--model", action="append", default=None,
                    help="name=stage2_ckpt:stage1_ckpt，可重复；缺省用内置 3 模型")
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    specs = [parse_model_spec(m) for m in args.model] if args.model \
        else DEFAULT_MODELS

    masks = load_or_run_masks(specs, args)
    print("加载 GT ...")
    gt = load_gt(args.data_root, args.cache_dir)
    # 名称对齐检查
    for tag, preds in masks.items():
        missing = [n for n in gt if n not in preds]
        assert not missing, f"[{tag}] 预测缺少 {len(missing)} 切片: {missing[:3]}"
    os.makedirs(args.out_dir, exist_ok=True)

    res_a = res_b = None
    if "A" in args.tasks:
        res_a = task_a(masks, gt)
        p = os.path.join(args.out_dir, "fp_burden.json")
        with open(p, "w") as f:
            json.dump(res_a, f, indent=2)
        print(f"Task A 结果 -> {p}")
    if "B" in args.tasks:
        res_b = task_b(masks, gt, args.n_boot, args.seed)
        p = os.path.join(args.out_dir, "metrics3d.json")
        with open(p, "w") as f:
            json.dump(res_b, f, indent=2)
        print(f"Task B 结果 -> {p}")
    print_tables(res_a, res_b)


if __name__ == "__main__":
    main()
