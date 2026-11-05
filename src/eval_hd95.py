"""审稿 M8 补充评估：HD95 与 NSD 容差敏感性（不改动 evaluate.py 的 Dice/IoU/NSD 定义）。

两种模式：
- ``--mode 2d``（CARE，无 spacing 元数据）：2D HD95（纯像素），只在含该类 GT 的
  切片上计算；GT 非空而预测为空的切片取图像对角线为最坏值并记 n_missed。
  病例级汇总 = 病例内逐切片 HD95 的 95 分位，再跨病例 bootstrap。
  同一次推理顺带输出 NSD τ∈{1,2,3}px 敏感性（容差敏感性，沿用 evaluate.py 的
  逐切片约定：GT 空且预测空 -> 剔除；GT 空预测非空 -> 0 分）。
- ``--mode 3d``（MSD Colon，nii 有 spacing）：按病例把 2D 预测堆叠为 3D 体数据，
  用 nii header 的 (sx, sy, sz) 计算真 3D HD95（mm）与 3D NSD（τ 默认 2mm，
  EDT sampling=spacing）。单侧为空 -> 物理体对角线（mm）最坏值；双侧为空 -> 剔除。

用法：
    python src/eval_hd95.py --ckpt runs_v2/sup_100_s42/best.pt \
        --stage1_ckpt runs_v2/s1_100_s42/best.pt --crop_size 256 \
        --data_root data/CARE --cache_dir data/cache \
        --out results/v2/hd95/sup_100_s42 --device cpu
    python src/eval_hd95.py --mode 3d --ckpt runs/msd_sup_50/best.pt \
        --stage1_ckpt runs/msd_s1_50/best.pt --crop_size 256 \
        --data_root data/MSD/colon_npz --nii_dir data/MSD/Task10_Colon/imagesTr \
        --out results/v2/hd95/msd_sup_50 --device cpu
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset
from common import load_model, resolve_device
from dataset import list_slices
from evaluate import _boundary, bootstrap_ci
from tensorstore import TensorStore

NSD_TAU_DEFAULT = [1.0, 2.0, 3.0]


def slice_index(name):
    """`case{id}_slice{nnn}.npz` / `case{id}_{nnnn}.npz` -> 切片序号（末尾数字）。"""
    base = name[:-4] if name.endswith(".npz") else name
    m = re.search(r"(\d+)$", base.rsplit("_", 1)[-1])
    if not m:
        raise ValueError(f"无法从文件名解析切片序号: {name}")
    return int(m.group(1))


@torch.no_grad()
def predict_with_offsets(model2, images, offsets, empty, crop_size,
                         batch_size):
    """用缓存的 stage-1 偏移做 stage-2 推理（与 cascade_predict 的 stage-2 部分
    逐步对应：裁剪 -> softmax argmax -> 贴回 -> 空预测切片走重叠平铺兜底）。"""
    from cascade import _tiled_predict, crop_batch, paste_batch
    n, _, h, w = images.shape
    dev = images.device
    pred_canvas = torch.zeros((n, h, w), dtype=torch.uint8, device=dev)
    for off in range(0, n, batch_size):
        crops, _ = crop_batch(images[off:off + batch_size],
                              offsets[off:off + batch_size], crop_size)
        if str(dev).startswith("cuda"):
            crops = crops.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(dev).startswith("cuda")):
            logits = model2(crops)
        probs = torch.softmax(logits.float(), dim=1)
        _, pred = probs.max(1)
        pred_canvas[off:off + batch_size] = paste_batch(
            pred.to(torch.uint8), offsets[off:off + batch_size], crop_size,
            h, w)
    if empty.any():
        idx = empty.nonzero(as_tuple=True)[0]
        tp, _ = _tiled_predict(model2, images[idx], crop_size, batch_size)
        pred_canvas[idx] = tp
    return pred_canvas


@torch.no_grad()
def infer_test(ckpt, data_root, device, batch_size, cache_dir=None,
               stage1_ckpt=None, crop_size=None, limit=None,
               offsets_npz=None):
    """级联（或单模型）推理全部 test 切片，返回 {slice_name: (pred uint8, gt uint8)}。

    offsets_npz 非空时用缓存的 stage-1 偏移（precompute_s1_offsets.py 产物），
    跳过 stage-1 全图前向；此时不需要 --stage1_ckpt。"""
    files = list_slices(data_root, "test")
    if limit:
        files = files[:limit]
    store = TensorStore.from_files(files, device, cache_dir=cache_dir or None)
    loader = store.loader(batch_size, shuffle=False)
    model = load_model(ckpt, device)  # 会按 ckpt 设置 num_classes
    off_map = None
    if offsets_npz:
        z = np.load(offsets_npz, allow_pickle=False)
        off_map = {n: (o, e) for n, o, e in
                   zip(z["names"], z["offsets"], z["empty"])}
        print(f"复用 stage-1 偏移缓存: {offsets_npz}（{len(off_map)} 切片）")
        model1 = None
        cascade = False
    else:
        model1 = load_model(stage1_ckpt, device) if stage1_ckpt else None
        cascade = model1 is not None and crop_size
    out = {}
    for bi, batch in enumerate(loader):
        if off_map is not None:
            offs = torch.as_tensor(np.stack([off_map[n][0]
                                             for n in batch["name"]]),
                                   dtype=torch.long, device=device)
            empt = torch.as_tensor(np.stack([off_map[n][1]
                                             for n in batch["name"]]),
                                   dtype=torch.bool, device=device)
            preds = predict_with_offsets(
                model, batch["image"].to(device), offs, empt, crop_size,
                batch_size).cpu().numpy()
        elif cascade:
            from cascade import cascade_predict
            p, _ = cascade_predict(model1, model, batch["image"].to(device),
                                   crop_size, batch_size)
            preds = p.cpu().numpy()
        else:
            preds = model(batch["image"].to(device)).argmax(1).cpu().numpy()
        gts = batch["label"].cpu().numpy()
        for i, name in enumerate(batch["name"]):
            out[name] = (preds[i].astype(np.uint8), gts[i].astype(np.uint8))
        if (bi + 1) % 50 == 0:
            print(f"  推理 {len(out)}/{len(store)} 切片", flush=True)
    return out


# ---------------- 2D 模式（CARE，像素单位） ----------------

def hd95_nsd_2d(pred, gt, taus):
    """单类 2D：GT 为空 -> None（不计）；GT 非空预测空 -> (对角线, {τ:0}, missed)。
    返回 (hd95_px, {tau: nsd}, missed)。"""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not gt.any():
        return None
    h, w = gt.shape
    diag = float(np.hypot(h, w))
    gb = _boundary(gt)
    pb = _boundary(pred)
    if pb.sum() == 0:
        return diag, {t: 0.0 for t in taus}, True
    dt_g = distance_transform_edt(~gb)
    dt_p = distance_transform_edt(~pb)
    dpg = dt_g[pb]  # pred 边界 -> gt 边界
    dgp = dt_p[gb]  # gt 边界 -> pred 边界
    hd = float(np.percentile(np.concatenate([dpg, dgp]), 95))
    denom = pb.sum() + gb.sum()
    nsd = {t: float((dpg <= t).sum() + (dgp <= t).sum()) / denom for t in taus}
    return hd, nsd, False


def run_2d(preds_gts, taus, out, ckpt, stage1_ckpt, crop_size, data_root,
           n_boot, offsets_npz=None):
    names = dataset.fg_class_names()
    rows = []
    for name, (pred, gt) in sorted(preds_gts.items()):
        row = {"slice": name}
        for c in dataset.FG_CLASSES:
            cn = names[c]
            r = hd95_nsd_2d(pred == c, gt == c, taus)
            if r is None:
                # GT 空：hd95 不计；NSD 沿用 evaluate 约定（预测非空记 0，否则剔除）
                row[f"hd95_px_{cn}"] = np.nan
                row[f"missed_{cn}"] = 0
                p_any = bool((pred == c).any())
                for t in taus:
                    row[f"nsd_tau{t:g}_{cn}"] = 0.0 if p_any else np.nan
            else:
                hd, nsd, missed = r
                row[f"hd95_px_{cn}"] = hd
                row[f"missed_{cn}"] = int(missed)
                for t in taus:
                    row[f"nsd_tau{t:g}_{cn}"] = nsd[t]
        rows.append(row)

    csv_path = f"{out}_per_slice.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 病例级：病例内逐切片 HD95 的 95 分位
    case_rows = []
    cases = {}
    for row in rows:
        cases.setdefault(dataset.parse_case_id(row["slice"]), []).append(row)
    for cid, rs in sorted(cases.items()):
        cr = {"case": cid}
        for c in dataset.FG_CLASSES:
            cn = names[c]
            vals = [r[f"hd95_px_{cn}"] for r in rs
                    if not np.isnan(r[f"hd95_px_{cn}"])]
            if vals:
                cr[f"hd95_p95_px_{cn}"] = float(np.percentile(vals, 95))
                cr[f"hd95_mean_px_{cn}"] = float(np.mean(vals))
                cr[f"n_slices_{cn}"] = len(vals)
                cr[f"n_missed_{cn}"] = int(sum(r[f"missed_{cn}"] for r in rs))
        case_rows.append(cr)
    case_csv = f"{out}_per_case.csv"
    with open(case_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        w.writeheader()
        w.writerows(case_rows)

    summary = {"ckpt": os.path.abspath(ckpt), "stage1_ckpt": stage1_ckpt,
               "offsets_npz": offsets_npz,
               "crop_size": crop_size, "data_root": os.path.abspath(data_root),
               "split": "test", "mode": "2d", "unit": "pixel",
               "hd95_definition": "逐切片对称边界距离 95 分位（px），仅含该类 GT 的"
                                  "切片；预测空 -> 图像对角线最坏值；病例级 = 病例内"
                                  "逐切片 HD95 的 95 分位，跨病例 bootstrap 均值",
               "nsd_convention": "与 evaluate.py 相同的逐切片 NSD，多 tau 敏感性",
               "n_slices": len(rows), "n_cases": len(cases),
               "nsd_taus_px": taus, "per_slice_csv": os.path.abspath(csv_path),
               "per_case_csv": os.path.abspath(case_csv), "metrics": {}}
    for c in dataset.FG_CLASSES:
        cn = names[c]
        m = {}
        vals = [r[f"hd95_px_{cn}"] for r in rows]
        mean, lo, hi, n = bootstrap_ci(vals, n_boot)
        m["hd95_px_slicewise"] = {"mean": mean, "ci95": [lo, hi], "n": n}
        m["n_missed_slices"] = int(sum(r[f"missed_{cn}"] for r in rows))
        cv = [cr[f"hd95_p95_px_{cn}"] for cr in case_rows
              if f"hd95_p95_px_{cn}" in cr]
        mean, lo, hi, n = bootstrap_ci(cv, n_boot)
        m["hd95_px_case_p95"] = {"mean": mean, "ci95": [lo, hi], "n": n,
                                 "median": float(np.median(cv)) if cv else None}
        for t in taus:
            nsd_vals = [r[f"nsd_tau{t:g}_{cn}"] for r in rows]
            mean, lo, hi, n = bootstrap_ci(nsd_vals, n_boot)
            m[f"nsd_tau{t:g}"] = {"mean": mean, "ci95": [lo, hi], "n": n}
        summary["metrics"][cn] = m
    json_path = f"{out}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["metrics"], indent=1))
    print(f"已保存: {json_path} / {csv_path} / {case_csv}")
    return summary


# ---------------- 3D 模式（MSD，毫米单位） ----------------

def surface3d(mask):
    er = binary_erosion(mask, structure=np.ones((3, 3, 3)), border_value=0)
    return mask & ~er


def hd95_nsd_3d(pred, gt, spacing, taus_mm):
    """3D 对称 HD95（mm）与 NSD（τ mm）。双侧空 -> None；单侧空 -> (体对角线, 0)。"""
    pred, gt = pred.astype(bool), gt.astype(bool)
    pe, ge = bool(pred.any()), bool(gt.any())
    if not pe and not ge:
        return None
    diag_mm = float(np.sqrt(sum((s * sp) ** 2
                                for s, sp in zip(pred.shape, spacing))))
    if not pe or not ge:
        return diag_mm, {t: 0.0 for t in taus_mm}, "worst"
    sp_p, sp_g = surface3d(pred), surface3d(gt)
    dt_g = distance_transform_edt(~sp_g, sampling=spacing)
    dt_p = distance_transform_edt(~sp_p, sampling=spacing)
    dpg = dt_g[sp_p]
    dgp = dt_p[sp_g]
    hd = float(np.percentile(np.concatenate([dpg, dgp]), 95))
    denom = sp_p.sum() + sp_g.sum()
    nsd = {t: float((dpg <= t).sum() + (dgp <= t).sum()) / denom
           for t in taus_mm}
    return hd, nsd, "ok"


def dice3d(pred, gt):
    ps, gs = int(pred.sum()), int(gt.sum())
    if ps == 0 and gs == 0:
        return None
    return 2.0 * int((pred & gt).sum()) / (ps + gs)


def nii_spacing_map(nii_dir):
    """{npz 病例 id（如 colon022）: (sx, sy, sz) mm}，反向 preprocess_msd.case_id_from。"""
    import nibabel as nib
    out = {}
    for f in os.listdir(nii_dir):
        if f.endswith(".nii.gz") and not f.startswith("._"):
            cid = f[:-len(".nii.gz")].replace("_", "")
            z = nib.load(os.path.join(nii_dir, f)).header.get_zooms()
            out[cid] = tuple(float(v) for v in z[:3])
    return out


def run_3d(preds_gts, taus_mm, out, ckpt, stage1_ckpt, crop_size, data_root,
           n_boot, nii_dir=None, spacing_fallback=None):
    names = dataset.fg_class_names()
    if nii_dir:
        spacings = nii_spacing_map(nii_dir)
    elif spacing_fallback:
        spacings = {}
    else:
        raise ValueError("3D 模式需要 --nii_dir 或 --spacing")

    cases = {}
    for name, (pred, gt) in preds_gts.items():
        cases.setdefault(dataset.parse_case_id(name), []).append(
            (slice_index(name), pred, gt))

    case_rows = []
    for cid, items in sorted(cases.items()):
        items.sort(key=lambda x: x[0])
        pvol = np.stack([p for _, p, _ in items])
        gvol = np.stack([g for _, _, g in items])
        if cid in spacings:
            sx, sy, sz = spacings[cid]
        else:
            sx, sy, sz = spacing_fallback
        # 堆叠数组轴序 (z, x, y)：npz 切片存的是 vol[:, :, z]（轴 x, y）
        spacing = (sz, sx, sy)
        cr = {"case": cid, "n_slices": len(items),
              "spacing_x_mm": sx, "spacing_y_mm": sy, "spacing_z_mm": sz}
        for c in dataset.FG_CLASSES:
            cn = names[c]
            pb, gb = pvol == c, gvol == c
            cr[f"dice3d_{cn}"] = dice3d(pb, gb)
            r = hd95_nsd_3d(pb, gb, spacing, taus_mm)
            if r is None:
                cr[f"hd95_mm_{cn}"] = np.nan
                cr[f"status_{cn}"] = "exclude"
                for t in taus_mm:
                    cr[f"nsd3d_tau{t:g}mm_{cn}"] = np.nan
            else:
                hd, nsd, st = r
                cr[f"hd95_mm_{cn}"] = hd
                cr[f"status_{cn}"] = st
                for t in taus_mm:
                    cr[f"nsd3d_tau{t:g}mm_{cn}"] = nsd[t]
        case_rows.append(cr)

    case_csv = f"{out}_per_case.csv"
    os.makedirs(os.path.dirname(case_csv) or ".", exist_ok=True)
    with open(case_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        w.writeheader()
        w.writerows(case_rows)

    summary = {"ckpt": os.path.abspath(ckpt), "stage1_ckpt": stage1_ckpt,
               "crop_size": crop_size, "data_root": os.path.abspath(data_root),
               "split": "test", "mode": "3d", "unit": "mm",
               "spacing_source": nii_dir or f"uniform {spacing_fallback}",
               "hd95_definition": "3D 对称表面距离 95 分位（EDT sampling=spacing）；"
                                  "单侧空 -> 物理体对角线最坏值（status=worst）；"
                                  "双侧空 -> 剔除；跨病例 bootstrap 均值",
               "n_cases": len(case_rows), "nsd3d_taus_mm": taus_mm,
               "per_case_csv": os.path.abspath(case_csv), "metrics": {}}
    for c in dataset.FG_CLASSES:
        cn = names[c]
        m = {}
        for key, vals in (
                ("dice3d", [r[f"dice3d_{cn}"] for r in case_rows]),
                ("hd95_mm", [r[f"hd95_mm_{cn}"] for r in case_rows])):
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            mean, lo, hi, n = bootstrap_ci(vals, n_boot)
            m[key] = {"mean": mean, "ci95": [lo, hi], "n": n,
                      "median": float(np.median(vals)) if vals else None}
        for t in taus_mm:
            vals = [r[f"nsd3d_tau{t:g}mm_{cn}"] for r in case_rows
                    if not np.isnan(r[f"nsd3d_tau{t:g}mm_{cn}"])]
            mean, lo, hi, n = bootstrap_ci(vals, n_boot)
            m[f"nsd3d_tau{t:g}mm"] = {"mean": mean, "ci95": [lo, hi], "n": n}
        m["n_worst_case"] = int(sum(r[f"status_{cn}"] == "worst"
                                    for r in case_rows))
        m["n_excluded_both_empty"] = int(sum(r[f"status_{cn}"] == "exclude"
                                             for r in case_rows))
        summary["metrics"][cn] = m
    json_path = f"{out}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["metrics"], indent=1))
    print(f"已保存: {json_path} / {case_csv}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d")
    ap.add_argument("--stage1_ckpt", default=None)
    ap.add_argument("--crop_size", type=int, default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--taus", type=float, nargs="+", default=NSD_TAU_DEFAULT,
                    help="NSD 容差：2D 模式单位 px，3D 模式单位 mm")
    ap.add_argument("--nii_dir", default=None,
                    help="3D 模式：原始 nii.gz 目录（读 header spacing）")
    ap.add_argument("--spacing", type=float, nargs=3, default=None,
                    metavar=("SX", "SY", "SZ"),
                    help="3D 模式无 nii 时的统一 spacing（mm），仅冒烟用")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 切片（调试用）")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--offsets_npz", default=None,
                    help="precompute_s1_offsets.py 的 stage-1 偏移缓存；"
                         "给了则跳过 stage-1 前向（仍需要 --crop_size）")
    args = ap.parse_args()

    device = resolve_device(args.device)
    preds_gts = infer_test(args.ckpt, args.data_root, device, args.batch_size,
                           args.cache_dir, args.stage1_ckpt, args.crop_size,
                           args.limit, args.offsets_npz)
    if args.mode == "2d":
        run_2d(preds_gts, args.taus, args.out, args.ckpt, args.stage1_ckpt,
               args.crop_size, args.data_root, args.n_boot, args.offsets_npz)
    else:
        run_3d(preds_gts, args.taus, args.out, args.ckpt, args.stage1_ckpt,
               args.crop_size, args.data_root, args.n_boot,
               args.nii_dir, args.spacing)


if __name__ == "__main__":
    main()
