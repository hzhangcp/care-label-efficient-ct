"""审稿意见 M4：全切片口径（all-slice）指标——把 FP 负担计入 Dice 分数。

背景：现有口径（evaluate.py / reanalyze_v2.py）只在 GT 含该类的切片上报告
Dice/IoU/NSD，空 GT 切片只报 FP 切片率。results_v2/*_per_slice.csv 只存
dice/iou/nsd（dice 与 iou 数学上互相确定），无法反推像素级 tp/fp/fn，
故全切片口径必须重新推理逐切片计数（LightUNet 需 GPU）或从已有预测
文件直读（nnU-Net 预测 nii.gz 在本地，纯 CPU 即可）。

两级流水线：
1) counts 生成：每切片每类 (tp, fp, fn) 像素计数 → {tag}_counts.csv
   - LightUNet 级联：`--infer`（headline 集，GPU）/ `--infer --all`（全档）。
     ckpt/stage1_ckpt/crop_size 一律从 results_v2/{tag}_summary.json 读，
     与已评估口径完全一致；推理管线与 evaluate.run_eval 同源（TensorStore +
     cascade_predict）。test 集只读。
   - 单个显式推理（冒烟/调试）：`--infer_one CKPT[:STAGE1_CKPT] --crop_size N`。
   - nnU-Net：`--infer_nnunet`（读 data/nnunet/predictions/Dataset50* 的
     病例级 nii.gz，CPU；展开逻辑与 eval_nnunet.py 相同）。
2) 聚合（纯 CPU）：`--aggregate`
   - 病例级全切片 pooled Dice（主口径）：每例每类 2Σtp/(2Σtp+Σfp+Σfn)，
     Σ 对该病例全部切片（含空 GT 切片，FP 由此计入）；肿瘤类只在 GT 含瘤
     病例上平均（json 记录 n_cases），直肠类同理。某类整例 GT 与预测全空
     （分母为 0）→ 该例该类记 nan 并从病例平均剔除（json 记录 n_excluded；
     惯例：真阴性病例既不奖励也不惩罚）。
   - 全切片 micro Dice（次口径）：全部切片汇总 tp/fp/fn 算一个 Dice。
   - 附带交叉校验量：切片级 FP-inclusive 平均 Dice（tp+fp+fn>0 的切片逐
     切片 Dice 均值，等价于把 per_slice.csv 非 nan 行平均）与 FP 切片率
     （GT 空切片中预测非空的比例，应与 reanalyze_v2 / aggregate_v2_seeds
     的 fp_rate 一致）。
   - 三种子 mean±std（tag 以 _s{seed} 后缀归组；无种子后缀如 nnunet 为
     单种子参照，std=null，tex 标 †）。
   输出 results/v2/allslice_metrics.json + paper/manuscript/tab_allslice.tex
   （默认复制到 paper/sr/tab_allslice.tex）。

用法（生产）：
  # GPU 节点：headline = sup_100 + sup/semi/mt/cps/ours/bcp 50% 档 × 3 种子
  python src/eval_allslice.py --infer --device cuda
  python src/eval_allslice.py --infer --all --device cuda   # 扩展到所有档
  # CPU：nnU-Net counts + 聚合出表
  python src/eval_allslice.py --infer_nnunet
  python src/eval_allslice.py --aggregate
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset
from dataset import list_slices

RESULTS_V2 = "results_v2"
ALLSLICE_DIR = os.path.join("results", "v2", "allslice")
OUT_JSON = os.path.join("results", "v2", "allslice_metrics.json")
OUT_TEX = os.path.join("paper", "manuscript", "tab_allslice.tex")
SR_TEX = os.path.join("paper", "sr", "tab_allslice.tex")

METHOD_ORDER = [("sup", "Supervised"), ("semi", "Pseudo-label ST"),
                ("mt", "Mean Teacher"), ("cps", "CPS"), ("bcp", "BCP"),
                ("ours", "Ours")]
RATIOS = [1, 5, 10, 25, 50]
SEEDS = [42, 43, 44]
SINGLE_SEED_REFS = ["mtcons_50_s42", "sup180_50_s42", "sup180_100_s42"]
NNUNET_TAGS = {"nnunet_sup_100": "Dataset501_CARE100",
               "nnunet_sup_50": "Dataset502_CARE50"}


def prefer_fixed_mt_tags(tags):
    """正式结果全部具备 mtfixed 时，推理队列也使用修复版而不是旧 mt。"""
    complete = all(os.path.exists(os.path.join(
        RESULTS_V2, f"mtfixed_{r}_s{sd}_summary.json"))
        for r in RATIOS for sd in SEEDS)
    if not complete:
        return tags
    return ["mtfixed_" + tag[len("mt_"):] if tag.startswith("mt_") else tag
            for tag in tags]


def headline_tags():
    tags = [f"sup_100_s{s}" for s in SEEDS]
    tags += [f"{m}_50_s{s}" for m, _ in METHOD_ORDER for s in SEEDS]
    return prefer_fixed_mt_tags(tags)


def all_tags():
    tags = [f"{m}_{r}_s{s}" for m, _ in METHOD_ORDER for r in RATIOS
            for s in SEEDS]
    tags += [f"sup_100_s{s}" for s in SEEDS]
    tags += SINGLE_SEED_REFS
    return prefer_fixed_mt_tags(sorted(set(tags), key=tags.index))


# ---------------------------------------------------------------- counts 生成

@torch.no_grad()
def infer_counts(ckpt, data_root, out_csv, device, batch_size=32,
                 cache_dir=None, stage1_ckpt=None, crop_size=None):
    """与 evaluate.run_eval 同源的推理循环，但逐切片累计 tp/fp/fn 像素计数。"""
    import torch
    from common import load_model
    from tensorstore import TensorStore

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    files = list_slices(data_root, "test")
    loader = TensorStore.from_files(files, device, cache_dir=cache_dir) \
        .loader(batch_size, shuffle=False)
    model = load_model(ckpt, device)
    model1 = load_model(stage1_ckpt, device) if stage1_ckpt else None
    cascade = model1 is not None and crop_size
    names = dataset.fg_class_names()  # load_model 已按 ckpt 设置类别数

    rows = []
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
            row = {"slice": name}
            for c in dataset.FG_CLASSES:
                pm, gm = pred == c, gt == c
                row[f"tp_{names[c]}"] = int((pm & gm).sum())
                row[f"fp_{names[c]}"] = int((pm & ~gm).sum())
                row[f"fn_{names[c]}"] = int((~pm & gm).sum())
            rows.append(row)

    _write_counts(out_csv, rows)
    print(f"[infer] {os.path.basename(out_csv)}: {len(rows)} 切片, "
          f"ckpt={ckpt}, stage1={stage1_ckpt}, crop={crop_size}")


def infer_counts_from_summary(tag, device, batch_size=32,
                              results_dir=RESULTS_V2, out_dir=ALLSLICE_DIR):
    """生产模式：从 results_v2/{tag}_summary.json 读 ckpt/stage1/crop_size。"""
    with open(os.path.join(results_dir, f"{tag}_summary.json")) as f:
        s = json.load(f)
    infer_counts(s["ckpt"], s["data_root"],
                 os.path.join(out_dir, f"{tag}_counts.csv"), device,
                 batch_size, cache_dir="data/cache",
                 stage1_ckpt=s.get("stage1_ckpt"),
                 crop_size=s.get("crop_size"))


@torch.no_grad()
def infer_counts_nnunet(pred_dir, data_root, out_csv):
    """nnU-Net 病例级 nii.gz -> 逐切片 counts（展开逻辑同 eval_nnunet）。"""
    import SimpleITK as sitk
    from prepare_nnunet_care import group_by_case

    dataset.set_num_classes(3)  # 本脚本只服务 CARE 三分类
    names = dataset.fg_class_names()
    rows = []
    for case_id, files in sorted(group_by_case(list_slices(data_root, "test")).items()):
        pred_path = os.path.join(pred_dir, f"{case_id}.nii.gz")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"缺预测文件: {pred_path}")
        pred_vol = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        assert pred_vol.shape[0] == len(files), \
            f"{case_id}: 预测 {pred_vol.shape[0]} 层 vs 源 {len(files)} 层"
        for i, p in enumerate(files):
            with np.load(p) as z:
                gt = dataset.remap_label(z["label"])
            pred = pred_vol[i]
            assert pred.shape == gt.shape, f"{case_id}[{i}]"
            row = {"slice": os.path.basename(p)}
            for c in dataset.FG_CLASSES:
                pm, gm = pred == c, gt == c
                row[f"tp_{names[c]}"] = int((pm & gm).sum())
                row[f"fp_{names[c]}"] = int((pm & ~gm).sum())
                row[f"fn_{names[c]}"] = int((~pm & gm).sum())
            rows.append(row)
    _write_counts(out_csv, rows)
    print(f"[nnunet] {os.path.basename(out_csv)}: {len(rows)} 切片, pred_dir={pred_dir}")


def _write_counts(out_csv, rows):
    import csv as _csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- 聚合

def parse_case_id(fname):
    """与 dataset.parse_case_id 一致。"""
    base = os.path.basename(fname)
    if base.endswith(".npz"):
        base = base[:-4]
    return base.rsplit("_", 1)[0] if "_" in base else base


def per_seed_metrics(counts_csv):
    """单个 counts csv -> 该种子的全切片口径指标（逐类）。"""
    import pandas as pd
    df = pd.read_csv(counts_csv)
    classes = [c[len("tp_"):] for c in df.columns if c.startswith("tp_")]
    df["case"] = [parse_case_id(s) for s in df["slice"]]
    out = {"tag": os.path.basename(counts_csv)[:-len("_counts.csv")],
           "n_slices": int(len(df)), "classes": {}}
    for cls in classes:
        tp, fp, fn = df[f"tp_{cls}"], df[f"fp_{cls}"], df[f"fn_{cls}"]
        # 病例级 pooled Dice
        grp = df.groupby("case")[[f"tp_{cls}", f"fp_{cls}", f"fn_{cls}"]].sum()
        denom = 2 * grp[f"tp_{cls}"] + grp[f"fp_{cls}"] + grp[f"fn_{cls}"]
        pooled = 2 * grp[f"tp_{cls}"] / denom.where(denom > 0)
        gt_pos_cases = grp[(grp[f"tp_{cls}"] + grp[f"fn_{cls}"]) > 0].index
        case_vals = pooled.loc[gt_pos_cases].dropna()
        # 对照口径（量化 FP 计入的掉点）：病例内只汇总 GT 含该类的切片，
        # 即同病例集上的 pooled-without-FP
        gt_rows = df[(df[f"tp_{cls}"] + df[f"fn_{cls}"]) > 0]
        grp_gt = gt_rows.groupby("case")[[f"tp_{cls}", f"fp_{cls}", f"fn_{cls}"]].sum()
        denom_gt = 2 * grp_gt[f"tp_{cls}"] + grp_gt[f"fp_{cls}"] + grp_gt[f"fn_{cls}"]
        pooled_gt = (2 * grp_gt[f"tp_{cls}"] / denom_gt.where(denom_gt > 0)) \
            .loc[gt_pos_cases].dropna()
        # micro Dice（全切片汇总）
        d_micro = 2 * tp.sum() + fp.sum() + fn.sum()
        micro = float(2 * tp.sum() / d_micro) if d_micro > 0 else float("nan")
        # 切片级 FP-inclusive 平均（交叉校验口径）
        d_slice = 2 * tp + fp + fn
        slice_mean = float((2 * tp[d_slice > 0] / d_slice[d_slice > 0]).mean())
        # FP 切片率（GT 空切片中预测非空的比例）
        empty = (tp + fn) == 0
        fp_rate = float((fp[empty] > 0).mean()) if empty.any() else float("nan")
        out["classes"][cls] = {
            "patient_pooled_dice": {
                "mean": float(case_vals.mean()),
                "n_cases": int(len(case_vals)),
                "n_cases_gt_positive": int(len(gt_pos_cases)),
                "n_excluded_true_negative": int((pooled.isna()).sum()),
                "per_case": {k: float(v) for k, v in case_vals.items()},
            },
            "patient_pooled_dice_gt_only": {
                "mean": float(pooled_gt.mean()),
                "n_cases": int(len(pooled_gt)),
                "note": "对照：病例内仅汇总 GT 含该类切片（不计入空切片 FP）",
            },
            "micro_dice": micro,
            "slice_mean_dice_fp_inclusive": slice_mean,
            "fp_slice_rate": fp_rate,
            "n_empty_gt_slices": int(empty.sum()),
            "n_fp_slices": int((fp[empty] > 0).sum()),
        }
    return out


def prefer_fixed_mt_counts(per_seed):
    """完整的 mtfixed 三种子组替换同名 mt 组；未齐时保留旧组并明确告警。"""
    groups = {}
    for tag in list(per_seed):
        if not tag.startswith("mtfixed_"):
            continue
        canonical = "mt_" + tag[len("mtfixed_"):]
        base = canonical.rsplit("_s", 1)[0]
        groups.setdefault(base, []).append((canonical, tag))
    sources = {}
    for base, pairs in groups.items():
        expected = {f"{base}_s{sd}" for sd in SEEDS}
        if {canonical for canonical, _ in pairs} != expected:
            print(f"[warn] {base} 的 mtfixed counts 未齐，不替换旧 mt 组")
            continue
        for canonical, fixed in pairs:
            per_seed[canonical] = per_seed.pop(fixed)
            per_seed[canonical]["source_tag"] = fixed
            sources[canonical] = fixed
    return sources


def aggregate(counts_dir, out_json, out_tex, sr_copy=True):
    counts = sorted(f for f in os.listdir(counts_dir) if f.endswith("_counts.csv"))
    assert counts, f"{counts_dir} 下无 *_counts.csv"
    per_seed = {}
    for f in counts:
        tag = f[:-len("_counts.csv")]
        per_seed[tag] = per_seed_metrics(os.path.join(counts_dir, f))
    mt_sources = prefer_fixed_mt_counts(per_seed)

    # GT 一致性校验：所有 counts 的 (tp+fn) 逐切片模式必须一致（同一份 GT）
    import pandas as pd
    ref = None
    for f in counts:
        df = pd.read_csv(os.path.join(counts_dir, f))
        sig = {c: (df[f"tp_{c}"] + df[f"fn_{c}"]).values
               for c in ("rectum", "tumor") if f"tp_{c}" in df}
        if ref is None:
            ref, ref_name, ref_slices = sig, f, list(df["slice"])
        else:
            assert list(df["slice"]) == ref_slices, f"{f} 切片顺序与 {ref_name} 不同"
            for c in ref:
                assert np.array_equal(sig[c], ref[c]), \
                    f"{f} 与 {ref_name} 的 GT 模式不一致（{c}）"
    print(f"GT 一致性校验通过：{len(counts)} 个 counts 文件")

    # 按实验（去 _s{seed} 后缀）分组聚合 mean±std
    experiments = {}
    for tag, m in sorted(per_seed.items()):
        exp = re.sub(r"_s\d+$", "", tag)
        experiments.setdefault(exp, {})[tag] = m
    result = {"protocol": "allslice", "counts_dir": os.path.abspath(counts_dir),
              "mt_source": mt_sources or "legacy mt_* counts (no complete mtfixed set)",
              "experiments": {}}
    for exp, seed_map in experiments.items():
        classes = next(iter(seed_map.values()))["classes"]
        e = {"n_seeds": len(seed_map), "single_seed": len(seed_map) == 1,
             "classes": {}}
        for cls in classes:
            c = {}
            for key in ("patient_pooled_dice", "patient_pooled_dice_gt_only",
                        "micro_dice", "slice_mean_dice_fp_inclusive",
                        "fp_slice_rate"):
                vals = {t: (s["classes"][cls][key]["mean"] if key.startswith(
                            "patient_pooled_dice") else s["classes"][cls][key])
                        for t, s in seed_map.items()}
                arr = np.array(list(vals.values()), dtype=float)
                c[key] = {"seed_means": vals, "mean": float(arr.mean()),
                          "std": float(arr.std(ddof=1)) if len(arr) > 1 else None}
            c["n_cases_gt_positive"] = next(iter(
                seed_map.values()))["classes"][cls]["patient_pooled_dice"][
                    "n_cases_gt_positive"]
            c["n_fp_slices"] = next(iter(
                seed_map.values()))["classes"][cls]["n_fp_slices"]
            c["n_empty_gt_slices"] = next(iter(
                seed_map.values()))["classes"][cls]["n_empty_gt_slices"]
            e["classes"][cls] = c
        result["experiments"][exp] = e

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"已保存 {out_json}（{len(result['experiments'])} 个实验）")

    if out_tex:
        tex = make_table(result)
        os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)
        with open(out_tex, "w") as f:
            f.write(tex)
        print(f"已保存 {out_tex}")
        if sr_copy:
            os.makedirs(os.path.dirname(SR_TEX), exist_ok=True)
            shutil.copyfile(out_tex, SR_TEX)
            print(f"已复制 {SR_TEX}")
    return result


# ---------------------------------------------------------------- 表格

def _fmt(mean, std):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "--"
    return f"{mean * 100:.2f}$\\pm${std * 100:.2f}" if std is not None \
        else f"{mean * 100:.2f}$^{{\\dagger}}$"


def make_table(result):
    exps = result["experiments"]
    lines = [
        "% 全切片口径（审稿意见 M4）：病例级 pooled Dice 与 micro Dice，FP 负担计入分数。"
        "由 src/eval_allslice.py 生成",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{All-slice Dice on the CARE test set (81 cases, 6{,}461 slices), "
        "counting false-positive burden into the score. Patient-level pooled Dice "
        "(primary): per-case $2\\sum\\mathrm{tp}/(2\\sum\\mathrm{tp}+\\sum\\mathrm{fp}"
        "+\\sum\\mathrm{fn})$ with sums over all slices of the case including "
        "class-negative slices; averaged over cases whose ground truth contains the "
        "class (cases with both empty ground truth and empty prediction contribute "
        "neither reward nor penalty and are excluded). Micro Dice (secondary): "
        "tp/fp/fn summed over the whole test set before computing Dice. "
        "Mean$\\pm$std over three seeds; $^{\\dagger}$single-seed reference.}",
        "\\label{tab:allslice}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * 4 + "}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Rectum} & \\multicolumn{2}{c}{Tumor} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "Method & Labeled & Pooled & Micro & Pooled & Micro \\\\",
        "\\midrule",
    ]

    def row(exp, label, ratio):
        e = exps[exp]
        cells = []
        for cls in ("rectum", "tumor"):
            c = e["classes"][cls]
            cells.append(_fmt(c["patient_pooled_dice"]["mean"],
                              c["patient_pooled_dice"]["std"]))
            cells.append(_fmt(c["micro_dice"]["mean"], c["micro_dice"]["std"]))
        return f"{label} & {ratio}\\% & " + " & ".join(cells) + " \\\\"

    n_body_rows = 0
    for key, label in METHOD_ORDER:
        first = True
        for r in RATIOS:
            exp = f"{key}_{r}"
            if exp not in exps:
                continue
            lines.append(row(exp, label if first else "", r))
            first = False
            n_body_rows += 1
        if not first:
            lines.append("\\midrule")
    if "sup_100" in exps:
        lines.append(row("sup_100", "Supervised", 100))
        n_body_rows += 1
    # 单种子参照（nnunet / sup180 / mtcons），仅数值呈现
    refs = [e for e in ("sup180_50", "sup180_100", "mtcons_50",
                        "nnunet_sup_50", "nnunet_sup_100") if e in exps]
    if refs:
        if n_body_rows:
            lines.append("\\midrule")
        pretty = {"sup180_50": "Supervised-180", "sup180_100": "Supervised-180",
                  "mtcons_50": "MT-cons", "nnunet_sup_50": "nnU-Net",
                  "nnunet_sup_100": "nnU-Net"}
        for exp in refs:
            ratio = exp.rsplit("_", 1)[-1]
            lines.append(row(exp, pretty[exp], ratio))
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infer", action="store_true",
                    help="生产推理：headline 集（--all 扩展全档），需 GPU")
    ap.add_argument("--all", action="store_true", help="--infer 覆盖所有档 + 单种子参照")
    ap.add_argument("--tags", nargs="*", default=None, help="--infer 只跑指定 tag")
    ap.add_argument("--infer_one", default=None, metavar="CKPT[:STAGE1_CKPT]",
                    help="显式单个推理（冒烟/调试）")
    ap.add_argument("--infer_nnunet", action="store_true",
                    help="nnU-Net counts（CPU，读 data/nnunet/predictions/Dataset50*；"
                         "给 --pred_dir + --out_csv 则只跑单个目录，冒烟用）")
    ap.add_argument("--pred_dir", default=None)
    ap.add_argument("--aggregate", action="store_true", help="聚合 counts 出 json + tex")
    ap.add_argument("--crop_size", type=int, default=None)
    ap.add_argument("--data_root", default="data/CARE")
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--counts_dir", default=ALLSLICE_DIR)
    ap.add_argument("--out_csv", default=None,
                    help="--infer_one / --infer_nnunet --pred_dir 的输出 csv")
    ap.add_argument("--out_json", default=OUT_JSON)
    ap.add_argument("--out_tex", default=OUT_TEX)
    ap.add_argument("--no_sr_copy", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not (args.infer or args.infer_one or args.infer_nnunet or args.aggregate):
        ap.error("需要 --infer / --infer_one / --infer_nnunet / --aggregate 之一")

    if args.infer_one:
        from common import resolve_device
        parts = args.infer_one.split(":")
        assert args.out_csv, "--infer_one 需要 --out_csv"
        infer_counts(parts[0], args.data_root, args.out_csv,
                     resolve_device(args.device), args.batch_size,
                     cache_dir=args.cache_dir or None,
                     stage1_ckpt=parts[1] if len(parts) > 1 else None,
                     crop_size=args.crop_size)

    if args.infer:
        from common import resolve_device
        device = resolve_device(args.device)
        tags = args.tags or (all_tags() if args.all else headline_tags())
        for i, tag in enumerate(tags, 1):
            print(f"=== [{i}/{len(tags)}] {tag} ===")
            infer_counts_from_summary(tag, device, args.batch_size,
                                      out_dir=args.counts_dir)

    if args.infer_nnunet:
        os.makedirs(args.counts_dir, exist_ok=True)
        if args.pred_dir:
            assert args.out_csv, "--infer_nnunet --pred_dir 需要 --out_csv"
            infer_counts_nnunet(args.pred_dir, args.data_root, args.out_csv)
        else:
            for tag, ds in NNUNET_TAGS.items():
                infer_counts_nnunet(os.path.join("data", "nnunet", "predictions", ds),
                                    args.data_root,
                                    os.path.join(args.counts_dir, f"{tag}_counts.csv"))

    if args.aggregate:
        aggregate(args.counts_dir, args.out_json, args.out_tex,
                  sr_copy=not args.no_sr_copy)


if __name__ == "__main__":
    main()
