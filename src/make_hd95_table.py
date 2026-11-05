"""M8 rebuttal 汇总：HD95 单位修正（px/mm）+ NSD 容差敏感性。

读 results/v2/hd95/{tag}_summary.json（src/eval_hd95.py 产物），输出：
- results/v2/hd95_units.json       结构化汇总（含 spacing 证据、逐种子值、MSD 3D 结果）
- paper/manuscript/tab_hd95.tex    三个 booktabs 表（CARE HD95 px / NSD τ 敏感性 / MSD 3D mm），
                                   自动复制到 paper/sr/
- paper/snippet_nsd_justification.txt  补充材料英文段落（数字由本脚本填入）

CARE 的 NSD 不再使用 summary 中的逐切片均值，而是从 per_slice CSV 统一按
GT 阳性切片→病例内均值→跨病例均值聚合，并以百分数报告，与主统计表一致；
Mean Teacher 使用 mtfixed_50 的坐标对齐修复版评估结果。

用法：
    python src/make_hd95_table.py --hd95_dir results/v2/hd95 \
        --out_json results/v2/hd95_units.json --out_tex paper/manuscript/tab_hd95.tex
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import parse_case_id
from reanalyze_v2 import care_gt_presence

SEEDS = (42, 43, 44)
# (方法标签, 表内行名)；sup_100 用自身 s1_100，其余共用同比例 s1_50（见 queue_hd95_care.sh）
CARE_CONFIGS = [
    ("sup_100", "Supervised 100\\%"),
    ("sup_50", "Supervised 50\\%"),
    ("semi_50", "Pseudo-label ST 50\\%"),
    ("mtfixed_50", "Mean Teacher 50\\%"),
    ("cps_50", "CPS 50\\%"),
    ("bcp_50", "BCP 50\\%"),
    ("ours_50", "Ours 50\\%"),
]
MSD_CONFIGS = [
    ("msd_sup_50", "Supervised 50\\%"),
    ("msd_ours_50", "Ours 50\\%"),
    ("msd_sup_25", "Supervised 25\\%"),
]
TAUS = ("1", "2", "3")


def load_json(p):
    with open(p) as f:
        return json.load(f)


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def fmt_ms(m, s, nd=2):
    if m is None:
        return "--"
    return f"{m:.{nd}f}$\\pm${s:.{nd}f}"


def fmt_ci(d, nd=2):
    if not d or d.get("mean") is None or np.isnan(d["mean"]):
        return "--"
    return f"{d['mean']:.{nd}f} [{d['ci95'][0]:.{nd}f}, {d['ci95'][1]:.{nd}f}]"


def tumor_size_stats(cache_dir):
    """CARE test 肿瘤/直肠等效直径（px），供 snippet 论证 τ=2px 的相对严格性。"""
    lab = np.load(os.path.join(cache_dir, "test_labels.npy"), mmap_mode="r")
    stats = {}
    for c, name in ((1, "rectum"), (2, "tumor")):
        diams = []
        areas = []
        for i in range(lab.shape[0]):
            a = int((lab[i] == c).sum())
            if a > 0:
                areas.append(a)
                diams.append(2.0 * np.sqrt(a / np.pi))
        stats[name] = {
            "n_slices": len(diams),
            "diam_px_median": float(np.median(diams)),
            "diam_px_iqr": [float(np.percentile(diams, 25)),
                            float(np.percentile(diams, 75))],
            "area_px_median": float(np.median(areas)),
        }
    return stats


def patient_level_nsd(per_seed, gt):
    """从逐切片 NSD 重算与主表一致的 GT 阳性切片→病例均值→跨病例均值。"""
    out = {}
    for seed, summary in per_seed.items():
        df = pd.read_csv(summary["per_slice_csv"]).set_index("slice")
        assert set(df.index) == set(gt), f"seed {seed}: NSD CSV 与 CARE GT 切片集不一致"
        out[seed] = {}
        for ci, cn in enumerate(("rectum", "tumor")):
            scored = df[[bool(gt[name][ci]) for name in df.index]]
            out[seed][cn] = {}
            for tau in TAUS:
                col = f"nsd_tau{tau}_{cn}"
                values = scored[col]
                assert not values.isna().any(), f"seed {seed}/{cn}/tau{tau}: GT 阳性切片 NSD 为 nan"
                per_case = values.groupby([parse_case_id(name) for name in scored.index]).mean()
                out[seed][cn][f"nsd_tau{tau}"] = float(per_case.mean())
    return out


def aggregate_care(hd95_dir, gt=None):
    """HD95 取病例级 p95；NSD 全部 τ 统一为主表的患者级聚合协议。"""
    if gt is None:
        gt = care_gt_presence()
    out = {}
    missing = []
    for stem, _row in CARE_CONFIGS:
        per_seed = {}
        for s in SEEDS:
            p = os.path.join(hd95_dir, f"{stem}_s{s}_summary.json")
            if os.path.exists(p):
                per_seed[s] = load_json(p)
            else:
                missing.append(p)
        if not per_seed:
            continue
        nsd_patient = patient_level_nsd(per_seed, gt)
        agg = {}
        for cn in ("rectum", "tumor"):
            hd_seeds = [v["metrics"][cn]["hd95_px_case_p95"]["mean"]
                        for v in per_seed.values()]
            rec = {"hd95_px_case_p95": mean_std(hd_seeds),
                   "hd95_px_case_p95_seeds": hd_seeds}
            for t in TAUS:
                key = f"nsd_tau{t}"
                ns = [nsd_patient[s][cn][key] for s in per_seed]
                rec[key] = mean_std(ns)
                rec[key + "_seeds"] = ns
            rec["n_missed_slices_seeds"] = [
                v["metrics"][cn]["n_missed_slices"] for v in per_seed.values()]
            agg[cn] = rec
        out[stem] = {"n_seeds": len(per_seed), "agg": agg,
                     "nsd_protocol": "GT-positive slices -> within-case mean -> across-case mean",
                     "summaries": {str(s): os.path.abspath(
                         os.path.join(hd95_dir, f"{stem}_s{s}_summary.json"))
                         for s in per_seed}}
    return out, missing


def aggregate_msd(hd95_dir):
    out = {}
    missing = []
    spacings = {}
    for stem, _row in MSD_CONFIGS:
        p = os.path.join(hd95_dir, f"{stem}_summary.json")
        if not os.path.exists(p):
            missing.append(p)
            continue
        s = load_json(p)
        case_csv = s["per_case_csv"]
        import csv as _csv
        with open(case_csv) as f:
            for r in _csv.DictReader(f):
                spacings[r["case"]] = [float(r["spacing_x_mm"]),
                                       float(r["spacing_y_mm"]),
                                       float(r["spacing_z_mm"])]
        out[stem] = {"metrics": s["metrics"]["tumor"],
                     "n_cases": s["n_cases"], "summary": os.path.abspath(p)}
    return out, missing, spacings


def make_tex(care, msd, size_stats):
    L = []
    L.append("% M8 rebuttal：HD95 单位修正（CARE 2D px / MSD 3D mm）+ NSD 容差敏感性。"
             "由 src/make_hd95_table.py 生成")
    # --- 表 A：CARE 2D HD95 (px) ---
    L.append("\\begin{table}[!htbp]\n\\centering")
    L.append("\\caption{Two-dimensional 95th-percentile Hausdorff distance (HD95) "
             "on the CARE test set, in pixels (the CARE release provides no voxel "
             "spacing; see text). Values are case-level HD95 (95th percentile of "
             "per-slice HD95 within each case), reported as mean$\\pm$std across "
             "three training seeds (42/43/44). Slices whose ground truth lacks the "
             "class are excluded; slices with an empty prediction despite "
             "present ground truth contribute the image diagonal (724 px) as the "
             "worst case.}")
    L.append("\\label{tab:hd95_care}\n\\small")
    L.append("\\begin{tabular}{lcc}\n\\toprule")
    L.append("Method & Rectum HD95 (px) & Tumor HD95 (px) \\\\")
    L.append("\\midrule")
    for stem, row in CARE_CONFIGS:
        if stem not in care:
            continue
        a = care[stem]["agg"]
        hr, ht = a["rectum"]["hd95_px_case_p95"], a["tumor"]["hd95_px_case_p95"]
        L.append(f"{row} & {fmt_ms(*hr)} & {fmt_ms(*ht)} \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # --- 表 B：NSD τ 敏感性（CARE，px） ---
    L.append("\\begin{table}[!htbp]\n\\centering")
    L.append("\\caption{Sensitivity of the 2D normalized surface Dice (NSD, \\%) to the "
             "tolerance $\\tau$ on the CARE test set. The aggregation matches the main "
             "statistical tables: within each seed, GT-positive slices are averaged "
             "within each case and then across cases; values are mean$\\pm$std across "
             "three seeds. Values increase monotonically with tolerance; the broad "
             "pattern is similar, although individual method orderings can move.}")
    L.append("\\label{tab:nsd_tau}\n\\small")
    L.append("\\begin{tabular}{lccc}\n\\toprule")
    L.append("Method & Tumor NSD $\\tau$=1 px & $\\tau$=2 px & $\\tau$=3 px \\\\")
    L.append("\\midrule")
    for stem, row in CARE_CONFIGS:
        if stem not in care:
            continue
        t = care[stem]["agg"]["tumor"]
        vals = []
        for key in ("nsd_tau1", "nsd_tau2", "nsd_tau3"):
            mean, std = t[key]
            vals.append(fmt_ms(mean * 100, std * 100))
        L.append(f"{row} & " + " & ".join(vals) + " \\\\")
    L.append("\\bottomrule\n\\end{tabular}")
    rectum_ref = care.get("sup_100", {}).get("agg", {}).get("rectum", {}) \
        .get("nsd_tau2", (None, None))[0]
    ref_txt = f"(Supervised 100\\%: {rectum_ref * 100:.2f}\\% at $\\tau$=2 px); " \
        if rectum_ref is not None else ""
    L.append("\\begin{minipage}{\\columnwidth}\\vspace{2pt}\\footnotesize\n"
             "Rectum NSD shows the same pattern "
             f"{ref_txt}"
             "full per-class values are in \\texttt{results/v2/hd95\\_units.json}.\n"
             "\\end{minipage}\n\\end{table}\n")
    # --- 表 C：MSD 3D（mm） ---
    L.append("\\begin{table}[!htbp]\n\\centering")
    L.append("\\caption{Three-dimensional boundary metrics on the MSD Task10 "
             "Colon held-out test subset (19 cases), computed with the true voxel "
             "spacing from the NIfTI headers (in-plane 0.54--0.98 mm, "
             "through-plane 3--5 mm). HD95 and 3D NSD ($\\tau$=2 mm) are "
             "case-level (mean [bootstrap 95\\% CI]; median also shown for "
             "HD95); cases with empty prediction but non-empty ground truth "
             "contribute the physical volume diagonal as the worst case "
             "(counted in the last column).}")
    L.append("\\label{tab:hd95_msd}\n\\small\n\\resizebox{\\textwidth}{!}{%")
    L.append("\\begin{tabular}{lccccc}\n\\toprule")
    L.append("Method & 3D Dice & HD95 mean (mm) & HD95 median (mm) "
             "& NSD$_{3D}$ $\\tau$=2 mm & $n$ worst \\\\\n\\midrule")
    for stem, row in MSD_CONFIGS:
        if stem not in msd:
            continue
        m = msd[stem]["metrics"]
        med = m['hd95_mm'].get('median')
        med_s = f"{med:.1f}" if med is not None else "--"
        L.append(f"{row} & {fmt_ci(m['dice3d'], 3)} & {fmt_ci(m['hd95_mm'], 1)} "
                 f"& {med_s} & {fmt_ci(m['nsd3d_tau2mm'], 3)} "
                 f"& {m['n_worst_case']} \\\\")
    L.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}")
    return "\n".join(L) + "\n"


def make_snippet(care, msd, spacings, size_stats):
    t = size_stats["tumor"]
    r = size_stats["rectum"]
    sp = np.array(list(spacings.values())) if spacings else np.zeros((1, 3))
    lines = []
    A = lines.append
    A("Supplementary note: units of the boundary-distance metrics (HD95) and the "
      "choice of the NSD tolerance")
    A("=" * 78)
    A("")
    A("Why CARE boundary metrics are reported in pixels.")
    A("The CARE dataset is distributed as anonymized 2D slices in NumPy .npz "
      "files that contain only the `image` and `label` arrays; the accompanying "
      "train/test bounding-box CSV files store only the slice id and the tumor "
      "bounding box. No DICOM headers, NIfTI affine matrices, or voxel-spacing "
      "fields are included in the release, so neither the in-plane pixel size "
      "nor the slice thickness is recoverable. Consequently, physical "
      "(millimetre) distances cannot be computed on CARE, and we report the 2D "
      "HD95 in pixels and the 2D NSD with a tolerance of tau = 2 px as "
      "boundary-agreement proxy metrics, with the unit explicitly stated. This "
      "replaces the voxel/slice mixed-unit 3D HD95 reported in the original "
      "submission, which stacked slices without spacing information and was "
      "therefore not physically interpretable.")
    A("")
    A("Why tau = 2 px is a reasonable tolerance on CARE.")
    A(f"On the CARE test set, the median tumor cross-section has an equivalent "
      f"diameter of {t['diam_px_median']:.1f} px (IQR "
      f"{t['diam_px_iqr'][0]:.1f}-{t['diam_px_iqr'][1]:.1f}) and the median "
      f"rectum cross-section {r['diam_px_median']:.1f} px. A tolerance of "
      f"2 px therefore corresponds to roughly "
      f"{2.0 / t['diam_px_median'] * 100:.0f}% of a typical tumor diameter, "
      "i.e. the metric rewards only close boundary agreement and penalizes "
      "coarse outlines; under the isotropic ~1 mm/px assumption commonly "
      "observed for pelvic CT (and used by the U-SAM reference protocol on the "
      "same data), 2 px further corresponds to approximately 2 mm, the standard "
      "NSD tolerance in the literature. Sensitivity analysis over "
      "tau in {1, 2, 3} px (Table \\ref{tab:nsd_tau}) shows that absolute "
      "NSD values increase monotonically and the broad pattern is similar; "
      "individual method orderings can move and are not over-interpreted.")
    A("")
    if msd:
        A("Millimetre-scale verification on the external MSD Colon set.")
        A(f"To verify that the conclusions carry over to physically calibrated "
          f"metrics, we computed true 3D HD95 and 3D NSD on the MSD Task10 Colon "
          f"held-out test subset (19 cases), for which voxel spacing is available "
          f"in the NIfTI headers (median in-plane {np.median(sp[:, 0]):.3f} mm, "
          f"range {sp[:, 0].min():.3f}-{sp[:, 0].max():.3f} mm; through-plane "
          f"{np.median(sp[:, 2]):.1f} mm, range {sp[:, 2].min():.1f}-"
          f"{sp[:, 2].max():.1f} mm). Two-dimensional per-slice predictions "
          f"were stacked per case and distances were computed with the "
          f"anisotropic spacing (Table \\ref{{tab:hd95_msd}}). Because "
          f"through-plane spacing (3-5 mm) is much coarser than in-plane "
          f"resolution, 3D HD95 is dominated by the slice direction; this "
          f"anisotropy is exactly why slice-wise 2D boundary metrics are a "
          f"reasonable proxy for such data. On this 19-case replication set, "
          f"3D Dice and 3D NSD (tau = 2 mm) provide physical-unit secondary "
          f"measurements rather than validation of a method ordering, whereas 3D HD95 "
          f"is dominated by rare complete misses (cases with an empty "
          f"prediction contribute the volume diagonal; see the miss counts "
          f"in Table \\ref{{tab:hd95_msd}}) and its bootstrap confidence "
          f"intervals are wide and overlapping. We therefore report both "
          f"the mean and the median HD95 and caution against "
          f"over-interpreting millimetre HD95 differences at this sample "
          f"size; these secondary metrics are reported jointly without "
          f"using them to claim a preserved method ranking.")
    A("")
    A("Key numbers (auto-generated from results/v2/hd95_units.json):")
    for stem, row in CARE_CONFIGS:
        if stem not in care:
            continue
        a = care[stem]["agg"]
        hr, hs = a["tumor"]["hd95_px_case_p95"]
        n1, _ = a["tumor"]["nsd_tau1"]
        n2, _ = a["tumor"]["nsd_tau2"]
        n3, _ = a["tumor"]["nsd_tau3"]
        A(f"  CARE {row.replace(chr(92), '')}: tumor HD95 {hr:.2f}+/-{hs:.2f} px "
          f"(case-level); tumor NSD {n1:.3f}/{n2:.3f}/{n3:.3f} at tau=1/2/3 px")
    for stem, row in MSD_CONFIGS:
        if stem not in msd:
            continue
        m = msd[stem]["metrics"]
        med = m['hd95_mm'].get('median')
        med_s = f", median {med:.1f} mm" if med is not None else ""
        A(f"  MSD  {row.replace(chr(92), '')}: 3D HD95 {m['hd95_mm']['mean']:.1f} mm "
          f"[{m['hd95_mm']['ci95'][0]:.1f}, {m['hd95_mm']['ci95'][1]:.1f}]"
          f"{med_s}, "
          f"NSD3D(tau=2mm) {m['nsd3d_tau2mm']['mean']:.3f}, "
          f"Dice3D {m['dice3d']['mean']:.3f}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hd95_dir", default="results/v2/hd95")
    ap.add_argument("--out_json", default="results/v2/hd95_units.json")
    ap.add_argument("--out_tex", default="paper/manuscript/tab_hd95.tex")
    ap.add_argument("--snippet", default="paper/snippet_nsd_justification.txt")
    ap.add_argument("--cache_dir", default="data/cache",
                    help="CARE test 标签缓存（肿瘤直径统计，供 snippet 论证）")
    ap.add_argument("--sr_dir", default="paper/sr")
    args = ap.parse_args()

    care, miss_care = aggregate_care(args.hd95_dir)
    msd, miss_msd, spacings = aggregate_msd(args.hd95_dir)
    size_stats = tumor_size_stats(args.cache_dir) \
        if os.path.exists(os.path.join(args.cache_dir, "test_labels.npy")) else None

    doc = {
        "note": "M8 rebuttal：CARE 无 spacing（npz 仅 image/label 两键，bbox csv "
                "仅 pid,pos）-> CARE 报 2D HD95(px) + NSD tau(px) 敏感性；MSD 有 "
                "nii spacing -> 3D HD95(mm) + 3D NSD(tau=2mm)。"
                "单侧空预测取体/图对角线最坏值；双侧空剔除。",
        "care_spacing_available": False,
        "care_spacing_evidence": "npz keys=['image','label']; train_bbox.csv "
                                 "columns=['pid','pos']; 发布版无 DICOM/NIfTI 头",
        "care_configs": care,
        "msd_configs": msd,
        "msd_test_case_spacings_mm": spacings,
        "care_test_size_stats_px": size_stats,
        "missing_inputs": miss_care + miss_msd,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"已保存: {args.out_json}")

    if size_stats:
        tex = make_tex(care, msd, size_stats)
        os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
        with open(args.out_tex, "w") as f:
            f.write(tex)
        dst = os.path.join(args.sr_dir, os.path.basename(args.out_tex))
        os.makedirs(args.sr_dir, exist_ok=True)
        shutil.copyfile(args.out_tex, dst)
        print(f"已保存: {args.out_tex} (+{dst})")

        snippet = make_snippet(care, msd, spacings, size_stats)
        with open(args.snippet, "w") as f:
            f.write(snippet)
        print(f"已保存: {args.snippet}")
    else:
        print("警告：无 cache 标签，跳过 tex/snippet（仅出 json）")
    if miss_care or miss_msd:
        print(f"缺失 {len(miss_care) + len(miss_msd)} 个输入 summary"
              "（队列未跑完时会部分缺席）")


if __name__ == "__main__":
    main()
