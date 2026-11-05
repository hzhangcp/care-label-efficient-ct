"""v2 评审重分析（Major Revision）：固定切片集协议 + 病例级统计。

不训练、不推理；仅从 results/*_per_slice.csv + GT 标签重算全部数字：
1. 前景指标：Dice/IoU/NSD 仅在 GT 含该类的切片上计算（所有方法共享同一固定切片集，
   解决 v1 协议"不同方法在不同切片集上平均"的审稿意见）。
2. 假阳性率（FP rate）：GT 不含该类的切片中预测非空的比例。
   v1 CSV 编码（见 evaluate.slice_metrics）：GT 空 + 预测非空 -> 0.0；GT 空 + 预测空 -> nan。
3. 病例级统计：病例内取该类已评分切片的均值，然后
   - cluster bootstrap 95%CI（按病例有放回重抽样，10000 次，seed 42）；
   - 配对双侧 Wilcoxon 符号秩（病例级数值）；
   - Holm 校正：family = 每个 (dataset, ratio, class, metric) 下 5 组比较
     {Sup vs ST, Sup vs MT, Sup vs CPS, Sup vs Ours, ST vs Ours}。

产出：
  results/v2/{tag}_summary_v2.json     每个实验的前景指标 + FP 率 + 切片/病例数
  paper/manuscript/table1_care_v2.tex  CARE 主结果表（v2 数字 + 每类 FP% 列）
  paper/manuscript/table2_msd_v2.tex   MSD Colon 跨数据集复现表（同上）
  paper/manuscript/tabS6_wilcoxon_msd_v2.tex  MSD 患者级 Wilcoxon 表
  paper/sr/table2_msd_v2_zh.tex        中文 MSD 表（同步生成）
  paper/v2_stats.md                    全部 Wilcoxon 原始 p 与 Holm 校正 p
  终端打印校验报告

用法：python src/reanalyze_v2.py
只更新 MSD（如标注内验证重跑）：
  python src/reanalyze_v2.py --msd_only --msd_results_dir results_msd_lb \
      --stats_md paper/v2_stats_msd_labeled.md
"""

import argparse
import json
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "results"
OUT_DIR = os.path.join(RESULTS, "v2")
MAN = os.path.join("paper", "manuscript")
STATS_MD = os.path.join("paper", "v2_stats.md")

METHODS = [("sup", "Supervised"), ("semi", "Pseudo-label ST"),
           ("mt", "Mean Teacher"), ("cps", "CPS"), ("ours", "Ours")]
RATIOS = [1, 5, 10, 25, 50]
METRICS = ["dice", "iou", "nsd"]
COMPARISONS = [("sup", "semi", "Supervised vs ST"),
               ("sup", "mt", "Supervised vs MT"),
               ("sup", "cps", "Supervised vs CPS"),
               ("sup", "ours", "Supervised vs Ours"),
               ("semi", "ours", "ST vs Ours")]
N_BOOT = 10000
SEED = 42

REPORT = []  # 校验报告行


def log(msg):
    print(msg)
    REPORT.append(msg)


def parse_case_id(fname):
    """与 dataset.parse_case_id 一致：`case{id}_slice{nnn}.npz` -> `case{id}`。"""
    base = os.path.basename(fname)
    if base.endswith(".npz"):
        base = base[:-4]
    if "_" in base:
        return base.rsplit("_", 1)[0]
    return base


# ---------------------------------------------------------------- GT 类存在性

def care_gt_presence():
    """CARE test：从 data/cache 读标签（与 make_figures.fig3 相同的加载方式）。

    返回 dict: slice_name -> (has_rectum, has_tumor)
    """
    labels = np.load("data/cache/test_labels.npy", mmap_mode="r")
    with open("data/cache/test_names.json") as f:
        names = json.load(f)
    pres = {}
    for i, n in enumerate(names):
        lab = np.asarray(labels[i])
        pres[n] = (bool((lab == 1).any()), bool((lab == 2).any()))
    return pres


def _msd_one(path):
    with np.load(path) as z:
        lab = z["label"]
        return os.path.basename(path), (bool((lab == 1).any()),)


def msd_gt_presence():
    """MSD Colon test：无现成缓存，直接从 npz 计算 GT 肿瘤存在性并缓存。

    返回 dict: slice_name -> (has_tumor,)，与 CARE 的元组结构对齐。
    """
    cache_path = os.path.join(OUT_DIR, "msd_gt_presence.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        return {k: (v if isinstance(v, (list, tuple)) else (v,)) for k, v in raw.items()}
    npz_dir = os.path.join("data", "MSD", "colon_npz", "test", "test_npz")
    paths = sorted(os.path.join(npz_dir, f) for f in os.listdir(npz_dir)
                   if f.endswith(".npz"))
    with ThreadPoolExecutor(max_workers=16) as ex:
        pres = dict(ex.map(_msd_one, paths))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(pres, f)
    return pres


# ---------------------------------------------------------------- 统计工具

def cluster_boot_ci(case_vals, n_boot=N_BOOT, seed=SEED):
    """病例级均值的 cluster bootstrap 95%CI（按病例有放回重抽样）。"""
    v = np.asarray(case_vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def wilcoxon_p(a, b):
    """配对双侧 Wilcoxon；全部差值为 0 时返回 1.0。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == b):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)


def holm_adjust(pvals):
    """Holm 校正（保持输入顺序返回校正后 p）。"""
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- 核心计算

def analyze_tag(tag, df, gt, classes):
    """单个实验的 v2 指标。df: index=slice；gt: name -> tuple[bool]。

    返回 (summary_dict, case_vals)：case_vals[cls][metric] = {case: 病例均值}
    """
    summary = {"tag": tag, "protocol": "v2", "n_slices": len(df), "classes": {}}
    case_vals = {}
    for ci, cls in enumerate(classes):
        has_cls = np.array([gt[s][ci] for s in df.index], dtype=bool)
        scored = df[has_cls]          # GT 含该类：固定切片集
        empty = df[~has_cls]          # GT 不含该类：FP 分析
        cinfo = {"n_gt_slices": int(has_cls.sum()),
                 "n_empty_gt_slices": int((~has_cls).sum()),
                 "metrics": {}}
        # FP 率：GT 空切片中 score==0（预测非空）的比例；以 dice 列解码
        fp_col = empty[f"dice_{cls}"]
        n_fp = int((fp_col == 0).sum())
        cinfo["fp"] = {"n_fp_slices": n_fp,
                       "fp_rate": n_fp / len(empty) if len(empty) else float("nan")}
        case_vals[cls] = {}
        for m in METRICS:
            col = scored[f"{m}_{cls}"]
            per_case = col.groupby([parse_case_id(s) for s in scored.index]).mean()
            lo, hi = cluster_boot_ci(per_case.values)
            cinfo["metrics"][m] = {
                "slice_mean": float(col.mean()),
                "patient_mean": float(per_case.mean()),
                "ci95": [lo, hi],
                "n_scored_slices": int(col.notna().sum()),
                "n_cases": int(len(per_case)),
            }
            case_vals[cls][m] = per_case.to_dict()
        cinfo["n_cases"] = cinfo["metrics"]["dice"]["n_cases"]
        summary["classes"][cls] = cinfo
    return summary, case_vals


def validate(tag, df, gt, classes):
    """CSV 解码校验：GT 含该类的切片不应有 nan；GT 空的切片 score 只能为 0 或 nan；
    且三个指标列的 0/nan 模式必须一致。"""
    msgs = []
    for ci, cls in enumerate(classes):
        has_cls = np.array([gt[s][ci] for s in df.index], dtype=bool)
        scored, empty = df[has_cls], df[~has_cls]
        nan_in_gt = {m: int(scored[f"{m}_{cls}"].isna().sum()) for m in METRICS}
        if any(nan_in_gt.values()):
            msgs.append(f"  [VIOLATION] {tag}/{cls}: GT 含类切片存在 nan {nan_in_gt}")
        bad = {m: int(((empty[f"{m}_{cls}"].notna())
                       & (empty[f"{m}_{cls}"] != 0)).sum()) for m in METRICS}
        if any(bad.values()):
            msgs.append(f"  [VIOLATION] {tag}/{cls}: GT 空切片出现非 0 非 nan {bad}")
        pat = [ (empty[f"{m}_{cls}"] == 0) | empty[f"{m}_{cls}"].isna() for m in METRICS ]
        if not (pat[0].equals(pat[1]) and pat[1].equals(pat[2])):
            msgs.append(f"  [VIOLATION] {tag}/{cls}: 三指标 0/nan 模式不一致")
    return msgs


# ---------------------------------------------------------------- 表格生成

def fmt_cell(m):
    return (f"{m['patient_mean'] * 100:.2f} "
            f"[{m['ci95'][0] * 100:.2f}, {m['ci95'][1] * 100:.2f}]")


def table_care(summaries):
    lines = [
        "% CARE 主结果 v2：病例级 mean% [cluster bootstrap 95%CI]，前景指标仅在 GT 含该类的切片上计算；"
        "FP%% = GT 阴性切片中预测非空的比例。由 src/reanalyze_v2.py 生成",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{CARE test set results (81 cases, 6{,}461 slices), revised protocol. "
        "Foreground metrics are computed only on slices whose ground truth contains the "
        "class (an identical slice set for all methods); values are patient-level "
        "mean\\% {[}cluster bootstrap 95\\% CI, 10{,}000 replicates over cases{]}. "
        "FP\\% = fraction of class-negative slices with a non-empty prediction for that "
        "class. R = normal rectum, T = tumor.}",
        "\\label{tab:care_v2}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * 8 + "}",
        "\\toprule",
        " & & \\multicolumn{4}{c}{Rectum} & \\multicolumn{4}{c}{Tumor} \\\\",
        "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
        "Method & Labeled & Dice & IoU & NSD & FP\\% & Dice & IoU & NSD & FP\\% \\\\",
        "\\midrule",
    ]

    def row(name, label, ratio):
        s = summaries[name]
        cells = []
        for cls in ("rectum", "tumor"):
            for m in METRICS:
                cells.append(fmt_cell(s["classes"][cls]["metrics"][m]))
            cells.append(f"{s['classes'][cls]['fp']['fp_rate'] * 100:.2f}")
        return f"{label} & {ratio}\\% & " + " & ".join(cells) + " \\\\"

    for key, label in METHODS:
        for i, r in enumerate(RATIOS):
            lines.append(row(f"{key}_{r}", label if i == 0 else "", r))
        lines.append("\\midrule")
    lines.append(row("sup_100", "Supervised", 100))
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)


def table_msd(summaries, chinese=False):
    if chinese:
        caption = (
            "\\caption{MSD Task10 Colon 上的跨数据集复现（19 例留出病例，1{,}680 "
            "切片），单一训练种子（42）。所有模型从头训练；模型选择仅使用标注子集"
            "内部验证，1\\% 档采用固定日程末轮。肿瘤 Dice/IoU/NSD 仅在含肿瘤切片上"
            "计算（所有方法使用完全相同的切片集合）；数值为患者级 mean\\% [聚类自助法 "
            "95\\% CI]。FP\\% = 无瘤切片中肿瘤预测非空的比例。}")
    else:
        caption = (
            "\\caption{Cross-dataset replication on MSD Task10 Colon (19 held-out "
            "cases, 1{,}680 slices), single training seed 42. All models are retrained "
            "from scratch; model selection uses labeled-subset validation only, with "
            "fixed-schedule final checkpoints at 1\\%. Tumor Dice/IoU/NSD are computed "
            "only on tumor-containing slices (identical slice set for all methods); "
            "values are patient-level mean\\% {[}cluster bootstrap 95\\% CI{]}. FP\\% = "
            "fraction of tumor-free slices with a non-empty tumor prediction.}")
    lines = [
        "% MSD Colon 跨数据集复现 v2：病例级 mean% [cluster bootstrap 95%CI]，仅肿瘤类；"
        "FP%% = 无瘤切片中预测非空的比例。由 src/reanalyze_v2.py 生成",
        "\\begin{table}[!htbp]", "\\centering",
        "\\small", "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * 4 + "}", "\\toprule",
        "Method & Labeled & Dice & IoU & NSD & FP\\% \\\\", "\\midrule",
    ]
    for key, label in METHODS:
        for i, r in enumerate(RATIOS):
            s = summaries[f"msd_{key}_{r}"]
            cells = [fmt_cell(s["classes"]["tumor"]["metrics"][m]) for m in METRICS]
            cells.append(f"{s['classes']['tumor']['fp']['fp_rate'] * 100:.2f}")
            lines.append(f"{label if i == 0 else ''} & {r}\\% & "
                         + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}%", "}", caption,
                          "\\label{tab:msd_v2}", "\\end{table}", ""]
    return "\n".join(lines)


def fmt_p_latex(p):
    if p < 0.001:
        text = "$<$0.001"
    elif p >= 0.995:
        text = "1.00"
    elif p < 0.1:
        text = f"{p:.3f}"
    else:
        text = f"{p:.2f}"
    return f"\\textbf{{{text}}}" if p < 0.05 else text


def table_msd_tests(stats):
    lines = [
        "% MSD 患者级 Wilcoxon；本文件由 src/reanalyze_v2.py 从同一统计结果生成。",
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Paired two-sided Wilcoxon signed-rank $p$-values on patient-level "
        "tumor Dice, MSD Task10 Colon held-out test set (19 cases; identical "
        "tumor-containing slice set for all methods; single training seed 42). Holm "
        "adjustment within each labeled-ratio family (5 comparisons). All models use "
        "labeled-subset validation only, with fixed-schedule final checkpoints at "
        "1\\%. $p<0.05$ in bold. ST = pseudo-label self-training, MT = Mean Teacher.}",
        "\\label{tabS:wilcoxon_msd_v2}", "\\small",
        "\\begin{tabular}{llcc}", "\\toprule",
        "Labeled ratio & Comparison & raw $p$ & Holm $p$ \\\\", "\\midrule"]
    for r in RATIOS:
        for i, (_a, _b, label) in enumerate(COMPARISONS):
            res = stats["tumor/dice"][f"{r}%/{label}"]
            prefix = f"\\multirow{{5}}{{*}}{{{r}\\%}}" if i == 0 else ""
            lines.append(f"{prefix} & {label} & {fmt_p_latex(res['raw_p'])} & "
                         f"{fmt_p_latex(res['holm_p'])} \\\\")
        if r != RATIOS[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程

def run_dataset(name, tags, gt, classes, results_dir=None, out_dir=None):
    """返回 (summaries, case_vals_all)：case_vals_all[tag][cls][metric] = {case: val}"""
    results_dir = results_dir or RESULTS
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    log(f"\n=== 数据集 {name} ===")
    summaries, case_vals_all = {}, {}
    slice_sets = []
    for tag in tags:
        csv_path = os.path.join(results_dir, f"{tag}_per_slice.csv")
        df = pd.read_csv(csv_path).set_index("slice")
        slice_sets.append((tag, set(df.index)))
        assert set(df.index) == set(gt.keys()), \
            f"{tag}: CSV 切片集与 GT 不一致（csv={len(df)}, gt={len(gt)}）"
        for msg in validate(tag, df, gt, classes):
            log(msg)
        s, cv = analyze_tag(tag, df, gt, classes)
        summaries[tag] = s
        case_vals_all[tag] = cv
        with open(os.path.join(out_dir, f"{tag}_summary_v2.json"), "w") as f:
            json.dump(s, f, indent=2)
    # 切片集一致性：所有方法 CSV 行集必须完全相同
    base_tag, base_set = slice_sets[0]
    for tag, ss in slice_sets[1:]:
        assert ss == base_set, f"{name}: {tag} 与 {base_tag} 切片集不同"
    log(f"所有 {len(tags)} 个方法的 CSV 切片集完全一致（{len(base_set)} 切片）")
    for ci, cls in enumerate(classes):
        n_gt = sum(1 for v in gt.values() if v[ci])
        n_cases = len({parse_case_id(s) for s, v in gt.items() if v[ci]})
        log(f"  {cls}: v2 固定评分切片集 = {n_gt} 切片（所有方法相同），"
            f"覆盖 {n_cases} 个病例；GT 空切片 {len(gt) - n_gt}")
    return summaries, case_vals_all


def run_stats(name, ratios, case_vals_all, classes, md_lines, prefix=""):
    """每个 (ratio, class, metric) 一个 family（5 组比较），Holm 校正。返回嵌套 dict。"""
    md_lines.append(f"\n## {name}\n")
    out = {}
    for cls in classes:
        for m in METRICS:
            md_lines.append(f"\n### {name} — {cls} {m.upper()}\n")
            md_lines.append("| Ratio | Comparison | n_cases | Mean A | Mean B "
                            "| raw p | Holm p |")
            md_lines.append("|---|---|---|---|---|---|---|")
            for r in ratios:
                pvals, rows = [], []
                for a, b, label in COMPARISONS:
                    va = case_vals_all[f"{prefix}{a}_{r}"][cls][m]
                    vb = case_vals_all[f"{prefix}{b}_{r}"][cls][m]
                    cases = sorted(set(va) & set(vb))
                    xa = np.array([va[c] for c in cases])
                    xb = np.array([vb[c] for c in cases])
                    p = wilcoxon_p(xa, xb)
                    pvals.append(p)
                    rows.append((label, len(cases), xa.mean(), xb.mean(), p))
                adj = holm_adjust(np.array(pvals))
                for (label, n, ma, mb, p), pa in zip(rows, adj):
                    out.setdefault(f"{cls}/{m}", {})[f"{r}%/{label}"] = {
                        "n_cases": n, "raw_p": p, "holm_p": float(pa)}
                    md_lines.append(
                        f"| {r}% | {label} | {n} | {ma * 100:.2f} | {mb * 100:.2f} "
                        f"| {p:.4g} | {pa:.4g} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msd_results_dir", default=None,
                    help="只替换 MSD 的 CSV 来源，如 results_msd_lb；缺省为 results")
    ap.add_argument("--msd_only", action="store_true",
                    help="仅重算 MSD，不触碰 CARE 表和统计")
    ap.add_argument("--out_dir", default=OUT_DIR,
                    help="*_summary_v2.json 输出目录")
    ap.add_argument("--man_dir", default=MAN)
    ap.add_argument("--sr_dir", default=os.path.join("paper", "sr"))
    ap.add_argument("--stats_md", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.man_dir, exist_ok=True)
    os.makedirs(args.sr_dir, exist_ok=True)
    stats_md = args.stats_md or (os.path.join("paper", "v2_stats_msd.md")
                                 if args.msd_only else STATS_MD)

    log("=== v2 重分析：CSV 解码校验 + 切片集一致性 ===")
    gt_care = None if args.msd_only else care_gt_presence()
    gt_msd = msd_gt_presence()
    if gt_care is not None:
        log(f"GT 加载：CARE {len(gt_care)} 切片（data/cache/test_labels.npy）")
    log(f"GT 加载：MSD {len(gt_msd)} 切片")

    care_tags = [f"{k}_{r}" for k, _ in METHODS for r in RATIOS] + ["sup_100"]
    msd_tags = [f"msd_{k}_{r}" for k, _ in METHODS for r in RATIOS]
    care_sum = care_cv = None
    if not args.msd_only:
        care_sum, care_cv = run_dataset("CARE", care_tags, gt_care,
                                        ("rectum", "tumor"), out_dir=args.out_dir)
    msd_sum, msd_cv = run_dataset(
        "MSD", msd_tags, gt_msd, ("tumor",),
        results_dir=args.msd_results_dir or RESULTS, out_dir=args.out_dir)

    if not any("VIOLATION" in line for line in REPORT):
        log("CSV 解码校验通过：所有方法所有类 —— GT 含类切片无 nan；"
            "GT 空切片 score ∈ {0, nan}；三指标列 0/nan 模式一致。")

    md = ["# v2 评审重分析：病例级 Wilcoxon 统计（raw p / Holm 校正 p）",
          "",
          "协议：前景指标仅在 GT 含该类的切片上计算（所有方法同一固定切片集）；"
          "病例级数值 = 病例内已评分切片均值；配对双侧 Wilcoxon 符号秩；"
          "Holm 校正在每个 (dataset, ratio, class, metric) 的 5 组比较 family 内进行。"
          "均值以 % 表示。由 `src/reanalyze_v2.py` 生成。"]
    if care_cv is not None:
        run_stats("CARE", RATIOS, care_cv, ("rectum", "tumor"), md)
    msd_stats = run_stats("MSD", RATIOS, msd_cv, ("tumor",), md, prefix="msd_")
    with open(stats_md, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"已保存 {stats_md}")

    if care_sum is not None:
        with open(os.path.join(args.man_dir, "table1_care_v2.tex"), "w") as f:
            f.write(table_care(care_sum))
    table2_path = os.path.join(args.man_dir, "table2_msd_v2.tex")
    with open(table2_path, "w") as f:
        f.write(table_msd(msd_sum))
    shutil.copyfile(table2_path, os.path.join(args.sr_dir, "table2_msd_v2.tex"))
    with open(os.path.join(args.sr_dir, "table2_msd_v2_zh.tex"), "w") as f:
        f.write(table_msd(msd_sum, chinese=True))
    tests_path = os.path.join(args.man_dir, "tabS6_wilcoxon_msd_v2.tex")
    with open(tests_path, "w") as f:
        f.write(table_msd_tests(msd_stats))
    shutil.copyfile(tests_path, os.path.join(args.sr_dir,
                                             "tabS6_wilcoxon_msd_v2.tex"))
    print(f"已保存 {table2_path} 及 SR 英文/中文副本和 MSD Wilcoxon 表")
    print(f"已保存 {len(msd_sum)} 个 MSD summary_v2.json（{args.out_dir}）")


if __name__ == "__main__":
    main()
