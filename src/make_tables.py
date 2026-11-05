"""从 results/*_summary.json 现算生成投稿用 LaTeX 三线表（booktabs）。

产出（paper/manuscript/）：
  table1_care.tex    CARE 主结果（方法 × 标注比例 × dice/iou/nsd × rectum/tumor）
  table2_msd.tex     MSD Colon 外部验证（二分类仅 tumor，全切片协议）
  table_ablation.tex stage-1 全图消融 + 10% 档阈值/轮数消融

用法：python src/make_tables.py
"""

import json
import os

RESULTS = "results"
OUT = os.path.join("paper", "manuscript")
RATIOS = [1, 5, 10, 25, 50]
METHODS = [("sup", "Supervised"), ("semi", "Pseudo-label ST"),
           ("mt", "Mean Teacher"), ("cps", "CPS"), ("ours", "Ours")]
METRICS = ["dice", "iou", "nsd"]


def cell(metrics, cls, metric):
    m = metrics[cls][metric]
    return f"{m['mean'] * 100:.2f} [{m['ci95'][0] * 100:.2f}, {m['ci95'][1] * 100:.2f}]"


def row_for(name, label, ratio, classes=("rectum", "tumor")):
    with open(os.path.join(RESULTS, f"{name}_summary.json")) as f:
        s = json.load(f)
    cells = [cell(s["metrics"], c, m) for c in classes for m in METRICS]
    return f"{label} & {ratio}\\% & " + " & ".join(cells) + " \\\\"


def table1_care():
    lines = [
        "% CARE 主结果：mean% [bootstrap 95%CI]，由 src/make_tables.py 从 results/*_summary.json 生成",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{CARE test set results (81 cases, 6{,}461 slices). "
        "Values are mean\\% [bootstrap 95\\% CI] computed per slice; "
        "slices with empty GT and empty prediction for a class are excluded "
        "from that class's average. R = normal rectum, T = tumor.}",
        "\\label{tab:care}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * 6 + "}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{Rectum} & \\multicolumn{3}{c}{Tumor} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
        "Method & Labeled & Dice & IoU & NSD & Dice & IoU & NSD \\\\",
        "\\midrule",
    ]
    for key, label in METHODS:
        for i, r in enumerate(RATIOS):
            row = row_for(f"{key}_{r}", label if i == 0 else "", r)
            lines.append(row)
        lines.append("\\midrule")
    lines.append(row_for("sup_100", "Supervised", 100))
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)


def table2_msd():
    lines = [
        "% MSD Colon 外部验证：mean% [bootstrap 95%CI]，二分类仅 tumor",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{External validation on MSD Task10 Colon (19 held-out cases, "
        "1{,}680 slices). Tumor Dice/IoU/NSD in mean\\% [bootstrap 95\\% CI] "
        "under the same full-slice protocol as CARE; because $\\sim$90\\% of "
        "slices contain no tumor and any false positive there scores zero, "
        "absolute values are systematically lower --- only the relative "
        "ordering across methods is meaningful.}",
        "\\label{tab:msd}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * 3 + "}",
        "\\toprule",
        "Method & Labeled & Dice & IoU & NSD \\\\",
        "\\midrule",
    ]
    for key, label in METHODS:
        for i, r in enumerate(RATIOS):
            lines.append(row_for(f"msd_{key}_{r}", label if i == 0 else "", r,
                                 classes=("tumor",)))
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)


def table_ablation():
    lines = [
        "% 消融：stage-1 全图基线（各比例）+ 10% 档阈值/轮数消融",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Ablations on the CARE test set (mean\\% [bootstrap 95\\% CI]). "
        "Top: stage-1 full-image model alone (no cascade refinement) at each "
        "labeled ratio. Bottom: pseudo-label self-training variants at 10\\% "
        "labels (default: 3 rounds, confidence threshold 0.9).}",
        "\\label{tab:ablation}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * 6 + "}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{Rectum} & \\multicolumn{3}{c}{Tumor} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}",
        "Variant & Labeled & Dice & IoU & NSD & Dice & IoU & NSD \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(RATIOS + [100]):
        lines.append(row_for(f"s1_{r}", "Stage-1 only" if i == 0 else "", r))
    lines.append("\\midrule")
    for tag, label in [("semi_10", "ST (default)"), ("semi_10_rounds1", "ST, 1 round"),
                       ("semi_10_rounds5", "ST, 5 rounds"),
                       ("semi_10_thr0.7", "ST, threshold 0.7"),
                       ("semi_10_thr0.95", "ST, threshold 0.95")]:
        lines.append(row_for(tag, label, 10))
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    for fname, gen in [("table1_care.tex", table1_care),
                       ("table2_msd.tex", table2_msd),
                       ("table_ablation.tex", table_ablation)]:
        path = os.path.join(OUT, fname)
        with open(path, "w") as f:
            f.write(gen())
        print(f"已保存 {path}")


if __name__ == "__main__":
    main()
