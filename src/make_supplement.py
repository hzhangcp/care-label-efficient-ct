"""补充材料生成（Computers in Biology and Medicine, supplement.tex 配套）。

全部数字从 results/*_summary.json、results/compare_*.json、runs/*/run_meta.json
现算，不手抄。产出（paper/manuscript/ 下，位图 300 dpi）：
  figS1_iou_nsd_curves.png   CARE 标注效率曲线（IoU/NSD × tumor/rectum，5 方法
                             + Supervised-100 参考点，bootstrap 95%CI 误差棒）
  figS2_pseudo_rounds.png    ST / Ours 各自训练轮次的伪标签覆盖率曲线
  tabS1_wilcoxon_care.tex    CARE 配对 Wilcoxon p 值表（tumor/rectum Dice+NSD）
  tabS2_wilcoxon_msd.tex     MSD 配对 Wilcoxon p 值表（tumor Dice+NSD）
  tabS3_pseudo_stats.tex     逐轮 pseudo_coverage 与 best_val_dice 统计表

用法（仓库根目录）：python src/make_supplement.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import (RESULTS, RATIOS, METHODS, METHOD_LABEL, METHOD_STYLE,
                          load_summary)

OUT = os.path.join("paper", "manuscript")
RUNS = "runs"

# CARE 表 S1 的比较组（A vs B，与 compare json 文件名方向一致）
CARE_PAIRS = [("sup", "semi"), ("sup", "mt"), ("sup", "cps"), ("sup", "ours"),
              ("semi", "ours")]
MSD_PAIRS = [("msd_sup", "msd_semi"), ("msd_sup", "msd_mt"),
             ("msd_sup", "msd_cps"), ("msd_sup", "msd_ours")]
PAIR_LABEL = {("sup", "semi"): "Supervised vs ST",
              ("sup", "mt"): "Supervised vs MT",
              ("sup", "cps"): "Supervised vs CPS",
              ("sup", "ours"): "Supervised vs Ours",
              ("semi", "ours"): "ST vs Ours",
              ("msd_sup", "msd_semi"): "Supervised vs ST",
              ("msd_sup", "msd_mt"): "Supervised vs MT",
              ("msd_sup", "msd_cps"): "Supervised vs CPS",
              ("msd_sup", "msd_ours"): "Supervised vs Ours"}


def metric_point(summary, cls, metric):
    """mean% 与 bootstrap 95%CI 半宽（%）。"""
    m = summary["metrics"][cls][metric]
    return (m["mean"] * 100, (m["mean"] - m["ci95"][0]) * 100,
            (m["ci95"][1] - m["mean"]) * 100)


def figS1_iou_nsd_curves():
    """2x2：行=IoU/NSD，列=tumor/rectum；风格对齐 make_figures.fig1。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True)
    for row, metric in enumerate(("iou", "nsd")):
        for col, (cls, title) in enumerate(
                [("tumor", "Tumor"), ("rectum", "Normal rectum")]):
            ax = axes[row][col]
            for m in METHODS:
                xs = RATIOS + ([100] if m == "sup" else [])
                ys, lo, hi = [], [], []
                for r in xs:
                    y, lo_, hi_ = metric_point(load_summary(f"{m}_{r}"),
                                               cls, metric)
                    ys.append(y)
                    lo.append(lo_)
                    hi.append(hi_)
                ax.errorbar(xs, ys, yerr=[lo, hi], capsize=2.5,
                            label=METHOD_LABEL[m], **METHOD_STYLE[m])
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 10, 25, 50, 100])
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
            ax.set_title(title)
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("Labeled ratio (% of training cases)")
            if col == 0:
                ax.set_ylabel(f"Test {metric.upper()} (%)")
    axes[0][1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Label efficiency on CARE test (81 cases, 6,461 slices): "
                 "IoU (top) and NSD (bottom)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "figS1_iou_nsd_curves.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"已保存 {out}")


def load_history(run):
    with open(os.path.join(RUNS, run, "run_meta.json")) as f:
        return json.load(f)["history"]


def figS2_pseudo_rounds():
    """ST / Ours 各标注比例的逐轮伪标签覆盖率（%）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    colors = {1: "tab:purple", 5: "tab:blue", 10: "tab:green",
              25: "tab:orange", 50: "tab:red"}
    for ax, (key, title) in zip(axes, [("semi", "Pseudo-label ST"),
                                       ("ours", "Ours")]):
        for r in RATIOS:
            h = load_history(f"{key}_{r}")
            xs = [e["round"] for e in h]
            ys = [e["pseudo_coverage"] * 100 for e in h]
            ax.plot(xs, ys, marker="o", lw=1.6, color=colors[r],
                    label=f"{r}% labels")
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Self-training round")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pseudo-label coverage (% of unlabeled pixels)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Pseudo-label coverage across self-training rounds (CARE)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT, "figS2_pseudo_rounds.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"已保存 {out}")


def _fmt_p(p):
    """科学计数法；p<0.001 记为 <0.001；p<0.05 加粗。"""
    if p is None:
        return "n/a"
    if p < 0.001:
        core = "$<$0.001"
    else:
        mant, exp = f"{p:.2e}".split("e")
        core = f"${mant}\\times10^{{{int(exp)}}}$"
    if p < 0.05:
        core = f"\\textbf{{{core}}}"
    return core


def _wilcoxon_rows(pairs, tests, ratios):
    """按 比较组 × 比例 展开行；tests 为 compare json tests 里的键序列。"""
    lines = []
    for a, b in pairs:
        label = PAIR_LABEL[(a, b)]
        for i, r in enumerate(ratios):
            path = os.path.join(RESULTS, f"compare_{a}_{r}_vs_{b}_{r}.json")
            if not os.path.exists(path):
                cells = ["n/a"] * len(tests)
            else:
                with open(path) as f:
                    c = json.load(f)
                cells = [_fmt_p(c["tests"].get(t, {}).get("p")) for t in tests]
            first = f"\\multirow{{{len(ratios)}}}{{*}}{{{label}}}" if i == 0 else ""
            lines.append(f"{first} & {r}\\% & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    return lines[:-1]  # 去掉最后一根 midrule，由调用方补 bottomrule


def tabS1_wilcoxon_care():
    tests = ["dice_tumor", "dice_rectum", "nsd_tumor", "nsd_rectum"]
    lines = [
        "% CARE 配对 Wilcoxon p 值表，由 src/make_supplement.py 从 results/compare_*.json 生成",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Paired two-sided Wilcoxon signed-rank $p$-values on per-slice "
        "scores, CARE test set (6{,}461 slices). Comparison direction A vs B "
        "follows the recorded file order; mean per-method scores are given in "
        "the main text. $p<0.05$ in bold; $p<0.001$ marked as such. "
        "ST = pseudo-label self-training, MT = Mean Teacher.}",
        "\\label{tabS:wilcoxon_care}",
        "\\small",
        "\\begin{tabular}{ll" + "c" * len(tests) + "}",
        "\\toprule",
        "Comparison & Labeled & \\multicolumn{2}{c}{Dice $p$} & "
        "\\multicolumn{2}{c}{NSD $p$} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        " & ratio & Tumor & Rectum & Tumor & Rectum \\\\",
        "\\midrule",
    ]
    lines += _wilcoxon_rows(CARE_PAIRS, tests, RATIOS)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tabS2_wilcoxon_msd():
    tests = ["dice_tumor", "nsd_tumor"]
    lines = [
        "% MSD Colon 配对 Wilcoxon p 值表（tumor），由 src/make_supplement.py 生成",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Paired two-sided Wilcoxon signed-rank $p$-values on per-slice "
        "tumor scores, MSD Task10 Colon held-out test set (19 cases, 1{,}680 "
        "slices). Conventions as in Table~\\ref{tabS:wilcoxon_care}.}",
        "\\label{tabS:wilcoxon_msd}",
        "\\small",
        "\\begin{tabular}{ll" + "c" * len(tests) + "}",
        "\\toprule",
        "Comparison & Labeled ratio & Tumor Dice $p$ & Tumor NSD $p$ \\\\",
        "\\midrule",
    ]
    lines += _wilcoxon_rows(MSD_PAIRS, tests, RATIOS)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tabS3_pseudo_stats():
    """ST / Ours 各比例逐轮 pseudo_coverage(%) 与 best_val_dice(%)。"""
    n_rounds = max(len(load_history(f"{k}_{r}"))
                   for k in ("semi", "ours") for r in RATIOS)
    lines = [
        "% 逐轮伪标签统计，由 src/make_supplement.py 从 runs/*/run_meta.json 生成",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Pseudo-label statistics per self-training round on CARE: "
        "coverage (fraction of unlabeled pixels passing the confidence "
        "threshold, \\%) and best validation Dice of the round (\\%). "
        "ST = pseudo-label self-training.}",
        "\\label{tabS:pseudo}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{ll" + "c" * (2 * n_rounds) + "}",
        "\\toprule",
        " & & \\multicolumn{%d}{c}{Coverage (\\%%), round} & "
        "\\multicolumn{%d}{c}{Val Dice (\\%%), round} \\\\" % (n_rounds, n_rounds),
        "\\cmidrule(lr){3-%d} \\cmidrule(lr){%d-%d}"
        % (2 + n_rounds, 3 + n_rounds, 2 + 2 * n_rounds),
        "Method & Labeled & " + " & ".join(str(i + 1) for i in range(n_rounds))
        + " & " + " & ".join(str(i + 1) for i in range(n_rounds)) + " \\\\",
        "\\midrule",
    ]
    for key, label in [("semi", "Pseudo-label ST"), ("ours", "Ours")]:
        for i, r in enumerate(RATIOS):
            h = load_history(f"{key}_{r}")
            cov = [f"{e['pseudo_coverage'] * 100:.2f}" for e in h]
            vd = [f"{e['best_val_dice'] * 100:.2f}" for e in h]
            pad = ["--"] * (n_rounds - len(h))
            first = (f"\\multirow{{{len(RATIOS)}}}{{*}}{{{label}}}"
                     if i == 0 else "")
            lines.append(f"{first} & {r}\\% & "
                         + " & ".join(cov + pad + vd + pad) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    figS1_iou_nsd_curves()
    figS2_pseudo_rounds()
    for fname, gen in [("tabS1_wilcoxon_care.tex", tabS1_wilcoxon_care),
                       ("tabS2_wilcoxon_msd.tex", tabS2_wilcoxon_msd),
                       ("tabS3_pseudo_stats.tex", tabS3_pseudo_stats)]:
        path = os.path.join(OUT, fname)
        with open(path, "w") as f:
            f.write(gen())
        print(f"已保存 {path}")


if __name__ == "__main__":
    main()
