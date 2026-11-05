#!/usr/bin/env python
"""make_table_1pct_perseed.py — 生成 1% 档逐种子患者级 Dice 与配对差值表（Table S23）。

回应审稿意见：1% 时"所有方法在所有种子均退化"的说法必须逐种子核查方向。
从 results/v2/agg_seeds.json（逐种子患者级均值）与 results/v2/agg_tests.json
（逐种子配对双侧 Wilcoxon p）读取既有统计结果，不重新计算指标。

产出：paper/sr/tabS_1pct_perseed.tex（并复制到 paper/manuscript/）。
冒烟：--smoke --seeds_json <json> --tests_json <json> --out <tex>（合成数据）。
"""

import argparse
import json
import shutil

METHODS = [("sup", "Supervised"), ("semi", "ST"), ("mt", "MT"),
           ("cps", "CPS"), ("bcp", "BCP"), ("ours", "Ours")]
SEEDS = ["42", "43", "44"]


def seed_means(exp, cls):
    """agg_seeds.json 中某实验某类的逐种子患者级 Dice 均值，键规范化为种子号。"""
    sm = exp["classes"][cls]["metrics"]["dice"]["seed_means"]
    return {k.rsplit("_s", 1)[1]: v for k, v in sm.items()}


def per_seed_p(test_entry):
    return test_entry["p_per_seed"]


def fmt_p(p):
    if p < 1e-3:
        return "$<$0.001"
    return f"{p:.3f}".lstrip("0") if p < 0.1 else f"{p:.3f}"


def fmt_cell(mean, diff=None, p=None, reverse=False):
    """单元格：Dice 均值（%），可选 Δ(pp) 与 p；方向反转加 dag。"""
    s = f"{mean * 100:.1f}"
    if diff is not None:
        dag = "$^{\\dagger}$" if reverse else ""
        s += f" ({diff * 100:+.1f}{dag}, {fmt_p(p)})"
    return s


def panel_rows(seeds_json, tests_json, cls, strict=True):
    exps = seeds_json["experiments"]
    fam_ours = tests_json["families"][f"{cls}/dice"]
    fam_sup = tests_json["sup_vs_others"]["families"][f"{cls}/dice"]
    rows = []
    sup = seed_means(exps["sup_1"], cls)
    rows.append("Supervised & " + " & ".join(fmt_cell(sup[s]) for s in SEEDS)
                + " \\\\")
    for key, label in METHODS[1:]:
        if f"{key}_1" not in exps:
            if strict:
                raise KeyError(f"缺实验 {key}_1（agg_seeds.json 不完整）")
            continue
        m = seed_means(exps[f"{key}_1"], cls)
        if key == "ours":
            p = per_seed_p(fam_ours["1%/sup vs ours"])
        else:
            p = per_seed_p(fam_sup[f"1%/sup vs {key}"])
        cells = []
        for s in SEEDS:
            diff = m[s] - sup[s]
            cells.append(fmt_cell(m[s], diff, p[s], reverse=diff > 0))
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(rows)


def build_tex(seeds_json, tests_json, strict=True):
    tumor = panel_rows(seeds_json, tests_json, "tumor", strict)
    rectum = panel_rows(seeds_json, tests_json, "rectum", strict)
    return r"""% 1% 档逐种子患者级 Dice 与配对差值（Table S23）。
% 本文件由 src/make_table_1pct_perseed.py 从 results/v2/agg_seeds.json 与
% results/v2/agg_tests.json 生成；指标与检验均不重新计算。
\begin{table}[!htbp]
\centering
\caption{Per-seed patient-level Dice (\%) at the 1\% labeled ratio (three
annotated cases) on the CARE test set (81 cases), with paired differences
against the same-seed supervised baseline. Each cell gives the patient-level
mean Dice and, for semi-supervised rows, ($\Delta$ vs Supervised in points,
two-sided paired Wilcoxon $p$ within that seed, unadjusted). $^{\dagger}$
marks seeds in which the semi-supervised method \emph{exceeds} the same-seed
supervised baseline (direction reversal); note that a significant max-$p$
across seeds does not imply a consistent direction --- MT (tumor, seed 43)
and BCP (tumor, seed 44) are significant in the \emph{opposite} direction.
ST = pseudo-label self-training, MT = Mean Teacher.}
\label{tabS:perseed_1pct}
\small
\begin{tabular}{lccc}
\toprule
Method & Seed 42 & Seed 43 & Seed 44 \\
\midrule
\multicolumn{4}{l}{\textit{Panel A: tumor}} \\
\midrule
__TUMOR__
\midrule
\multicolumn{4}{l}{\textit{Panel B: normal rectum}} \\
\midrule
__RECTUM__
\bottomrule
\end{tabular}
\end{table}
""".replace("__TUMOR__", tumor).replace("__RECTUM__", rectum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_json", default="results/v2/agg_seeds.json")
    ap.add_argument("--tests_json", default="results/v2/agg_tests.json")
    ap.add_argument("--out", default="paper/sr/tabS_1pct_perseed.tex")
    ap.add_argument("--man_copy", default="paper/manuscript/tabS_1pct_perseed.tex")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式：仅写出 --out，不复制稿件目录")
    args = ap.parse_args()

    seeds_json = json.load(open(args.seeds_json))
    tests_json = json.load(open(args.tests_json))
    tex = build_tex(seeds_json, tests_json, strict=not args.smoke)
    with open(args.out, "w") as f:
        f.write(tex)
    print(f"written {args.out}")
    if not args.smoke:
        shutil.copy(args.out, args.man_copy)
        print(f"copied -> {args.man_copy}")


if __name__ == "__main__":
    main()
