#!/usr/bin/env python
"""make_table_rounds_paired.py — ST vs Ours 配对轮数消融表（Table S24）。

回应审稿意见 point 4：Table 4 的轮数消融仅针对 ST 且用全训练池验证标签；
本表在同一初始教师（runs_v2/sup_10_s*）、同一 v2 主协议（labeled 验证）下，
配对比较 ST 与 Ours 在 rounds={1,3,5} 的患者级指标（三种子 mean±std），
检验过滤是否减缓确认误差随轮数积累（Ours 的轮间衰减应缓于 ST）。

rounds=3 复用主实验 runs_v2/{semi,ours}_10_s*；rounds={1,5} 来自
runs_v2/{semi,ours}_10_r{1,5}_s*（scripts/queue_ablation_rounds.sh）。

产出：results/v2/rounds_paired.json + paper/sr/tabS_rounds_paired.tex（复制到 manuscript）。
冒烟：--smoke --results_dir <dir> --gt_json <json> --out_json <json> --out_tex <tex>
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from reanalyze_v2 import analyze_tag, care_gt_presence, wilcoxon_p
from aggregate_v4_rebuttal import twoway_cluster_boot_ci

SEEDS = [42, 43, 44]
CLASSES = ("rectum", "tumor")
ROUNDS = [1, 3, 5]
METHODS = [("semi", "ST"), ("ours", "Ours")]


def stem_for(method, rounds, seed):
    if rounds == 3:
        return f"{method}_10_s{seed}"
    return f"{method}_10_r{rounds}_s{seed}"


def collect(results_dir, gt):
    """{(method, rounds): {seed: (summary, case_vals)}}；缺文件抛 FileNotFoundError。"""
    data = {}
    for mkey, _ in METHODS:
        for r in ROUNDS:
            per_seed = {}
            for s in SEEDS:
                stem = stem_for(mkey, r, s)
                csv = os.path.join(results_dir, f"{stem}_per_slice.csv")
                if not os.path.exists(csv):
                    raise FileNotFoundError(csv)
                df = pd.read_csv(csv).set_index("slice")
                per_seed[s] = analyze_tag(stem, df, gt, CLASSES)
            data[(mkey, r)] = per_seed
    return data


def seed_mean(case_vals, cls, metric="dice"):
    return float(np.mean(list(case_vals[cls][metric].values())))


def paired_p(data, m1, r1, m2, r2, cls, metric="dice"):
    """逐种子患者级配对 Wilcoxon，返回 {seed: p} 与 max-p。"""
    ps = {}
    for s in SEEDS:
        a = data[(m1, r1)][s][1][cls][metric]
        b = data[(m2, r2)][s][1][cls][metric]
        cases = sorted(set(a) & set(b))
        va = np.array([a[c] for c in cases])
        vb = np.array([b[c] for c in cases])
        ps[s] = wilcoxon_p(va, vb)
    return ps, max(ps.values())


def fmt_p(p):
    if p < 1e-3:
        return "$<$0.001"
    return f"{p:.3f}"


def diff_of_changes(data, cls, metric="dice"):
    """配对变化量之差 D[s,i] = (Ours_r5 - Ours_r1) - (ST_r5 - ST_r1)。

    患者级 Dice，shape (n_seeds, n_cases)。回应审稿意见：仅凭两个
    独立的 r1-vs-r5 检验（ST 显著、Ours 不显著）不能断言衰减幅度不同，
    需直接检验两者变化量之差。
    """
    mats = {}
    cases_ref = None
    for mkey, _ in METHODS:
        for r in (1, 5):
            rows = []
            for s in SEEDS:
                cv = data[(mkey, r)][s][1][cls][metric]
                cases = sorted(cv)
                if cases_ref is None:
                    cases_ref = cases
                assert cases == cases_ref, f"case 集合不一致: {mkey} r{r} s{s}"
                rows.append([cv[c] for c in cases])
            mats[(mkey, r)] = np.array(rows)
    return (mats[("ours", 5)] - mats[("ours", 1)]) - \
           (mats[("semi", 5)] - mats[("semi", 1)])


def build(data):
    """构造 json 结果与 tex 行。"""
    out = {"protocol": "v2 patient-level, 3 seeds mean±std; per-seed paired "
                       "Wilcoxon max-p; 同一教师 runs_v2/sup_10_s*",
           "cells": {}, "tests": {}}
    for mkey, _ in METHODS:
        for r in ROUNDS:
            cell = {}
            for cls in CLASSES:
                means = {s: seed_mean(data[(mkey, r)][s][1], cls) for s in SEEDS}
                arr = np.array(list(means.values()))
                cell[cls] = {"seed_means": means,
                             "mean": float(arr.mean()),
                             "std": float(arr.std(ddof=1))}
            out["cells"][f"{mkey}_r{r}"] = cell
    # Ours vs ST 每轮
    for r in ROUNDS:
        for cls in CLASSES:
            ps, pmax = paired_p(data, "ours", r, "semi", r, cls)
            out["tests"][f"ours_r{r}_vs_semi_r{r}/{cls}"] = {
                "p_per_seed": ps, "p_max": pmax}
    # 同方法轮间（衰减）
    for mkey, _ in METHODS:
        for r_lo, r_hi in ((1, 3), (3, 5), (1, 5)):
            for cls in CLASSES:
                ps, pmax = paired_p(data, mkey, r_lo, mkey, r_hi, cls)
                out["tests"][f"{mkey}_r{r_lo}_vs_r{r_hi}/{cls}"] = {
                    "p_per_seed": ps, "p_max": pmax}
    # 配对变化量之差（Ours 衰减 − ST 衰减），双路 cluster bootstrap
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        D = diff_of_changes(data, cls)
        out["tests"][f"diff_of_changes_r1_r5/{cls}"] = {
            "mean": float(D.mean()),
            "per_seed_mean": {str(s): float(D[i].mean())
                              for i, s in enumerate(SEEDS)},
            "ci95_twoway_boot": twoway_cluster_boot_ci(D, 10000, rng)}
    return out


def tex_table(res):
    label = dict(METHODS)

    def cell(mkey, r, cls):
        c = res["cells"][f"{mkey}_r{r}"][cls]
        return f"{c['mean'] * 100:.2f}$\\pm${c['std'] * 100:.2f}"

    def row(cls, r):
        st = res["cells"][f"semi_r{r}"][cls]
        ou = res["cells"][f"ours_r{r}"][cls]
        d = (ou["mean"] - st["mean"]) * 100
        p = res["tests"][f"ours_r{r}_vs_semi_r{r}/{cls}"]["p_max"]
        return (f"{r} & {cell('semi', r, cls)} & {cell('ours', r, cls)} "
                f"& {d:+.2f} & {fmt_p(p)} \\\\")

    def decay_lines(cls):
        lines = []
        for mkey in ("semi", "ours"):
            p13 = res["tests"][f"{mkey}_r1_vs_r3/{cls}"]["p_max"]
            p35 = res["tests"][f"{mkey}_r3_vs_r5/{cls}"]["p_max"]
            p15 = res["tests"][f"{mkey}_r1_vs_r5/{cls}"]["p_max"]
            lines.append(f"{label[mkey]} & {fmt_p(p13)} & {fmt_p(p35)} & {fmt_p(p15)} \\\\")
        return "\n".join(lines)

    tumor_rows = "\n".join(row("tumor", r) for r in ROUNDS)
    rectum_rows = "\n".join(row("rectum", r) for r in ROUNDS)
    tumor_decay = decay_lines("tumor")
    rectum_decay = decay_lines("rectum")
    doc_t = res["tests"]["diff_of_changes_r1_r5/tumor"]
    doc_r = res["tests"]["diff_of_changes_r1_r5/rectum"]
    def fmt_doc(e):
        lo, hi = e["ci95_twoway_boot"]
        return f"{e['mean'] * 100:+.2f} [${lo * 100:+.2f}, {hi * 100:+.2f}$]"

    template = r"""% ST vs Ours 配对轮数消融（Table S24），10% 标注、v2 主协议、同一初始教师、三种子。
% 由 src/make_table_rounds_paired.py 生成。
\begin{table}[!htbp]
\centering
\caption{Paired rounds ablation at 10\% labels under the main three-seed
protocol (same initial supervised teacher per seed, labeled-subset
validation; patient-level Dice \%, mean$\pm$std over seeds 42/43/44).
$\Delta$ is Ours minus ST; the $p$-value is the maximum per-seed paired
two-sided Wilcoxon $p$ across the three seeds. The lower block reports
within-method round-pair tests (confirmation-error accumulation). The
paired difference of round-wise changes, (Ours r5$-$r1) $-$ (ST r5$-$r1),
is __DOC_T__ points for tumor and __DOC_R__ points for normal rectum
(95\% CIs from a two-way cluster bootstrap resampling patients and seeds,
10{,}000 replicates). All $p$-values in this table are unadjusted maxima
across seeds; the analysis is exploratory. The
3-round rows are the main-protocol runs of
Tables~1--2 of the main text.}
\label{tabS:rounds_paired}
\small
\begin{tabular}{cccccc}
\toprule
Rounds & ST & Ours & $\Delta$ & max-$p$ \\
\midrule
\multicolumn{5}{l}{\textit{Panel A: tumor}} \\
\midrule
__TUMOR__
\midrule
\multicolumn{5}{l}{\textit{Panel B: normal rectum}} \\
\midrule
__RECTUM__
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{lccc}
\toprule
Within-method & r1 vs r3 & r3 vs r5 & r1 vs r5 \\
\midrule
\multicolumn{4}{l}{\textit{Tumor}} \\
__TUMOR_DECAY__
\midrule
\multicolumn{4}{l}{\textit{Normal rectum}} \\
__RECTUM_DECAY__
\bottomrule
\end{tabular}
\end{table}
"""
    return template.replace("__TUMOR__", tumor_rows).replace("__RECTUM__", rectum_rows) \
       .replace("__TUMOR_DECAY__", tumor_decay).replace("__RECTUM_DECAY__", rectum_decay) \
       .replace("__DOC_T__", fmt_doc(doc_t)).replace("__DOC_R__", fmt_doc(doc_r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_v2")
    ap.add_argument("--gt_json", default=None)
    ap.add_argument("--out_json", default="results/v2/rounds_paired.json")
    ap.add_argument("--out_tex", default="paper/sr/tabS_rounds_paired.tex")
    ap.add_argument("--man_copy", default="paper/manuscript/tabS_rounds_paired.tex")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：缺数据报错信息精简；不写 man_copy")
    args = ap.parse_args()

    gt = care_gt_presence() if not args.gt_json else \
        {k: tuple(v) for k, v in json.load(open(args.gt_json)).items()}
    try:
        data = collect(args.results_dir, gt)
    except FileNotFoundError as e:
        raise SystemExit(f"缺数据文件：{e}\n配对轮数消融尚未跑完，稍后再试。")
    res = build(data)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(res, open(args.out_json, "w"), indent=1)
    tex = tex_table(res)
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    open(args.out_tex, "w").write(tex)
    print(f"written {args.out_tex} + {args.out_json}")
    if not args.smoke:
        shutil.copy(args.out_tex, args.man_copy)
        print(f"copied -> {args.man_copy}")
    # 控制台摘要：肿瘤 Dice 随轮数变化
    for mkey, mlabel in METHODS:
        line = "  ".join(
            f"r{r}={res['cells'][f'{mkey}_r{r}']['tumor']['mean'] * 100:.2f}"
            for r in ROUNDS)
        print(f"{mlabel:5s} tumor Dice: {line}")


if __name__ == "__main__":
    main()
