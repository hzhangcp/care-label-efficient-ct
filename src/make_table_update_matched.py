#!/usr/bin/env python
"""相同优化器更新数监督对照（Table S27）的三种子汇总与配对分析。"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from reanalyze_v2 import analyze_tag, care_gt_presence, wilcoxon_p
from aggregate_v4_rebuttal import twoway_cluster_boot_ci

SEEDS = [42, 43, 44]
RATIOS = [50, 1]
METHODS = ["sup", "semi", "ours", "supupd"]
CLASSES = ["rectum", "tumor"]
LABELS = {"sup": "Supervised-100ep", "semi": "ST", "ours": "Ours",
          "supupd": "Supervised-update-matched"}


def collect(results_dir, gt):
    data = {}
    for ratio in RATIOS:
        for method in METHODS:
            for seed in SEEDS:
                stem = f"{method}_{ratio}_s{seed}"
                path = os.path.join(results_dir, f"{stem}_per_slice.csv")
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path).set_index("slice")
                data[(ratio, method, seed)] = analyze_tag(stem, frame, gt, CLASSES)
    return data


def case_values(data, ratio, method, seed, cls):
    return data[(ratio, method, seed)][1][cls]["dice"]


def effect(data, ratio, reference, cls):
    rows = []
    cases_ref = None
    for seed in SEEDS:
        control = case_values(data, ratio, "supupd", seed, cls)
        base = case_values(data, ratio, reference, seed, cls)
        cases = sorted(set(control) & set(base))
        if cases_ref is None:
            cases_ref = cases
        if cases != cases_ref:
            raise ValueError(f"case mismatch at ratio={ratio}, seed={seed}")
        rows.append([control[c] - base[c] for c in cases])
    return np.asarray(rows)


def build(data, updates=None):
    updates = updates or {}
    out = {"protocol": "patient-level Dice; three seeds; supervised baseline matched to "
                       "the Ours optimizer-update count at each seed",
           "cells": {}, "comparisons": {}, "updates": updates}
    rng = np.random.default_rng(42)
    for ratio in RATIOS:
        for method in METHODS:
            cell = {}
            for cls in CLASSES:
                means = {str(s): float(np.mean(list(case_values(data, ratio, method, s, cls).values())))
                         for s in SEEDS}
                values = np.asarray(list(means.values()))
                cell[cls] = {"seed_means": means, "mean": float(values.mean()),
                             "std": float(values.std(ddof=1))}
            out["cells"][f"{ratio}/{method}"] = cell
        for reference in ["sup", "semi", "ours"]:
            for cls in CLASSES:
                diff = effect(data, ratio, reference, cls)
                p_per_seed = {}
                for i, seed in enumerate(SEEDS):
                    p_per_seed[str(seed)] = wilcoxon_p(diff[i], np.zeros(diff.shape[1]))
                out["comparisons"][f"{ratio}/supupd_minus_{reference}/{cls}"] = {
                    "mean": float(diff.mean()),
                    "per_seed_mean": {str(s): float(diff[i].mean())
                                      for i, s in enumerate(SEEDS)},
                    "ci95_twoway_boot": twoway_cluster_boot_ci(diff, 10000, rng),
                    "p_per_seed": p_per_seed, "p_max": max(p_per_seed.values())}
    return out


def fmt_p(value):
    return "$<$0.001" if value < 0.001 else f"{value:.3f}"


def tex_table(result):
    def cell(ratio, method, cls):
        item = result["cells"][f"{ratio}/{method}"][cls]
        return f"{100 * item['mean']:.2f}$\\pm${100 * item['std']:.2f}"

    def comparison(ratio, reference):
        item = result["comparisons"][f"{ratio}/supupd_minus_{reference}/tumor"]
        lo, hi = item["ci95_twoway_boot"]
        return (f"{100 * item['mean']:+.2f} [{100 * lo:+.2f},{100 * hi:+.2f}] & "
                f"{fmt_p(item['p_max'])}")

    lines = []
    for ratio in RATIOS:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{ratio}\\% labels}}}} \\\\")
        for method in METHODS:
            update = result["updates"].get(f"{ratio}/{method}")
            update_text = "--" if update is None else f"{update / 1000:.1f}k"
            lines.append(f"{LABELS[method]} & {update_text} & {cell(ratio, method, 'tumor')} & "
                         f"{cell(ratio, method, 'rectum')} \\\\")
        lines.append("\\midrule")
        for reference in ["sup", "semi", "ours"]:
            lines.append(f"Update-matched $-$ {LABELS[reference]} (tumor) & -- & "
                         f"{comparison(ratio, reference)} \\\\")
        if ratio != RATIOS[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    return r"""% 由 src/make_table_update_matched.py 生成。
\begin{table}[!htbp]
\centering
\caption{Optimizer-update-matched supervised controls. The supervised-update-matched
runs use the same labeled subsets and validation protocol as the main experiments,
but their cosine schedule is extended to the Ours update count at each seed. Dice is
patient-level mean\% $\pm$ std across seeds 42/43/44. Effect cells show the
update-matched supervised-minus-reference difference in percentage points, with a
95\% two-way cluster-bootstrap interval over patients and seeds and the maximum
per-seed paired Wilcoxon $p$. These are exploratory robustness analyses with
unadjusted $p$-values.}
\label{tabS:update_matched}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
Method / comparison & Updates & Tumor / effect [95\% CI] & Rectum / max-$p$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def selftest(out_json, out_tex):
    rng = np.random.default_rng(0)
    data = {}
    cases = [f"case{i:03d}" for i in range(12)]
    for ratio in RATIOS:
        for method in METHODS:
            for seed in SEEDS:
                offset = {"sup": 0.0, "semi": 0.01, "ours": 0.02, "supupd": 0.005}[method]
                values = {}
                for cls in CLASSES:
                    vals = np.clip(rng.normal(0.5 + offset, 0.1, len(cases)), 0, 1)
                    values[cls] = {"dice": dict(zip(cases, vals))}
                data[(ratio, method, seed)] = (None, values)
    result = build(data, {f"{r}/{m}": 100000 for r in RATIOS for m in METHODS})
    assert len(result["comparisons"]) == 12
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    json.dump(result, open(out_json, "w"), indent=1)
    open(out_tex, "w").write(tex_table(result))
    print("make_table_update_matched SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_v2")
    ap.add_argument("--runs_dir", default="runs_v2")
    ap.add_argument("--gt_json", default=None)
    ap.add_argument("--out_json", default="results/v2/update_matched_controls.json")
    ap.add_argument("--out_tex", default="paper/sr/tabS_update_matched.tex")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.out_json, args.out_tex)
        return
    gt = (care_gt_presence() if not args.gt_json else
          {k: tuple(v) for k, v in json.load(open(args.gt_json)).items()})
    try:
        data = collect(args.results_dir, gt)
    except FileNotFoundError as exc:
        raise SystemExit(f"相同更新数对照尚未跑完，缺少：{exc}")
    updates = {}
    budget_path = os.path.join(os.path.dirname(args.out_json), "budget.json")
    budget = json.load(open(budget_path))["runs"] if os.path.isfile(budget_path) else {}
    for ratio in RATIOS:
        for method in METHODS:
            values = []
            for seed in SEEDS:
                meta_path = os.path.join(args.runs_dir, f"{method}_{ratio}_s{seed}", "run_meta.json")
                value = None
                if os.path.isfile(meta_path):
                    meta = json.load(open(meta_path))
                    value = meta.get("actual_optimizer_updates", meta.get("optimizer_updates"))
                if value is None:
                    value = budget.get(f"{method}_{ratio}_s{seed}", {}).get("optimizer_updates")
                if value is not None:
                    values.append(value)
            updates[f"{ratio}/{method}"] = float(np.median(values)) if values else None
    result = build(data, updates)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    json.dump(result, open(args.out_json, "w"), indent=1)
    open(args.out_tex, "w").write(tex_table(result))
    print(f"written {args.out_json} + {args.out_tex}")
    for ratio in RATIOS:
        for method in METHODS:
            cell = result["cells"][f"{ratio}/{method}"]["tumor"]
            print(f"{ratio}% {method}: tumor Dice {100 * cell['mean']:.2f}±{100 * cell['std']:.2f}")


if __name__ == "__main__":
    main()
