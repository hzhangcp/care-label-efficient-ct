"""v4 rebuttal 统计（审稿意见 M1/M3/M6）：复用 aggregate_v2_seeds 的患者级聚合，不改旧脚本。

M1 交叉比例检验（新族 F3）：所有半监督方法 {semi, mt, cps, ours, bcp} 在 5%/25%/50%
  三档分别 vs Sup-100 的逐种子患者级配对双侧 Wilcoxon（同种子配对，两类分别），3 种子取
  max-p（保守合并）+ Holm 校正。族定义：每个 metric 一个族 = 方法数(5) × 档位数(3)
  × 类别数(2) = 30 个比较（2026-10-15 起纳入 5% 档：bcp 低比例补跑完成）。
  缺失档记 NA。
M3 探索性等效/非劣效（TOST）：患者级配对 Dice 差。primary = Ours-50% vs Sup-100%（肿瘤+直肠）；
  secondary = 其余半监督方法 50% vs Sup-100% + bcp@5%（F3 新档位追加组）。
  δ=2 个 Dice 点（事后选择的主界值），δ∈{1,3} 敏感性。
  报告逐种子差值与 pooled 差值；等效判定、TOST p 和 90%/95% CI 统一使用患者×种子
  双路 cluster bootstrap（10000 次）。旧版仅患者重抽样区间留在 JSON 作敏感性参照。
MT 数据源规则与 aggregate_v2_seeds 一致（resolve_mt_stems，全有或全无，
  未齐则沿用旧 mt_* 并标注）。gate 打开时另加载旧 mt_*，在同一运行内重算
  新旧 F1/F2/F3，paper/v4_stats.md 增加"MT 对齐修复前后对比"节
  （逐档 Dice mean±std + 显著性翻转/仅方向反转清单）。
M6 效应量：关键比较（同档 Ours vs ST / Sup vs 各半监督 / 交叉比例各方法 vs Sup-100）
  的患者级配对 Dice 差均值 + 95% CI，CI 用双路 cluster bootstrap（同时重抽样患者 ×
  种子，10000 次）——对 max-p 的层级敏感性分析（三种子共享同一测试集、检验强相关，
  见 paper/snippet_stats_methods.txt 方法学说明）。

产出：
  results/v2/cross_ratio_tests.json        M1
  results/v2/equivalence_tost.json         M3
  results/v2/effect_sizes.json             M6
  paper/manuscript/tabS7b_cross_ratio.tex  M1（并复制到 paper/sr/）
  paper/manuscript/tab_equiv_tost.tex      M3（并复制到 paper/sr/）
  paper/manuscript/tab_effect_sizes.tex    M6 精选（并复制到 paper/sr/）
  paper/v4_stats.md                        人读汇总（每结论一句话+数字）
  paper/snippet_stats_methods.txt          补充材料方法节英文说明（Holm 族 F1/F2/F3 + 敏感性分析）

用法：python src/aggregate_v4_rebuttal.py
冒烟：--smoke --n_boot 200 --results_dir <dir> --gt_json <json> --out_dir <d>
      --man_dir <d> --sr_dir <d> --stats_md <md> --snippet <txt>
"""

import argparse
import json
import os
import shutil

import numpy as np

from reanalyze_v2 import METRICS, care_gt_presence, holm_adjust
from aggregate_v2_seeds import (CLASSES, MULTI, SEEDS, _fmt_p_tex, fmt_p,
                                agg_over_seeds, comparisons_for,
                                comparisons_sup_vs_others, load_and_analyze, log,
                                resolve_mt_stems, run_comparison, run_families)

METHOD_SHORT = {"semi": "ST", "mt": "MT", "cps": "CPS", "bcp": "BCP", "ours": "Ours"}
CROSS_METHODS = ["semi", "mt", "cps", "ours", "bcp"]   # 全部半监督方法
SECONDARY_TOST = ["semi", "mt", "cps", "bcp"]          # M3 secondary（primary 为 ours）
DELTAS = [0.01, 0.02, 0.03]                             # 1/2/3 个 Dice 点
DELTA_PRIMARY = 0.02
REF = "sup_100"
REF_LABEL = "Supervised-100\\%"
SAME_RATIOS_FULL = [1, 5, 10, 25, 50]
CROSS_RATIOS_FULL = [5, 25, 50]   # 2026-10-15：bcp_5 补跑完成，F3 纳入 5% 档

FAMILY_DEFS = [
    ("F1", "各方法 vs Ours 同档（tabS8/tabS5）：每个 (class, metric, ratio) 一族，"
           "5 个比较 {sup, semi, mt, cps, bcp} vs ours"),
    ("F2", "Sup vs 各半监督同档（tabS8b）：每个 (class, metric, ratio) 一族，"
           "4 个比较 sup vs {semi, mt, cps, bcp}；sup_50 vs sup_100 为单一比较族"),
    ("F3", "交叉比例 vs Sup-100（tabS7b，本脚本新增）：每个 metric 一族，"
           "方法数(5: semi/mt/cps/ours/bcp) × 档位数(3: 5%/25%/50%) × 类别数(2) "
           "= 30 个比较（2026-10-15 起含 5% 档，bcp 低比例补跑完成后纳入）"),
]


# ---------------------------------------------------------------- 通用工具

def fmt_pp(v):
    """0-1 Dice 差 → 带符号百分点字符串。"""
    return f"{v * 100:+.2f}"


def fmt_ci_pp(ci):
    return f"[{fmt_pp(ci[0])}, {fmt_pp(ci[1])}]"


def paired_diff_cube(per_seed_cv, name_a, name_b, cls, metric="dice"):
    """患者级配对差值矩阵 D[seed, case] = A - B（同种子配对，病例取所有种子交集）。"""
    by_a = {s.rsplit("_s", 1)[-1]: cv for s, cv in per_seed_cv[name_a].items()}
    by_b = {s.rsplit("_s", 1)[-1]: cv for s, cv in per_seed_cv[name_b].items()}
    seeds = sorted(set(by_a) & set(by_b))
    assert seeds, f"{name_a} vs {name_b}: 无共同种子"
    cases = None
    for sd in seeds:
        cs = set(by_a[sd][cls][metric]) & set(by_b[sd][cls][metric])
        cases = cs if cases is None else (cases & cs)
    cases = sorted(cases)
    D = np.array([[by_a[sd][cls][metric][c] - by_b[sd][cls][metric][c]
                   for c in cases] for sd in seeds])
    return D, seeds, cases


# ---------------------------------------------------------------- M1 交叉比例检验

def run_cross_ratio_tests(per_seed_cv, methods, ratios):
    """族 F3：methods × ratios × classes vs sup_100，逐种子 Wilcoxon → max-p → Holm。

    返回 {metric: {f"{cls}/{method}_{r}": res|None}}；None = 该档位缺失（NA）。
    Holm 在每个 metric 族（≤ 方法数×档位数×类别数 个非 NA 比较）内进行。
    """
    results = {}
    for m in METRICS:
        entries = []
        for cls in CLASSES:
            for method in methods:
                for r in ratios:
                    key = f"{cls}/{method}_{r}"
                    name = f"{method}_{r}"
                    if name in per_seed_cv and REF in per_seed_cv:
                        entries.append((key, run_comparison(per_seed_cv, name, REF, cls, m)))
                    else:
                        entries.append((key, None))
        adj = holm_adjust(np.array([res["p_max"] for _, res in entries
                                    if res is not None]))
        it = iter(adj)
        fam = {}
        for key, res in entries:
            if res is None:
                fam[key] = None
            else:
                res["holm_p"] = float(next(it))
                fam[key] = res
        results[m] = fam
    return results


# ---------------------------------------------------------------- MT 修复前后对比

def build_mt_repair_comparison(per_seed_summary, per_seed_cv, old_summary, old_cv,
                               multi, cross_methods, cross_ratios, new_cross):
    """同一运行内重算旧/新 MT 的 F1/F2/F3，比较 Holm 显著性与均值差方向。

    old_* 只含旧 mt_{r} 实验；new per_seed_* 的 MT 已来自 mtfixed。
    返回均值聚合与全部 MT 相关检验的翻转清单（两类 × 三指标）。
    """
    old_scenario = dict(per_seed_cv)
    for r in multi.get("mt", []):
        old_scenario[f"mt_{r}"] = old_cv[f"mt_{r}"]
    old_f1 = run_families(old_scenario, multi, comparisons_for)
    new_f1 = run_families(per_seed_cv, multi, comparisons_for)
    old_f2 = run_families(old_scenario, multi, comparisons_sup_vs_others)
    new_f2 = run_families(per_seed_cv, multi, comparisons_sup_vs_others)
    old_f3 = run_cross_ratio_tests(old_scenario, cross_methods, cross_ratios)

    means = {}
    for r in multi.get("mt", []):
        name = f"mt_{r}"
        means[r] = {"old": agg_over_seeds(old_summary[name], CLASSES),
                    "new": agg_over_seeds(per_seed_summary[name], CLASSES)}

    flips = []
    n_tests = 0

    def direction(res):
        d = res["mean_a"] - res["mean_b"]
        return "+" if d > 0 else ("-" if d < 0 else "0")

    def check(family, key, old_res, new_res, cls, metric):
        nonlocal n_tests
        if old_res is None or new_res is None:
            return
        n_tests += 1
        old_sig = old_res["holm_p"] < 0.05
        new_sig = new_res["holm_p"] < 0.05
        old_dir, new_dir = direction(old_res), direction(new_res)
        if old_sig != new_sig or old_dir != new_dir:
            flips.append({"family": family, "comparison": key,
                          "class": cls, "metric": metric,
                          "old": old_res, "new": new_res,
                          "old_sig": old_sig, "new_sig": new_sig,
                          "old_dir": old_dir, "new_dir": new_dir})

    for cls in CLASSES:
        for metric in METRICS:
            cm = f"{cls}/{metric}"
            for r in multi.get("mt", []):
                check("F1", f"{r}%/mt vs ours",
                      old_f1[cm].get(f"{r}%/mt vs ours"),
                      new_f1[cm].get(f"{r}%/mt vs ours"), cls, metric)
                check("F2", f"{r}%/sup vs mt",
                      old_f2[cm].get(f"{r}%/sup vs mt"),
                      new_f2[cm].get(f"{r}%/sup vs mt"), cls, metric)
            for r in cross_ratios:
                check("F3", f"{r}%/mt vs {REF}", old_f3[metric].get(f"{cls}/mt_{r}"),
                      new_cross[metric].get(f"{cls}/mt_{r}"), cls, metric)
    return {"means": means, "flips": flips, "n_tests": n_tests}


def build_mt_repair_md(mt_repair):
    """MT 对齐修复前后对比 markdown 节（含翻转点清单）。"""
    if not mt_repair:
        return []
    lines = ["## MT 对齐修复前后对比", "",
             "旧 `mt_*` vs 新 `mtfixed_*`，三种子患者级 Dice mean±std（%）。", "",
             "| Ratio | 旧 MT tumor | 新 MT tumor | 旧 MT rectum | 新 MT rectum |",
             "|---|---|---|---|---|"]
    for r, pair in mt_repair["means"].items():
        cells = []
        for cls in ("tumor", "rectum"):
            for version in ("old", "new"):
                d = pair[version]["classes"][cls]["metrics"]["dice"]
                cells.append(f"{d['mean'] * 100:.2f}±{(d['std'] or 0) * 100:.2f}")
        lines.append(f"| {r}% | " + " | ".join(cells) + " |")
    sig_flips = [f for f in mt_repair["flips"] if f["old_sig"] != f["new_sig"]]
    dir_only = [f for f in mt_repair["flips"] if f["old_sig"] == f["new_sig"]]
    main_sig = [f for f in sig_flips if f["family"] in ("F1", "F2")]
    main_dir = [f for f in dir_only if f["family"] in ("F1", "F2")]
    f3_sig = [f for f in sig_flips if f["family"] == "F3"]
    if main_sig or main_dir:
        conclusion = (
            f"**修复改变了主要比较结论**：Ours vs MT（F1）与 sup vs MT（F2）共有 "
            f"{len(main_sig)} 项显著性翻转、{len(main_dir)} 项仅方向反转；"
            f"F3 另有 {len(f3_sig)} 项显著性翻转，正文措辞需同步修订。")
    elif f3_sig:
        conclusion = ("Ours vs MT（F1）与 sup vs MT（F2）的主要比较结论不变，"
                      f"但 F3 有 {len(f3_sig)} 项显著性翻转。")
    else:
        conclusion = ("修复未改变主要比较结论：Ours vs MT（F1）与 sup vs MT（F2）"
                      "的方向和显著性在所有档位、类别与指标上均不变。")
    lines += ["", conclusion, ""]

    def render(f):
        old, new = f["old"], f["new"]
        return (f"- **{f['family']} {f['class']}/{f['metric']} "
                f"{f['comparison']}**：显著性 "
                f"{'显著' if f['old_sig'] else '不显著'}→"
                f"{'显著' if f['new_sig'] else '不显著'}"
                f"（Holm p {fmt_p(old['holm_p'])}→{fmt_p(new['holm_p'])}），"
                f"Δ={fmt_pp(old['mean_a'] - old['mean_b'])}→"
                f"{fmt_pp(new['mean_a'] - new['mean_b'])}pp"
                f"（方向 {f['old_dir']}→{f['new_dir']}）。")

    if sig_flips:
        lines += [f"**显著性翻转点清单（F1/F2/F3，共 {len(sig_flips)} 项；"
                  "方向按 A−B 的符号）：**"]
        lines += [render(f) for f in sig_flips]
    if dir_only:
        lines += ["", f"**仅方向反转（显著性不变，共 {len(dir_only)} 项）：**"]
        lines += [render(f) for f in dir_only]
    if not mt_repair["flips"]:
        lines.append(f"翻转点清单：**无翻转**（共核对 {mt_repair['n_tests']} 项 "
                     "MT 相关 F1/F2/F3 检验，含两类 Dice/IoU/NSD）。")
    lines.append("")
    return lines


# ---------------------------------------------------------------- M3 TOST

def twoway_cluster_boot_samples(D, n_boot, rng):
    """同时重抽样训练种子与测试患者，返回每次 bootstrap 的总体配对差值均值。"""
    ns, nc = D.shape
    s_idx = rng.integers(0, ns, size=(n_boot, ns))
    c_idx = rng.integers(0, nc, size=(n_boot, nc))
    return D[s_idx[:, :, None], c_idx[:, None, :]].mean(axis=(1, 2))


def tost_analysis(D, seeds, n_boot, rng, deltas=DELTAS):
    """患者级配对 Dice 差的探索性 TOST。D: (n_seeds, n_cases)。

    per-seed CI 仅重抽样该种子内患者。pooled 保留旧版仅患者重抽样结果作敏感性
    参照；正式等效判定、TOST p 和 pooled_twoway CI 均同时重抽样患者与种子，
    与效应量表的不确定性口径一致。等效 = pooled_twoway 90%CI ⊆ (-δ, +δ)。
    """
    ns, nc = D.shape
    per_seed = {}
    for i, sd in enumerate(seeds):
        d = D[i]
        idx = rng.integers(0, nc, size=(n_boot, nc))
        boots = d[idx].mean(axis=1)
        per_seed[sd] = {"mean": float(d.mean()),
                        "ci90": [float(np.percentile(boots, 5)),
                                 float(np.percentile(boots, 95))],
                        "ci95": [float(np.percentile(boots, 2.5)),
                                 float(np.percentile(boots, 97.5))]}
    idx = rng.integers(0, nc, size=(n_boot, nc))
    boots_patient = D[:, idx].mean(axis=(0, 2))
    pooled = {"mean": float(D.mean()),
              "ci90": [float(np.percentile(boots_patient, 5)),
                       float(np.percentile(boots_patient, 95))],
              "ci95": [float(np.percentile(boots_patient, 2.5)),
                       float(np.percentile(boots_patient, 97.5))]}
    boots_twoway = twoway_cluster_boot_samples(D, n_boot, rng)
    pooled_twoway = {
        "mean": float(D.mean()),
        "ci90": [float(np.percentile(boots_twoway, 5)),
                 float(np.percentile(boots_twoway, 95))],
        "ci95": [float(np.percentile(boots_twoway, 2.5)),
                 float(np.percentile(boots_twoway, 97.5))]}
    tost = {}
    for dlt in deltas:
        p_lo = float((boots_twoway <= -dlt).mean())
        p_hi = float((boots_twoway >= dlt).mean())
        lo90, hi90 = pooled_twoway["ci90"]
        tost[f"{dlt * 100:.0f}pp"] = {
            "delta": dlt, "p_lower": p_lo, "p_upper": p_hi,
            "tost_p": max(p_lo, p_hi),
            "equivalent": bool(lo90 > -dlt and hi90 < dlt),
            "non_inferior": bool(lo90 > -dlt)}
    return {"n_seeds": int(ns), "n_cases": int(nc),
            "diff_per_seed_mean": {sd: float(D[i].mean()) for i, sd in enumerate(seeds)},
            "per_seed": per_seed, "pooled": pooled,
            "pooled_twoway": pooled_twoway, "tost": tost}


def run_tost(per_seed_cv, primary_method, secondary_methods, ratio, n_boot, seed,
             extra_secondary=()):
    """primary = {primary_method}_{ratio} vs sup_100；secondary 同档其余方法 +
    extra_secondary 里的 (method, ratio) 追加组（如 bcp@5%）。Dice only。"""
    rng = np.random.default_rng(seed)
    out = {"primary": {}, "secondary": {}}

    def one(name):
        res = {}
        for cls in CLASSES:
            D, seeds, _ = paired_diff_cube(per_seed_cv, name, REF, cls)
            res[cls] = tost_analysis(D, seeds, n_boot, rng)
        return res

    prim = f"{primary_method}_{ratio}"
    out["primary"][f"{prim} vs {REF}"] = one(prim)
    pairs = [(m, ratio) for m in secondary_methods] + list(extra_secondary)
    for m, r in pairs:
        name = f"{m}_{r}"
        if name in per_seed_cv:
            out["secondary"][f"{name} vs {REF}"] = one(name)
        else:
            out["secondary"][f"{name} vs {REF}"] = None   # NA
    return out


# ---------------------------------------------------------------- M6 效应量

def twoway_cluster_boot_ci(D, n_boot, rng):
    """双路 cluster bootstrap 95%CI：同时重抽样种子（axis 0）与患者（axis 1）。"""
    boots = twoway_cluster_boot_samples(D, n_boot, rng)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def effect_size_entry(D, seeds, n_boot, rng):
    return {"n_seeds": int(D.shape[0]), "n_cases": int(D.shape[1]),
            "diff_mean": float(D.mean()),
            "diff_per_seed_mean": {sd: float(D[i].mean())
                                   for i, sd in enumerate(seeds)},
            "ci95_twoway_boot": twoway_cluster_boot_ci(D, n_boot, rng)}


def build_effect_sizes(per_seed_cv, same_ratios, cross_methods, cross_ratios,
                       n_boot, seed):
    """三组比较：G1 同档 Ours vs ST；G2 同档 Sup vs 各半监督；G3 交叉比例 vs Sup-100。"""
    rng = np.random.default_rng(seed)
    groups = {g: {cls: {} for cls in CLASSES}
              for g in ("ours_vs_st_same_ratio", "sup_vs_semis_same_ratio",
                        "cross_ratio_vs_sup100")}

    def add(group, cls, key, a, b):
        if a in per_seed_cv and b in per_seed_cv:
            D, seeds, _ = paired_diff_cube(per_seed_cv, a, b, cls)
            groups[group][cls][key] = effect_size_entry(D, seeds, n_boot, rng)

    for r in same_ratios:
        for cls in CLASSES:
            add("ours_vs_st_same_ratio", cls, f"{r}%", f"ours_{r}", f"semi_{r}")
            for m in CROSS_METHODS:
                add("sup_vs_semis_same_ratio", cls, f"{r}%/{m}", f"sup_{r}", f"{m}_{r}")
    for r in cross_ratios:
        for cls in CLASSES:
            for m in cross_methods:
                add("cross_ratio_vs_sup100", cls, f"{r}%/{m}", f"{m}_{r}", REF)
    return groups


# ---------------------------------------------------------------- 输出：LaTeX 表

def build_tabS7b_tex(cross, methods, ratios):
    """M1 交叉比例表（风格参照 tabS8b_sup_vs_others.tex）。Dice 主终点。"""
    d = cross["dice"]
    lines = [
        "% CARE 补充表：交叉比例检验（M1）。各半监督方法 5%/25%/50% 档 vs Supervised-100% 的"
        "患者级配对 Wilcoxon p（v3 三种子协议：逐种子 p 取最大；Holm 校正在新族 F3 内，"
        "F3 = 每 metric 5 方法 × 3 档位 × 2 类别 = 30 个比较）。"
        "由 src/aggregate_v4_rebuttal.py 生成",
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Cross-ratio comparison against the 100\\%-labeled fully supervised "
        "baseline (CARE test set, 81 cases): patient-level paired two-sided Wilcoxon "
        "$p$-values for each semi-supervised method trained with 5\\%, 25\\% or 50\\% "
        "of the "
        "labels vs Supervised-100\\%. A separate test is performed per training seed "
        "(42/43/44) and the maximum $p$ across the three seeds is reported "
        "(conservative combination), Holm-adjusted within the cross-ratio family F3 of "
        "5 methods $\\times$ 3 ratios $\\times$ 2 classes $=$ 30 "
        "comparisons per metric (primary endpoint Dice shown). Bold: $p<0.05$. "
        "ST = pseudo-label self-training, MT = Mean Teacher.}",
        "\\label{tabS:cross_ratio}", "\\small",
        "\\begin{tabular}{llcc}", "\\toprule",
        "Comparison & Labeled & Tumor Dice & Rectum Dice \\\\",
        " & ratio & Holm $p$ & Holm $p$ \\\\", "\\midrule",
    ]
    for method in methods:
        avail = [r for r in ratios if d.get(f"tumor/{method}_{r}") is not None]
        if not avail:
            lines.append(f"{METHOD_SHORT[method]} vs {REF_LABEL} & -- & NA & NA \\\\")
            lines.append("\\midrule")
            continue
        for i, r in enumerate(avail):
            pt = _fmt_p_tex(d[f"tumor/{method}_{r}"]["holm_p"])
            pr = _fmt_p_tex(d[f"rectum/{method}_{r}"]["holm_p"])
            prefix = (f"\\multirow{{{len(avail)}}}{{*}}{{{METHOD_SHORT[method]} vs "
                      f"{REF_LABEL}}}") if i == 0 else ""
            lines.append(f"{prefix} & {r}\\% & {pt} & {pr} \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _eq_cell(tost_entry, dlt_key, with_p=False):
    t = tost_entry["tost"][dlt_key]
    s = "Yes" if t["equivalent"] else "No"
    if with_p:
        s += f" ({_fmt_p_tex(t['tost_p'])})"
    return s


def build_tost_tex(tost_res, ratio):
    """M3 探索性 TOST：逐种子差值 + 患者×种子双路 90%CI + 等效判定。"""
    lines = [
        "% CARE 补充表：探索性 TOST。等效判定与 p 值均使用患者×种子双路 bootstrap；",
        "% 旧版仅患者重抽样区间仅保留在 JSON 作敏感性参照，不用于等效判定。",
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Exploratory equivalence testing (two one-sided tests, TOST) for "
        "the listed configurations versus Supervised-100\\% (CARE test, 81 cases, "
        "three seeds 42/43/44). $\\Delta$ = method $-$ Supervised-100\\% in Dice "
        "points (pp). Per-seed mean differences and the pooled difference with a "
        "90\\% CI from 10{,}000 two-way cluster bootstrap replicates (resampling "
        "both test patients and training seeds) are shown. Equivalence is declared "
        "when this 90\\% CI lies entirely within $(-\\delta,+\\delta)$; the TOST "
        "$p$-value is given for $\\delta$ = 2pp. The margin was chosen post hoc, "
        "with 1pp/3pp sensitivity analyses. ST = pseudo-label self-training, "
        "MT = Mean Teacher.}",
        "\\label{tabS:tost}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llccccc}", "\\toprule",
        "Method & Class & $\\Delta$ per seed (pp) & Pooled $\\Delta$ [90\\% CI] & "
        "\\multicolumn{3}{c}{Equivalent?} \\\\",
        "\\cmidrule(lr){5-7}",
        " & & s42 / s43 / s44 & method $-$ Sup-100\\% & "
        "$\\delta$=1pp & $\\delta$=2pp & $\\delta$=3pp \\\\",
        "\\midrule",
    ]

    def rows_for(comp, a, primary=False):
        method, r = comp.split(" vs ")[0].rsplit("_", 1)
        label = f"{METHOD_SHORT[method]}-{r}\\%"
        if primary:
            label += " (primary)"
        out = []
        for i, cls in enumerate(CLASSES):
            if a is None:
                out.append(f"{label if i == 0 else ''} & {cls.capitalize()} & "
                           "NA & NA & NA & NA & NA \\\\")
                continue
            e = a[cls]
            e2 = e["pooled_twoway"]
            per = " / ".join(f"{e['diff_per_seed_mean'][str(sd)] * 100:+.2f}"
                             for sd in SEEDS)
            out.append(
                f"{label if i == 0 else ''} & {cls.capitalize()} & {per} & "
                f"{fmt_pp(e2['mean'])} {fmt_ci_pp(e2['ci90'])} & "
                f"{_eq_cell(e, '1pp')} & {_eq_cell(e, '2pp', with_p=True)} & "
                f"{_eq_cell(e, '3pp')} \\\\")
        return out

    for comp, a in tost_res["primary"].items():
        lines += rows_for(comp, a, primary=True)
        lines.append("\\midrule")
    for comp, a in tost_res["secondary"].items():
        lines += rows_for(comp, a)
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _es_cell(es):
    return f"{fmt_pp(es['diff_mean'])} {fmt_ci_pp(es['ci95_twoway_boot'])}"


def build_effect_sizes_tex(groups, same_ratios, cross_methods, cross_ratios):
    """M6 精选效应量表：A 同档 Ours vs ST；B 同档 Sup vs 各半监督（最高两档）；
    C 交叉比例 vs Sup-100。Δ 单位 Dice 点，95%CI 双路 cluster bootstrap。"""
    g1, g2, g3 = (groups["ours_vs_st_same_ratio"],
                  groups["sup_vs_semis_same_ratio"],
                  groups["cross_ratio_vs_sup100"])
    panel_b_ratios = sorted(same_ratios)[-2:]
    lines = [
        "% CARE 补充表：关键比较的效应量（M6）。患者级配对 Dice 差均值（Dice 点）+ 95%CI"
        "（双路 cluster bootstrap：同时重抽样患者×种子，10000 次），作为逐种子 Wilcoxon "
        "max-p 之外的层级敏感性分析。由 src/aggregate_v4_rebuttal.py 生成",
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Effect sizes for key comparisons (CARE test set, 81 cases, seeds "
        "42/43/44): patient-level paired Dice differences in Dice points (pp) with "
        "95\\% CIs from a two-way cluster bootstrap (10{,}000 replicates) that "
        "simultaneously resamples test patients and training seeds, propagating both "
        "patient-level and seed-level variability. This serves as a hierarchical "
        "sensitivity analysis complementing the per-seed Wilcoxon max-$p$ tests "
        "(the three seeds share the same test set and are strongly correlated). "
        "Positive $\\Delta$ favors the first-listed method. "
        "ST = pseudo-label self-training, MT = Mean Teacher.}",
        "\\label{tabS:effect_sizes}", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcc}", "\\toprule",
        "Comparison & Tumor $\\Delta$Dice (pp) [95\\% CI] & "
        "Rectum $\\Delta$Dice (pp) [95\\% CI] \\\\", "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Panel A: Ours vs ST (same labeled ratio)}} \\\\",
    ]
    for r in same_ratios:
        k = f"{r}%"
        if k in g1["tumor"]:
            lines.append(f"Ours $-$ ST, {r}\\% & {_es_cell(g1['tumor'][k])} & "
                         f"{_es_cell(g1['rectum'][k])} \\\\")
    lines += ["\\midrule",
              "\\multicolumn{3}{l}{\\textit{Panel B: Supervised vs semi-supervised "
              "(same labeled ratio)}} \\\\"]
    for m in cross_methods:
        for r in panel_b_ratios:
            k = f"{r}%/{m}"
            if k in g2["tumor"]:
                lines.append(f"Supervised $-$ {METHOD_SHORT[m]}, {r}\\% & "
                             f"{_es_cell(g2['tumor'][k])} & "
                             f"{_es_cell(g2['rectum'][k])} \\\\")
    lines += ["\\midrule",
              "\\multicolumn{3}{l}{\\textit{Panel C: semi-supervised at "
              "5\\%/25\\%/50\\% "
              "vs Supervised-100\\%}} \\\\"]
    for m in cross_methods:
        for r in cross_ratios:
            k = f"{r}%/{m}"
            if k in g3["tumor"]:
                lines.append(f"{METHOD_SHORT[m]}-{r}\\% $-$ Supervised-100\\% & "
                             f"{_es_cell(g3['tumor'][k])} & "
                             f"{_es_cell(g3['rectum'][k])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- 输出：markdown + snippet

def build_stats_md(cross, tost_res, groups, methods, cross_ratios, same_ratios,
                   n_boot, seed, bcp_low=None, mt_repair=None):
    """人读汇总：每个结论一句话 + 数字（均值/差值以 Dice 点表示）。"""
    d = cross["dice"]
    md = ["# v4 rebuttal 统计汇总（M1 交叉比例 / M3 TOST 等效性 / M6 效应量）", "",
          "由 `src/aggregate_v4_rebuttal.py` 生成；数据源 results_v2/*_per_slice.csv，"
          "患者级聚合协议与 aggregate_v2_seeds 完全一致（v2 协议，逐种子 42/43/44）。",
          "检验：逐种子患者级配对双侧 Wilcoxon → 3 种子 max-p（保守合并；三种子共享同一"
          "测试集 81 例、检验强相关）→ Holm 校正。TOST/效应量 CI：bootstrap "
          f"（n={n_boot}，seed={seed}）。均值/差值以 Dice 点（pp）表示。", "",
          "## 头条结论", ""]

    # --- M3 primary
    prim_comp, prim = next(iter(tost_res["primary"].items()))
    for cls in CLASSES:
        e = prim[cls]
        t2 = e["tost"]["2pp"]
        md.append(
            f"- **M3 primary（{prim_comp}，{cls}）**：pooled Δ = "
            f"{fmt_pp(e['pooled_twoway']['mean'])}pp，90%CI {fmt_ci_pp(e['pooled_twoway']['ci90'])}，"
            f"95%CI {fmt_ci_pp(e['pooled_twoway']['ci95'])}；δ=2pp TOST "
            f"**等效{'成立' if t2['equivalent'] else '不成立'}**（p={fmt_p(t2['tost_p'])}；"
            f"非劣{'成立' if t2['non_inferior'] else '不成立'}）；"
            f"敏感性 δ=1pp {'成立' if e['tost']['1pp']['equivalent'] else '不成立'}、"
            f"δ=3pp {'成立' if e['tost']['3pp']['equivalent'] else '不成立'}。")

    # --- M1 50% / 25% 打平清单
    for r in cross_ratios:
        ties, sigs = [], []
        for m in methods:
            res = d.get(f"tumor/{m}_{r}")
            if res is None:
                continue
            (ties if res["holm_p"] >= 0.05 else sigs).append(
                f"{METHOD_SHORT[m]}(p={fmt_p(res['holm_p'])})")
        md.append(
            f"- **M1 {r}% 档 vs Sup-100（tumor Dice，F3 Holm p）**：打平 "
            f"({len(ties)}): {', '.join(ties) if ties else '无'}；"
            f"显著差异 ({len(sigs)}): {', '.join(sigs) if sigs else '无'}。")
    md.append("")

    # --- M1 详表
    n_fam = 5 * len(cross_ratios) * 2
    md += [f"## M1 交叉比例检验详表（Dice；F3 族 = 5 方法 × {len(cross_ratios)} 档 "
           f"× 2 类 = {n_fam} 比较/metric）",
           ""]
    for cls in CLASSES:
        md += [f"### {cls} Dice", "",
               "| Method | Ratio | A mean | Sup-100 mean | p_s42 | p_s43 | p_s44 | "
               "max p | Holm p |", "|---|---|---|---|---|---|---|---|---|"]
        for m in methods:
            for r in cross_ratios:
                res = d.get(f"{cls}/{m}_{r}")
                if res is None:
                    md.append(f"| {METHOD_SHORT[m]} | {r}% | NA | NA | NA | NA | NA "
                              "| NA | NA |")
                    continue
                ps = [res["p_per_seed"].get(str(sd)) for sd in SEEDS]
                md.append(
                    f"| {METHOD_SHORT[m]} | {r}% | {res['mean_a'] * 100:.2f} | "
                    f"{res['mean_b'] * 100:.2f} | "
                    + " | ".join(fmt_p(p) if p is not None else "-" for p in ps)
                    + f" | {fmt_p(res['p_max'])} | {fmt_p(res['holm_p'])} |")
        md.append("")

    # --- M3 详表
    md += ["## M3 TOST 详表（Δ = 方法 − Sup-100，Dice 点；等效 = pooled 90%CI ⊆ (−δ,+δ)）",
           "",
           "| Comparison | Class | Δ s42 | Δ s43 | Δ s44 | pooled Δ | 90% CI | "
           "95% CI | TOST p (δ=2pp) | Eq δ=1 | Eq δ=2 | Eq δ=3 | NI δ=2 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def md_tost_rows(comp, a):
        for cls in CLASSES:
            if a is None:
                md.append(f"| {comp} | {cls} | NA | NA | NA | NA | NA | NA | NA "
                          "| NA | NA | NA |")
                continue
            e = a[cls]
            ds = [e["diff_per_seed_mean"][str(sd)] for sd in SEEDS]
            md.append(
                f"| {comp} | {cls} | " + " | ".join(fmt_pp(x) for x in ds)
                + f" | {fmt_pp(e['pooled_twoway']['mean'])} "
                + f"| {fmt_ci_pp(e['pooled_twoway']['ci90'])} "
                + f"| {fmt_ci_pp(e['pooled_twoway']['ci95'])} "
                + f"| {fmt_p(e['tost']['2pp']['tost_p'])} "
                + f"| {'Y' if e['tost']['1pp']['equivalent'] else 'N'} "
                + f"| {'Y' if e['tost']['2pp']['equivalent'] else 'N'} "
                + f"| {'Y' if e['tost']['3pp']['equivalent'] else 'N'} "
                + f"| {'Y' if e['tost']['2pp']['non_inferior'] else 'N'} |")

    for comp, a in tost_res["primary"].items():
        md_tost_rows(comp + " (primary)", a)
    for comp, a in tost_res["secondary"].items():
        md_tost_rows(comp, a)
    md.append("")

    # --- M6 详表
    md += ["## M6 效应量详表（患者级配对 Dice 差均值 [双路 cluster bootstrap 95%CI]，pp）",
           ""]
    for gname, glabel, note in [
            ("ours_vs_st_same_ratio", "G1 同档 Ours − ST", ""),
            ("sup_vs_semis_same_ratio", "G2 同档 Sup − 各半监督", ""),
            ("cross_ratio_vs_sup100", "G3 交叉比例：方法档 − Sup-100", "")]:
        md += [f"### {glabel}", "",
               "| Comparison | Tumor Δ [95% CI] | Rectum Δ [95% CI] |",
               "|---|---|---|"]
        for k in groups[gname]["tumor"]:
            md.append(f"| {k} | {_es_cell(groups[gname]['tumor'][k])} | "
                      f"{_es_cell(groups[gname]['rectum'][k])} |")
        md.append("")

    # --- BCP 低比例补充结果
    if bcp_low:
        agg_low, tests_low = bcp_low
        md += ["## BCP 低比例补充结果（bcp_1/bcp_5，三种子 mean±std，患者级 v2 协议）",
               "",
               "| Ratio | Sup | ST | MT | CPS | Ours | BCP |",
               "|---|---|---|---|---|---|---|"]
        for r in (1, 5):
            cells = []
            for m in ("sup", "semi", "mt", "cps", "ours", "bcp"):
                g = agg_low.get(f"{m}_{r}")
                if g is None:
                    cells.append("--")
                    continue
                d = g["classes"]["tumor"]["metrics"]["dice"]
                cells.append(f"{d['mean'] * 100:.2f}±{(d['std'] or 0) * 100:.2f}")
            md.append(f"| {r}%（tumor） | " + " | ".join(cells) + " |")
        for r in (1, 5):
            cells = []
            for m in ("sup", "semi", "mt", "cps", "ours", "bcp"):
                g = agg_low.get(f"{m}_{r}")
                if g is None:
                    cells.append("--")
                    continue
                d = g["classes"]["rectum"]["metrics"]["dice"]
                cells.append(f"{d['mean'] * 100:.2f}±{(d['std'] or 0) * 100:.2f}")
            md.append(f"| {r}%（rectum） | " + " | ".join(cells) + " |")
        md.append("")
        for r in (1, 5):
            semi_means = {m: agg_low[f"{m}_{r}"]["classes"]["tumor"]["metrics"]
                          ["dice"]["mean"]
                          for m in ("semi", "mt", "cps", "ours", "bcp")
                          if f"{m}_{r}" in agg_low}
            if semi_means:
                best = max(semi_means, key=semi_means.get)
                ranking = " > ".join(f"{METHOD_SHORT[m]}({semi_means[m] * 100:.2f})"
                                     for m in sorted(semi_means,
                                                     key=semi_means.get,
                                                     reverse=True))
                md.append(
                    f"- {r}% 档半监督肿瘤 Dice 均值排序：{ranking}；"
                    f"均值最高 = **{METHOD_SHORT[best]}**"
                    f"（'Ours 最高' 声称{'仍成立' if best == 'ours' else '不成立'}）。")
        for key, res in tests_low.items():
            md.append(
                f"- {key}（Dice，逐种子 max-p）：max p = {fmt_p(res['p_max'])}，"
                f"means = {res['mean_a'] * 100:.2f} vs {res['mean_b'] * 100:.2f}"
                f"（n={res['n_cases']}）。")
        md.append("")

    md += build_mt_repair_md(mt_repair)

    # --- 族定义
    md += ["## Holm 族定义清单（M6 ii）", ""]
    for fid, desc in FAMILY_DEFS:
        md.append(f"- **{fid}**：{desc}")
    md += ["",
           "方法学说明：三种子检验共享同一测试集（81 例），相互强相关，max-p 为保守合并；"
           "双路 cluster bootstrap（患者×种子）效应量作为层级敏感性分析。"
           "英文方法节文字见 paper/snippet_stats_methods.txt。", ""]
    return "\n".join(md)


def build_snippet():
    """补充材料方法节英文说明（Holm 族 F1/F2/F3 + 敏感性分析 + TOST）。"""
    return r"""% Statistical analysis snippet for the supplement (multiplicity control,
% effect sizes, equivalence testing). Generated by src/aggregate_v4_rebuttal.py.

\paragraph{Multiplicity control and comparison families.}
All inferential analyses are performed on patient-level metrics (per-patient means
over the scored slices) on the identical held-out CARE test set (81 cases). For each
comparison, a paired two-sided Wilcoxon signed-rank test is computed separately for
every training seed (42/43/44), and the maximum $p$-value across the three seeds is
reported. Because the three seed-wise tests share the same test set, they are strongly
positively correlated; taking the maximum $p$-value is a conservative combination
rule that remains valid under arbitrary dependence among the seeds. Holm adjustment
is then applied within pre-specified families of comparisons:
\textbf{F1} (same-ratio baselines vs Ours; Table~S8): for each
(class, metric, labeled ratio), the family comprises the 5 comparisons
\{Supervised, ST, MT, CPS, BCP\} vs Ours at that ratio;
\textbf{F2} (Supervised vs semi-supervised; Table~S8b): for each
(class, metric, labeled ratio), the 4 comparisons Supervised vs \{ST, MT, CPS,
BCP\}, with the Supervised-50\% vs Supervised-100\% comparison forming a
single-comparison family;
\textbf{F3} (cross-ratio vs Supervised-100\%; Table~S7b): each semi-supervised
method (ST, MT, CPS, Ours, BCP) trained with 5\%, 25\% or 50\% of the labels vs
the 100\%-labeled supervised baseline; per evaluation metric the family comprises
5 methods $\times$ 3 ratios $\times$ 2 classes $=$ 30 comparisons.

\paragraph{Effect sizes (hierarchical sensitivity analysis).}
Because max-$p$ pooling over only three correlated seeds is conservative, and
$p$-values alone do not quantify magnitude, we additionally report effect sizes:
patient-level paired Dice differences (first-listed method minus second), averaged
over patients and seeds, with 95\% confidence intervals from a two-way cluster
bootstrap (10{,}000 replicates) that simultaneously resamples test patients
(81 cases) and training seeds (3). This propagates both patient-level and
seed-level variability into the interval estimate and serves as a hierarchical
sensitivity analysis complementing the max-$p$ significance tests
(Table~S\ref{tabS:effect_sizes}).

\paragraph{Exploratory equivalence and non-inferiority testing (TOST).}
For the key question of whether a semi-supervised method trained with 50\% of the
labels approaches the 100\%-label supervised baseline, we apply two one-sided tests
(TOST) to patient-level paired Dice differences. The post-hoc equivalence margin is
$\delta = 2$ Dice points, with $\delta \in \{1, 3\}$ points as sensitivity analyses.
The 90\% and 95\% percentile intervals use the same two-way cluster bootstrap as the
effect-size analysis (10{,}000 replicates, resampling both test patients and training
seeds), so that training variability is not excluded from the equivalence claim.
Equivalence is declared when this two-way 90\% CI lies entirely within
$(-\delta, +\delta)$, and non-inferiority when its lower bound exceeds $-\delta$.
These analyses are exploratory rather than confirmatory. The primary comparison is
Ours-50\% vs Supervised-100\% (tumor and normal rectum); the remaining
semi-supervised methods at 50\% labels are secondary comparisons.
"""


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_v2")
    ap.add_argument("--out_dir", default=os.path.join("results", "v2"))
    ap.add_argument("--man_dir", default=os.path.join("paper", "manuscript"))
    ap.add_argument("--sr_dir", default=os.path.join("paper", "sr"))
    ap.add_argument("--stats_md", default=os.path.join("paper", "v4_stats.md"))
    ap.add_argument("--snippet", default=os.path.join("paper",
                                                    "snippet_stats_methods.txt"))
    ap.add_argument("--gt_json", default=None,
                    help="默认 data/cache CARE test GT；冒烟时传 {name: [has_r, has_t]}")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="迷你实验集（sup/semi/mt/cps/ours@1%% + sup_100；bcp 缺失→NA）")
    args = ap.parse_args()

    for d in (args.out_dir, args.man_dir, args.sr_dir):
        os.makedirs(d, exist_ok=True)

    if args.gt_json:
        with open(args.gt_json) as f:
            gt = {k: tuple(v) for k, v in json.load(f).items()}
    else:
        gt = care_gt_presence()
    log(f"GT 加载：{len(gt)} 切片（source={args.gt_json or 'data/cache/test_labels.npy'}）")

    if args.smoke:
        multi = {"sup": [1, 100], "semi": [1], "mt": [1], "cps": [1], "ours": [1]}
        same_ratios, cross_ratios = [1], [1]
    else:
        multi, same_ratios, cross_ratios = MULTI, SAME_RATIOS_FULL, CROSS_RATIOS_FULL
    tost_ratio = max(cross_ratios)
    # F3 最低档的 bcp 追加进 TOST secondary（2026-10-15 bcp 低比例补跑后）
    tost_extra = [("bcp", min(cross_ratios))]

    use_mtfixed, mt_note = resolve_mt_stems(multi, args.results_dir)
    log(f"MT 数据源：{mt_note}")
    stems = {}
    for method, ratios in multi.items():
        for r in ratios:
            base = "mtfixed" if (method == "mt" and use_mtfixed) else method
            stems[f"{method}_{r}"] = [f"{base}_{r}_s{sd}" for sd in SEEDS]
    log(f"=== 加载并逐种子分析 {len(stems)} 个实验（复用 aggregate_v2_seeds 协议） ===")
    per_seed_summary, per_seed_cv = load_and_analyze(args.results_dir, stems, gt,
                                                     CLASSES)

    # ---- M1 交叉比例检验（族 F3）
    log("=== M1：交叉比例检验（F3：方法×档位×类别 vs sup_100，max-p + Holm） ===")
    cross = run_cross_ratio_tests(per_seed_cv, CROSS_METHODS, cross_ratios)
    n_fam = sum(1 for v in cross["dice"].values() if v is not None)
    cross_json = {
        "protocol": "patient-level paired two-sided Wilcoxon per seed (same-seed "
                    "pairing); combined p = max over seeds (conservative: seeds "
                    "share the same test set); Holm within family F3",
        "family_definition": ("F3 (per metric): 5 semi-supervised methods "
                              "(semi/mt/cps/ours/bcp) x 3 labeled ratios "
                              f"({'/'.join(f'{r}%' for r in cross_ratios)}) x 2 classes "
                              f"= {5 * len(cross_ratios) * 2} comparisons vs sup_100; "
                              f"non-NA comparisons here: {n_fam}"),
        "reference": REF, "seeds": SEEDS,
        "methods": CROSS_METHODS, "ratios": cross_ratios,
        "results": cross}
    with open(os.path.join(args.out_dir, "cross_ratio_tests.json"), "w") as f:
        json.dump(cross_json, f, indent=2)
    log(f"已保存 {os.path.join(args.out_dir, 'cross_ratio_tests.json')}")

    # ---- M3 TOST
    log("=== M3：探索性 TOST（患者×种子双路 cluster bootstrap） ===")
    tost_res = run_tost(per_seed_cv, "ours", SECONDARY_TOST, tost_ratio,
                        args.n_boot, args.seed, extra_secondary=tost_extra)
    tost_json = {
        "protocol": "exploratory patient-level paired Dice differences (method - "
                    "sup_100); per-seed and pooled diff means; equivalence decisions, "
                    "TOST p-values and pooled_twoway 90%/95% CIs use a two-way "
                    "cluster bootstrap resampling both patients and seeds; the legacy "
                    "pooled field resamples patients only and is sensitivity output, "
                    f"not the equivalence basis; n_boot={args.n_boot}, seed={args.seed}; "
                    "equivalence = pooled_twoway 90% CI within (-delta, +delta); "
                    "non-inferiority = lower bound of pooled_twoway 90% CI > -delta; "
                    "the 2-pp margin is post hoc",
        "delta_primary_pp": DELTA_PRIMARY * 100,
        "deltas_pp": [d * 100 for d in DELTAS],
        "primary_comparison": f"ours_{tost_ratio} vs {REF}",
        "results": tost_res}
    with open(os.path.join(args.out_dir, "equivalence_tost.json"), "w") as f:
        json.dump(tost_json, f, indent=2)
    log(f"已保存 {os.path.join(args.out_dir, 'equivalence_tost.json')}")

    # ---- M6 效应量
    log("=== M6：效应量（双路 cluster bootstrap：患者×种子） ===")
    groups = build_effect_sizes(per_seed_cv, same_ratios, CROSS_METHODS,
                                cross_ratios, args.n_boot, args.seed)
    es_json = {
        "protocol": "patient-level paired Dice difference mean (first-listed minus "
                    "second), averaged over seeds and cases; 95% CI = two-way "
                    "cluster bootstrap resampling patients x seeds simultaneously, "
                    f"n_boot={args.n_boot}, seed={args.seed}. NOTE: the three seeds "
                    "share the same test set, so the per-seed max-p tests are "
                    "strongly correlated; this two-way bootstrap is a hierarchical "
                    "sensitivity analysis to the max-p significance tests.",
        "groups": groups}
    with open(os.path.join(args.out_dir, "effect_sizes.json"), "w") as f:
        json.dump(es_json, f, indent=2)
    log(f"已保存 {os.path.join(args.out_dir, 'effect_sizes.json')}")

    # ---- LaTeX 表（manuscript + sr 两处）
    texs = {"tabS7b_cross_ratio.tex": build_tabS7b_tex(cross, CROSS_METHODS,
                                                       cross_ratios),
            "tab_equiv_tost.tex": build_tost_tex(tost_res, tost_ratio),
            "tab_effect_sizes.tex": build_effect_sizes_tex(groups, same_ratios,
                                                           CROSS_METHODS,
                                                           cross_ratios)}
    for fname, content in texs.items():
        path = os.path.join(args.man_dir, fname)
        with open(path, "w") as f:
            f.write(content)
        shutil.copyfile(path, os.path.join(args.sr_dir, fname))
        log(f"已保存 {path}（并复制到 {args.sr_dir}/）")

    # ---- BCP 低比例补充数据（bcp_1/bcp_5 入表后的汇总与声称核验）
    bcp_low = None
    low_names = [f"{m}_{r}" for r in (1, 5)
                 for m in ("sup", "semi", "mt", "cps", "ours", "bcp")]
    if any(n in per_seed_summary for n in low_names):
        agg_low = {n: agg_over_seeds(per_seed_summary[n], CLASSES)
                   for n in low_names if n in per_seed_summary}
        tests_low = {}
        for r in (1, 5):
            for a, b in (("bcp", "ours"), ("bcp", "semi"), ("bcp", "sup")):
                na, nb = f"{a}_{r}", f"{b}_{r}"
                if na in per_seed_cv and nb in per_seed_cv:
                    for cls in CLASSES:
                        tests_low[f"{na} vs {nb}/{cls}"] = run_comparison(
                            per_seed_cv, na, nb, cls, "dice")
        bcp_low = (agg_low, tests_low)

    # ---- MT 对齐修复前后对比（只有 gate 打开时才需要对比旧/新数据源）
    mt_repair = None
    if use_mtfixed:
        old_stems = {f"mt_{r}": [f"mt_{r}_s{sd}" for sd in SEEDS]
                     for r in multi.get("mt", [])}
        log("=== MT 修复前后对比：加载旧 mt_* 并重算 F1/F2/F3 ===")
        old_summary, old_cv = load_and_analyze(args.results_dir, old_stems, gt,
                                               CLASSES)
        mt_repair = build_mt_repair_comparison(
            per_seed_summary, per_seed_cv, old_summary, old_cv, multi,
            CROSS_METHODS, cross_ratios, cross)
        log(f"MT 修复翻转检测：{len(mt_repair['flips'])} 项翻转 / "
            f"{mt_repair['n_tests']} 项检验")
        for f in mt_repair["flips"]:
            log(f"  翻转 {f['family']} {f['class']}/{f['metric']} "
                f"{f['comparison']}: Holm p {f['old']['holm_p']:.4g}→"
                f"{f['new']['holm_p']:.4g}, 方向 {f['old_dir']}→{f['new_dir']}")

    # ---- markdown 汇总 + 方法节 snippet
    md = build_stats_md(cross, tost_res, groups, CROSS_METHODS, cross_ratios,
                        same_ratios, args.n_boot, args.seed, bcp_low=bcp_low,
                        mt_repair=mt_repair)
    with open(args.stats_md, "w") as f:
        f.write(md + "\n")
    log(f"已保存 {args.stats_md}")
    with open(args.snippet, "w") as f:
        f.write(build_snippet())
    log(f"已保存 {args.snippet}")

    # ---- 终端校验报告：关键结论数字
    log("\n=== 校验报告：关键结论 ===")
    prim_comp, prim = next(iter(tost_res["primary"].items()))
    for cls in CLASSES:
        e = prim[cls]
        t2 = e["tost"]["2pp"]
        log(f"M3 primary {prim_comp} [{cls}]: pooled Δ = "
            f"{fmt_pp(e['pooled_twoway']['mean'])}pp, 90%CI {fmt_ci_pp(e['pooled_twoway']['ci90'])}, "
            f"95%CI {fmt_ci_pp(e['pooled_twoway']['ci95'])}, "
            f"δ=2pp 等效={'成立' if t2['equivalent'] else '不成立'} "
            f"(p={t2['tost_p']:.4g}), 非劣={'成立' if t2['non_inferior'] else '不成立'}; "
            f"δ=1pp {'Y' if e['tost']['1pp']['equivalent'] else 'N'}/"
            f"δ=3pp {'Y' if e['tost']['3pp']['equivalent'] else 'N'}")
    for comp, a in tost_res["secondary"].items():
        if a is None:
            log(f"M3 secondary {comp}: NA")
            continue
        e = a["tumor"]
        t2 = e["tost"]["2pp"]
        log(f"M3 secondary {comp} [tumor]: pooled Δ = "
            f"{fmt_pp(e['pooled_twoway']['mean'])}pp, 90%CI {fmt_ci_pp(e['pooled_twoway']['ci90'])}, "
            f"δ=2pp 等效={'成立' if t2['equivalent'] else '不成立'} "
            f"(p={t2['tost_p']:.4g})")
    dd = cross["dice"]
    for r in cross_ratios:
        for m in CROSS_METHODS:
            res = dd.get(f"tumor/{m}_{r}")
            if res is None:
                log(f"M1 {m}_{r} vs sup_100 [tumor]: NA")
                continue
            resr = dd[f"rectum/{m}_{r}"]
            log(f"M1 {m}_{r} vs sup_100: tumor Δ="
                f"{(res['mean_a'] - res['mean_b']) * 100:+.2f}pp "
                f"Holm p={res['holm_p']:.4g} | rectum Δ="
                f"{(resr['mean_a'] - resr['mean_b']) * 100:+.2f}pp "
                f"Holm p={resr['holm_p']:.4g}")


if __name__ == "__main__":
    main()
