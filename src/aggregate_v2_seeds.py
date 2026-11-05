"""v3 三种子统计汇总（rebuttal 最终交付）：对 results_v2/ 的多 seed 实验做患者级聚合。

协议（与 reanalyze_v2 完全一致，仅多"跨种子"一层）：
1. 每个 (method, ratio, seed)：v2 协议患者级均值（前景指标仅在 GT 含该类的切片上计算，
   病例内取均值）+ FP 率（GT 阴性切片中预测非空比例）；复用 reanalyze_v2.analyze_tag。
2. 主表：3 种子患者级均值的 mean±std（std 为样本标准差 ddof=1）。
3. 检验：每个种子分别做患者级配对双侧 Wilcoxon，3 种子 p 取最大（保守合并），
   Holm 校正在每个 (class, metric, ratio) family 内进行
   （family = {sup,semi,mt,cps,bcp} vs ours 同档比较）。
4. 头条比较 ours_50 vs sup_100 单独报告（每种子 p + max p）。
5. 单种子参照（sup180_*/mtcons_50/nnunet_sup_*）只做数值呈现，表中标 †，不参与检验。
6. bcp 已覆盖 1/5/10/25/50 全档（runs_v2/bcp_{1,5}_s* 补跑完成，2026-10-15）。
7. MT 替换规则（mtfixed，2026-10-15）：mtfixed_{r}_s{s} 是 MT 的修复重跑。
   仅当 multi["mt"] 全部档位 × 全部种子的 mtfixed summary+csv 都齐全时，
   MT 实验名 mt_{r} 的 stem 才整体切换为 mtfixed_{r}_s{s}（全有或全无，
   避免三种子 mean±std 混入不同训练轮次）；未齐则用旧 mt_* 并在日志/md 标注。

产出：
  results/v2/agg_seeds.json            全部实验的逐种子 + 聚合指标
  results/v2/agg_tests.json            全部 Wilcoxon（逐种子 p / max p / Holm p）
  paper/manuscript/table1_care_v3.tex  主表（肿瘤表 + 直肠表，Dice mean±std）
  paper/manuscript/tabS5_wilcoxon_care_v3.tex  头条 + 同档检验（含 BCP 全档）
  paper/manuscript/tabS8b_sup_vs_others.tex    Supervised 与各半监督同档检验
  paper/v3_stats.md                    可读统计报告
正式运行时三张表同步复制到 paper/sr/，避免不同稿件目录使用不同统计版本。

用法：python src/aggregate_v2_seeds.py
冒烟：--smoke --results_dir <dir> --gt_json <json> --out_dir <dir> --man_dir <dir>
      --sr_dir <dir> --stats_md <md>
"""

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from reanalyze_v2 import (METRICS, analyze_tag, care_gt_presence, holm_adjust,
                          validate, wilcoxon_p)

SEEDS = [42, 43, 44]
CLASSES = ("rectum", "tumor")

# 三种子实验：method -> ratios（s1 为 stage-1 中间产物，不入表）
MULTI = {"sup": [1, 5, 10, 25, 50, 100],
         "semi": [1, 5, 10, 25, 50],
         "mt": [1, 5, 10, 25, 50],
         "cps": [1, 5, 10, 25, 50],
         "bcp": [1, 5, 10, 25, 50],
         "ours": [1, 5, 10, 25, 50]}
METHOD_LABEL = {"sup": "Supervised", "semi": "Pseudo-label ST",
                "mt": "Mean Teacher", "cps": "CPS", "bcp": "BCP", "ours": "Ours"}
# 单种子参照：(文件 stem, 表内标签, ratio)
SINGLE = [("sup180_50_s42", "Supervised (180 ep)", 50),
          ("sup180_100_s42", "Supervised (180 ep)", 100),
          ("mtcons_50_s42", "MT + consistency filter", 50),
          ("nnunet_sup_50", "nnU-Net", 50),
          ("nnunet_sup_100", "nnU-Net", 100)]
BCP_RATIOS = [1, 5, 10, 25, 50]
HEADLINE = ("ours", 50, "sup", 100)  # (方法A, ratioA, 方法B, ratioB)
TABLE_RATIOS = [1, 5, 10, 25, 50, 100]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- 数据加载与聚合

def resolve_mt_stems(multi, results_dir):
    """MT 替换规则（见模块 docstring 第 7 条）。

    返回 (use_mtfixed: bool, note: str)。use_mtfixed=True 时调用方应把
    mt_{r} 的 stem 换成 mtfixed_{r}_s{seed}。全有或全无：multi["mt"] 全部
    档位 × SEEDS 的 summary+csv 齐全才切换，否则保留旧 mt_* 并标注。
    """
    ratios = multi.get("mt", [])
    total = len(ratios) * len(SEEDS)
    have = 0
    for r in ratios:
        for sd in SEEDS:
            stem = f"mtfixed_{r}_s{sd}"
            if (os.path.exists(os.path.join(results_dir, f"{stem}_summary.json"))
                    and os.path.exists(os.path.join(results_dir,
                                                    f"{stem}_per_slice.csv"))):
                have += 1
    if ratios and have == total:
        return True, f"MT 行全部来自 mtfixed 重跑（{have}/{total}）"
    note = f"MT 行暂为旧版 mt_*（mtfixed 重跑未齐：{have}/{total}），待齐后切换"
    return False, note


def load_and_analyze(results_dir, stems, gt, classes):
    """stems: {显示名: [csv stem, ...]}（多 seed 则多个 stem）。

    返回 (per_seed_summary, per_seed_case_vals)：
    per_seed_summary[name][stem] = analyze_tag summary；
    per_seed_case_vals[name][stem] = case_vals[cls][metric] = {case: val}
    """
    per_seed_summary, per_seed_cv = {}, {}
    for name, stem_list in stems.items():
        per_seed_summary[name], per_seed_cv[name] = {}, {}
        for stem in stem_list:
            df = pd.read_csv(os.path.join(results_dir, f"{stem}_per_slice.csv")).set_index("slice")
            assert set(df.index) == set(gt.keys()), \
                f"{stem}: CSV 切片集与 GT 不一致（csv={len(df)}, gt={len(gt)}）"
            for msg in validate(stem, df, gt, classes):
                log(msg)
            s, cv = analyze_tag(stem, df, gt, classes)
            per_seed_summary[name][stem] = s
            per_seed_cv[name][stem] = cv
    return per_seed_summary, per_seed_cv


def agg_over_seeds(seed_summaries, classes):
    """跨种子聚合：每个指标/类的患者级均值 mean±std + FP 率 mean±std。"""
    out = {"n_seeds": len(seed_summaries), "classes": {}}
    for cls in classes:
        c = {"metrics": {}}
        for m in METRICS:
            vals = {stem: s["classes"][cls]["metrics"][m]["patient_mean"]
                    for stem, s in seed_summaries.items()}
            arr = np.array(list(vals.values()), dtype=float)
            c["metrics"][m] = {
                "seed_means": vals, "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "n_cases": seed_summaries[next(iter(seed_summaries))][
                    "classes"][cls]["metrics"][m]["n_cases"],
                "per_seed_ci95": {stem: s["classes"][cls]["metrics"][m]["ci95"]
                                  for stem, s in seed_summaries.items()}}
        fp = {stem: s["classes"][cls]["fp"]["fp_rate"]
              for stem, s in seed_summaries.items()}
        arr = np.array(list(fp.values()), dtype=float)
        c["fp"] = {"seed_rates": fp, "mean": float(arr.mean()),
                   "std": float(arr.std(ddof=1)) if len(arr) > 1 else None}
        out["classes"][cls] = c
    return out


# ---------------------------------------------------------------- 统计检验

def paired_test(cv_a, cv_b, cls, m):
    """单种子的患者级配对 Wilcoxon。返回 (p, n_cases, mean_a, mean_b)。"""
    va, vb = cv_a[cls][m], cv_b[cls][m]
    cases = sorted(set(va) & set(vb))
    xa = np.array([va[c] for c in cases])
    xb = np.array([vb[c] for c in cases])
    return wilcoxon_p(xa, xb), len(cases), float(xa.mean()), float(xb.mean())


def run_comparison(per_seed_cv, name_a, name_b, cls, m):
    """三种子分别检验，返回逐种子 p 与 max p（按 stem 的 _s{seed} 后缀配对）。"""
    by_seed_a = {s.rsplit("_s", 1)[-1]: cv for s, cv in per_seed_cv[name_a].items()}
    by_seed_b = {s.rsplit("_s", 1)[-1]: cv for s, cv in per_seed_cv[name_b].items()}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    assert seeds, f"{name_a} vs {name_b}: 无共同种子"
    pvals, means = [], []
    n_cases = None
    for sd in seeds:
        p, n, ma, mb = paired_test(by_seed_a[sd], by_seed_b[sd], cls, m)
        pvals.append(p)
        means.append((ma, mb))
        n_cases = n
    seed_key = {sd: p for sd, p in zip(seeds, pvals)}
    return {"n_cases": n_cases,
            "mean_a": float(np.mean([x[0] for x in means])),
            "mean_b": float(np.mean([x[1] for x in means])),
            "p_per_seed": seed_key, "p_max": float(max(pvals))}


def comparisons_for(ratio, multi_methods):
    pairs = [("sup", "ours"), ("semi", "ours"), ("mt", "ours"), ("cps", "ours")]
    if ratio in BCP_RATIOS and "bcp" in multi_methods:
        pairs.append(("bcp", "ours"))
    return [(a, b) for a, b in pairs
            if a in multi_methods and b in multi_methods
            and ratio in multi_methods[a] and ratio in multi_methods[b]]


def comparisons_sup_vs_others(ratio, multi_methods):
    """sup 与各半监督同档直接配对（family = 每 class×metric×ratio 内 3-4 个比较）。"""
    pairs = [("sup", "semi"), ("sup", "mt"), ("sup", "cps")]
    if ratio in BCP_RATIOS and "bcp" in multi_methods:
        pairs.append(("sup", "bcp"))
    return [(a, b) for a, b in pairs
            if a in multi_methods and b in multi_methods
            and ratio in multi_methods[a] and ratio in multi_methods[b]]


def run_families(per_seed_cv, multi, pair_fn):
    """通用 family 计算：pair_fn(ratio, multi) -> [(a, b), ...]，Holm 在同 ratio family 内。"""
    families = {}
    for cls in CLASSES:
        for m in METRICS:
            fam = {}
            for r in sorted({r for rs in multi.values() for r in rs}):
                pairs = pair_fn(r, multi)
                if not pairs:
                    continue
                rows = []
                for a, b in pairs:
                    res = run_comparison(per_seed_cv, f"{a}_{r}", f"{b}_{r}", cls, m)
                    rows.append((r, a, b, res))
                adj = holm_adjust(np.array([x[3]["p_max"] for x in rows]))
                for (r, a, b, res), pa in zip(rows, adj):
                    res["holm_p"] = float(pa)
                    fam[f"{r}%/{a} vs {b}"] = res
            families[f"{cls}/{m}"] = fam
    return families


def run_headline(per_seed_cv, name_a, name_b):
    """单一比较 family（holm_p = p_max），name 为实验全名（如 sup_50）。"""
    out = {}
    for cls in CLASSES:
        for m in METRICS:
            res = run_comparison(per_seed_cv, name_a, name_b, cls, m)
            res["holm_p"] = res["p_max"]
            out[f"{cls}/{m}"] = res
    return out


def run_all_tests(per_seed_cv, multi, headline):
    """返回 (families, headline_result)。

    families[cls/metric][f"{ratio}%/{a} vs {b}"] = {n, means, p_per_seed, p_max, holm_p}
    """
    families = run_families(per_seed_cv, multi, comparisons_for)
    a, ra, b, rb = headline
    headline_result = run_headline(per_seed_cv, f"{a}_{ra}", f"{b}_{rb}")
    return families, headline_result


# ---------------------------------------------------------------- 输出：markdown

def fmt_p(p):
    return f"{p:.4g}"


def _md_pair_table(title, name_a, name_b, agg, result):
    """单比较（headline 式）markdown 表。"""
    md = [f"## {title}（患者级，3 种子）", "",
          f"| Class | Metric | {name_a} mean±std | {name_b} mean±std | "
          "p_s42 | p_s43 | p_s44 | max p |",
          "|---|---|---|---|---|---|---|---|"]
    for cls in CLASSES:
        for m in METRICS:
            ga = agg[name_a]["classes"][cls]["metrics"][m]
            gb = agg[name_b]["classes"][cls]["metrics"][m]
            res = result[f"{cls}/{m}"]
            ps = [res["p_per_seed"].get(str(sd)) for sd in SEEDS]
            md.append(
                f"| {cls} | {m.upper()} | "
                f"{ga['mean'] * 100:.2f}±{(ga['std'] or 0) * 100:.2f} | "
                f"{gb['mean'] * 100:.2f}±{(gb['std'] or 0) * 100:.2f} | "
                + " | ".join(fmt_p(p) if p is not None else "-" for p in ps)
                + f" | {fmt_p(res['p_max'])} |")
    md += ["", "（单一比较 family，Holm 校正后 p = max p。）", ""]
    return md


def _md_family_section(families):
    """family 比较的 markdown 表（每 class×metric 一张）。"""
    md = []
    for cls in CLASSES:
        for m in METRICS:
            md += [f"### {cls} {m.upper()}", "",
                   "| Ratio | Comparison | n_cases | A mean | B mean | "
                   "p_s42 | p_s43 | p_s44 | max p | Holm p |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
            for key, res in families[f"{cls}/{m}"].items():
                ratio, comp = key.split("/", 1)
                ps = [res["p_per_seed"].get(str(sd)) for sd in SEEDS]
                md.append(
                    f"| {ratio} | {comp} | {res['n_cases']} | "
                    f"{res['mean_a'] * 100:.2f} | {res['mean_b'] * 100:.2f} | "
                    + " | ".join(fmt_p(p) if p is not None else "-" for p in ps)
                    + f" | {fmt_p(res['p_max'])} | {fmt_p(res['holm_p'])} |")
            md.append("")
    return md


def build_stats_md(agg, families, headline_result, headline, single_stems, single_agg,
                   sup_families, sup50v100_pair, sup50v100_result, mt_note=None):
    md = ["# v3 三种子统计汇总（rebuttal 最终版）", "",
          "协议：每个种子先按 v2 协议算患者级均值（前景指标仅在 GT 含该类的切片上计算，",
          "病例内取已评分切片均值）；跨种子报 mean±std（n=3）。",
          "检验：每个种子分别做患者级配对双侧 Wilcoxon，**3 种子 p 取最大值（保守合并）**，",
          "Holm 校正在每个 (class, metric, ratio) family 内进行",
          "（主检验 family = 同档 {sup, semi, mt, cps, bcp} vs ours；",
          "补充检验 family = 同档 sup vs {semi, mt, cps, bcp}）。",
          "均值以 % 表示。由 `src/aggregate_v2_seeds.py` 生成。", ""]
    if mt_note:
        md += [f"**MT 数据源标注**：{mt_note}。", ""]
    a, ra, b, rb = headline
    md += _md_pair_table(f"头条：{a}_{ra} vs {b}_{rb}", f"{a}_{ra}", f"{b}_{rb}",
                         agg, headline_result)
    md += ["## 主检验：各方法 vs Ours（同档，患者级 Wilcoxon）", ""]
    md += _md_family_section(families)

    md += ["## 单种子参照（† 仅单次运行，无种子方差，不参与显著性检验，仅作数值参照）", "",
           "| Experiment | Ratio | Rectum Dice | Rectum IoU | Rectum NSD | "
           "Tumor Dice | Tumor IoU | Tumor NSD | Rectum FP% | Tumor FP% |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for stem, label, ratio in single_stems:
        g = single_agg[stem]["classes"]
        md.append(
            f"| {label}† (`{stem}`) | {ratio}% | "
            + " | ".join(f"{g[cls]['metrics'][m]['mean'] * 100:.2f}"
                        for cls in CLASSES for m in METRICS)
            + f" | {g['rectum']['fp']['mean'] * 100:.2f} "
            + f"| {g['tumor']['fp']['mean'] * 100:.2f} |")
    md += ["",
           "说明：sup180 = epoch-matched 180 epoch 全监督对照（seed 42 单次）；"
           "mtcons = Mean Teacher + 一致性过滤组合（seed 42 单次）；"
           "nnU-Net = nnU-Net v2 参照（单次训练）。上述行均无法估计种子间方差，"
           "与其对比仅以三种子方法的 mean±std 数值参照，不做配对检验。", ""]

    h2a, h2b = sup50v100_pair
    md += [f"## 补充检验：{h2a} vs {h2b}（支撑「50% 与 100% 监督打平」）", ""]
    md += _md_pair_table(f"{h2a} vs {h2b}", h2a, h2b, agg, sup50v100_result)[2:]

    md += ["## 补充检验：Supervised vs 各半监督（同档，患者级 Wilcoxon）", "",
           "family = 每个 (class, metric, ratio) 内 sup vs {semi, mt, cps, bcp}",
           "（3-4 个比较）的 Holm 校正。", ""]
    md += _md_family_section(sup_families)
    return "\n".join(md)


# ---------------------------------------------------------------- 输出：LaTeX 主表

def tex_table(agg, single_stems, single_agg, cls, label_suffix, caption_cls):
    """单个类别的主表：行=方法，列=标注档，单元格=患者级 Dice mean%±std%。"""
    lines = [
        f"% CARE 主结果 v3（{caption_cls} Dice）：三种子（42/43/44）患者级 mean%±std%；"
        "每种子内为 v2 协议患者级均值（仅在 GT 含该类的切片上评分）。"
        "† = 单次运行参照（无种子方差）。由 src/aggregate_v2_seeds.py 生成",
        # sn-jnl 兼容：caption/label 置于 resizebox 之后，字号命令移入 resizebox 内
        "\\begin{table}[!htbp]", "\\centering",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\footnotesize",
        "\\begin{tabular}{l" + "c" * len(TABLE_RATIOS) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"{r}\\%" for r in TABLE_RATIOS) + " \\\\",
        "\\midrule",
    ]

    def cell(name):
        if name not in agg:
            return "--"
        g = agg[name]["classes"][cls]["metrics"]["dice"]
        if g["std"] is None:
            return f"{g['mean'] * 100:.2f}$^\\dagger$"
        return f"{g['mean'] * 100:.2f}$\\pm${g['std'] * 100:.2f}"

    for method in ["sup", "semi", "mt", "cps", "bcp", "ours"]:
        cells = [cell(f"{method}_{r}") for r in TABLE_RATIOS]
        lines.append(f"{METHOD_LABEL[method]} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    singles_by_label = {}
    for stem, label, ratio in single_stems:
        singles_by_label.setdefault(label, {})[ratio] = stem
    for label, by_ratio in singles_by_label.items():
        cells = []
        for r in TABLE_RATIOS:
            if r in by_ratio:
                g = single_agg[by_ratio[r]]["classes"][cls]["metrics"]["dice"]
                cells.append(f"{g['mean'] * 100:.2f}$^\\dagger$")
            else:
                cells.append("--")
        lines.append(f"{label}$^\\dagger$ & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}",
              f"\\caption{{CARE test set {caption_cls} Dice (patient-level, \\%), "
              "mean $\\pm$ std over three training seeds (42/43/44). For each seed, "
              "the patient-level mean is computed under the revised protocol "
              "(only slices whose ground truth contains the class are scored). "
              "$^\\dagger$single-run references (no seed variability; values are "
              "the single patient-level mean, shown for numerical reference only).}",
              f"\\label{{tab:care_v3_{label_suffix}}}",
              "\\end{table}", ""]
    return "\n".join(lines)


def build_tables_tex(agg, single_stems, single_agg):
    return (tex_table(agg, single_stems, single_agg, "tumor", "tumor", "Tumor")
            + "\n"
            + tex_table(agg, single_stems, single_agg, "rectum", "rectum",
                        "Normal rectum"))


# ---------------------------------------------------------------- 输出：LaTeX 补充表

def _fmt_p_tex(p):
    """tabS1 风格：p<0.001 记 <$0.001，其余科学计数；p<0.05 加粗。"""
    if p < 0.001:
        s = "$<$0.001"
    elif p >= 0.995:
        s = "$1.0$"
    else:
        exp = int(np.floor(np.log10(p)))
        s = f"${p / 10 ** exp:.2f}\\times10^{{{exp}}}$"
    return f"\\textbf{{{s}}}" if p < 0.05 else s


def _fmt_p_short(p):
    """补充主检验表的紧凑 p 值格式；与科学计数版一样对 p<0.05 加粗。"""
    if p < 0.001:
        s = "$<$0.001"
    elif p >= 0.995:
        s = "1.00"
    elif p < 0.1:
        s = f"{p:.3f}"
    else:
        s = f"{p:.2f}"
    return f"\\textbf{{{s}}}" if p < 0.05 else s


def build_tabS5_wilcoxon_tex(agg, families, headline_result, headline, multi):
    """由同一次聚合结果生成头条与同档检验表，避免手工复制造成版本混用。"""
    a, ra, b, rb = headline
    name_a, name_b = f"{a}_{ra}", f"{b}_{rb}"
    label_a = f"{METHOD_LABEL.get(a, a)}-{ra}\\%"
    label_b = f"{METHOD_LABEL.get(b, b)}-{rb}\\%"
    lines = [
        "% CARE 配对 Wilcoxon（患者级，3 种子）：本文件完全由 agg_tests/agg_seeds 生成，",
        "% 同档 family 自动包含当前 multi 中的 BCP 全档与修复版 MT，禁止手工补行。",
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Headline comparison, " + label_a + " vs.\\ " + label_b +
        " (CARE test, 81 cases). Patient-level paired two-sided Wilcoxon signed-rank "
        "tests are run independently for each training seed; means are cross-seed "
        "mean\\% under the revised protocol. The conservative summary is the maximum "
        "$p$ across seeds (single-comparison family).}",
        "\\label{tabS:headline_v3}", "\\small",
        "\\begin{tabular}{llccccc}", "\\toprule",
        f"Class & Metric & {label_a} & {label_b} & $p$ (s42) & $p$ (s43) & $p$ (s44) \\\\",
        "\\midrule"]
    for cls in ("tumor", "rectum"):
        for i, metric in enumerate(METRICS):
            res = headline_result[f"{cls}/{metric}"]
            ps = [_fmt_p_short(res["p_per_seed"][str(sd)]) for sd in SEEDS]
            class_cell = "Tumor" if cls == "tumor" else "Normal rectum"
            lines.append(
                f"{class_cell if i == 0 else ''} & {metric.upper()} & "
                f"{res['mean_a'] * 100:.1f} & {res['mean_b'] * 100:.1f} & "
                + " & ".join(ps) + " \\\\")
        if cls == "tumor":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "",
              "\\begin{table}[!htbp]", "\\centering",
              "\\caption{Same-ratio paired comparisons on CARE (patient-level tumor/"
              "normal-rectum Dice). Each method is compared against Ours at the same "
              "labeled ratio, independently per seed; reported are the maximum $p$ "
              "across the three seeds (max-$p$) and the Holm-adjusted $p$ within each "
              "(ratio, class) family. Positive direction: Ours scores higher. "
              "$p<0.05$ in bold.}",
              "\\label{tabS:wilcoxon_care_v3}", "\\small",
              "\\setlength{\\tabcolsep}{4pt}",
              "\\begin{tabular}{llcccc}", "\\toprule",
              " & & \\multicolumn{2}{c}{Tumor Dice} & \\multicolumn{2}{c}{Rectum Dice} \\\\",
              "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
              "Labeled & Comparison & max $p$ & Holm $p$ & max $p$ & Holm $p$ \\\\",
              "\\midrule"]
    ratios = sorted({r for rs in multi.values() for r in rs})
    for r in ratios:
        pairs = comparisons_for(r, multi)
        if not pairs:
            continue
        for i, (ma, mb) in enumerate(pairs):
            key = f"{r}%/{ma} vs {mb}"
            rt = families[f"tumor/dice"][key]
            rr = families[f"rectum/dice"][key]
            prefix = f"\\multirow{{{len(pairs)}}}{{*}}{{{r}\\%}}" if i == 0 else ""
            comp = f"{METHOD_LABEL.get(ma, ma)} vs {METHOD_LABEL.get(mb, mb)}"
            lines.append(
                f"{prefix} & {comp} & {_fmt_p_short(rt['p_max'])} & "
                f"{_fmt_p_short(rt['holm_p'])} & {_fmt_p_short(rr['p_max'])} & "
                f"{_fmt_p_short(rr['holm_p'])} \\\\")
        if r != ratios[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def build_tabS8b_tex(sup_families, sup50v100_pair, sup50v100_result):
    """tabS8b：sup vs 各半监督同档 + sup_50 vs sup_100 的患者级 Wilcoxon Holm p。"""
    label = {"semi": "ST", "mt": "MT", "cps": "CPS", "bcp": "BCP"}
    ratios_of = {"semi": [1, 5, 10, 25, 50], "mt": [1, 5, 10, 25, 50],
                 "cps": [1, 5, 10, 25, 50], "bcp": [1, 5, 10, 25, 50]}
    lines = [
        "% CARE 补充表：Supervised 与各半监督方法同档对比 + sup_50 vs sup_100 的"
        "患者级 Wilcoxon p（v3 三种子协议：逐种子 p 取最大，Holm 校正）。"
        "由 src/aggregate_v2_seeds.py 生成",
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Patient-level paired two-sided Wilcoxon $p$-values comparing "
        "the fully supervised baseline with each semi-supervised method at the "
        "same labeled ratio (CARE test set, 81 cases), and the 50\\% vs 100\\% "
        "supervised comparison. A separate test is performed per training seed "
        "(42/43/44) and the maximum $p$ across the three seeds is reported "
        "(conservative combination), Holm-adjusted within each "
        "(class, ratio) family of 4 comparisons; the 50\\% vs 100\\% row is "
        "a single-comparison family. Bold: $p<0.05$. ST = pseudo-label "
        "self-training, MT = Mean Teacher.}",
        "\\label{tabS:sup_vs_others}", "\\small",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Comparison & Labeled & Tumor Dice & Rectum Dice \\\\",
        " & ratio & Holm $p$ & Holm $p$ \\\\",
        "\\midrule",
    ]
    for method in ["semi", "mt", "cps", "bcp"]:
        ratios = [r for r in ratios_of[method]
                  if any(k.startswith(f"{r}%/sup vs {method}")
                         for k in sup_families["tumor/dice"])]
        if not ratios:
            continue
        for i, r in enumerate(ratios):
            key = f"{r}%/sup vs {method}"
            pt = _fmt_p_tex(sup_families["tumor/dice"][key]["holm_p"])
            pr = _fmt_p_tex(sup_families["rectum/dice"][key]["holm_p"])
            prefix = (f"\\multirow{{{len(ratios)}}}{{*}}{{Supervised vs "
                      f"{label[method]}}}") if i == 0 else ""
            lines.append(f"{prefix} & {r}\\% & {pt} & {pr} \\\\")
        lines.append("\\midrule")
    h2a, h2b = sup50v100_pair
    pt = _fmt_p_tex(sup50v100_result["tumor/dice"]["holm_p"])
    pr = _fmt_p_tex(sup50v100_result["rectum/dice"]["holm_p"])
    label2 = (f"{h2a} vs {h2b}").replace("_", "\\_")
    lines.append(f"{label2} & -- & {pt} & {pr} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results_v2")
    ap.add_argument("--out_dir", default=os.path.join("results", "v2"))
    ap.add_argument("--man_dir", default=os.path.join("paper", "manuscript"))
    ap.add_argument("--sr_dir", default=os.path.join("paper", "sr"))
    ap.add_argument("--stats_md", default=os.path.join("paper", "v3_stats.md"))
    ap.add_argument("--gt_json", default=None,
                    help="默认用 data/cache CARE test GT；冒烟时传 {name: [has_rectum, has_tumor]}")
    ap.add_argument("--smoke", action="store_true", help="迷你实验集（sup/ours@1%% + 单种子参照）")
    args = ap.parse_args()

    for directory in (args.out_dir, args.man_dir, args.sr_dir):
        os.makedirs(directory, exist_ok=True)

    if args.gt_json:
        with open(args.gt_json) as f:
            raw = json.load(f)
        gt = {k: tuple(v) for k, v in raw.items()}
    else:
        gt = care_gt_presence()
    log(f"GT 加载：{len(gt)} 切片（source={args.gt_json or 'data/cache/test_labels.npy'}）")

    multi = ({"sup": [1], "semi": [1], "mt": [1], "ours": [1]}
             if args.smoke else MULTI)
    single = [("refsup_1_s42", "RefSup", 1)] if args.smoke else SINGLE
    headline = ("ours", 1, "sup", 1) if args.smoke else HEADLINE
    headline2 = ("sup_1", "refsup_1_s42") if args.smoke else ("sup_50", "sup_100")

    stems, name_meta = {}, {}
    use_mtfixed, mt_note = resolve_mt_stems(multi, args.results_dir)
    log(f"MT 数据源：{mt_note}")
    for method, ratios in multi.items():
        for r in ratios:
            base = "mtfixed" if (method == "mt" and use_mtfixed) else method
            stems[f"{method}_{r}"] = [f"{base}_{r}_s{sd}" for sd in SEEDS]
            name_meta[f"{method}_{r}"] = (method, r)
    for stem, label, r in single:
        stems[stem] = [stem]
        name_meta[stem] = (label, r)

    log(f"=== 加载并逐种子分析 {len(stems)} 个实验 ===")
    per_seed_summary, per_seed_cv = load_and_analyze(args.results_dir, stems, gt, CLASSES)

    agg = {}
    for name, seed_sums in per_seed_summary.items():
        agg[name] = agg_over_seeds(seed_sums, CLASSES)
        agg[name]["stems"] = list(seed_sums)
        agg[name]["n_seeds"] = len(seed_sums)
    single_agg = {stem: agg[stem] for stem, _, _ in single}

    with open(os.path.join(args.out_dir, "agg_seeds.json"), "w") as f:
        json.dump({"protocol": "v2 patient-level per seed; mean±std over seeds "
                               f"{SEEDS} (std: sample ddof=1); FP rate per seed "
                               "then mean±std",
                   "mt_source": mt_note, "experiments": agg}, f, indent=2)
    log(f"已保存 {os.path.join(args.out_dir, 'agg_seeds.json')}")

    log("=== 统计检验（逐种子 Wilcoxon + max p + Holm） ===")
    families, headline_result = run_all_tests(per_seed_cv, multi, headline)
    a, ra, b, rb = headline
    sup_families = run_families(per_seed_cv, multi, comparisons_sup_vs_others)
    sup50v100_result = run_headline(per_seed_cv, *headline2)
    tests = {"protocol": "patient-level paired two-sided Wilcoxon per seed; "
                         "combined p = max over seeds (conservative); Holm within "
                         "each (class, metric, ratio) family",
             "seeds": SEEDS,
             "headline": {"pair": f"{a}_{ra} vs {b}_{rb}", "result": headline_result},
             "families": families,
             "sup_vs_others": {
                 "protocol": "same as above; family = sup vs {semi, mt, cps, bcp} "
                             "at each (class, metric, ratio)",
                 "families": sup_families},
             "sup50_vs_sup100": {"pair": f"{headline2[0]} vs {headline2[1]}",
                                 "result": sup50v100_result}}
    with open(os.path.join(args.out_dir, "agg_tests.json"), "w") as f:
        json.dump(tests, f, indent=2)
    log(f"已保存 {os.path.join(args.out_dir, 'agg_tests.json')}")

    md = build_stats_md(agg, families, headline_result, headline, single, single_agg,
                        sup_families, headline2, sup50v100_result,
                        mt_note=mt_note)
    with open(args.stats_md, "w") as f:
        f.write(md + "\n")
    log(f"已保存 {args.stats_md}")

    texs = {
        "table1_care_v3.tex": build_tables_tex(agg, single, single_agg),
        "tabS5_wilcoxon_care_v3.tex": build_tabS5_wilcoxon_tex(
            agg, families, headline_result, headline, multi),
        "tabS8b_sup_vs_others.tex": build_tabS8b_tex(
            sup_families, headline2, sup50v100_result),
    }
    for fname, content in texs.items():
        tex_path = os.path.join(args.man_dir, fname)
        with open(tex_path, "w") as f:
            f.write(content)
        shutil.copyfile(tex_path, os.path.join(args.sr_dir, fname))
        log(f"已保存 {tex_path}（并同步到 {args.sr_dir}/）")

    # 终端校验报告：关键数字
    log("\n=== 校验报告（肿瘤 Dice，患者级 %，三种子 mean±std） ===")
    for method in ["sup", "semi", "mt", "cps", "bcp", "ours"]:
        for r in multi.get(method, []):
            g = agg[f"{method}_{r}"]["classes"]["tumor"]["metrics"]["dice"]
            std = f"±{g['std'] * 100:.2f}" if g["std"] is not None else " (single)"
            log(f"  {method}_{r}: {g['mean'] * 100:.2f}{std}")
    for stem, label, r in single:
        g = agg[stem]["classes"]["tumor"]["metrics"]["dice"]
        log(f"  {label}† @{r}%: {g['mean'] * 100:.2f}")
    res = headline_result["tumor/dice"]
    log(f"头条 {a}_{ra} vs {b}_{rb}（tumor Dice）："
        f"max p = {res['p_max']:.4g}，逐种子 p = "
        + ", ".join(f"{k}={v:.4g}" for k, v in res["p_per_seed"].items()))
    log("\n=== 补充检验关键数字 ===")
    for cls in ("tumor", "rectum"):
        for key, res in sup_families[f"{cls}/dice"].items():
            if key.startswith("1%/"):
                log(f"  sup vs 半监督 @1% [{cls} dice] {key.split('/')[1]}: "
                    f"max p = {res['p_max']:.4g}, Holm p = {res['holm_p']:.4g}, "
                    f"means = {res['mean_a'] * 100:.2f} vs {res['mean_b'] * 100:.2f}")
        res = sup50v100_result[f"{cls}/dice"]
        log(f"  {headline2[0]} vs {headline2[1]} [{cls} dice]: "
            f"max p = {res['p_max']:.4g}, "
            f"means = {res['mean_a'] * 100:.2f} vs {res['mean_b'] * 100:.2f}")


if __name__ == "__main__":
    main()
