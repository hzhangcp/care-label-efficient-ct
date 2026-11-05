"""Fig 3（病灶大小分层）审稿补充：层内显著性检验 + 分层边界依据。

不重跑推理；仅从 results/*_per_slice.csv（逐切片 dice_tumor）+ GT 肿瘤面积重算：
1. 按 GT 肿瘤面积分层（默认与 make_figures.fig3_size_strata 相同的 paper 边界：
   (0,50] / (50,200] / (200,1000] / (1000,inf] 像素；空 GT 切片剔除）。
2. 层内肿瘤 Dice 的患者级统计（与 reanalyze_v2 同原则：病例内取均值 ->
   cluster bootstrap 95%CI；配对双侧 Wilcoxon：baseline vs 各方法；
   Holm 校正在每层的比较 family 内进行）。
3. 分层边界依据：报告每个边界在肿瘤切片面积分布中的分位数；同时用四分位边界
   （Q1/Q2/Q3）重算一套完整结果，两套均写入 JSON，供正文选择。

产出：
  results/v2/stratified_stats.json        两套分层的全部数字 + 边界分析
  paper/manuscript/tab_stratified.tex     paper 边界：层 x 方法 Dice[CI] + p 值表
  paper/manuscript/tab_stratified_quartiles.tex  四分位边界版本（备选）

三种子模式（--seeds 42,43,44，审稿 M7；与 aggregate_v2_seeds 同一 max-p/Holm 协议）：
  数据源改为 results_v2/{tag}_s{seed}_per_slice.csv（50% 档全部方法 + sup_100，
  各 3 种子），只做 paper 边界分层：
  - 主比较（headline）sup_100 vs ours_50：层内逐种子患者级配对 Wilcoxon，
    3 种子 p 取最大（保守合并），Holm 跨 4 层校正；效应量 = 患者级配对 Dice 差
    （ours - sup_100，每病例先跨种子平均）均值 + cluster bootstrap 95%CI；
  - 次要 family：各半监督（semi/mt/cps/bcp_50）vs ours_50，逐种子 max-p，
    Holm 在每层 family（4 个比较）内校正；
  - 每层每方法报三种子患者级均值 mean±std。
产出：
  results/v2/stratified_stats_3seed.json     三种子全部分层数字
  paper/manuscript/tab_stratified_3seed.tex  三种子分层表（同步复制到 --sr_dir）

用法：
  python src/stratified_stats.py                 # 真实数据（CARE test, 50% 档 + sup_100）
  python src/stratified_stats.py --seeds 42,43,44   # 三种子模式（results_v2）
  # 冒烟（假数据）见 scripts/smoke_test.sh
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset  # noqa: E402  remap_label
from reanalyze_v2 import (cluster_boot_ci, holm_adjust, parse_case_id,  # noqa: E402
                          wilcoxon_p)
from aggregate_v4_rebuttal import twoway_cluster_boot_ci  # noqa: E402

# paper 分层（与 make_figures.fig3_size_strata 完全一致：左开右闭，单位=像素）
PAPER_STRATA = [("\u226450", 0, 50), ("51\u2013200", 50, 200),
                ("201\u20131000", 200, 1000), (">1000", 1000, float("inf"))]

DEFAULT_TAGS = ["sup_100", "sup_50", "semi_50", "mt_50", "cps_50", "ours_50"]
DEFAULT_TAGS_V2 = ["sup_100", "sup_50", "semi_50", "mt_50", "cps_50", "bcp_50",
                   "ours_50"]
METHOD_LABELS = {"sup": "Supervised", "semi": "Pseudo-label ST",
                 "mt": "Mean Teacher", "cps": "CPS", "bcp": "BCP", "ours": "Ours"}


def tag_label(tag):
    if tag.startswith("mtfixed_"):
        return f"Mean Teacher {tag[len('mtfixed_'):]}\\%"
    for k, v in METHOD_LABELS.items():
        if tag.startswith(k + "_"):
            return f"{v} {tag[len(k) + 1:]}\\%"
    return tag


# ---------------------------------------------------------------- GT 肿瘤面积

def tumor_areas(cache_dir="data/cache", data_root=None, areas_json=None):
    """slice_name -> GT 肿瘤面积（像素）。优先 areas_json，其次 cache，否则扫 data_root。"""
    if areas_json:
        with open(areas_json) as f:
            return {k: int(v) for k, v in json.load(f).items()}
    lab_path = os.path.join(cache_dir, "test_labels.npy") if cache_dir else None
    if lab_path and os.path.exists(lab_path):
        labels = np.load(lab_path, mmap_mode="r")
        with open(os.path.join(cache_dir, "test_names.json")) as f:
            names = json.load(f)
        return {n: int((np.asarray(labels[i]) == 2).sum())
                for i, n in enumerate(names)}
    assert data_root, "无 cache 时必须给 --data_root"
    npz_dir = os.path.join(data_root, "test", "test_npz")
    area = {}
    for f in sorted(os.listdir(npz_dir)):
        if not f.endswith(".npz"):
            continue
        with np.load(os.path.join(npz_dir, f)) as z:
            lab = dataset.remap_label(z["label"])
        area[f] = int((lab == 2).sum())
    return area


# ---------------------------------------------------------------- 分层统计

def assign_stratum(a, strata):
    """左开右闭：lo < a <= hi。a<=0（空 GT）返回 None。"""
    if a <= 0:
        return None
    for si, (_, lo, hi) in enumerate(strata):
        if lo < a <= hi:
            return si
    return None


def per_case_by_stratum(csv_path, area, strata):
    """返回 [{case: 层内病例均值}, ...]（按层）+ 各层切片数。"""
    df = pd.read_csv(csv_path)
    cases = {s: parse_case_id(s) for s in df["slice"]}
    out = [dict() for _ in strata]
    n_slices = [0] * len(strata)
    sums = [dict() for _ in strata]
    cnts = [dict() for _ in strata]
    for s, v in zip(df["slice"], df["dice_tumor"]):
        a = area.get(s)
        if a is None:
            continue
        si = assign_stratum(a, strata)
        if si is None or pd.isna(v):  # 空 GT 剔除；nan 防御性跳过
            continue
        c = cases[s]
        sums[si][c] = sums[si].get(c, 0.0) + float(v)
        cnts[si][c] = cnts[si].get(c, 0) + 1
    for si in range(len(strata)):
        n_slices[si] = sum(cnts[si].values())
        out[si] = {c: sums[si][c] / cnts[si][c] for c in sums[si]}
    return out, n_slices


def quartile_strata(areas_pos):
    """由肿瘤切片面积分布的四分位数构造层边界（左开右闭）。"""
    q = np.percentile(np.asarray(areas_pos, float), [25, 50, 75])
    bounds = sorted({int(round(x)) for x in q})
    edges = [0] + bounds + [float("inf")]
    strata = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi == float("inf"):
            label = f">{lo}"
        elif lo == 0:
            label = f"\u2264{hi}"
        else:
            label = f"{lo + 1}\u2013{hi}"
        strata.append((label, lo, hi))
    return strata, bounds


def analyze(strata, area, tags, baseline, comparisons, extra_pair, n_boot, seed):
    """一套分层下的全部统计。返回 dict（可直接 JSON 序列化）。"""
    pc, ns = {}, None
    for tag in tags:
        pc[tag], ns_tag = per_case_by_stratum(
            os.path.join(ARGS.results_dir, f"{tag}_per_slice.csv"), area, strata)
        ns = ns_tag if ns is None else ns
        assert ns == ns_tag, f"{tag}: 层内切片数与其他方法不一致"
    res = {"bins": [{"label": l, "lo": lo,
                     "hi": (None if hi == float("inf") else hi)}
                    for l, lo, hi in strata],
           "strata": []}
    for si, (label, lo, hi) in enumerate(strata):
        s = {"label": label, "n_slices": int(ns[si]), "methods": {}, "tests": []}
        for tag in tags:
            cv = pc[tag][si]
            if not cv:
                s["methods"][tag] = None
                continue
            lo_ci, hi_ci = cluster_boot_ci(list(cv.values()), n_boot=n_boot, seed=seed)
            s["methods"][tag] = {"patient_mean": float(np.mean(list(cv.values()))),
                                 "ci95": [lo_ci, hi_ci], "n_cases": len(cv)}
        # family：baseline vs 各方法，Holm 在层内校正
        pvals, rows = [], []
        vb_all = pc[baseline][si]
        for other in comparisons:
            va, vb = vb_all, pc[other][si]
            cases = sorted(set(va) & set(vb))
            if len(cases) < 2:
                p = None
            else:
                p = wilcoxon_p([va[c] for c in cases], [vb[c] for c in cases])
            pvals.append(np.nan if p is None else p)
            rows.append({"comparison": f"{baseline} vs {other}",
                         "n_cases": len(cases), "raw_p": p})
        valid = [i for i, p in enumerate(pvals) if not np.isnan(p)]
        if valid:
            adj = holm_adjust(np.array([pvals[i] for i in valid]))
            for i, pa in zip(valid, adj):
                rows[i]["holm_p"] = float(pa)
        s["tests"] = rows
        # 额外单比较（如 sup_100 vs ours_50，对应 fig3 主对比），不做多重校正
        if extra_pair:
            a_tag, b_tag = extra_pair
            va, vb = pc[a_tag][si], pc[b_tag][si]
            cases = sorted(set(va) & set(vb))
            p = (wilcoxon_p([va[c] for c in cases], [vb[c] for c in cases])
                 if len(cases) >= 2 else None)
            s["extra_test"] = {"comparison": f"{a_tag} vs {b_tag}",
                               "n_cases": len(cases), "raw_p": p,
                               "note": "单一比较，未做多重校正"}
        res["strata"].append(s)
    return res


# ---------------------------------------------------------------- 边界分析

def boundary_analysis(area):
    pos = np.array([a for a in area.values() if a > 0], dtype=float)
    if len(pos) == 0:  # 防御：假数据可能无肿瘤切片
        return {"n_tumor_slices": 0, "n_slices_total": len(area),
                "paper_boundary_percentiles": {}, "paper_strata_n": {}}
    pa = {"n_tumor_slices": int(len(pos)), "n_slices_total": len(area),
          "area_min": float(pos.min()), "area_median": float(np.median(pos)),
          "area_max": float(pos.max()),
          "area_percentiles": {f"p{q}": float(np.percentile(pos, q))
                               for q in (5, 10, 25, 50, 75, 90, 95)},
          "paper_boundary_percentiles": {},
          "paper_strata_n": {}}
    for _, lo, hi in PAPER_STRATA:
        key = hi if hi != float("inf") else None
        if key is not None:
            pa["paper_boundary_percentiles"][str(int(key))] = \
                float((pos <= key).mean() * 100)
    for label, lo, hi in PAPER_STRATA:
        n = int(((pos > lo) & (pos <= hi)).sum())
        pa["paper_strata_n"][label] = {"n_slices": n,
                                       "pct": n / len(pos) * 100}
    return pa


# ---------------------------------------------------------------- LaTeX

def fmt_p(p):
    if p is None:
        return "--"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def make_tex(strat, tags, comparisons, extra_pair, title_note, label_sfx=""):
    strata = strat["strata"]
    lines = [
        "% 病灶大小分层统计：层内肿瘤 Dice（患者级 mean% [cluster bootstrap 95%CI]）"
        "+ 配对 Wilcoxon（Holm 校正）。由 src/stratified_stats.py 生成",
        f"% {title_note}",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Tumor Dice stratified by ground-truth lesion area "
        "(CARE test, tumor-containing slices only). Values are patient-level "
        "mean\\% [cluster bootstrap 95\\% CI].}",
        f"\\label{{tab:stratified{label_sfx}}}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(strata) + "}",
        "\\toprule",
        "Method & " + " & ".join(s["label"] + " px" for s in strata) + " \\\\",
        "\\midrule",
        "$n$ slices & " + " & ".join(str(s["n_slices"]) for s in strata) + " \\\\",
        "$n$ cases & " + " & ".join(
            str(next((m["n_cases"] for m in (s["methods"][t] for t in tags)
                      if m is not None), 0)) for s in strata) + " \\\\",
        "\\midrule",
    ]
    for tag in tags:
        cells = []
        for s in strata:
            m = s["methods"][tag]
            cells.append("--" if m is None else
                         f"{m['patient_mean'] * 100:.1f} "
                         f"[{m['ci95'][0] * 100:.1f}, {m['ci95'][1] * 100:.1f}]")
        lines.append(f"{tag_label(tag)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    # p 值表：行=层，列=各比较
    short = {"sup": "Sup", "semi": "ST", "mt": "MT", "cps": "CPS", "ours": "Ours"}

    def tag_short(t):
        if t.startswith("mtfixed_"):
            return f"MT$_{{{t[len('mtfixed_'):]}}}$"
        for k, v in short.items():
            if t.startswith(k + "_"):
                return f"{v}$_{{{t[len(k) + 1:]}}}$"
        return t

    cols = [f"{tag_short(ARGS.baseline)} vs {tag_short(c)}" for c in comparisons]
    if extra_pair:
        cols.append(f"{tag_short(extra_pair[0])} vs {tag_short(extra_pair[1])}")
    lines += [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Paired two-sided Wilcoxon signed-rank tests on patient-level "
        "tumor Dice within each size stratum. $p$ values are Holm-adjusted within "
        f"each stratum (family of {len(comparisons)} comparisons vs. the "
        f"{tag_label(ARGS.baseline)} baseline)" +
        ("; the last column is a single unadjusted test corresponding to "
         "Fig.\\,3" if extra_pair else "") + ".}",
        f"\\label{{tab:stratified{label_sfx}_p}}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Stratum & " + " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for s in strata:
        cells = [fmt_p(t.get("holm_p")) for t in s["tests"]]
        if extra_pair and "extra_test" in s:
            cells.append(fmt_p(s["extra_test"]["raw_p"]))
        lines.append(f"{s['label']} px & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- 三种子模式（M7）

def run_pair_3seed(pc, a_tag, b_tag, si, seeds, n_boot, seed):
    """层 si 内 a_tag vs b_tag 的三种子统计。

    逐种子患者级配对双侧 Wilcoxon -> p_max（保守合并）；效应量 = b - a 的
    患者级配对 Dice 差（每病例先跨种子取均值）的均值 + cluster bootstrap 95%CI。
    """
    p_per_seed, diff_per_seed, diffs_by_case, n_cases = {}, {}, {}, None
    for sd in seeds:
        va, vb = pc[a_tag][sd][si], pc[b_tag][sd][si]
        cases = sorted(set(va) & set(vb))
        assert n_cases in (None, len(cases)), f"{a_tag} vs {b_tag}: 各种子病例数不一致"
        n_cases = len(cases)
        p = (wilcoxon_p([va[c] for c in cases], [vb[c] for c in cases])
             if len(cases) >= 2 else None)
        p_per_seed[str(sd)] = p
        diff_per_seed[str(sd)] = (float(np.mean([vb[c] - va[c] for c in cases]))
                                  if cases else None)
        for c in cases:
            diffs_by_case.setdefault(c, []).append(vb[c] - va[c])
    pooled = {c: float(np.mean(d)) for c, d in diffs_by_case.items()
              if len(d) == len(seeds)}
    vals = np.array(list(pooled.values()), dtype=float)
    pvals = [p for p in p_per_seed.values() if p is not None]
    res = {"comparison": f"{a_tag} vs {b_tag}", "direction": f"{b_tag} - {a_tag}",
           "n_cases": n_cases, "p_per_seed": p_per_seed,
           "p_max": float(max(pvals)) if pvals else None,
           "diff_per_seed": diff_per_seed}
    if len(vals) >= 2:
        lo_ci, hi_ci = cluster_boot_ci(vals, n_boot=n_boot, seed=seed)
        common = sorted(pooled)
        D = np.array([[pc[b_tag][sd][si][c] - pc[a_tag][sd][si][c]
                       for c in common] for sd in seeds])
        rng = np.random.default_rng(seed)
        res["effect"] = {"diff_mean": float(vals.mean()),
                         "diff_ci95": [lo_ci, hi_ci],
                         "diff_ci95_twoway": twoway_cluster_boot_ci(
                             D, n_boot, rng),
                         "n_cases_pooled": len(vals)}
    else:  # 防御：病例不足（冒烟极小集）
        res["effect"] = {"diff_mean": float(vals.mean()) if len(vals) else None,
                         "diff_ci95": [None, None], "n_cases_pooled": len(vals)}
    return res


def analyze_3seed(results_dir_v2, strata, area, tags, seeds, headline, vs_ours,
                  n_boot, seed):
    """paper 分层下的三种子统计。headline=(a,b) 跨层 Holm；vs_ours 层内 Holm。"""
    pc, ns = {}, None
    for tag in tags:
        pc[tag] = {}
        for sd in seeds:
            pc_tag_sd, ns_tag_sd = per_case_by_stratum(
                os.path.join(results_dir_v2, f"{tag}_s{sd}_per_slice.csv"),
                area, strata)
            ns = ns_tag_sd if ns is None else ns
            assert ns == ns_tag_sd, f"{tag}_s{sd}: 层内切片数与其他方法/种子不一致"
            pc[tag][sd] = pc_tag_sd
    res = {"seeds": list(seeds),
           "bins": [{"label": l, "lo": lo,
                     "hi": (None if hi == float("inf") else hi)}
                    for l, lo, hi in strata],
           "strata": []}
    for si, (label, lo, hi) in enumerate(strata):
        s = {"label": label, "n_slices": int(ns[si]), "methods": {}}
        for tag in tags:
            seed_means, ncs = {}, {}
            for sd in seeds:
                cv = pc[tag][sd][si]
                seed_means[str(sd)] = (float(np.mean(list(cv.values())))
                                       if cv else None)
                ncs[str(sd)] = len(cv)
            arr = np.array([v for v in seed_means.values() if v is not None])
            s["methods"][tag] = {
                "seed_means": seed_means,
                "mean": float(arr.mean()) if len(arr) else None,
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "n_cases": ncs}
        s["headline"] = run_pair_3seed(pc, headline[0], headline[1], si, seeds,
                                       n_boot, seed)
        rows, pvals = [], []
        for o_tag in vs_ours:
            r = run_pair_3seed(pc, o_tag, "ours_50", si, seeds, n_boot, seed)
            rows.append(r)
            pvals.append(np.nan if r["p_max"] is None else r["p_max"])
        valid = [i for i, p in enumerate(pvals) if not np.isnan(p)]
        if valid:
            adj = holm_adjust(np.array([pvals[i] for i in valid]))
            for i, pa in zip(valid, adj):
                rows[i]["holm_p"] = float(pa)
        s["tests_vs_ours"] = rows
        res["strata"].append(s)
    p_h = [s["headline"]["p_max"] for s in res["strata"]]
    valid = [i for i, p in enumerate(p_h) if p is not None]
    if valid:
        adj = holm_adjust(np.array([p_h[i] for i in valid]))
        for i, pa in zip(valid, adj):
            res["strata"][i]["headline"]["holm_p"] = float(pa)
    return res


def fmt_delta(e):
    if e["diff_mean"] is None:
        return "--"
    lo, hi = e["diff_ci95"]
    if lo is None:
        return f"${e['diff_mean'] * 100:+.1f}$"
    return (f"${e['diff_mean'] * 100:+.1f}$ "
            f"[${lo * 100:+.1f}$, ${hi * 100:+.1f}$]")


def tex_stratum_label(label):
    """分层标签转 LaTeX 形式（与 tab_stratified.tex 一致）：≤50 -> $\\leq$50 等。"""
    if label.startswith("≤"):
        return f"$\\leq${label[1:]}"
    if label.startswith(">"):
        return f"$>${label[1:]}"
    return label.replace("–", "--")


def make_tex_3seed(res3, tags, headline, vs_ours, title_note, n_boot):
    """三张表：分层 Dice（mean±std）/ headline 检验 + 效应量 / 次要 family。"""
    strata, seeds = res3["strata"], res3["seeds"]
    seed_str = "/".join(str(s) for s in seeds)
    nb_str = f"{n_boot:,}".replace(",", "{,}")
    short = {"sup": "Sup", "semi": "ST", "mt": "MT", "cps": "CPS", "bcp": "BCP",
             "ours": "Ours"}

    def tag_short(t):
        if t.startswith("mtfixed_"):
            return f"MT$_{{{t[len('mtfixed_'):]}}}$"
        for k, v in short.items():
            if t.startswith(k + "_"):
                return f"{v}$_{{{t[len(k) + 1:]}}}$"
        return t

    lines = [
        "% 病灶大小分层统计（三种子 max-p/Holm 协议，审稿 M7）：层内肿瘤 Dice"
        "（患者级 mean%±std%，种子 42/43/44）+ 逐种子配对 Wilcoxon 取 max-p。"
        "由 src/stratified_stats.py --seeds 生成",
        f"% {title_note}",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Tumor Dice stratified by ground-truth lesion area (CARE test, "
        f"tumor-containing slices only; three training seeds {seed_str}). Values "
        "are patient-level mean\\% $\\pm$ std\\% across seeds (each seed first "
        "reduced to a patient-level mean).}",
        "\\label{tab:stratified3}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(strata) + "}",
        "\\toprule",
        "Method & " + " & ".join(tex_stratum_label(s["label"]) + " px"
                                 for s in strata) + " \\\\",
        "\\midrule",
        "$n$ slices & " + " & ".join(str(s["n_slices"]) for s in strata) + " \\\\",
        "$n$ cases & " + " & ".join(
            str(next((m["n_cases"][str(seeds[0])]
                      for m in (s["methods"][t] for t in tags)
                      if m["mean"] is not None), 0)) for s in strata) + " \\\\",
        "\\midrule",
    ]
    for tag in tags:
        cells = []
        for s in strata:
            m = s["methods"][tag]
            cells.append("--" if m["mean"] is None else
                         f"{m['mean'] * 100:.1f}$\\pm${(m['std'] or 0) * 100:.1f}")
        lines.append(f"{tag_label(tag)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    ha, hb = headline
    lines += [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Headline comparison "
        f"{tag_label(hb)} vs.\\ {tag_label(ha)} on patient-level tumor Dice "
        "within each size stratum (three training seeds "
        f"{seed_str}). A paired two-sided Wilcoxon test is performed per seed "
        "and the maximum $p$ across seeds is reported (conservative "
        "combination), Holm-adjusted across the four strata. "
        f"$\\Delta$ = patient-level paired Dice difference ({tag_short(hb)} "
        f"$-$ {tag_short(ha)}), mean over cases [cluster bootstrap 95\\% CI, "
        f"{nb_str} replicates]; two-way cluster bootstrap CIs (resampling "
        "patients and seeds jointly) are reported in the JSON as a sensitivity "
        "analysis.}",
        "\\label{tab:stratified3_p}",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Stratum & $\\Delta$ Dice [95\\% CI] & max $p$ & Holm $p$ \\\\",
        "\\midrule",
    ]
    for s in strata:
        h = s["headline"]
        lines.append(f"{tex_stratum_label(s['label'])} px & "
                     f"{fmt_delta(h['effect'])} & "
                     f"{fmt_p(h['p_max'])} & {fmt_p(h.get('holm_p'))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    lines += [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Secondary comparisons: each semi-supervised baseline vs.\\ "
        f"{tag_label('ours_50')} on patient-level tumor Dice within each "
        f"stratum (three training seeds {seed_str}). $p$ values are the max "
        "across seeds, Holm-adjusted within each stratum (family of "
        f"{len(vs_ours)} comparisons).}}",
        "\\label{tab:stratified3_vs_ours}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(vs_ours) + "}",
        "\\toprule",
        "Stratum & " + " & ".join(
            f"{tag_short(o)} vs {tag_short('ours_50')}" for o in vs_ours)
        + " \\\\",
        "\\midrule",
    ]
    for s in strata:
        lines.append(f"{tex_stratum_label(s['label'])} px & " + " & ".join(
            fmt_p(t.get("holm_p")) for t in s["tests_vs_ours"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def run_3seed(area):
    """三种子模式主流程（--seeds 非空时调用）。"""
    seeds = [int(s) for s in ARGS.seeds.split(",") if s]
    tags = ARGS.tags_v2.split(",")
    headline = tuple(ARGS.headline.split(":"))
    vs_ours = [t for t in ARGS.vs_ours.split(",") if t]
    assert headline[0] in tags and headline[1] in tags, \
        f"headline {headline} 必须包含在 --tags_v2 中"
    assert "ours_50" in tags, "--tags_v2 必须含 ours_50（次要 family 的比较基准）"
    for tag in tags:
        for sd in seeds:
            csv_path = os.path.join(ARGS.results_dir_v2,
                                    f"{tag}_s{sd}_per_slice.csv")
            df_slices = set(pd.read_csv(csv_path)["slice"])
            missing = df_slices - set(area)
            assert not missing, \
                f"{tag}_s{sd}: {len(missing)} 个切片无 GT 面积（例：{sorted(missing)[:3]}）"
    print(f"三种子模式：seeds={seeds}，tags={tags}，headline={headline}，"
          f"vs_ours={vs_ours}")

    res3 = analyze_3seed(ARGS.results_dir_v2, PAPER_STRATA, area, tags, seeds,
                         headline, vs_ours, ARGS.n_boot, ARGS.seed)
    out = {"protocol": {
        "dice": "results_v2/{tag}_s{seed}_per_slice.csv 的 dice_tumor 列；"
                "仅 GT 含肿瘤切片（面积>0），paper 边界分层",
        "assignment": "左开右闭 (lo, hi]，单位=512x512 切片像素（与单种子版一致）",
        "stats": "层内病例均值 -> 逐种子配对双侧 Wilcoxon，3 种子 p 取最大；"
                 "headline 跨 4 层 Holm 校正，次要 family 层内 Holm 校正；"
                 "效应量 = 患者级配对 Dice 差（每病例先跨种子平均）的均值 + "
                 f"cluster bootstrap 95%CI（n_boot={ARGS.n_boot}, seed={ARGS.seed}）；"
                 "另报双路 cluster bootstrap（患者×种子，与 aggregate_v4_rebuttal "
                 "M6 一致）95%CI 作敏感性",
        "seeds": seeds, "tags": tags, "headline": list(headline),
        "vs_ours": vs_ours, "baseline_note": "主比较为 sup_100 vs ours_50"},
        "results": res3}

    os.makedirs(os.path.dirname(ARGS.out_json_3seed), exist_ok=True)
    with open(ARGS.out_json_3seed, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已保存 {ARGS.out_json_3seed}")

    os.makedirs(os.path.dirname(ARGS.out_tex_3seed), exist_ok=True)
    tex = make_tex_3seed(res3, tags, headline, vs_ours,
                         f"paper 边界 (0,50]/(50,200]/(200,1000]/(1000,inf] px；"
                         f"数字见 {ARGS.out_json_3seed}", ARGS.n_boot)
    with open(ARGS.out_tex_3seed, "w") as f:
        f.write(tex)
    print(f"已保存 {ARGS.out_tex_3seed}")
    if ARGS.sr_dir and os.path.isdir(ARGS.sr_dir):
        dst = os.path.join(ARGS.sr_dir, os.path.basename(ARGS.out_tex_3seed))
        shutil.copyfile(ARGS.out_tex_3seed, dst)
        print(f"已同步复制到 {dst}")

    print("\n=== 三种子分层校验报告（肿瘤 Dice，患者级 %，mean±std over seeds） ===")
    for s in res3["strata"]:
        mline = " ".join(
            f"{t}={s['methods'][t]['mean'] * 100:.1f}±{(s['methods'][t]['std'] or 0) * 100:.1f}"
            if s["methods"][t]["mean"] is not None else f"{t}=--" for t in tags)
        print(f"  {s['label']:>10}px n={s['n_slices']:5d}: {mline}")
        h = s["headline"]
        e = h["effect"]
        if e["diff_mean"] is not None and e["diff_ci95"][0] is not None:
            d_txt = (f"{e['diff_mean'] * 100:+.1f} "
                     f"[{e['diff_ci95'][0] * 100:+.1f}, {e['diff_ci95'][1] * 100:+.1f}]")
        else:
            d_txt = "--"
        print(f"    headline {h['comparison']} ({h['direction']}): "
              f"Δ={d_txt} (n={h['n_cases']}), "
              f"p/seed=" + ", ".join(f"{k}={fmt_p(v)}" for k, v in h["p_per_seed"].items())
              + f" -> max={fmt_p(h['p_max'])}, Holm={fmt_p(h.get('holm_p'))}")
        for t in s["tests_vs_ours"]:
            print(f"    {t['comparison']}: max={fmt_p(t['p_max'])}, "
                  f"Holm={fmt_p(t.get('holm_p'))} (n={t['n_cases']})")


# ---------------------------------------------------------------- main

ARGS = None


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    ap.add_argument("--baseline", default="sup_50")
    ap.add_argument("--compare", default="semi_50,mt_50,cps_50,ours_50",
                    help="与 baseline 配对比较的方法（family 内 Holm 校正）")
    ap.add_argument("--extra_pair", default="sup_100:ours_50",
                    help="额外单一比较 a:b（不校正）；空串跳过")
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--data_root", default="data/CARE")
    ap.add_argument("--out_json", default="results/v2/stratified_stats.json")
    ap.add_argument("--out_tex", default="paper/manuscript/tab_stratified.tex")
    ap.add_argument("--out_tex_quartiles",
                    default="paper/manuscript/tab_stratified_quartiles.tex")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="",
                    help="逗号分隔（如 42,43,44）；非空则进入三种子模式，"
                         "跳过旧单种子流程")
    ap.add_argument("--results_dir_v2", default="results_v2",
                    help="三种子模式的 per_slice.csv 目录（{tag}_s{seed}_per_slice.csv）")
    ap.add_argument("--tags_v2", default=",".join(DEFAULT_TAGS_V2),
                    help="三种子模式的方法 tag（不含 _s{seed} 后缀）")
    ap.add_argument("--headline", default="sup_100:ours_50",
                    help="三种子模式主比较 a:b（跨层 Holm）")
    ap.add_argument("--vs_ours", default="semi_50,mt_50,cps_50,bcp_50",
                    help="三种子模式次要 family：各方法 vs ours_50（层内 Holm）")
    ap.add_argument("--areas_json", default="",
                    help="slice->GT 肿瘤面积 json（冒烟用；优先于 cache/data_root）")
    ap.add_argument("--out_json_3seed",
                    default="results/v2/stratified_stats_3seed.json")
    ap.add_argument("--out_tex_3seed",
                    default="paper/manuscript/tab_stratified_3seed.tex")
    ap.add_argument("--sr_dir", default="paper/sr",
                    help="目录存在时同步复制 3seed tex 到此；空串跳过")
    ARGS = ap.parse_args()

    area = tumor_areas(ARGS.cache_dir, ARGS.data_root, ARGS.areas_json or None)
    if ARGS.seeds:
        run_3seed(area)
        return

    tags = ARGS.tags.split(",")
    comparisons = [c for c in ARGS.compare.split(",") if c]
    extra_pair = tuple(ARGS.extra_pair.split(":")) if ARGS.extra_pair else None

    n_miss = 0
    for tag in tags:
        csv_path = os.path.join(ARGS.results_dir, f"{tag}_per_slice.csv")
        df_slices = set(pd.read_csv(csv_path)["slice"])
        missing = df_slices - set(area)
        n_miss += len(missing)
        assert not missing, f"{tag}: {len(missing)} 个切片无 GT 面积（例：{sorted(missing)[:3]}）"
    print(f"GT 面积加载完成：{len(area)} 切片，"
          f"肿瘤切片 {sum(1 for a in area.values() if a > 0)}")

    pa = boundary_analysis(area)
    print("\n=== 分层边界依据 ===")
    if pa["n_tumor_slices"] == 0:
        print("⚠️ GT 无肿瘤切片，仅输出空统计（冒烟防御路径）")
    else:
        print(f"肿瘤切片 {pa['n_tumor_slices']}/{pa['n_slices_total']}；"
              f"面积中位数 {pa['area_median']:.0f} px，"
              f"范围 [{pa['area_min']:.0f}, {pa['area_max']:.0f}]")
    for b, pct in pa["paper_boundary_percentiles"].items():
        print(f"  paper 边界 {b}px = 肿瘤切片面积分布的第 {pct:.1f} 百分位")
    pos = [a for a in area.values() if a > 0]
    if pos:
        qstrata, qbounds = quartile_strata(pos)
    else:
        qstrata, qbounds = PAPER_STRATA, []
    print(f"  四分位边界（Q1/Q2/Q3）= {qbounds} px")
    for label, d in pa["paper_strata_n"].items():
        print(f"  paper 层 {label}: n={d['n_slices']} ({d['pct']:.1f}%)")

    out = {"protocol": {
        "dice": "results/*_per_slice.csv 的 dice_tumor 列；仅 GT 含肿瘤切片（面积>0）",
        "assignment": "左开右闭 (lo, hi]，单位=512x512 切片像素",
        "stats": "病例内层内均值 -> cluster bootstrap 95%CI（seed=42）;"
                 "配对双侧 Wilcoxon；Holm 在每层 family 内校正",
        "baseline": ARGS.baseline, "comparisons": comparisons,
        "extra_pair": list(extra_pair) if extra_pair else None},
        "boundary_analysis": pa,
        "quartile_boundaries_px": qbounds,
        "stratifications": {}}

    print("\n=== paper 边界分层 ===")
    out["stratifications"]["paper"] = analyze(
        PAPER_STRATA, area, tags, ARGS.baseline, comparisons, extra_pair,
        ARGS.n_boot, ARGS.seed)
    print("=== 四分位边界分层 ===")
    out["stratifications"]["quartiles"] = analyze(
        qstrata, area, tags, ARGS.baseline, comparisons, extra_pair,
        ARGS.n_boot, ARGS.seed)

    for name in ("paper", "quartiles"):
        strat = out["stratifications"][name]
        print(f"\n[{name}]")
        for s in strat["strata"]:
            mline = " ".join(
                f"{t}={s['methods'][t]['patient_mean'] * 100:.1f}%"
                if s["methods"][t] else f"{t}=--" for t in tags)
            print(f"  {s['label']:>10}px n={s['n_slices']:5d}: {mline}")
            for t in s["tests"]:
                print(f"    {t['comparison']}: raw_p={fmt_p(t['raw_p'])} "
                      f"holm_p={fmt_p(t.get('holm_p'))} (n={t['n_cases']})")
            if "extra_test" in s:
                t = s["extra_test"]
                print(f"    {t['comparison']}: raw_p={fmt_p(t['raw_p'])} "
                      f"(n={t['n_cases']}, 未校正)")

    os.makedirs(os.path.dirname(ARGS.out_json), exist_ok=True)
    with open(ARGS.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 {ARGS.out_json}")

    os.makedirs(os.path.dirname(ARGS.out_tex), exist_ok=True)
    with open(ARGS.out_tex, "w") as f:
        f.write(make_tex(out["stratifications"]["paper"], tags, comparisons,
                         extra_pair,
                         f"paper 边界 (0,50]/(50,200]/(200,1000]/(1000,inf] px；"
                         f"边界百分位见 {ARGS.out_json}", label_sfx=""))
    if ARGS.out_tex_quartiles:
        with open(ARGS.out_tex_quartiles, "w") as f:
            f.write(make_tex(out["stratifications"]["quartiles"], tags,
                             comparisons, extra_pair,
                             f"四分位边界 Q1/Q2/Q3={qbounds} px（按肿瘤切片面积分布）",
                             label_sfx="_q"))
    print(f"已保存 {ARGS.out_tex}"
          + (f", {ARGS.out_tex_quartiles}" if ARGS.out_tex_quartiles else ""))


if __name__ == "__main__":
    main()
