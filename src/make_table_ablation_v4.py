"""消融表 v4：旧单种子消融 run 的 per_slice.csv 按 v2/v3 患者级协议重聚合。

背景（审稿一致性修复）：旧 table_ablation.tex 由 make_tables.py 从
results/*_summary.json 生成，指标是**逐切片池化** mean + 逐切片 bootstrap CI；
而主表 table1_care_v3.tex 是**患者级**（病例内均值→跨病例均值，cluster bootstrap
按病例重抽样）。两种口径下同一 ckpt 的肿瘤 Dice 可差约 6 个点（如 ST-10%
55.31 vs 61.02），导致正文/表间矛盾。本脚本不重跑推理，只把旧消融
（runs/ 单种子，seed=42）的 per_slice.csv 重聚合到患者级口径，使消融表与
表 1 的每种子协议完全一致；单种子性质在 caption 注明。

产出：
  paper/manuscript/table_ablation_v4.tex（同步复制到 paper/sr/）
  results/v2/ablation_v4_patient_level.json（含 slice_mean 对照 + ours_25 参照）

用法：python src/make_table_ablation_v4.py
"""

import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reanalyze_v2 import METRICS, analyze_tag, care_gt_presence, validate  # noqa: E402

RESULTS = "results"
OUT_JSON = os.path.join("results", "v2", "ablation_v4_patient_level.json")
OUT_TEX = os.path.join("paper", "manuscript", "table_ablation_v4.tex")
OUT_TEX_SR = os.path.join("paper", "sr", "table_ablation_v4.tex")
CLASSES = ("rectum", "tumor")

# 上半块：stage-1 全图单模型；下半块：10% 档自训练消融（均为 runs/ 旧单种子 run）
ROWS_S1 = [("s1_1", 1), ("s1_5", 5), ("s1_10", 10), ("s1_25", 25),
           ("s1_50", 50), ("s1_100", 100)]
ROWS_ST = [("semi_10", "ST (default)"), ("semi_10_rounds1", "ST, 1 round"),
           ("semi_10_rounds5", "ST, 5 rounds"),
           ("semi_10_thr0.7", "ST, threshold 0.7"),
           ("semi_10_thr0.95", "ST, threshold 0.95")]
# 仅入 JSON 不入表：§2.5「stage-1 低于完整 Ours 管线」同 run 家族（旧单种子）逐比例核对
EXTRA_TAGS = ["ours_1", "ours_5", "ours_10", "ours_25", "ours_50"]


def cell(summary, cls, metric):
    m = summary["classes"][cls]["metrics"][metric]
    return (f"{m['patient_mean'] * 100:.2f} "
            f"[{m['ci95'][0] * 100:.2f}, {m['ci95'][1] * 100:.2f}]")


def row_line(label, ratio, summary, cls):
    cells = [cell(summary, cls, m) for m in METRICS]
    return f"{label} & {ratio}\\% & " + " & ".join(cells) + " \\\\"


def make_tex(summaries, chinese=False):
    lines = [
        "% 消融 v4：同一患者级聚合结果分为正常直肠/肿瘤两个面板，避免八列缩放。"
        "由 src/make_table_ablation_v4.py 生成",
        "\\begin{table}[!htbp]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
    ]
    for panel_idx, cls in enumerate(CLASSES):
        if chinese:
            panel = "A 部分：正常直肠" if cls == "rectum" else "B 部分：肿瘤"
            header = "方案 & 标注比例 & Dice & IoU & NSD \\\\"
            s1_label = "仅 Stage-1"
        else:
            panel = ("Panel A: Normal rectum" if cls == "rectum"
                     else "Panel B: Tumor")
            header = "Variant & Labeled & Dice & IoU & NSD \\\\"
            s1_label = "Stage-1 only"
        lines += ["\\begin{tabular}{llccc}", "\\toprule",
                  f"\\multicolumn{{5}}{{l}}{{\\textit{{{panel}}}}} \\\\",
                  header, "\\midrule"]
        for i, (tag, r) in enumerate(ROWS_S1):
            lines.append(row_line(s1_label if i == 0 else "", r,
                                  summaries[tag], cls))
        lines.append("\\midrule")
        for tag, label in ROWS_ST:
            lines.append(row_line(label, 10, summaries[tag], cls))
        lines += ["\\bottomrule", "\\end{tabular}"]
        if panel_idx == 0:
            lines.append("\\vspace{8pt}")
    if chinese:
        caption = (
            "\\caption{CARE 测试集（81 例）消融实验，按表~\\ref{tab:care_v3_tumor} "
            "的患者级协议重新聚合：前景指标仅在真实标注包含该类的切片上计算，"
            "先在每例内平均再跨病例平均；方括号为重采样\\emph{病例}的聚类自助法 "
            "95\\% CI（10{,}000 次）。每行均为一次单独训练（种子 42），不显示跨种子"
            "变异；``ST (default)'' 是主表对应配置的单次运行。这些消融使用完整训练池"
            "中的验证标签，与三种子主协议的标注内验证不同，因此与主表的小幅偏移"
            "在预期之内；这些行为辅助性探索分析，主协议下的配对轮数比较见补充表~S24。"
            "A/B 两部分分别报告正常直肠和肿瘤；每部分上段为仅 "
            "stage-1 全图模型，下段为 10\\% 标注的自训练变体。}")
    else:
        caption = (
            "\\caption{Ablations on the CARE test set (81 cases), re-aggregated "
            "under the patient-level protocol of Table~\\ref{tab:care_v3_tumor}: "
            "foreground metrics are computed only on slices whose ground truth "
            "contains the class, averaged within each case and then across cases; "
            "brackets are cluster-bootstrap 95\\% CIs resampling \\emph{cases} "
            "(10{,}000 replicates). Every row is a single training run (seed 42), "
            "so no across-seed variability is shown. These ablations used validation "
            "labels from the full training pool, unlike the labeled-only validation "
            "of the three-seed main protocol, so small offsets are expected; these "
            "rows are an auxiliary exploratory analysis, and the paired rounds "
            "comparison under the main protocol is Supplementary "
            "Table~S24. Panels "
            "A and B report normal rectum and tumor, respectively; within each "
            "panel, the upper rows show stage-1 alone and the lower rows show "
            "self-training variants at 10\\% labels.}")
    lines += [caption, "\\label{tab:ablation_v4}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    gt = care_gt_presence()
    tags = [t for t, _ in ROWS_S1] + [t for t, _ in ROWS_ST] + EXTRA_TAGS
    summaries = {}
    for tag in tags:
        csv_path = os.path.join(RESULTS, f"{tag}_per_slice.csv")
        df = pd.read_csv(csv_path).set_index("slice")
        assert set(df.index) == set(gt.keys()), \
            f"{tag}: CSV 切片集与 GT 不一致（csv={len(df)}, gt={len(gt)}）"
        for msg in validate(tag, df, gt, CLASSES):
            print(msg)
        s, _ = analyze_tag(tag, df, gt, CLASSES)
        # 附带 provenance：旧 summary 里的 ckpt 路径 + 逐切片池化均值（旧口径对照）
        with open(os.path.join(RESULTS, f"{tag}_summary.json")) as f:
            old = json.load(f)
        s["ckpt"] = old.get("ckpt")
        s["stage1_ckpt"] = old.get("stage1_ckpt")
        s["slice_pooled_mean"] = {
            cls: {m: old["metrics"][cls][m]["mean"] for m in METRICS}
            for cls in CLASSES}
        summaries[tag] = s
        t = s["classes"]["tumor"]["metrics"]["dice"]
        print(f"{tag:20s} tumor dice: patient {t['patient_mean'] * 100:.2f} "
              f"[{t['ci95'][0] * 100:.2f}, {t['ci95'][1] * 100:.2f}] "
              f"(旧逐切片口径 {old['metrics']['tumor']['dice']['mean'] * 100:.2f})")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"protocol": "v2 patient-level re-aggregation of legacy "
                               "single-seed (42) runs/ ablation CSVs; CI = "
                               "cluster bootstrap over cases (10000 reps, "
                               "seed 42); slice_pooled_mean = 旧逐切片口径对照",
                   "tags": summaries}, f, indent=2)
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write(make_tex(summaries))
    shutil.copyfile(OUT_TEX, OUT_TEX_SR)
    zh_path = os.path.join("paper", "sr", "table_ablation_v4_zh.tex")
    with open(zh_path, "w") as f:
        f.write(make_tex(summaries, chinese=True))
    print(f"已保存 {OUT_JSON}")
    print(f"已保存 {OUT_TEX} + {OUT_TEX_SR} + {zh_path}")


if __name__ == "__main__":
    main()
