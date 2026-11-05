"""汇总 results/*_summary.json -> 论文主表（Markdown）。

python src/aggregate_results.py --results_dir results --out paper/table1.md

命名约定：{method}_{ratio}（CARE，method ∈ sup/semi/mt/cps/ours）；
msd_{method}_{ratio}（MSD Colon 外部验证，二分类只报 tumor）；
s1_{ratio}（stage-1 消融）与 semi_10_thr*/rounds*（超参消融）入附表。
"""

import argparse
import glob
import json
import os
import re

RATIOS = ["1", "5", "10", "25", "50", "100"]
SEMI_RATIOS = RATIOS[:-1]  # 100% 无未标注数据，不跑半监督
METHODS = [("sup", "全监督"), ("semi", "自训练"), ("mt", "Mean Teacher"),
           ("cps", "CPS"), ("ours", "Ours")]
CLASS_NAMES = ["rectum", "tumor"]
METRICS = ["dice", "iou", "nsd"]


def fmt(cell):
    if cell is None:
        return "-"
    return f"{cell['mean']*100:.2f} [{cell['ci95'][0]*100:.2f}, {cell['ci95'][1]*100:.2f}]"


def row_cells(s):
    row = [str(s["n_slices"])]
    for m in METRICS:
        for c in CLASS_NAMES:
            row.append(fmt(s["metrics"].get(c, {}).get(m)))
    return row


HEADER = ["方法", "标注比例", "n_slices"] + \
         [f"{m}_{c}" for m in METRICS for c in CLASS_NAMES]


def table(rows):
    lines = ["| " + " | ".join(HEADER) + " |",
             "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="paper/table1.md")
    args = ap.parse_args()

    summaries = {}
    for path in glob.glob(os.path.join(args.results_dir, "*_summary.json")):
        name = os.path.basename(path)[:-len("_summary.json")]
        with open(path) as f:
            summaries[name] = json.load(f)

    lines = ["# 主结果表（自动生成，勿手改）", "",
             "指标为 mean% [bootstrap 95%CI]，test 集 n_slices 见各 summary。",
             "rectum=正常直肠(类1), tumor=肿瘤(类2)；MSD 为二分类仅 tumor。",
             "MSD 采用与 CARE 完全相同的全切片协议（空 GT 且有假阳性的切片计 0），",
             "该协议在 90% 空切片的 MSD 上系统性压低绝对值，横向比较方法间相对序才有意义。",
             "", "## 表 1：CARE 主结果", ""]

    rows = []
    for r in RATIOS:
        for key, label in METHODS:
            name = f"{key}_{r}"
            if name in summaries:
                rows.append([label, f"{r}%"] + row_cells(summaries[name]))
    lines += table(rows)

    msd_rows = []
    for r in SEMI_RATIOS:
        for key, label in METHODS:
            name = f"msd_{key}_{r}"
            if name in summaries:
                msd_rows.append([label, f"{r}%"] + row_cells(summaries[name]))
    if msd_rows:
        lines += ["", "## 表 2：MSD Colon 外部验证（全切片协议）", ""]
        lines += table(msd_rows)

    s1_rows = [["stage-1 全图（消融）", f"{r}%"] + row_cells(summaries[f"s1_{r}"])
               for r in RATIOS if f"s1_{r}" in summaries]
    abl_rows = []
    for name in sorted(summaries):
        m = re.fullmatch(r"semi_10_(thr[\d.]+|rounds\d+)", name)
        if m:
            abl_rows.append([f"半监督消融（{m.group(1)}）", "10%"] +
                            row_cells(summaries[name]))
    if s1_rows or abl_rows:
        lines += ["", "## 附表：消融", ""]
        lines += table(s1_rows + abl_rows)

    missing = [f"{k}_{r}" for k, _ in METHODS for r in SEMI_RATIOS
               if f"{k}_{r}" not in summaries and not (k == "sup" and False)] + \
              [f"sup_100" if "sup_100" not in summaries else None] + \
              [f"msd_{k}_{r}" for k, _ in METHODS for r in SEMI_RATIOS
               if f"msd_{k}_{r}" not in summaries]
    missing = [m for m in missing if m]
    if missing:
        lines += ["", f"⚠️ 缺失实验: {', '.join(missing)}"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n已保存: {args.out}")


if __name__ == "__main__":
    main()
