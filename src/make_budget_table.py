"""训练预算表 v4（审稿：名义 epoch 匹配口径下的实际计算量差异）。

论点：三种子主结果里所有方法"stage-2 训练 100 epoch、自训练 3 轮 × 60 epoch"
（sup180 为 180 epoch 匹配对照），但名义 epoch 相同 ≠ 计算量相同：
  * sup：每 epoch 只扫标注子集（L 切片）；
  * mt/cps/bcp：每 epoch 扫整个未标注池（U 切片）作主循环，逐步另含标注 batch
    （+ teacher/第二网络前向）；
  * semi/ours：每轮在 标注+伪标注 联合集（L+U）上训 60 epoch，轮间还要对 U 做
    级联 teacher 推理生成伪标签（含在 wall-clock 内）。
本脚本从 runs_v2/*/run_meta.json（各方法记录的实际 L/U/val 切片数）+ log.csv
行数（实际 epoch 数；sup180 的 run_meta epochs 字段是 config 回显，以 log 为准）
+ 文件 btime/mtime（wall-clock；自训练含轮间伪标签生成）计算：
  optimizer 更新次数、student 训练前向切片数、teacher/辅助前向切片数、
  伪标签生成遍数、wall-clock 小时。

口径约定（batch_size=32，各训练脚本均 drop_last=True）：
  sup/s1/sup180: steps/ep = L//32；updates = E×steps；每步 32 切片前向+反向。
  semi/ours:     每轮 steps/ep = (L+U)//32；updates = rounds×60×steps；
                 轮间伪标签生成 = rounds 遍 × U 切片 × 2 次前向（级联两级）。
  mt:            steps/ep = U//32；updates = 100×steps；每步前向 96 切片
                 （student 标注 32 + student 未标注增广 32 + teacher 未标注 32）。
  cps:           steps/ep = U//32；每步两网络各 1 次更新（updates=2×steps）；
                 每步前向 128 切片（两网络 ×（标注 32 + 未标注 32））。
  bcp:           steps/ep = U//32；updates = 100×steps；每步前向 128 切片
                 （teacher 未标注 32 + student 标注 32 + student 双向混合 64）。
  验证前向：sup/semi/ours 每 epoch 1 遍 val；mt/cps/bcp 2 遍（teacher+student /
  两网络），仅记入 JSON，tex 以脚注说明。

产出：
  results/v2/budget.json            全部 run 明细 + 推导量
  paper/manuscript/tab_budget_v4.tex（同步复制 paper/sr/）

用法：python src/make_budget_table.py
"""

import json
import os
import re
import subprocess

import numpy as np

RUNS = "runs_v2"
OUT_JSON = os.path.join("results", "v2", "budget.json")
OUT_TEX = os.path.join("paper", "manuscript", "tab_budget_v4.tex")
OUT_TEX_SR = os.path.join("paper", "sr", "tab_budget_v4.tex")
BATCH = 32
RATIOS = [1, 5, 10, 25, 50]
METHOD_LABEL = {"sup": "Supervised", "s1": "Stage-1 (shared)", "sup180": "Supervised (180 ep)",
                "semi": "Pseudo-label ST", "mt": "Mean Teacher", "cps": "CPS",
                "bcp": "BCP", "ours": "Ours"}
TAG_RE = re.compile(r"^(sup|semi|ours|mt|cps|bcp|s1|sup180)_(\d+)_s(\d+)$")


def _stat_times(path):
    out = subprocess.run(["stat", "-c", "%W %Y", path],
                         capture_output=True, text=True, check=True).stdout.split()
    b, m = int(out[0]), int(out[1])
    return (None if b <= 0 else b), m


def _count_epochs(run_dir):
    single = os.path.join(run_dir, "log.csv")
    rounds = sorted(f for f in os.listdir(run_dir)
                    if re.fullmatch(r"r\d+_log\.csv", f))
    logs = [single] if os.path.exists(single) else \
        [os.path.join(run_dir, f) for f in rounds]
    n = 0
    for lp in logs:
        with open(lp) as f:
            n += sum(1 for _ in f) - 1  # 减表头
    return n, logs


def run_budget(run_dir, tag):
    meta_p = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(meta_p):
        return None
    with open(meta_p) as f:
        rm = json.load(f)
    m = TAG_RE.match(tag)
    method, ratio, seed = m.group(1), int(m.group(2)), int(m.group(3))
    n_epochs, logs = _count_epochs(run_dir)
    if not logs:
        return None
    # wall-clock：首个 log 的 btime -> run_meta/last.pt 的 mtime（与 timing_table 同口径）
    t0 = _stat_times(logs[0])[0]
    train_s = 0.0
    btime_ok = True
    for lp in logs:
        b, mt_ = _stat_times(lp)
        if b is None:
            btime_ok = False
            break
        train_s += mt_ - b
    end = _stat_times(meta_p if os.path.exists(meta_p)
                      else os.path.join(run_dir, "last.pt"))[1]
    wall_s = (end - t0) if (t0 is not None and btime_ok) else None

    val_mode = rm.get("val_mode", "pool")  # runs/ 旧 run 无此字段=pool；runs_v2 均有
    if method in ("sup", "s1", "sup180"):
        L = rm["n_labeled_slices"]
        U = 0
    else:
        if val_mode == "labeled":
            # 标注切片再划 15% 验证：L = n_labeled - n_val；semi/ours 未记 n_val，
            # 用同 ratio+seed 的 sup run 的 n_labeled_slices（同一 split、同一 seed）
            if rm.get("n_val_slices") is not None:
                L = rm["n_labeled"] - rm["n_val_slices"]
            else:
                sup_meta = os.path.join(RUNS, f"sup_{ratio}_s{seed}", "run_meta.json")
                with open(sup_meta) as f:
                    L = json.load(f)["n_labeled_slices"]
        else:  # fixed：标注全集即训练集（val 仅日志）
            L = rm["n_labeled"]
        U = rm["n_unlabeled"]

    rounds = rm.get("rounds")
    epr = rm.get("epochs_per_round")
    if method in ("sup", "s1", "sup180"):
        E = n_epochs
        steps = L // BATCH
        updates = E * steps
        fwd_student = updates * BATCH
        fwd_aux = 0
        pseudo_passes = 0
        val_passes = E  # 每 epoch 1 遍 val
    elif method in ("semi", "ours"):
        E = n_epochs  # = rounds×60
        steps = (L + U) // BATCH
        updates = E * steps
        fwd_student = updates * BATCH
        pseudo_passes = rounds or 0
        fwd_aux = pseudo_passes * U * 2  # 级联 teacher：两级各一次前向
        val_passes = E
    elif method == "mt":
        E = n_epochs
        steps = U // BATCH
        updates = E * steps
        fwd_student = updates * 2 * BATCH   # 标注 32 + 未标注增广 32
        fwd_aux = updates * BATCH           # teacher 未标注 32
        pseudo_passes = 0
        val_passes = 2 * E
    elif method == "cps":
        E = n_epochs
        steps = U // BATCH
        updates = 2 * E * steps             # 两个 optimizer
        fwd_student = updates * 2 * BATCH   # 每网络：标注 32 + 未标注 32
        fwd_aux = 0
        pseudo_passes = 0
        val_passes = 2 * E
    elif method == "bcp":
        E = n_epochs
        steps = U // BATCH
        updates = E * steps
        fwd_student = updates * 3 * BATCH   # 标注 32 + 双向混合 64
        fwd_aux = updates * BATCH           # teacher 未标注 32
        pseudo_passes = 0
        val_passes = 2 * E
    else:
        return None
    return {"method": method, "ratio": ratio, "seed": seed,
            "val_mode": val_mode, "L": L, "U": U, "epochs_actual": E,
            "steps_per_epoch": steps, "optimizer_updates": updates,
            "fwd_slices_student": fwd_student, "fwd_slices_aux": fwd_aux,
            "pseudo_label_passes": pseudo_passes, "val_passes": val_passes,
            "train_seconds": train_s if btime_ok else None,
            "wall_seconds": wall_s,
            "pseudo_gen_seconds": (wall_s - train_s)
            if (wall_s is not None and btime_ok) else None}


def fmt_m(x):
    return f"{x / 1e6:.2f}M" if x >= 1e6 else f"{x / 1e3:.1f}k"


def main():
    runs = {}
    for d in sorted(os.listdir(RUNS)):
        if not TAG_RE.match(d) or not os.path.isdir(os.path.join(RUNS, d)):
            continue
        b = run_budget(os.path.join(RUNS, d), d)
        if b:
            runs[d] = b
    print(f"扫描 {RUNS}: {len(runs)} 个 run")

    # 按 (method, ratio) 聚合三种子：推导量取均值，wall 取中位 [min,max]
    agg = {}
    for tag, b in runs.items():
        key = (b["method"], b["ratio"])
        agg.setdefault(key, []).append(b)
    table = {}
    for (method, ratio), bs in sorted(agg.items()):
        def med(f):
            vals = [b[f] for b in bs if b[f] is not None]
            return (float(np.median(vals)), float(min(vals)), float(max(vals)),
                    len(vals)) if vals else None
        table[f"{method}_{ratio}"] = {
            "n_seeds": len(bs),
            "L_mean": float(np.mean([b["L"] for b in bs])),
            "U_mean": float(np.mean([b["U"] for b in bs])),
            "epochs_actual": int(np.median([b["epochs_actual"] for b in bs])),
            "optimizer_updates": med("optimizer_updates"),
            "fwd_slices_student": med("fwd_slices_student"),
            "fwd_slices_aux": med("fwd_slices_aux"),
            "pseudo_label_passes": int(np.median([b["pseudo_label_passes"]
                                                  for b in bs])),
            "wall_hours": tuple(w / 3600 for w in med("wall_seconds")[:3])
            if med("wall_seconds") else None,
            "pseudo_gen_hours": tuple(p / 3600 for p in med("pseudo_gen_seconds")[:3])
            if med("pseudo_gen_seconds") else None,
        }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"protocol": __doc__.split("产出")[0].strip(),
                   "runs": runs, "aggregated": table}, f, indent=2)

    # ---------------- tex：表 A（50% 档明细）+ 表 B（updates×ratio）+ 表 C（wall×ratio）
    def cell50(method, field, fmt):
        e = table.get(f"{method}_50")
        if not e or e[field] is None:
            return "--"
        return fmt(e[field][0])

    lines = [
        "% 训练预算表 v4：名义 epoch 匹配下的实际计算量。推导量由 runs_v2 的 "
        "run_meta.json 切片计数 + log.csv 实际 epoch 数按训练循环结构解析计算；"
        "wall-clock 为三种子中位数（文件 btime/mtime，含调度噪声）。"
        "由 src/make_budget_table.py 生成",
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Training compute per method at 50\\% labels (CARE; median "
        "over the three seeds 42/43/44 — slice counts vary $\\lesssim$5\\% "
        "across seeds). "
        "\\emph{Updates} = optimizer step count; \\emph{Student fwd} = slices "
        "passed through the trained network during optimization; "
        "\\emph{Aux fwd} = additional forward passes (EMA teacher in MT/BCP, "
        "cascade teacher pseudo-labeling between rounds in ST/Ours); "
        "\\emph{PL gen.} = full pseudo-labeling passes over the unlabeled "
        "pool. Wall = median wall-clock hours over three seeds on one A100 "
        "(includes inter-round pseudo-labeling; excludes one-time data "
        "preload). All methods share the stage-1 model per ratio "
        "(100 epochs over the labeled subset).}",
        "\\label{tab:budget}", "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Epochs & Updates & Student fwd & Aux fwd & PL gen. & Wall (h) \\\\",
        "\\midrule",
    ]
    rows50 = ["sup", "sup180", "semi", "mt", "cps", "bcp", "ours"]
    for m in rows50:
        e = table.get(f"{m}_50")
        if not e:
            continue
        wall = f"{e['wall_hours'][0]:.2f}" if e["wall_hours"] else "--"
        lines.append(
            f"{METHOD_LABEL[m]} & {e['epochs_actual']} & "
            f"{fmt_m(e['optimizer_updates'][0])} & "
            f"{fmt_m(e['fwd_slices_student'][0])} & "
            f"{fmt_m(e['fwd_slices_aux'][0])} & "
            f"{e['pseudo_label_passes']} & {wall} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # 表 B：updates（千次）× ratio
    lines += [
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Optimizer updates (thousands) at each labeled ratio. "
        "Supervised rows scale with the labeled subset; consistency/self-"
        "training rows are dominated by the unlabeled pool "
        "(${\sim}$22.6k slices), so nominal epoch parity (100 or 180) hides "
        "an order-of-magnitude difference in actual compute.}",
        "\\label{tab:budget_updates}", "\\small",
        "\\begin{tabular}{l" + "c" * len(RATIOS) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"{r}\\%" for r in RATIOS) + " \\\\",
        "\\midrule",
    ]
    for m in ["sup", "sup180", "semi", "mt", "cps", "bcp", "ours"]:
        cells = []
        for r in RATIOS:
            e = table.get(f"{m}_{r}")
            cells.append("--" if not e or not e["optimizer_updates"]
                         else f"{e['optimizer_updates'][0] / 1e3:.1f}")
        label = METHOD_LABEL[m] + (" (180 ep)" if m == "sup180" and
                                   "180" not in METHOD_LABEL[m] else "")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # 表 C：wall h（中位 [min,max]）× ratio
    lines += [
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Stage-2 wall-clock hours per labeled ratio (median "
        "[min, max] over three seeds; A100). Self-training walls include the "
        "inter-round pseudo-labeling passes (three passes totaling "
        "${\\sim}$0.05--0.15 h).}",
        "\\label{tab:budget_wall}", "\\small",
        "\\begin{tabular}{l" + "c" * len(RATIOS) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"{r}\\%" for r in RATIOS) + " \\\\",
        "\\midrule",
    ]
    for m in ["s1", "sup", "semi", "mt", "cps", "bcp", "ours"]:
        cells = []
        for r in RATIOS:
            e = table.get(f"{m}_{r}")
            if not e or not e["wall_hours"]:
                cells.append("--")
                continue
            med, lo, hi = e["wall_hours"]
            cells.append(f"{med:.2f} [{lo:.2f}, {hi:.2f}]")
        lines.append(f"{METHOD_LABEL[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines))
    import shutil
    shutil.copyfile(OUT_TEX, OUT_TEX_SR)
    print(f"已保存 {OUT_JSON}")
    print(f"已保存 {OUT_TEX} + {OUT_TEX_SR}")
    # 终端校验
    print("\n=== 50% 档校验 ===")
    for m in rows50:
        e = table.get(f"{m}_50")
        if e:
            print(f"  {m:7s} ep={e['epochs_actual']:3d} "
                  f"upd={e['optimizer_updates'][0]:>9.0f} "
                  f"fwd={e['fwd_slices_student'][0] / 1e6:6.2f}M "
                  f"aux={e['fwd_slices_aux'][0] / 1e6:5.2f}M "
                  f"wall={e['wall_hours'][0]:.2f}h")


if __name__ == "__main__":
    main()
