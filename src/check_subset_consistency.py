"""审稿 M9 复现性：标注子集跨方法一致性核验 + 子集定义导出。

所有训练脚本（train_supervised / train_semi / train_mt / train_cps / train_bcp）
走同一条确定性划分路径（均为病人级、random.Random(seed) 洗牌）：

    split_train_val(all_train, 0.15, seed)      # dataset.py:79  318 -> 270 pool + 48 val
    labeled_subset(pool, label_ratio, seed)     # dataset.py:92  按病人抽标注子集
    split_labeled_val(labeled, 0.15, seed)      # dataset.py:109 val_mode=labeled 时
                                                # 从标注病例后缀划验证集（<4 例回退 fixed）

因此同一 (seed, label_ratio) 下各方法的标注病例集合必然一致。本脚本提供可审计
证据：(a) 独立重算每个 (seed, ratio) 的划分；(b) 逐个核验 runs_dir 下每个 run
的 run_meta.json 中运行期记录的切片计数（n_labeled_slices / n_labeled /
n_unlabeled / n_val_slices）与重算值完全一致——不同病例集合在全部比例×全部
种子下切片数逐一相等的概率可忽略，计数全等 + 同一确定性代码路径即集合全等。

用法：
    python src/check_subset_consistency.py \
        --data_root data/CARE --runs_dir runs_v2 \
        --report results/v2/subset_consistency.md \
        --export_subsets dist/anon_repo/subsets
    python src/check_subset_consistency.py --selftest   # 冒烟自测（假数据端到端）
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import labeled_subset, list_slices, parse_case_id, split_labeled_val, split_train_val

# run 目录名：{method}_{ratio_pct}_s{seed}（method 可含数字，如 s1/sup180/mtcons）
RUN_RE = re.compile(r"^([a-z]+\d*)_(\d{1,3})(?:_s(\d+))?$")

# train_supervised.py 系：meta 记 n_labeled_slices（val_mode=labeled 时为去掉
# 验证病例后的训练切片数）与 n_val_slices
SUP_FAMILY = {"sup", "s1", "sup180"}
# train_semi/train_mt/train_cps/train_bcp 系：meta 记 n_labeled（全部标注切片，
# 含划入验证的病例）与 n_unlabeled；mt/cps/bcp 另记 n_val_slices
SEMI_FAMILY = {"semi", "ours", "mt", "mtcons", "cps", "bcp"}


def md5_of(items):
    return hashlib.md5("\n".join(sorted(items)).encode()).hexdigest()


def case_ids(files):
    return sorted({parse_case_id(p) for p in files})


def recompute_split(all_train, label_ratio, seed, val_ratio=0.15, val_mode_requested="labeled"):
    """按 dataset.py 的确定性逻辑重算一个 (ratio, seed) 的完整划分。"""
    pool_files, pool_val_files = split_train_val(all_train, val_ratio, seed)
    labeled_files = labeled_subset(pool_files, label_ratio, seed)
    labeled_cases = case_ids(labeled_files)
    # 与 train_*.py 相同的回退规则：val_mode=labeled 且标注病例 <4 -> fixed
    if val_mode_requested == "labeled" and len(labeled_cases) < 4:
        resolved = "fixed"
    else:
        resolved = val_mode_requested
    out = {
        "label_ratio": label_ratio, "seed": seed, "val_ratio": val_ratio,
        "pool_cases": case_ids(pool_files), "pool_n_slices": len(pool_files),
        "pool_val_cases": case_ids(pool_val_files), "pool_val_n_slices": len(pool_val_files),
        "labeled_cases": labeled_cases, "labeled_n_slices": len(labeled_files),
        "val_mode_requested": val_mode_requested, "val_mode": resolved,
        "labeled_train_cases": None, "labeled_val_cases": None,
        "labeled_train_n_slices": None, "labeled_val_n_slices": None,
    }
    if resolved == "labeled":
        tr_files, va_files = split_labeled_val(labeled_files, val_ratio, seed)
        out["labeled_train_cases"] = case_ids(tr_files)
        out["labeled_val_cases"] = case_ids(va_files)
        out["labeled_train_n_slices"] = len(tr_files)
        out["labeled_val_n_slices"] = len(va_files)
    elif resolved == "fixed":
        # train_*.py: val_files = labeled_files（仅日志用，不做模型选择）
        out["labeled_train_cases"] = labeled_cases
        out["labeled_val_cases"] = labeled_cases
        out["labeled_train_n_slices"] = len(labeled_files)
        out["labeled_val_n_slices"] = len(labeled_files)
    else:  # pool
        out["labeled_train_cases"] = labeled_cases
        out["labeled_val_cases"] = case_ids(pool_val_files)
        out["labeled_train_n_slices"] = len(labeled_files)
        out["labeled_val_n_slices"] = len(pool_val_files)
    return out


def check_run(name, meta, split):
    """核验单个 run 的 run_meta.json 与重算划分，返回 (checks, problems)。"""
    method, ratio_pct, seed_s = parse_run_name(name)
    checks = []

    def add(field, recorded, expected):
        checks.append({"field": field, "recorded": recorded,
                       "expected": expected, "ok": recorded == expected})

    add("seed", meta.get("seed"), split["seed"])
    add("label_ratio", round(float(meta.get("label_ratio", -1)), 6),
        round(split["label_ratio"], 6))
    add("val_mode", meta.get("val_mode"), split["val_mode"])

    if method in SUP_FAMILY:
        add("n_labeled_slices", meta.get("n_labeled_slices"),
            split["labeled_train_n_slices"])
        add("n_val_slices", meta.get("n_val_slices"), split["labeled_val_n_slices"])
    elif method in SEMI_FAMILY:
        add("n_labeled", meta.get("n_labeled"), split["labeled_n_slices"])
        add("n_unlabeled", meta.get("n_unlabeled"),
            split["pool_n_slices"] - split["labeled_n_slices"])
        if "n_val_slices" in meta:  # mt/cps/bcp 记录；semi/ours/mtcons 不记录
            add("n_val_slices", meta.get("n_val_slices"),
                split["labeled_val_n_slices"])
    problems = [c for c in checks if not c["ok"]]
    return checks, problems


def parse_run_name(name):
    m = RUN_RE.match(name)
    if not m:
        return None, None, None
    method, ratio_pct, seed = m.group(1), int(m.group(2)), m.group(3)
    return method, ratio_pct, (int(seed) if seed else None)


def scan_runs(runs_dir):
    """扫描 runs_dir -> {(ratio_pct, seed): {run_name: meta}}。seed 以目录名为准，
    目录名无 _sXX 后缀时回退用 meta 的 seed。"""
    groups = {}
    skipped = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        method, ratio_pct, seed = parse_run_name(name)
        meta_path = os.path.join(run_dir, "run_meta.json")
        if method is None or not os.path.exists(meta_path):
            skipped.append(name)
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if seed is None:
            seed = meta.get("seed")
        groups.setdefault((ratio_pct, seed), {})[name] = meta
    return groups, skipped


def run_checks(data_root, runs_dir, report_path=None, export_subsets=None, verbose=True):
    all_train = list_slices(data_root, "train")
    n_total_cases = len(case_ids(all_train))
    groups, skipped = scan_runs(runs_dir)
    lines = []
    all_problems = []
    summary_rows = []
    export_index = []

    for (ratio_pct, seed) in sorted(groups, key=lambda k: (k[0], k[1] if k[1] is not None else -1)):
        metas = groups[(ratio_pct, seed)]
        # 以 meta 的 val_mode_requested 为准（v2 全部为 labeled）；组内不一致即报错
        reqs = {m.get("val_mode_requested", "labeled") for m in metas.values()}
        if len(reqs) > 1:
            all_problems.append((f"(group {ratio_pct}% s{seed})", [
                {"field": "val_mode_requested", "recorded": sorted(reqs),
                 "expected": "组内一致", "ok": False}]))
        split = recompute_split(all_train, ratio_pct / 100.0, seed,
                                val_mode_requested=sorted(reqs)[0])
        split["labeled_md5"] = md5_of(split["labeled_cases"])
        group_ok = True
        detail = []
        for name in sorted(metas):
            checks, problems = check_run(name, metas[name], split)
            status = "PASS" if not problems else "FAIL"
            if problems:
                group_ok = False
                all_problems.append((name, problems))
            rec = {c["field"]: c["recorded"] for c in checks}
            detail.append((name, rec, status))
        n_lab = len(split["labeled_cases"])
        summary_rows.append({
            "ratio_pct": ratio_pct, "seed": seed,
            "n_labeled_cases": n_lab,
            "n_labeled_slices": split["labeled_n_slices"],
            "labeled_md5": split["labeled_md5"],
            "val_mode": split["val_mode"],
            "n_runs": len(metas), "ok": group_ok,
        })
        lines.append((split, detail))
        if export_subsets is not None:
            os.makedirs(export_subsets, exist_ok=True)
            unlabeled_cases = sorted(set(split["pool_cases"]) - set(split["labeled_cases"]))
            payload = {
                "description": "CARE DataV6 train 病人级标注子集定义（dataset.py "
                               "split_train_val -> labeled_subset -> split_labeled_val "
                               "确定性重算；与本仓库全部训练脚本同一代码路径）",
                "label_ratio": ratio_pct / 100.0, "seed": seed,
                "val_ratio": split["val_ratio"],
                "n_train_cases_total": n_total_cases,
                "n_pool_cases": len(split["pool_cases"]),
                "pool_val_cases_15pct_holdout": split["pool_val_cases"],
                "n_labeled_cases": n_lab,
                "labeled_cases": split["labeled_cases"],
                "unlabeled_pool_cases": unlabeled_cases,
                "val_mode_requested": split["val_mode_requested"],
                "val_mode_resolved": split["val_mode"],
                "labeled_val_cases": (split["labeled_val_cases"]
                                      if split["val_mode"] == "labeled" else None),
                "labeled_train_cases": split["labeled_train_cases"],
                "n_labeled_slices": split["labeled_n_slices"],
                "n_labeled_train_slices": split["labeled_train_n_slices"],
                "n_labeled_val_slices": (split["labeled_val_n_slices"]
                                         if split["val_mode"] == "labeled" else None),
                "md5_labeled_cases": split["labeled_md5"],
            }
            out_path = os.path.join(
                export_subsets, f"labeled_ratio{ratio_pct:03d}_s{seed}.json")
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            export_index.append({"file": os.path.basename(out_path),
                                 "label_ratio": ratio_pct / 100.0, "seed": seed,
                                 "n_labeled_cases": n_lab,
                                 "md5_labeled_cases": split["labeled_md5"]})

    if export_subsets is not None:
        with open(os.path.join(export_subsets, "index.json"), "w") as f:
            json.dump({"subsets": export_index}, f, indent=2, ensure_ascii=False)

    ok_all = not all_problems
    if verbose:
        print(f"runs_dir={runs_dir} 共 {sum(len(v) for v in groups.values())} 个 run，"
              f"{len(groups)} 个 (ratio, seed) 组；skipped={skipped}")
        for row in summary_rows:
            mark = "OK " if row["ok"] else "FAIL"
            print(f"[{mark}] ratio={row['ratio_pct']:>3}% seed={row['seed']} "
                  f"labeled={row['n_labeled_cases']:>3} cases "
                  f"({row['n_labeled_slices']:>6} slices) "
                  f"md5={row['labeled_md5'][:12]} val_mode={row['val_mode']:<8} "
                  f"runs={row['n_runs']}")
        print("RESULT:", "ALL PASS" if ok_all else "FAILURES PRESENT")

    if report_path:
        write_report(report_path, data_root, runs_dir, n_total_cases,
                     summary_rows, lines, skipped, ok_all)
    return ok_all, summary_rows, all_problems


def write_report(path, data_root, runs_dir, n_total_cases, summary_rows, lines,
                 skipped, ok_all):
    """Markdown 证据报告（中文正文 + 英文摘要，供 rebuttal 引用）。"""
    out = []
    out.append("# 标注子集跨方法一致性核验（审稿 M9 复现性）\n")
    out.append("**English summary.** All training entry points "
               "(supervised stage-1/stage-2, pseudo-label self-training, Mean "
               "Teacher, CPS, BCP, and our consistency-filtered self-training) draw "
               "their labeled case sets from the *same deterministic patient-level "
               "splitting pipeline* (`dataset.split_train_val` -> `labeled_subset` -> "
               "`split_labeled_val`, all seeded shuffles). This report recomputes "
               "every (seed, label ratio) split independently and verifies that the "
               "slice counts recorded at run time in each `run_meta.json` match the "
               "recomputed values exactly. Identical recorded counts across all "
               "methods, ratios and seeds, produced by one deterministic code path, "
               "establish that the labeled case sets are identical across methods. "
               f"Result: **{'ALL CHECKS PASSED' if ok_all else 'FAILURES PRESENT'}**.\n")
    out.append("## 核验方法\n")
    out.append(f"- 数据：`{data_root}` train（{n_total_cases} 例）；runs：`{runs_dir}`")
    out.append("- 划分代码：`src/dataset.py:79` `split_train_val`（318 例 → 270 例池 + "
               "48 例 15% 池级留出）、`src/dataset.py:92` `labeled_subset`（按病人抽"
               "标注子集）、`src/dataset.py:109` `split_labeled_val`（val_mode=labeled "
               "时从标注病例后缀划 15% 验证集；标注病例 <4 时各训练脚本回退 fixed）")
    out.append("- 调用点：`src/train_supervised.py:57-76`、`src/train_semi.py:217-237`、"
               "`src/train_mt.py:202-225`、`src/train_cps.py:216-240`、"
               "`src/train_bcp.py:223-243`——全部使用同一组函数、同一 (seed, ratio)")
    out.append("- 字段语义：sup 系（sup/s1/sup180）`n_labeled_slices` = 实际训练切片数"
               "（val_mode=labeled 时为标注子集去掉验证病例后）；semi 系"
               "（semi/ours/mt/mtcons/cps/bcp）`n_labeled` = 全部标注切片、"
               "`n_unlabeled` = 池内未标注切片；mt/cps/bcp 另记 `n_val_slices`")
    out.append("- 判定：逐 run 核对 seed / label_ratio / val_mode / 各切片计数与独立"
               "重算值完全一致；组内所有 run 对得上同一份重算划分 ⟹ 标注病例集合相同"
               "（计数由运行期实际文件列表得出，且集合不同则切片计数几乎必不同）\n")
    out.append("## 总表（每比例 × 每种子）\n")
    out.append("| ratio | seed | 标注病例数 | 标注切片数 | labeled 列表 md5(前12) "
               "| val_mode | 核验 run 数 | 结果 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        out.append(f"| {r['ratio_pct']}% | {r['seed']} | {r['n_labeled_cases']} "
                   f"| {r['n_labeled_slices']} | `{r['labeled_md5'][:12]}` "
                   f"| {r['val_mode']} | {r['n_runs']} "
                   f"| {'PASS' if r['ok'] else '**FAIL**'} |")
    out.append("")
    out.append("## 各组明细\n")
    for split, detail in lines:
        ratio_pct = int(round(split["label_ratio"] * 100))
        out.append(f"### ratio={ratio_pct}% seed={split['seed']} "
                   f"（标注 {len(split['labeled_cases'])} 例 / "
                   f"{split['labeled_n_slices']} 切片，val_mode={split['val_mode']}）\n")
        if len(split["labeled_cases"]) <= 30:
            out.append("标注病例列表：" + ", ".join(split["labeled_cases"]) + "\n")
        else:
            out.append(f"标注病例列表 md5：`{split['labeled_md5']}`"
                       f"（完整清单见导出 subsets/labeled_ratio{ratio_pct:03d}"
                       f"_s{split['seed']}.json）\n")
        if split["val_mode"] == "labeled":
            out.append(f"标注内验证集：{len(split['labeled_val_cases'])} 例 / "
                       f"{split['labeled_val_n_slices']} 切片；训练用标注："
                       f"{len(split['labeled_train_cases'])} 例 / "
                       f"{split['labeled_train_n_slices']} 切片\n")
        else:
            out.append("val_mode=fixed：不做验证集模型选择（best.pt=最终 epoch）；"
                       "日志 val 在训练标注切片上计算，仅作参考\n")
        out.append("| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |")
        out.append("|---|---|---|---|---|---|")
        for name, rec, status in detail:
            counts = ", ".join(f"{k}={v}" for k, v in rec.items()
                               if k not in ("seed", "label_ratio", "val_mode"))
            out.append(f"| {name} | {rec['seed']} | {rec['label_ratio']} "
                       f"| {rec['val_mode']} | {counts} | {status} |")
        out.append("")
    if skipped:
        out.append("## 跳过的目录（无 run_meta.json 或命名不符）\n")
        out.append(", ".join(skipped) + "\n")
    out.append("## 结论\n")
    out.append("**全部核验通过**。" if ok_all else "**存在不一致，见上表 FAIL 行**。")
    out.append("同一 (比例, 种子) 下 sup / semi / mt / cps / ours / bcp（含 stage-1 "
               "的 s1）使用的标注病例集合完全一致；集合由 `src/dataset.py` 的确定性"
               "病人级划分唯一决定，逐 (比例, 种子) 的病例清单已随匿名代码包导出"
               "（subsets/）。")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"报告已写入 {path}")


def selftest():
    """假数据端到端自测：真实跑 3 个微型训练，再核验 + 篡改检测。"""
    import numpy as np

    import make_dummy
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        data_root = os.path.join(tmp, "dummy")
        rng = np.random.default_rng(0)
        make_dummy.write_split(data_root, "train", rng, 8, 4)
        make_dummy.write_split(data_root, "test", rng, 3, 4)
        runs_dir = os.path.join(tmp, "runs")
        py = sys.executable

        def run(script, config, *extra):
            cmd = [py, os.path.join(repo_root, "src", script),
                   "--config", os.path.join(repo_root, "configs", config),
                   "--data_root", data_root, "--batch_size", "4",
                   "--device", "cpu", "--val_mode", "labeled"] + list(extra)
            # 父进程 import numpy 后 os.environ 会带上 MKL_THREADING_LAYER=
            # INTEL，传给子进程会与 torch 的 GNU OpenMP(libgomp) 冲突报错，
            # 子进程环境中剔除该变量
            env = {k: v for k, v in os.environ.items()
                   if k != "MKL_THREADING_LAYER"}
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           cwd=repo_root, env=env)

        # labeled 模式（100% 档：7 个池病例全标注，>=4）+ fixed 回退（10% 档：1 例）
        run("train_supervised.py", "label_ratio_100.json",
            "--out_dir", os.path.join(runs_dir, "sup_100_s42"),
            "--seed", "42", "--epochs", "1")
        run("train_supervised.py", "label_ratio_10.json",
            "--out_dir", os.path.join(runs_dir, "sup_10_s42"),
            "--seed", "42", "--epochs", "1")
        run("train_semi.py", "label_ratio_50.json",
            "--teacher_ckpt", os.path.join(runs_dir, "sup_100_s42", "best.pt"),
            "--out_dir", os.path.join(runs_dir, "semi_50_s42"),
            "--seed", "42", "--rounds", "1", "--epochs", "1")
        ok, rows, problems = run_checks(data_root, runs_dir, verbose=False)
        assert ok and len(rows) == 3, f"自测应全过且有 3 组: {problems}"
        # 篡改检测：改坏一个 meta 的 n_labeled，必须被查出
        meta_path = os.path.join(runs_dir, "semi_50_s42", "run_meta.json")
        meta = json.load(open(meta_path))
        meta["n_labeled"] += 1
        json.dump(meta, open(meta_path, "w"))
        ok2, _, problems2 = run_checks(data_root, runs_dir, verbose=False)
        assert not ok2 and problems2 and problems2[0][0] == "semi_50_s42", \
            "篡改 n_labeled 未被检出"
        print("SELFTEST OK: 3 runs PASS，篡改检测生效")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/CARE")
    ap.add_argument("--runs_dir", default="runs_v2")
    ap.add_argument("--report", default=None, help="Markdown 证据报告输出路径")
    ap.add_argument("--export_subsets", default=None,
                    help="导出每 (ratio, seed) 标注病例清单 JSON 的目录")
    ap.add_argument("--selftest", action="store_true",
                    help="假数据端到端自测（含篡改检测），冒烟测试用")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    ok, _, _ = run_checks(args.data_root, args.runs_dir, args.report,
                          args.export_subsets)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
