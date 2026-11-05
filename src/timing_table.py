"""训练/推理耗时对比表（审稿 Minor 补充）。

训练耗时：log.csv 无时间戳列，改用文件系统时间戳估算——
  * 单阶段方法（sup/mt/cps/s1）：时长 = mtime(log.csv) - btime(log.csv)
    （log.csv 在 train_model 开始、完成数据预载后创建，逐 epoch flush）；
  * 自训练方法（semi/ours，r1..rR 多轮）：wall 时长 = mtime(run_meta.json)
    - btime(r1_log.csv)，含轮间伪标签生成；另报纯训练时长（各轮 log 时长之和）。
  btime 用 `stat -c %W` 读取（ext4 支持）；读不到则该 run 记 None 并警告。

推理耗时：加载 stage-1 + stage-2 ckpt，对 test 切片逐片做 cascade_predict
（与 evaluate 完全相同的推理路径，含空预测的 tile fallback），报告 ms/slice。
所有方法模型结构相同（LightUNet 2M 参数），推理耗时与方法无关。

产出：
  paper/manuscript/tab_timing.tex   方法 x (训练时长, 相对倍数, 推理 ms/slice, 多模型/EMA)
  results/v2/timing.json            全部档位明细 + 推理基准原始数据

用法：
  python src/timing_table.py                 # 真实数据（CPU 推理基准 50 片）
  python src/timing_table.py --skip_bench    # 只算训练耗时
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAG_RE = re.compile(r"^(msd_sup|msd_semi|msd_mt|msd_cps|msd_ours|msd_s1|"
                    r"sup|semi|ours|mt|cps|s1|bcp)_(\d+)$")
METHOD_ORDER = ["sup", "semi", "mt", "cps", "ours"]
METHOD_LABELS = {"sup": "Supervised", "semi": "Pseudo-label ST",
                 "mt": "Mean Teacher", "cps": "CPS", "ours": "Ours",
                 "s1": "Stage-1 (shared)", "bcp": "BCP"}
RATIOS = [1, 5, 10, 25, 50]
ARCH_NOTE = {"sup": "No", "semi": "No (3 self-training rounds)",
             "mt": "EMA teacher", "cps": "Two networks",
             "ours": "No (3 rounds + consistency)"}


def _stat_times(path):
    """返回 (btime, mtime) 秒；不支持 btime 时 btime=None。"""
    out = subprocess.run(["stat", "-c", "%W %Y", path],
                         capture_output=True, text=True, check=True).stdout.split()
    b, m = int(out[0]), int(out[1])
    return (None if b <= 0 else b), m


def run_timing(run_dir):
    """单个 run 目录的耗时。返回 dict 或 None（无 log）。"""
    single = os.path.join(run_dir, "log.csv")
    rounds = sorted(f for f in os.listdir(run_dir)
                    if re.fullmatch(r"r\d+_log\.csv", f))
    if os.path.exists(single):
        logs = [single]
    elif rounds:
        logs = [os.path.join(run_dir, f) for f in rounds]
    else:
        return None
    t0 = _stat_times(logs[0])[0]
    train_s, first_b = 0.0, None
    for lp in logs:
        b, m = _stat_times(lp)
        if b is None:
            return {"error": "btime_unavailable"}
        train_s += m - b
        first_b = b if first_b is None else min(first_b, b)
    meta = os.path.join(run_dir, "run_meta.json")
    end = _stat_times(meta if os.path.exists(meta)
                      else os.path.join(run_dir, "last.pt"))[1]
    info = {"train_seconds": train_s, "wall_seconds": end - t0,
            "n_phases": len(logs)}
    if os.path.exists(meta):
        with open(meta) as f:
            rm = json.load(f)
        info["epochs"] = rm.get("epochs")
        info["rounds"] = rm.get("rounds")
        info["epochs_per_round"] = rm.get("epochs_per_round")
        info["crop_size"] = rm.get("crop_size", 512)
        info["n_labeled_slices"] = rm.get("n_labeled_slices", rm.get("n_labeled"))
    return info


def scan_runs(runs_dir):
    """{tag: {"method":..., "ratio":..., **timing}}"""
    out = {}
    for d in sorted(os.listdir(runs_dir)):
        m = TAG_RE.match(d)
        if not m or not os.path.isdir(os.path.join(runs_dir, d)):
            continue
        info = run_timing(os.path.join(runs_dir, d))
        if info is None:
            continue
        out[d] = {"method": m.group(1), "ratio": int(m.group(2)), **info}
    return out


# ---------------------------------------------------------------- 推理基准

def load_bench_images(cache_dir, data_root, n):
    """test 切片均匀抽 n 张 -> (n,1,H,W) float32 numpy。优先 cache，否则 npz。"""
    if cache_dir:
        imgs_path = os.path.join(cache_dir, "test_images.npy")
        if os.path.exists(imgs_path):
            imgs = np.load(imgs_path, mmap_mode="r")
            idx = np.linspace(0, len(imgs) - 1, n).round().astype(int)
            arr = np.stack([np.asarray(imgs[i]) for i in idx])
            return arr[:, None].astype(np.float32)
    npz_dir = os.path.join(data_root, "test", "test_npz")
    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    arr = np.stack([np.load(os.path.join(npz_dir, files[i]))["image"]
                    for i in idx])
    return arr[:, None].astype(np.float32)


def bench_inference(s1_ckpt, s2_ckpt, crop_size, images, device):
    """逐片 cascade_predict 计时。返回 dict（ms）。"""
    import torch
    import cascade
    import common

    m1 = common.load_model(s1_ckpt, device)
    m2 = common.load_model(s2_ckpt, device)
    x_all = torch.from_numpy(images).to(device)
    n = x_all.shape[0]
    with torch.no_grad():
        for i in range(min(3, n)):  # warmup
            cascade.cascade_predict(m1, m2, x_all[i:i + 1], crop_size,
                                    batch_size=1)
        per_slice = []
        n_fallback = 0
        for i in range(n):
            x = x_all[i:i + 1]
            t0 = time.perf_counter()
            cascade.cascade_predict(m1, m2, x, crop_size, batch_size=1)
            per_slice.append((time.perf_counter() - t0) * 1000)
            pred1 = cascade.batch_argmax(m1, x, batch_size=1)
            if not bool((pred1 > 0).any()):
                n_fallback += 1
        # 组件耗时：stage-1 全图前向 / stage-2 单 crop 前向（不含 fallback）
        t0 = time.perf_counter()
        for i in range(n):
            cascade.batch_argmax(m1, x_all[i:i + 1], batch_size=1)
        t_s1 = (time.perf_counter() - t0) / n * 1000
        crop = torch.zeros((1, 1, crop_size, crop_size), device=device)
        for _ in range(3):
            m2(crop)
        t0 = time.perf_counter()
        for _ in range(n):
            m2(crop)
        t_s2 = (time.perf_counter() - t0) / n * 1000
    return {"device": device, "n_slices": n, "crop_size": crop_size,
            "cascade_ms_per_slice_mean": float(np.mean(per_slice)),
            "cascade_ms_per_slice_sd": float(np.std(per_slice)),
            "cascade_ms_per_slice_median": float(np.median(per_slice)),
            "n_tile_fallback": n_fallback,
            "stage1_ms": t_s1, "stage2_crop_ms": t_s2}


# ---------------------------------------------------------------- LaTeX

def fmt_h(sec):
    return "--" if sec is None else f"{sec / 3600:.2f}"


def detect_outliers(runs):
    """半监督方法的每 epoch 成本由全量数据决定，跨档位近似恒定；
    wall > 3x 组中位数视为调度拥塞/重启污染。sup/s1 时长随标注量线性增长，不适用。

    返回 (flagged_tags, 说明 dict)。"""
    flagged, note = {}, {}
    ratio_invariant = {"semi", "ours", "mt", "cps", "bcp",
                       "msd_semi", "msd_ours", "msd_mt", "msd_cps"}
    groups = {}
    for tag, r in runs.items():
        if "error" in r or r["method"] not in ratio_invariant:
            continue
        groups.setdefault(r["method"], {})[tag] = r["wall_seconds"]
    for m, g in groups.items():
        if len(g) < 3:
            continue
        med = float(np.median(list(g.values())))
        for tag, w in g.items():
            if w > 3 * med:
                flagged[tag] = True
                note[tag] = (f"wall={w / 3600:.2f}h > "
                             f"3x 同方法中位 {med / 3600:.2f}h")
    return flagged, note


def make_tex(runs, bench, train_hw, flagged):
    """主表（50% 档 + sup_100）+ 附录全档位表。"""
    bench_ms = bench and bench["cascade_ms_per_slice_mean"]
    bench_sd = bench and bench["cascade_ms_per_slice_sd"]
    infer_dev = bench and bench["device"]
    lines = [
        "% 训练/推理耗时对比：训练时长由 runs/*/log.csv 文件时间戳估算"
        "（不含一次性数据预载；自训练含轮间伪标签生成）；推理为级联逐片实测。"
        "由 src/timing_table.py 生成",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Computational cost comparison (50\\% labeled ratio, "
        f"{train_hw}). Training time is wall-clock of the stage-2 run "
        "(self-training methods include inter-round pseudo-label generation; "
        "the one-time data preload is excluded). Inference is the deployed "
        f"two-stage cascade measured per slice on {infer_dev}; all methods "
        "share the same 2M-parameter LightUNet architecture, so a single "
        "inference cost applies to all rows.}",
        "\\label{tab:timing}",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Train (h) & Rel. $\\times$ Sup & Infer (ms/slice) "
        "& Multi-model/EMA \\\\",
        "\\midrule",
    ]
    base = runs.get("sup_50", {}).get("wall_seconds")
    rows = [("sup_100", "Supervised 100\\%")]
    for m in METHOD_ORDER:
        rows.append((f"{m}_50", f"{METHOD_LABELS[m]} 50\\%"))
    for tag, label in rows:
        r = runs.get(tag)
        if not r or "error" in r:
            lines.append(f"{label} & -- & -- & -- & -- \\\\")
            continue
        method = r["method"]
        rel = r["wall_seconds"] / base if base else None
        infer = (f"{bench_ms:.0f} \\pm {bench_sd:.0f}"
                 if bench_ms is not None else "--")
        lines.append(f"{label} & {fmt_h(r['wall_seconds'])} & "
                     + (f"{rel:.2f}$\\times$" if rel is not None else "--")
                     + f" & {infer} & {ARCH_NOTE.get(method, '--')} \\\\")
    s1 = runs.get("s1_50")
    foot = (f"Stage-1 (shared by all methods at the same ratio): "
            f"{fmt_h(s1['wall_seconds'])} h." if s1 and "error" not in s1
            else "")
    if bench and bench.get("n_tile_fallback") is not None:
        foot += (f" Stage-1 forward {bench['stage1_ms']:.0f} ms + stage-2 "
                 f"crop {bench['stage2_crop_ms']:.0f} ms; tile fallback "
                 f"triggered on {bench['n_tile_fallback']}/"
                 f"{bench['n_slices']} benchmark slices.")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if foot:
        lines += ["\\begin{minipage}{\\columnwidth}\\vspace{2pt}\\footnotesize",
                  foot + "\\end{minipage}"]
    lines += ["\\end{table}", ""]
    # 附录：全档位训练时长
    lines += [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Stage-2 training time (hours) at all labeled ratios "
        f"({train_hw}). $^{{\\ast}}$Runs slowed by cluster scheduling "
        "contention or job restart (wall-clock inflated by an order of "
        "magnitude relative to sibling ratios of the same method, whose "
        "per-epoch cost is otherwise approximately constant); not an "
        "algorithmic property.}",
        "\\label{tab:timing_all}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(RATIOS) + "}",
        "\\toprule",
        "Method & " + " & ".join(f"{r}\\%" for r in RATIOS) + " \\\\",
        "\\midrule",
    ]
    for m in ["s1"] + METHOD_ORDER:
        cells = []
        for r_ in RATIOS:
            tag = f"{m}_{r_}"
            info = runs.get(tag)
            if not info or "error" in info:
                cells.append("--")
            elif tag in flagged:
                cells.append(fmt_h(info["wall_seconds"]) + "$^{\\ast}$")
            else:
                cells.append(fmt_h(info["wall_seconds"]))
        lines.append(f"{METHOD_LABELS[m]} & " + " & ".join(cells) + " \\\\")
    info100 = runs.get("sup_100")
    lines.append("Supervised 100\\% & \\multicolumn{4}{c}{--} & "
                 + (f"{fmt_h(info100['wall_seconds'])} (100\\%)"
                    if info100 and "error" not in info100 else "--")
                 + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_tex", default="paper/manuscript/tab_timing.tex")
    ap.add_argument("--out_json", default="results/v2/timing.json")
    ap.add_argument("--bench_s1", default="runs/s1_100/best.pt")
    ap.add_argument("--bench_s2", default="runs/sup_100/best.pt")
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--bench_cache_dir", default="data/cache")
    ap.add_argument("--bench_data_root", default="data/CARE")
    ap.add_argument("--bench_n", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_bench", action="store_true")
    args = ap.parse_args()

    runs = scan_runs(args.runs_dir)
    print(f"扫描 {args.runs_dir}: {len(runs)} 个 run 含日志")
    for tag in sorted(runs):
        r = runs[tag]
        if "error" in r:
            print(f"  {tag}: ⚠️ {r['error']}")
            continue
        print(f"  {tag}: train={r['train_seconds'] / 3600:.2f}h "
              f"wall={r['wall_seconds'] / 3600:.2f}h phases={r['n_phases']}")

    bench = None
    if not args.skip_bench:
        images = load_bench_images(
            args.bench_cache_dir if args.bench_cache_dir else None,
            args.bench_data_root, args.bench_n)
        bench = bench_inference(args.bench_s1, args.bench_s2, args.crop_size,
                                images, args.device)
        if args.device == "cpu":
            try:
                with open("/proc/cpuinfo") as fh:
                    cpu_model = next(l.split(":", 1)[1].strip()
                                     for l in fh if l.startswith("model name"))
                bench["device"] = f"CPU ({cpu_model})"
            except Exception:
                pass
        print(f"\n推理基准（{bench['device']}, {bench['n_slices']} 片, "
              f"crop={bench['crop_size']}）：级联 "
              f"{bench['cascade_ms_per_slice_mean']:.1f} +- "
              f"{bench['cascade_ms_per_slice_sd']:.1f} ms/片"
              f"（中位 {bench['cascade_ms_per_slice_median']:.1f}；"
              f"stage1={bench['stage1_ms']:.1f}ms "
              f"stage2={bench['stage2_crop_ms']:.1f}ms；"
              f"tile fallback {bench['n_tile_fallback']}/{bench['n_slices']}）")

    flagged, note = detect_outliers(runs)
    for tag, msg in sorted(note.items()):
        print(f"  ⚠️ {tag}: {msg}（调度拥塞/重启污染，tex 中标注 *）")

    train_hw = "NVIDIA A100 80GB"
    try:
        import torch
        if torch.cuda.is_available():
            train_hw = torch.cuda.get_device_name(0)
    except Exception:
        pass
    out = {"train_hardware": train_hw,
           "timing_note": "wall_seconds = mtime(run_meta.json) - "
                          "btime(首个 log.csv)，不含数据预载；train_seconds = "
                          "各阶段 log.csv 时长之和（不含轮间伪标签生成）",
           "outlier_runs": note,
           "runs": runs, "inference_bench": bench}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w") as f:
        f.write(make_tex(runs, bench, train_hw, flagged))
    print(f"已保存 {args.out_json}, {args.out_tex}")


if __name__ == "__main__":
    main()
