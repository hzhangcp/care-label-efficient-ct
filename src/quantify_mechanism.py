"""量化跨阶段一致性过滤的机制证据（reviewer major comment 6）。

核心论点：跨阶段一致性过滤能去除置信度阈值无法去除的"高置信度假阳性"伪标签。
CARE 训练池的"未标注"切片实际有 GT，因此可以直接度量伪标签质量。

做法（严格镜像 train_semi.py 的伪标签生成路径）：
- 对每个标注比例 {10,50}% 重建未标注池（seed 42，病人级，同 train_semi.main），
  按固定 stride 确定性子采样（默认每 4 片取 1，约 5k 片/配置）；
- 第 r 轮 teacher 定义与训练一致：r=1 用 run_meta.json 的 teacher_init（sup 模型），
  r>=2 用 runs/{method}_{ratio}/teacher_r{r-1}.pt；stage-1 = meta 的 stage1_ckpt；
- 用 cascade.cascade_predict（tile_fallback 默认开，同 fill_pseudo_cascade）推理，
  keep = conf >= threshold；Ours 额外 keep &= (pred1 == pred)；
- 对隐藏 GT 统计（按类 rectum/tumor 及 pooled 前景）：
  * 过滤前（仅置信度）/过滤后（各方法完整过滤）的伪标签像素 precision/recall；
  * Ours 被移除像素的组成：真前景被误删 / 假阳性（预测前景≠GT，细分
    背景被标为前景、前景类间混淆）/ 前景被标为背景 / 正确背景被误删；
  * 保留像素覆盖率与像素错误率（逐轮演化：过滤是否减缓错误累积）。

产出（数字全部现算，JSON 落盘后可无 GPU 重生成图表）：
  results/mechanism/{method}_{ratio}_round{r}.json   每配置原始计数 + 派生指标
  results/mechanism/summary.json                     汇总
  paper/tabS4_pseudo_quality.tex  + paper/manuscript/ 同名副本
  paper/figS3_pseudo_quality.png  + paper/manuscript/ 同名副本

三种子模式（v2 收尾）：--runs_dir runs_v2 --seeds 42 43 44
  --results_dir results/mechanism_v2 时，逐种子跑
  runs_v2/{method}_{ratio}_s{seed}（每种子独立病人级划分与未标注池），
  JSON 命名为 {method}_{ratio}_s{seed}_round{r}.json，tabS4/figS3
  输出三种子 mean±std，并额外落 agg_seeds.json。

用法：
  python src/quantify_mechanism.py                      # GPU 全量 + 图表
  python src/quantify_mechanism.py --tables_only        # 仅用已有 JSON 重出图表
  python src/quantify_mechanism.py --runs_dir runs_v2 --seeds 42 43 44 \
      --results_dir results/mechanism_v2 [--tables_only]

覆盖率分解模式（审稿意见：25% 覆盖率是否被 256² 裁剪窗/画布填充值主导）：
  python src/quantify_mechanism.py --coverage_breakdown --runs_dir runs_v2 \
      --seeds 42 43 44 --device cpu
  对三种子逐 (ratio,seed) 精确计算 stage-2 视野像素数（裁剪窗 256²；
  stage-1 空预测触发 tile_fallback 的切片为全画布），并对 --breakdown_seed
  （默认 42）重跑扩展级联累积（新增 raw=不过滤变体与分类别 GT 保留率），
  合并 results/mechanism_v2 已有 kept 计数，输出：
    results/v2/coverage_breakdown.json       全图/裁剪内覆盖率 + 分类别保留比例
    results/v2/coverage_breakdown/*.json     seed42 扩展逐配置原始计数
    paper/snippet_coverage.txt               英文说明段（rebuttal/补充材料用）
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dataset
from cascade import cascade_predict
from common import load_model, resolve_device
from dataset import IGNORE_INDEX, labeled_subset, list_slices, split_train_val
from tensorstore import TensorStore

METHODS = ["semi", "ours"]
METHOD_LABEL = {"semi": "Pseudo-label ST", "ours": "Ours"}
FG_CLASSES = (1, 2)            # 1=rectum, 2=tumor
CLASS_LABEL = {1: "rectum", 2: "tumor"}
RESULTS_DIR = os.path.join("results", "mechanism")
PAPER_DIRS = ["paper", os.path.join("paper", "manuscript")]


def unlabeled_pool(meta, stride):
    """按 train_semi.py 的逻辑重建未标注池并确定性子采样。"""
    all_train = list_slices(meta.get("resolved_data_root") or meta["data_root"],
                            "train")
    pool_files, _ = split_train_val(all_train, meta.get("val_ratio", 0.15),
                                    meta.get("seed", 42))
    labeled_files = labeled_subset(pool_files, meta["label_ratio"],
                                   meta.get("seed", 42))
    labeled_set = set(labeled_files)
    unlabeled = [p for p in pool_files if p not in labeled_set]
    return unlabeled[::stride]


def empty_counts():
    """单配置累加器：kept 三种口径（raw/conf/full）+ Ours 移除像素组成。

    raw = 不过滤（conf>0，即 stage-2 视野内全部像素）；conf = 仅置信度阈值
    （等价 ST 的完整过滤）；full = 各方法完整过滤（Ours = conf + 一致性）。
    """
    return {
        "n_pixels": 0,
        "crop_pixels": 0,        # conf>0 像素数 = stage-2 视野（裁剪窗/兜底平铺）
        "fallback_slices": 0,    # stage-1 空预测触发 tile_fallback 的切片数
        "gt": {c: 0 for c in FG_CLASSES},
        "variants": {
            v: {"kept": 0, "correct": 0,
                "pred": {c: 0 for c in FG_CLASSES},
                "tp": {c: 0 for c in FG_CLASSES},
                "gt_kept": {c: 0 for c in FG_CLASSES}}
            for v in ("raw", "conf", "full")
        },
        "removed": {"total": 0, "true_fg": 0, "fp_bg": 0, "wrong_cls": 0,
                    "bg_on_fg": 0, "true_bg": 0},
    }


@torch.no_grad()
def accumulate_config(model1, model2, store, conf_threshold, crop_size,
                      consistency, batch_size):
    """镜像 fill_pseudo_cascade 的 keep 逻辑，对 GT 累计像素计数。"""
    model1.eval()
    model2.eval()
    cnt = empty_counts()
    n = len(store)
    for off in range(0, n, batch_size):
        end = min(off + batch_size, n)
        pred, conf, pred1 = cascade_predict(model1, model2,
                                            store.images[off:end], crop_size,
                                            end - off, return_stage1=True)
        gt = store.labels[off:end]
        keep_raw = conf > 0
        keep_conf = conf >= conf_threshold
        keep_full = keep_conf & (pred1 == pred) if consistency else keep_conf
        cnt["n_pixels"] += pred.numel()
        cnt["crop_pixels"] += int(keep_raw.sum())
        # 与 cascade_predict 内 empty 判定一致：stage-1 无任何前景像素
        cnt["fallback_slices"] += int((~(pred1 > 0).flatten(1)
                                       .any(dim=1)).sum())
        for c in FG_CLASSES:
            cnt["gt"][c] += int((gt == c).sum())
        for v, keep in (("raw", keep_raw), ("conf", keep_conf),
                        ("full", keep_full)):
            vv = cnt["variants"][v]
            vv["kept"] += int(keep.sum())
            vv["correct"] += int((keep & (pred == gt)).sum())
            for c in FG_CLASSES:
                pc = keep & (pred == c)
                vv["pred"][c] += int(pc.sum())
                vv["tp"][c] += int((pc & (gt == c)).sum())
                vv["gt_kept"][c] += int((keep & (gt == c)).sum())
        if consistency:
            rem = keep_conf & ~keep_full     # 过了置信度但被一致性过滤删掉
            rr = cnt["removed"]
            fg_pred = pred > 0
            rr["total"] += int(rem.sum())
            rr["true_fg"] += int((rem & (pred == gt) & fg_pred).sum())
            rr["fp_bg"] += int((rem & fg_pred & (gt == 0)).sum())
            rr["wrong_cls"] += int((rem & fg_pred & (gt > 0)
                                    & (pred != gt)).sum())
            rr["bg_on_fg"] += int((rem & (pred == 0) & (gt > 0)).sum())
            rr["true_bg"] += int((rem & (pred == gt) & (pred == 0)).sum())
    return cnt


def derive_metrics(cnt):
    """由原始计数派生 precision/recall/coverage/error 等指标。

    coverage = kept/全画布像素（补充材料表 S3 口径）；coverage_in_crop =
    kept/stage-2 视野像素（裁剪窗或兜底平铺覆盖区）；retention = 各类
    GT 前景像素被保留的比例（gt_kept/gt，不问预测标签是否正确）。
    """
    crop_px = cnt.get("crop_pixels", 0)
    out = {"n_pixels": cnt["n_pixels"], "crop_pixels": crop_px,
           "fallback_slices": cnt.get("fallback_slices", 0),
           "crop_area_frac": crop_px / max(cnt["n_pixels"], 1),
           "gt": cnt["gt"], "variants": {},
           "removed": cnt["removed"]}
    for v, vv in cnt["variants"].items():
        m = {"kept": vv["kept"],
             "coverage": vv["kept"] / max(cnt["n_pixels"], 1),
             "coverage_in_crop": vv["kept"] / max(crop_px, 1),
             "error_rate": 1 - vv["correct"] / max(vv["kept"], 1),
             "per_class": {}}
        tp_pool = pred_pool = gt_pool = 0
        for c in FG_CLASSES:
            tp, pr, gt = vv["tp"][c], vv["pred"][c], cnt["gt"][c]
            gk = vv.get("gt_kept", {}).get(c, 0)
            m["per_class"][CLASS_LABEL[c]] = {
                "precision": tp / max(pr, 1), "recall": tp / max(gt, 1),
                "retention": gk / max(gt, 1),
                "tp": tp, "pred": pr, "gt": gt, "gt_kept": gk}
            tp_pool += tp
            pred_pool += pr
            gt_pool += gt
        m["pooled"] = {"precision": tp_pool / max(pred_pool, 1),
                       "recall": tp_pool / max(gt_pool, 1)}
        out["variants"][v] = m
    rr = cnt["removed"]
    if rr["total"] > 0:
        fp_total = rr["fp_bg"] + rr["wrong_cls"]
        out["removed_frac"] = {
            "of_conf_kept": rr["total"]
            / max(cnt["variants"]["conf"]["kept"], 1),
            "true_fg": rr["true_fg"] / rr["total"],
            "fp": fp_total / rr["total"],
            "fp_bg": rr["fp_bg"] / rr["total"],
            "wrong_cls": rr["wrong_cls"] / rr["total"],
            "bg_on_fg": rr["bg_on_fg"] / rr["total"],
            "true_bg": rr["true_bg"] / rr["total"],
            "correct": (rr["true_fg"] + rr["true_bg"]) / rr["total"]}
    return out


def run_gpu(args):
    """逐 (seed, ratio, method, round) 推理并保存 results_dir/*.json。"""
    device = resolve_device(args.device)
    dataset.set_num_classes(3)          # CARE 三分类
    os.makedirs(args.results_dir, exist_ok=True)
    seeds = args.seeds or [None]
    summary = {}
    for ratio in args.ratios:
        for seed in seeds:
            suffix = f"_s{seed}" if seed is not None else ""
            with open(os.path.join(args.runs_dir, f"semi_{ratio}{suffix}",
                                   "run_meta.json")) as f:
                meta = json.load(f)
            files = unlabeled_pool(meta, args.stride)
            print(f"[ratio {ratio}%{suffix}] 未标注池 {meta['n_unlabeled']} 片，"
                  f"子采样 stride={args.stride} -> {len(files)} 片")
            store = TensorStore.from_files(files, device, meta.get("cache_dir"))
            model1 = load_model(meta["stage1_ckpt"], device)
            for method in METHODS:
                run_dir = os.path.join(args.runs_dir,
                                       f"{method}_{ratio}{suffix}")
                with open(os.path.join(run_dir, "run_meta.json")) as f:
                    mmeta = json.load(f)
                consistency = bool(mmeta.get("consistency"))
                for r in args.rounds:
                    teacher = (mmeta["teacher_init"] if r == 1 else
                               os.path.join(run_dir, f"teacher_r{r - 1}.pt"))
                    model2 = load_model(teacher, device)
                    cnt = accumulate_config(model1, model2, store,
                                            mmeta["conf_threshold"],
                                            mmeta["crop_size"], consistency,
                                            args.batch_size)
                    rec = {"method": method, "ratio": ratio, "seed": seed,
                           "round": r,
                           "teacher": teacher,
                           "stage1_ckpt": mmeta["stage1_ckpt"],
                           "consistency": consistency,
                           "conf_threshold": mmeta["conf_threshold"],
                           "crop_size": mmeta["crop_size"],
                           "n_unlabeled_full": mmeta["n_unlabeled"],
                           "stride": args.stride, "n_slices": len(files),
                           "metrics": derive_metrics(cnt)}
                    path = os.path.join(
                        args.results_dir,
                        f"{method}_{ratio}{suffix}_round{r}.json")
                    with open(path, "w") as f:
                        json.dump(rec, f, indent=2)
                    summary[f"{method}_{ratio}{suffix}_round{r}"] = rec
                    m = rec["metrics"]
                    tum = m["variants"]["full"]["per_class"]["tumor"]
                    print(f"  {method}_{ratio}{suffix} r{r}: "
                          f"cov={m['variants']['full']['coverage']:.2%} "
                          f"tumor P={tum['precision']:.4f} "
                          f"R={tum['recall']:.4f} "
                          f"err={m['variants']['full']['error_rate']:.2%}")
                    del model2
            del store, model1
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"已保存 {len(summary)} 个配置到 {args.results_dir}/")


def load_records(results_dir, ratios, rounds):
    """读取 {(method, ratio, seed, round): record}（旧 JSON 无 seed 键=None）。"""
    recs = {}
    for path in glob.glob(os.path.join(results_dir, "*_round*.json")):
        with open(path) as f:
            rec = json.load(f)
        if rec["ratio"] in ratios and rec["round"] in rounds:
            recs[(rec["method"], rec["ratio"], rec.get("seed"),
                  rec["round"])] = rec
    return recs


# ------------------------------------------------------- 覆盖率分解（审稿意见 A）
@torch.no_grad()
def stage1_crop_area(model1, store, crop_size, batch_size):
    """仅跑 stage-1，精确统计 stage-2 视野像素数。

    非空预测切片 = crop_size²；stage-1 空预测切片触发 tile_fallback = 全画布。
    与 cascade_predict 的 empty 判定逐切片一致。
    """
    from cascade import batch_argmax
    h, w = store.images.shape[2], store.images.shape[3]
    crop_px = fb = 0
    for off in range(0, len(store), batch_size):
        pred1 = batch_argmax(model1, store.images[off:off + batch_size],
                             batch_size)
        empty = ~(pred1 > 0).flatten(1).any(dim=1)
        fb += int(empty.sum())
        crop_px += (int((~empty).sum()) * crop_size * crop_size
                    + int(empty.sum()) * h * w)
    return {"crop_pixels": crop_px, "fallback_slices": fb,
            "n_pixels": len(store) * h * w}


def _mean_std(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals), "values": vals}


def run_coverage_breakdown(args):
    """覆盖率分解：全图 vs 裁剪内覆盖率 + 各前景类别 GT 保留比例。

    - 逐 (ratio, seed) 仅跑 stage-1 精确计算视野像素数（所有 seed）；
    - 对 --breakdown_seed 重跑扩展级联累积（raw/conf/full 三口径 +
      分类别 retention），逐配置 JSON 落 <breakdown_out 同名目录>/；
    - 合并 mechanism_dir 已有 36 配置的 kept 计数换算裁剪内覆盖率，
      三种子 mean±std 聚合，写 breakdown_out + 英文 snippet。
    """
    device = resolve_device(args.device)
    dataset.set_num_classes(3)          # CARE 三分类
    seeds = args.seeds or [42, 43, 44]
    out_path = args.breakdown_out
    snippet_path = args.snippet_out
    if args.smoke:                      # 冒烟：单配置、单轮、假数据
        specs = [(10, None, os.path.join(args.runs_dir, "semi_10_s2"),
                  os.path.join(args.runs_dir, "semi_10_s2"))]
        rounds = [1]
        out_path = os.path.join("results", "smoke",
                                "coverage_breakdown_smoke.json")
        snippet_path = os.path.join("results", "smoke",
                                    "snippet_coverage_smoke.txt")
    else:
        specs = []
        for ratio in args.ratios:
            for seed in seeds:
                suffix = f"_s{seed}"
                specs.append((ratio, seed,
                              os.path.join(args.runs_dir,
                                           f"semi_{ratio}{suffix}"),
                              os.path.join(args.runs_dir,
                                           f"ours_{ratio}{suffix}")))
        rounds = args.rounds
    raw_dir = os.path.splitext(out_path)[0]      # 逐配置原始 JSON 目录
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    crop_area, extended = {}, {}
    for ratio, seed, semi_dir, ours_dir in specs:
        suffix = f"_s{seed}" if seed is not None else "_smoke"
        # 断点续跑：seed42 的扩展逐配置 JSON 已存在则直接加载，
        # crop_area 由 rec.metrics 精确还原，不再重复任何推理
        rec_paths = {r: os.path.join(raw_dir,
                                     f"ours_{ratio}{suffix}_round{r}.json")
                     for r in rounds}
        if all(os.path.exists(p) for p in rec_paths.values()):
            for r, p in rec_paths.items():
                with open(p) as f:
                    rec = json.load(f)
                extended[(ratio, seed, r)] = rec
            m0 = extended[(ratio, seed, rounds[0])]["metrics"]
            crop_area[f"{ratio}{suffix}"] = {
                "ratio": ratio, "seed": seed,
                "n_slices": extended[(ratio, seed, rounds[0])]["n_slices"],
                "crop_size": extended[(ratio, seed, rounds[0])]["crop_size"],
                "crop_pixels": m0["crop_pixels"],
                "fallback_slices": m0["fallback_slices"],
                "n_pixels": m0["n_pixels"],
                "crop_area_frac": m0["crop_area_frac"],
                "resumed_from": raw_dir}
            print(f"[ratio {ratio}%{suffix}] 已有扩展 JSON，直接续用")
            continue
        with open(os.path.join(semi_dir, "run_meta.json")) as f:
            meta = json.load(f)
        files = unlabeled_pool(meta, args.stride)
        print(f"[ratio {ratio}%{suffix}] 未标注池 {meta['n_unlabeled']} 片，"
              f"子采样 stride={args.stride} -> {len(files)} 片")
        store = TensorStore.from_files(files, device, meta.get("cache_dir"))
        model1 = load_model(meta["stage1_ckpt"], device)
        crop_size = meta.get("crop_size", 256)
        ca = stage1_crop_area(model1, store, crop_size, args.batch_size)
        ca.update({"ratio": ratio, "seed": seed, "n_slices": len(files),
                   "crop_size": crop_size,
                   "crop_area_frac": ca["crop_pixels"] / ca["n_pixels"]})
        crop_area[f"{ratio}{suffix}"] = ca
        print(f"  stage-2 视野 {ca['crop_pixels']} px "
              f"({ca['crop_area_frac']:.4%} of canvas), "
              f"fallback {ca['fallback_slices']}/{len(files)} 片")
        if seed == args.breakdown_seed or args.smoke:
            with open(os.path.join(ours_dir, "run_meta.json")) as f:
                mmeta = json.load(f)
            consistency = bool(mmeta.get("consistency"))
            for r in rounds:
                teacher = (mmeta["teacher_init"] if r == 1 else
                           os.path.join(ours_dir, f"teacher_r{r - 1}.pt"))
                model2 = load_model(teacher, device)
                cnt = accumulate_config(model1, model2, store,
                                        mmeta["conf_threshold"],
                                        mmeta["crop_size"], consistency,
                                        args.batch_size)
                m = derive_metrics(cnt)
                # 交叉验证：conf>0 计数须与 stage-1 推算的视野像素一致
                diff = abs(m["crop_pixels"] - ca["crop_pixels"])
                if diff > max(1, ca["crop_pixels"] * 1e-3):
                    print(f"  ⚠️ crop_pixels 不一致: cascade={m['crop_pixels']} "
                          f"vs stage1={ca['crop_pixels']}")
                rec = {"method": "ours", "ratio": ratio, "seed": seed,
                       "round": r, "teacher": teacher,
                       "stage1_ckpt": mmeta["stage1_ckpt"],
                       "consistency": consistency,
                       "conf_threshold": mmeta["conf_threshold"],
                       "crop_size": mmeta["crop_size"],
                       "n_unlabeled_full": mmeta["n_unlabeled"],
                       "stride": args.stride, "n_slices": len(files),
                       "device": str(device), "metrics": m}
                path = os.path.join(raw_dir,
                                    f"ours_{ratio}{suffix}_round{r}.json")
                with open(path, "w") as f:
                    json.dump(rec, f, indent=2)
                extended[(ratio, seed, r)] = rec
                print(f"  ours_{ratio}{suffix} r{r}: "
                      f"cov_full={m['variants']['full']['coverage']:.4%} "
                      f"in_crop conf={m['variants']['conf']['coverage_in_crop']:.4%} "
                      f"full={m['variants']['full']['coverage_in_crop']:.4%}")
                del model2
        del store, model1

    write_breakdown_outputs(args, crop_area, extended, seeds, rounds,
                            out_path, snippet_path)


def write_breakdown_outputs(args, crop_area, extended, seeds, rounds,
                            out_path, snippet_path):
    """合并 mechanism_dir 已有计数 + 新算视野面积，写 JSON 与英文 snippet。"""
    per_config = []
    if not args.smoke and os.path.isdir(args.mechanism_dir):
        mech = load_records(args.mechanism_dir, set(args.ratios),
                            set(args.rounds))
        for (method, ratio, seed, r), rec in sorted(mech.items()):
            ca = crop_area.get(f"{ratio}_s{seed}")
            if ca is None:
                continue
            entry = {"method": method, "ratio": ratio, "seed": seed,
                     "round": r, "n_slices": rec["n_slices"],
                     "stride": rec["stride"],
                     "crop_pixels": ca["crop_pixels"],
                     "fallback_slices": ca["fallback_slices"],
                     "source": os.path.join(args.mechanism_dir,
                                            f"{method}_{ratio}_s{seed}"
                                            f"_round{r}.json")}
            for v in ("conf", "full"):
                kept = rec["metrics"]["variants"][v]["kept"]
                entry[v] = {
                    "coverage_full_image": kept / rec["metrics"]["n_pixels"],
                    "coverage_in_crop": kept / ca["crop_pixels"]}
            per_config.append(entry)
    # 三种子聚合（逐 method/ratio/round）
    aggregate = {}
    for e in per_config:
        key = f"{e['method']}_{e['ratio']}_round{e['round']}"
        aggregate.setdefault(key, []).append(e)
    agg_out = {}
    for key, entries in aggregate.items():
        agg_out[key] = {
            "n_seeds": len({e["seed"] for e in entries}),
            "coverage_full_image_conf": _mean_std(
                e["conf"]["coverage_full_image"] for e in entries),
            "coverage_full_image_full": _mean_std(
                e["full"]["coverage_full_image"] for e in entries),
            "coverage_in_crop_conf": _mean_std(
                e["conf"]["coverage_in_crop"] for e in entries),
            "coverage_in_crop_full": _mean_std(
                e["full"]["coverage_in_crop"] for e in entries),
            "crop_area_frac": _mean_std(
                crop_area[f"{e['ratio']}_s{e['seed']}"]["crop_area_frac"]
                for e in entries),
        }
    # seed42 扩展：raw/conf/full 逐轮口径
    ext_out = {}
    for (ratio, seed, r), rec in sorted(extended.items()):
        m = rec["metrics"]
        va = {}
        for v in ("raw", "conf", "full"):
            vm = m["variants"][v]
            va[v] = {"coverage_full_image": vm["coverage"],
                     "coverage_in_crop": vm["coverage_in_crop"],
                     "error_rate": vm["error_rate"],
                     "per_class": {cls: {
                         "retention": vm["per_class"][cls]["retention"],
                         "recall": vm["per_class"][cls]["recall"],
                         "precision": vm["per_class"][cls]["precision"]}
                         for cls in ("rectum", "tumor")}}
        ext_out[f"ours_{ratio}_s{seed}_round{r}"] = {
            "variants": va, "removed_frac": m.get("removed_frac"),
            "crop_area_frac": m["crop_area_frac"],
            "fallback_slices": m["fallback_slices"]}
    result = {
        "generated_by": "src/quantify_mechanism.py --coverage_breakdown",
        "device": args.device, "stride": args.stride,
        "runs_dir": args.runs_dir, "mechanism_dir": args.mechanism_dir,
        "definitions": {
            "coverage_full_image": "kept pseudo-label pixels / all canvas "
                "pixels (the Supplement coverage metric)",
            "coverage_in_crop": "kept pixels / pixels inside the stage-2 "
                "field of view (256x256 crop window; full canvas for "
                "tile-fallback slices with empty stage-1 prediction)",
            "crop_area_frac": "stage-2 field-of-view pixels / canvas pixels",
            "retention": "kept GT-class pixels / all GT-class pixels "
                "(regardless of predicted label correctness)",
            "variants": "raw = no filter (entire field of view); conf = "
                "confidence >= 0.9 (identical to ST's full filter); full = "
                "conf + cross-stage agreement (Ours)"},
        "crop_area": crop_area,
        "per_config": per_config,
        "aggregate_3seed": agg_out,
        "extended_breakdown_seed": ext_out,
        "notes": [
            "ST (semi) 的完整过滤等价于 conf 变体；第 1 轮 teacher 与 Ours "
            "完全相同（sup 模型），第 2/3 轮 teacher 各自独立但覆盖率模式一致。",
            "kept 计数（conf/full）来自 results/mechanism_v2 的既有 36 配置"
            "（GPU, stride-4）；crop_pixels 为本模式 stage-1 重算（CPU）。"
            "extended 部分（raw 变体/分类别 retention）仅 seed "
            f"{args.breakdown_seed} 重算。"],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"已保存 {out_path}")
    write_snippet(result, snippet_path, args)


def write_snippet(result, path, args):
    """英文说明段：覆盖率分解关键数字（rebuttal / 补充材料直接引用）。"""
    cas = result["crop_area"]
    ca_fracs = [ca["crop_area_frac"] for ca in cas.values()]
    fb = sum(ca["fallback_slices"] for ca in cas.values())
    ns = sum(ca["n_slices"] for ca in cas.values())
    pcs = result["per_config"]

    def stat2(field, variant):
        vs = [e[variant][field] for e in pcs]
        return (min(vs), max(vs), float(np.mean(vs)),
                float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0)

    ext = result["extended_breakdown_seed"]
    lines = []
    if pcs:
        lo, hi, mean, std = stat2("coverage_full_image", "conf")
        lo_f, hi_f, mean_f, std_f = stat2("coverage_full_image", "full")
        ic_lo, ic_hi, ic_m, ic_s = stat2("coverage_in_crop", "conf")
        icf_lo, icf_hi, icf_m, icf_s = stat2("coverage_in_crop", "full")
        lines.append("Pseudo-label coverage decomposition (CARE unlabeled "
                     "training pool; three patient-level seeds 42/43/44, "
                     "deterministic stride-%d subsample; %d method x ratio x "
                     "seed x round configurations)." % (args.stride, len(pcs)))
        lines.append("")
        lines.append(
            "The pseudo-label coverage reported in the Supplement is computed "
            "over the full 512x512 canvas, whereas the stage-2 teacher only "
            "labels pixels inside the stage-1-guided 256x256 crop window "
            "(exactly 25%% of the canvas); pixels outside the window carry "
            "confidence 0 and can never pass the threshold. The measured "
            "stage-2 field of view covers %.2f%% of the canvas on average "
            "(range %.2f-%.2f%%; %d of %d subsampled slices had an empty "
            "stage-1 prediction and triggered tiled fallback covering the "
            "whole image). Full-image coverage is therefore geometrically "
            "bounded at ~25%% and is NOT governed by the confidence "
            "threshold: %.2f+/-%.2f%% (confidence filter) and %.2f+/-%.2f%% "
            "(confidence + cross-stage agreement) across all configurations."
            % (100 * float(np.mean(ca_fracs)), 100 * min(ca_fracs),
               100 * max(ca_fracs), fb, ns,
               100 * mean, 100 * std, 100 * mean_f, 100 * std_f))
        lines.append("")
        lines.append(
            "Within the crop window, coverage is %.2f+/-%.2f%% for the "
            "confidence filter and %.2f+/-%.2f%% after adding cross-stage "
            "agreement, i.e. the threshold removes only %.2f%% and the "
            "agreement filter another %.2f%% of in-crop pixels. The teacher "
            "is confident almost everywhere inside the crop, so coverage is "
            "dominated by the crop geometry rather than by filtering; the "
            "selectivity of the filters should instead be read from the "
            "pseudo-label quality metrics (Supplementary Table S4)."
            % (100 * ic_m, 100 * ic_s, 100 * icf_m, 100 * icf_s,
               100 * (1 - ic_m), 100 * (ic_m - icf_m)))
    if ext:
        lines.append("")
        for key in sorted(ext):
            e = ext[key]
            r = e["variants"]
            rf = e.get("removed_frac") or {}
            lines.append(
                "%s: in-crop coverage raw %.2f%% -> conf %.2f%% -> full "
                "%.2f%%; GT retention rectum %.2f%% -> %.2f%% -> %.2f%%, "
                "tumor %.2f%% -> %.2f%% -> %.2f%% (raw -> conf -> full); "
                "pixel error rate among kept %.2f%% -> %.2f%% -> %.2f%%. "
                "Of the pixels additionally removed by the agreement filter, "
                "%.1f%% were false-positive foreground and %.1f%% were "
                "correct labels (vs %.2f%% baseline error among kept "
                "pixels)." %
                (key,
                 100 * r["raw"]["coverage_in_crop"],
                 100 * r["conf"]["coverage_in_crop"],
                 100 * r["full"]["coverage_in_crop"],
                 100 * r["raw"]["per_class"]["rectum"]["retention"],
                 100 * r["conf"]["per_class"]["rectum"]["retention"],
                 100 * r["full"]["per_class"]["rectum"]["retention"],
                 100 * r["raw"]["per_class"]["tumor"]["retention"],
                 100 * r["conf"]["per_class"]["tumor"]["retention"],
                 100 * r["full"]["per_class"]["tumor"]["retention"],
                 100 * r["raw"]["error_rate"], 100 * r["conf"]["error_rate"],
                 100 * r["full"]["error_rate"],
                 100 * rf.get("fp", 0.0), 100 * rf.get("correct", 0.0),
                 100 * r["conf"]["error_rate"]))
        lines.append("")
        lines.append(
            "Extended metrics (raw variant and per-class retention) were "
            "recomputed on CPU for seed %d; crop-window areas were measured "
            "exactly for all seeds via stage-1 inference. GT labels of the "
            "unlabeled pool were used only to measure pseudo-label quality, "
            "never for training." % args.breakdown_seed)
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"已保存 {path}")


def metric_paths():
    """表格/图用到的派生指标路径（variants × per-class + removed_frac）。"""
    paths = []
    for v in ("full", "conf"):
        paths.append(("variants", v, "coverage"))
        paths.append(("variants", v, "error_rate"))
        for cls in ("rectum", "tumor"):
            paths.append(("variants", v, "per_class", cls, "precision"))
            paths.append(("variants", v, "per_class", cls, "recall"))
    for k in ("of_conf_kept", "true_fg", "fp", "fp_bg", "wrong_cls",
              "bg_on_fg", "true_bg", "correct"):
        paths.append(("removed_frac", k))
    return paths


def get_metric(metrics, path):
    """按路径取派生指标；缺失（如 ST 无 removed_frac）返回 None。"""
    node = metrics
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def aggregate(recs, ratios, rounds, seeds):
    """{(method, ratio, round): {"mean": {path: m}, "std": {path: s}}}。

    seeds=[None] 时退化为旧单次布局（std 全 0，表格不显示 ±）。
    """
    paths = metric_paths()
    agg = {}
    for method in METHODS:
        for ratio in ratios:
            for r in rounds:
                vals = {p: [] for p in paths}
                n = 0
                for s in seeds:
                    rec = recs.get((method, ratio, s, r))
                    if rec is None:
                        continue
                    n += 1
                    for p in paths:
                        vals[p].append(get_metric(rec["metrics"], p))
                mean, std = {}, {}
                for p, vs in vals.items():
                    vs = [v for v in vs if v is not None]
                    if not vs:
                        mean[p] = std[p] = None
                    else:
                        mean[p] = float(np.mean(vs))
                        std[p] = (float(np.std(vs, ddof=1))
                                  if len(vs) > 1 else 0.0)
                agg[(method, ratio, r)] = {"mean": mean, "std": std,
                                           "n_seeds": n}
    return agg


def fmt(mean, std, seeded, scale=100.0):
    """百分比格式化；seeded 时输出 mean$\\pm$std。"""
    if mean is None:
        return "n/a"
    if not seeded:
        return f"{mean * scale:.2f}"
    return f"{mean * scale:.2f}$\\pm${(std or 0.0) * scale:.2f}"


def make_table(agg, ratios, rounds, seeded):
    """booktabs 表：过滤后质量（ST vs Ours）+ Ours 一致性过滤效果列。"""
    src = ("results/mechanism_v2/*.json（runs_v2 三种子 mean±std）"
           if seeded else "results/mechanism/*.json")
    caption_extra = (" Values are the mean $\\pm$ std across the three "
                     "training seeds (42/43/44), each with its own "
                     "patient-level labeled/unlabeled split." if seeded else "")
    lines = [
        "% 伪标签质量定量分析，由 src/quantify_mechanism.py 从 " + src + " 生成",
        "% 未标注池按固定 stride 子采样（见各 json 的 stride/n_slices 字段）",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Pseudo-label quality on the CARE unlabeled training pool, "
        "measured against the hidden ground truth (pixel level, \\%). "
        "Precision (P) and recall (R) are reported after each method's full "
        "filter (ST: confidence $\\geq 0.9$; Ours: confidence + cross-stage "
        "agreement). For Ours, ``Tumor P$_{\\mathrm{conf}}$'' is the "
        "precision before the agreement filter (same teacher), ``Rem.\\ FP'' "
        "and ``Rem.\\ corr.'' give the fraction of removed pixels that were "
        "false-positive foreground (predicted foreground $\\neq$ GT) and "
        "correct pseudo-labels, respectively. Err = pixel error rate among "
        "kept pseudo-labels; Cov = kept-pixel coverage." + caption_extra + "}",
        "\\label{tabS:pseudo_quality}",
        "\\small" if not seeded else "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}" if not seeded else
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\begin{tabular}{ll" + "c" * 10 + "}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{Precision} & "
        "\\multicolumn{2}{c}{Recall} & & & \\multicolumn{3}{c}{Ours filter} "
        "\\\\",
        "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){10-12}",
        "Method & Round & Cov & Rectum & Tumor & Rectum & Tumor & Err "
        "& Tumor P$_{\\mathrm{conf}}$ & Rem.\\ FP & Rem.\\ corr. \\\\",
        "\\midrule",
    ]
    p_cov = ("variants", "full", "coverage")
    p_err = ("variants", "full", "error_rate")
    p_tp_conf = ("variants", "conf", "per_class", "tumor", "precision")
    for ratio in ratios:
        lines.append(f"\\multicolumn{{12}}{{l}}{{\\emph{{Labeled ratio "
                     f"{ratio}\\%}}}} \\\\")
        lines.append("\\midrule")
        for method in METHODS:
            for i, r in enumerate(rounds):
                a = agg.get((method, ratio, r))
                first = (f"\\multirow{{{len(rounds)}}}{{*}}"
                         f"{{{METHOD_LABEL[method]}}}") if i == 0 else ""
                if a is None or a["n_seeds"] == 0:
                    cells = ["n/a"] * 10
                else:
                    m, s = a["mean"], a["std"]

                    def pc(cls, stat):
                        return fmt(m[("variants", "full", "per_class",
                                      cls, stat)],
                                   s[("variants", "full", "per_class",
                                      cls, stat)], seeded)

                    cells = [fmt(m[p_cov], s[p_cov], seeded),
                             pc("rectum", "precision"),
                             pc("tumor", "precision"),
                             pc("rectum", "recall"),
                             pc("tumor", "recall"),
                             fmt(m[p_err], s[p_err], seeded)]
                    if method == "ours":
                        cells += [
                            fmt(m[p_tp_conf], s[p_tp_conf], seeded),
                            fmt(m[("removed_frac", "fp")],
                                s[("removed_frac", "fp")], seeded),
                            fmt(m[("removed_frac", "correct")],
                                s[("removed_frac", "correct")], seeded)]
                    else:
                        cells += ["--"] * 3
                lines.append(f"{first} & {r} & " + " & ".join(cells) + " \\\\")
            if method != METHODS[-1]:
                lines.append("\\midrule")
        if ratio != ratios[-1]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def make_figure(agg, ratios, rounds, seeded):
    """2x2：行=rectum/tumor，列=标注比例；ST vs Ours 的 precision/recall 逐轮曲线。

    另画 Ours 过滤前（仅置信度，灰色点线）precision，直观显示一致性过滤的增益。
    seeded 时曲线为三种子均值，误差棒 ±1 std。
    """
    fig, axes = plt.subplots(2, len(ratios), figsize=(5.6 * len(ratios), 8.6),
                             sharex=True)
    style = {"semi": dict(color="tab:blue", marker="o"),
             "ours": dict(color="tab:red", marker="s")}

    def series(method, ratio, cls, stat, variant="full"):
        p = ("variants", variant, "per_class", cls, stat)
        ys = [agg[(method, ratio, r)]["mean"][p] * 100 for r in rounds]
        es = [agg[(method, ratio, r)]["std"][p] * 100 for r in rounds]
        return ys, es

    for col, ratio in enumerate(ratios):
        for row, cls in enumerate(("rectum", "tumor")):
            ax = axes[row][col] if len(ratios) > 1 else axes[row]
            for method in METHODS:
                for stat, ls, tag in (("precision", "-", "P"),
                                      ("recall", "--", "R")):
                    ys, es = series(method, ratio, cls, stat)
                    kw = dict(ls=ls, lw=1.8,
                              label=f"{METHOD_LABEL[method]} {tag}",
                              **style[method])
                    if seeded:
                        ax.errorbar(rounds, ys, yerr=es, capsize=2.5, **kw)
                    else:
                        ax.plot(rounds, ys, **kw)
            # Ours 过滤前 precision（同一 teacher，仅置信度阈值）
            ys0, es0 = series("ours", ratio, cls, "precision", "conf")
            kw = dict(ls=":", lw=1.6, color="gray", marker="^",
                      label="Ours P (conf only)")
            if seeded:
                ax.errorbar(rounds, ys0, yerr=es0, capsize=2.5, **kw)
            else:
                ax.plot(rounds, ys0, **kw)
            ax.set_xticks(list(rounds))
            ax.set_title(f"{CLASS_LABEL[1 if cls == 'rectum' else 2].capitalize()} "
                         f"— {ratio}% labels")
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("Self-training round")
            if col == 0:
                ax.set_ylabel("Pseudo-label pixel precision / recall (%)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Pseudo-label quality against hidden ground truth (CARE "
                 "unlabeled pool):\nprecision (solid) and recall (dashed)"
                 + ("; mean ± std over three seeds" if seeded else ""))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outs = [os.path.join(d, "figS3_pseudo_quality.png") for d in PAPER_DIRS]
    for out in outs:
        fig.savefig(out, dpi=300)
        print(f"已保存 {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", type=int, nargs="+", default=[10, 50])
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--stride", type=int, default=4,
                    help="未标注池确定性子采样步长（每 stride 片取 1）")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs_dir", default="runs",
                    help="训练 run 根目录（runs 或 runs_v2）")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="训练种子；给了则跑 {method}_{ratio}_s{seed} 并聚合 "
                         "mean±std（缺省为旧单次布局）")
    ap.add_argument("--results_dir", default=RESULTS_DIR,
                    help="机制量化 JSON 输出目录")
    ap.add_argument("--tables_only", action="store_true",
                    help="跳过 GPU 推理，仅用已有 JSON 重出图表")
    ap.add_argument("--coverage_breakdown", action="store_true",
                    help="覆盖率分解模式（审稿意见 A）：全图/裁剪内覆盖率 + "
                         "分类别保留比例；CPU 可跑，不触碰 tabS4/figS3")
    ap.add_argument("--breakdown_seed", type=int, default=42,
                    help="覆盖率分解中重算扩展口径（raw 变体/分类别 retention）"
                         "的种子；其余 seed 仅跑 stage-1 算视野面积")
    ap.add_argument("--mechanism_dir", default="results/mechanism_v2",
                    help="已有机制量化 JSON 目录（合并其 kept 计数）")
    ap.add_argument("--breakdown_out",
                    default=os.path.join("results", "v2",
                                         "coverage_breakdown.json"),
                    help="覆盖率分解汇总 JSON 输出")
    ap.add_argument("--snippet_out",
                    default=os.path.join("paper", "snippet_coverage.txt"),
                    help="英文说明段输出")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：runs/smoke/semi_10_s2 单配置单轮，输出落 "
                         "results/smoke/")
    args = ap.parse_args()

    if args.coverage_breakdown:
        run_coverage_breakdown(args)
        return

    if not args.tables_only:
        run_gpu(args)
    seeds = args.seeds or [None]
    recs = load_records(args.results_dir, set(args.ratios), set(args.rounds))
    missing = [(m, r, s, q) for m in METHODS for r in args.ratios
               for s in seeds for q in args.rounds
               if (m, r, s, q) not in recs]
    if missing:
        raise SystemExit(f"缺少配置结果: {missing}（先跑 GPU 部分）")
    seeded = args.seeds is not None
    agg = aggregate(recs, args.ratios, args.rounds, seeds)
    if seeded:
        agg_path = os.path.join(args.results_dir, "agg_seeds.json")
        ser = {f"{m}_{r}_round{q}": {
            "n_seeds": a["n_seeds"],
            "mean": {".".join(p): v for p, v in a["mean"].items()
                     if v is not None},
            "std": {".".join(p): v for p, v in a["std"].items()
                    if v is not None}}
            for (m, r, q), a in agg.items()}
        with open(agg_path, "w") as f:
            json.dump(ser, f, indent=2)
        print(f"已保存 {agg_path}")
    tex = make_table(agg, args.ratios, args.rounds, seeded)
    for d in PAPER_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "tabS4_pseudo_quality.tex")
        with open(path, "w") as f:
            f.write(tex)
        print(f"已保存 {path}")
    make_figure(agg, args.ratios, args.rounds, seeded)


if __name__ == "__main__":
    main()
