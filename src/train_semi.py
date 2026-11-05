"""半监督伪标签自训练。

流程（每轮）：
1. 用当前 teacher 在未标注训练切片上推理，softmax 置信度 > 阈值的像素保留其预测类别，
   其余像素置 IGNORE_INDEX(255)，直接写回联合 TensorStore 的标签区（不落盘）；
2. 在 [标注切片(真标签) + 未标注切片(伪标签)] 上从头训练新模型；
3. 验证集（带真标签）Dice 选出的新 best 作为下一轮 teacher。

用法：
    python src/train_semi.py --config configs/label_ratio_10.json \
        --teacher_ckpt runs/sup_10/best.pt --out_dir runs/semi_10
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset
from common import (build_loaders, load_config, load_model, resolve_device,
                    set_seed, train_model)
from dataset import (CAREDataset, IGNORE_INDEX, labeled_subset, list_slices,
                     parse_case_id, split_labeled_val, split_train_val)
from models import LightUNet
from tensorstore import TensorStore


@torch.no_grad()
def fill_pseudo_from_store(model, store, start, conf_threshold, batch_size=32):
    """对 store[start:] 的切片推理并把伪标签写回 store.labels[start:]。

    返回伪标签像素覆盖率。
    """
    model.eval()
    kept, total = 0, 0
    n = len(store)
    for off in range(start, n, batch_size):
        end = min(off + batch_size, n)
        probs = torch.softmax(model(store.images[off:end]), dim=1)
        conf, pred = probs.max(1)
        pseudo = pred.to(torch.uint8)
        pseudo[conf < conf_threshold] = IGNORE_INDEX
        store.labels[off:end] = pseudo
        kept += (conf >= conf_threshold).sum().item()
        total += conf.numel()
    return kept / max(total, 1)


@torch.no_grad()
def generate_pseudo_labels(model, files, device, conf_threshold, out_dir, batch_size=32):
    """CPU 路径（preload=False）：伪标签落盘 out_dir/*.npz（键 'pseudo'）。"""
    import numpy as np
    from torch.utils.data import DataLoader
    os.makedirs(out_dir, exist_ok=True)
    ds = CAREDataset(files, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    kept, total = 0, 0
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        probs = torch.softmax(model(images), dim=1)
        conf, pred = probs.max(1)
        pseudo = pred.to(torch.int64)
        pseudo[conf < conf_threshold] = IGNORE_INDEX
        kept += (conf >= conf_threshold).sum().item()
        total += conf.numel()
        pseudo_np = pseudo.cpu().numpy().astype(np.uint8)
        for i, name in enumerate(batch["name"]):
            np.savez(os.path.join(out_dir, name), pseudo=pseudo_np[i])
    return kept / max(total, 1)


@torch.no_grad()
def fill_pseudo_cascade(model1, model2, store, start, conf_threshold, crop_size,
                        batch_size=32, thr_rectum=None, thr_tumor=None,
                        small_cc_area=None, small_cc_thr=None,
                        consistency=False):
    """级联版伪标签：stage1 定位 + stage2 精分割，置信度写回 store.labels[start:]。

    裁剪窗外像素 conf=0，会被阈值滤为 IGNORE。返回伪标签像素覆盖率。

    可选组件（默认全关，退化为基线行为）：
    - thr_rectum/thr_tumor：类别选择性阈值（按预测类别分别设阈值，背景用 conf_threshold）；
    - small_cc_area/small_cc_thr：肿瘤连通域面积 <= small_cc_area 时阈值放宽为 small_cc_thr；
    - consistency：跨阶段一致性过滤，stage-1 与 stage-2 预测不一致的像素置 IGNORE。
    """
    from cascade import cascade_predict
    model1.eval()
    model2.eval()
    need_s1 = consistency or small_cc_area
    kept, total = 0, 0
    n = len(store)
    for off in range(start, n, batch_size):
        end = min(off + batch_size, n)
        if need_s1:
            pred, conf, pred1 = cascade_predict(
                model1, model2, store.images[off:end], crop_size, end - off,
                return_stage1=True)
        else:
            pred, conf = cascade_predict(model1, model2, store.images[off:end],
                                         crop_size, end - off)
        thr_map = torch.full_like(conf, conf_threshold)
        if thr_rectum is not None:
            thr_map[pred == 1] = thr_rectum
        if thr_tumor is not None:
            thr_map[pred == 2] = thr_tumor
        keep = conf >= thr_map
        if consistency:
            keep &= pred1 == pred
        pseudo = pred.clone()
        pseudo[~keep] = IGNORE_INDEX
        if small_cc_area:
            pseudo = _relax_small_tumor_cc(pseudo, pred, conf, small_cc_area,
                                           small_cc_thr or conf_threshold,
                                           pred1 if consistency else None)
        store.labels[off:end] = pseudo
        kept += (pseudo != IGNORE_INDEX).sum().item()
        total += pseudo.numel()
    return kept / max(total, 1)


def _relax_small_tumor_cc(pseudo, pred, conf, area, thr, pred1=None):
    """肿瘤小连通域阈值放宽：面积 <= area 的连通域内 conf >= thr 的像素收为肿瘤。

    pred1 非 None 时额外要求 stage-1 也预测为肿瘤（一致性约束）。
    """
    from scipy import ndimage
    pn = pseudo.cpu().numpy()
    predn = pred.cpu().numpy()
    confn = conf.cpu().numpy()
    p1n = pred1.cpu().numpy() if pred1 is not None else None
    for i in range(pn.shape[0]):
        lab, ncc = ndimage.label(predn[i] == 2)
        for j in range(1, ncc + 1):
            m = lab == j
            if m.sum() <= area:
                acc = m & (confn[i] >= thr)
                if p1n is not None:
                    acc &= p1n[i] == 2
                pn[i][acc] = 2
    return torch.from_numpy(pn).to(device=pseudo.device, dtype=pseudo.dtype)


def cropped_copy(store, crop_size):
    """按当前标签前景质心（标注=GT，未标注=伪标签）生成裁剪副本 store。"""
    from cascade import crop_batch, crop_offsets, fg_centers
    h, w = store.images.shape[2], store.images.shape[3]
    offsets = crop_offsets(fg_centers(store.labels), crop_size, h, w)
    n = len(store)
    # 预分配输出再逐块写入：避免 torch.cat 的 2 倍显存峰值（本机 NVML 故障下高峰值易被杀）
    imgs = torch.empty((n, store.images.shape[1], crop_size, crop_size),
                       dtype=store.images.dtype, device=store.device)
    labs = torch.empty((n, crop_size, crop_size),
                       dtype=store.labels.dtype, device=store.device)
    bs = 1024
    for off in range(0, n, bs):
        end = min(off + bs, n)
        im, lb = crop_batch(store.images[off:end], offsets[off:end], crop_size,
                            labels=store.labels[off:end])
        imgs[off:end] = im
        labs[off:end] = lb
        del im, lb
    return TensorStore(imgs, labs, store.names, store.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--rounds", type=int, default=None, help="自训练轮数（消融用）")
    ap.add_argument("--conf_threshold", type=float, default=None, help="置信度阈值（消融用）")
    ap.add_argument("--epochs", type=int, default=None, help="每轮 epoch（冒烟测试用）")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--crop_size", type=int, default=None,
                    help="级联 stage-2：裁剪边长（需 --stage1_ckpt，teacher 为级联）")
    ap.add_argument("--stage1_ckpt", default=None, help="级联 stage-1 ckpt")
    ap.add_argument("--thr_rectum", type=float, default=None,
                    help="类别选择性：直肠类伪标签阈值（默认=conf_threshold）")
    ap.add_argument("--thr_tumor", type=float, default=None,
                    help="类别选择性：肿瘤类伪标签阈值（默认=conf_threshold）")
    ap.add_argument("--small_cc_area", type=int, default=None,
                    help="小病灶挖掘：肿瘤连通域面积 <= 该值时放宽阈值")
    ap.add_argument("--small_cc_thr", type=float, default=None,
                    help="小病灶挖掘：放宽后的阈值（默认=conf_threshold）")
    ap.add_argument("--consistency", action="store_true",
                    help="跨阶段一致性：stage1 与 stage2 不一致的像素置 IGNORE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config 中的 seed（多 seed 重复实验；标注子集随 seed 重抽）")
    ap.add_argument("--val_mode", choices=["pool", "labeled", "fixed"], default="pool",
                    help="模型选择验证集来源：pool=全训练池 15%%（默认，原行为）；"
                         "labeled=仅从标注病例后缀划 15%%（标注预算一致，<4 例回退 fixed）；"
                         "fixed=不做验证选择，每轮最终 epoch 模型为该轮 best")
    ap.add_argument("--start_round", type=int, default=None,
                    help="断点续跑：从第 N 轮开始（此时 --teacher_ckpt 应为第 N-1 轮的 teacher_rN-1.pt）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg["data_root"]
    semi_cfg = cfg.get("semi", {})
    rounds = args.rounds or semi_cfg.get("rounds", 3)
    start_round = args.start_round or 1
    conf_threshold = args.conf_threshold or semi_cfg.get("conf_threshold", 0.9)
    epochs = args.epochs or semi_cfg.get("epochs_per_round", 60)
    batch_size = args.batch_size or cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = resolve_device(args.device)
    set_seed(seed)
    n_cls = cfg.get("num_classes", 3)  # 默认 3=CARE；MSD Colon 配 2
    dataset.set_num_classes(n_cls)

    all_train = list_slices(data_root, "train")
    pool_files, val_files = split_train_val(all_train, cfg.get("val_ratio", 0.15), seed)
    labeled_files = labeled_subset(pool_files, cfg["label_ratio"], seed)
    labeled_set = set(labeled_files)
    unlabeled_files = [p for p in pool_files if p not in labeled_set]
    val_mode = args.val_mode
    if val_mode == "labeled":
        n_lab_cases = len({parse_case_id(p) for p in labeled_files})
        if n_lab_cases < 4:
            print(f"⚠️ val_mode=labeled 但标注病例仅 {n_lab_cases} 例（<4），"
                  "无法划出验证集，回退 val_mode=fixed")
            val_mode = "fixed"
    final_as_best = val_mode == "fixed"
    if val_mode == "labeled":
        train_labeled_files, val_files = split_labeled_val(
            labeled_files, cfg.get("val_ratio", 0.15), seed)
    elif val_mode == "fixed":
        train_labeled_files = labeled_files
        val_files = labeled_files  # 仅日志用（训练标注切片上的 Dice），不做模型选择
        print("val_mode=fixed：每轮不做验证集模型选择，最终 epoch 模型为该轮 best；"
              "log.csv 的 val_dice 在训练标注切片上计算，仅作参考")
    else:  # pool：原行为
        train_labeled_files = labeled_files
    ov = cfg.get("oversample_tumor", 1)  # 肿瘤切片过采样（MSD 用；仅训练列表）
    train_labeled = train_labeled_files
    if ov > 1:
        from dataset import oversample_positive_files
        train_labeled = oversample_positive_files(train_labeled_files, ov)
        print(f"过采样标注肿瘤切片 x{ov}: {len(train_labeled_files)} -> {len(train_labeled)}")

    print(f"device={device} seed={seed} val_mode={val_mode} "
          f"标注切片 {len(labeled_files)} / 未标注 {len(unlabeled_files)} "
          f"/ 验证 {len(val_files)}；rounds={rounds} 阈值={conf_threshold} 每轮 {epochs} epoch")

    preload = cfg.get("preload", False)
    crop_size = args.crop_size or cfg.get("crop_size")
    stage1_ckpt = args.stage1_ckpt or cfg.get("stage1_ckpt")
    if crop_size and not (preload and stage1_ckpt):
        raise SystemExit("--crop_size 需要 preload=true 且 --stage1_ckpt")
    if (args.thr_rectum is not None or args.thr_tumor is not None
            or args.small_cc_area or args.consistency) and not crop_size:
        raise SystemExit("伪标签组件（thr_rectum/thr_tumor/small_cc/consistency）需要级联模式")
    joint_store = val_loader = None
    labeled_ds = unlabeled_ds = infer_files = None
    teacher1 = None
    if preload:
        # 联合 TensorStore：前段标注切片（真标签），后段未标注切片（每轮覆写伪标签）
        cache_dir = cfg.get("cache_dir")
        joint_store = TensorStore.from_files(train_labeled + unlabeled_files,
                                             device, cache_dir)
        val_store = TensorStore.from_files(val_files, device, cache_dir)
        if crop_size:
            from cascade import batch_argmax, fg_centers
            teacher1 = load_model(stage1_ckpt, device)
            # 验证集按 stage-1 预测质心裁剪（一次性，模拟测试行为）
            val_pred1 = batch_argmax(teacher1, val_store.images, batch_size)
            val_store.crop_in_place(fg_centers(val_pred1), crop_size)
            del val_pred1
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        val_loader = val_store.loader(batch_size, shuffle=False)
    else:
        labeled_ds = CAREDataset(train_labeled, augment=True)
        unlabeled_ds = CAREDataset(unlabeled_files, augment=True)
        val_ds = CAREDataset(val_files, augment=False)
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)

    teacher_path = args.teacher_ckpt
    history = []
    for r in range(start_round, rounds + 1):
        print(f"===== 自训练第 {r}/{rounds} 轮，teacher={teacher_path} =====")
        teacher = load_model(teacher_path, device)
        round_store = None
        if preload and crop_size:
            coverage = fill_pseudo_cascade(teacher1, teacher, joint_store,
                                           len(train_labeled), conf_threshold,
                                           crop_size, batch_size,
                                           thr_rectum=args.thr_rectum,
                                           thr_tumor=args.thr_tumor,
                                           small_cc_area=args.small_cc_area,
                                           small_cc_thr=args.small_cc_thr,
                                           consistency=args.consistency)
            round_store = cropped_copy(joint_store, crop_size)
            joint_loader = round_store.loader(batch_size, shuffle=True, augment=True,
                                              drop_last=True)
        elif preload:
            coverage = fill_pseudo_from_store(teacher, joint_store, len(train_labeled),
                                              conf_threshold, batch_size)
            joint_loader = joint_store.loader(batch_size, shuffle=True, augment=True,
                                              drop_last=True)
        else:
            pseudo_dir = os.path.join(args.out_dir, f"pseudo_r{r}")
            coverage = generate_pseudo_labels(teacher, unlabeled_files, device,
                                              conf_threshold, pseudo_dir, batch_size)
            unlabeled_ds.set_pseudo_dir(pseudo_dir)
            from torch.utils.data import ConcatDataset, DataLoader
            joint = ConcatDataset([labeled_ds, unlabeled_ds])
            joint_loader = DataLoader(joint, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, drop_last=True)
        print(f"伪标签像素覆盖率: {coverage:.2%}")
        if coverage < 0.01:
            print("⚠️ 覆盖率不足 1%，建议降低阈值或先用更多 warmup epoch 训练 teacher")

        model = LightUNet(num_classes=n_cls, dropout=cfg.get("dropout", 0.0)).to(device)  # 每轮从头训练
        best = train_model(model, joint_loader, val_loader, device, epochs,
                           cfg["lr"], cfg.get("weight_decay", 1e-5),
                           args.out_dir, log_prefix=f"r{r}_",
                           class_weights=cfg.get("class_weights"),
                           final_as_best=final_as_best)
        # 本轮 best 改名为 teacher_r{r}.pt，作为下一轮 teacher
        round_best = os.path.join(args.out_dir, "best.pt")
        new_teacher = os.path.join(args.out_dir, f"teacher_r{r}.pt")
        os.replace(round_best, new_teacher)
        teacher_path = new_teacher
        history.append({"round": r, "pseudo_coverage": coverage, "best_val_dice": best})
        print(f"第 {r} 轮 best val dice={best:.4f}")
        del round_store, joint_loader, teacher
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # 最终对外发布的 best.pt = 最后一轮 teacher（即最后一轮验证最优模型）
    final_best = os.path.join(args.out_dir, "best.pt")
    if os.path.exists(final_best):
        os.remove(final_best)
    os.link(teacher_path, final_best)

    meta = dict(cfg)
    meta.update({"teacher_init": args.teacher_ckpt, "rounds": rounds,
                 "start_round": start_round,
                 "conf_threshold": conf_threshold, "epochs_per_round": epochs,
                 "seed": seed, "val_mode": val_mode,
                 "val_mode_requested": args.val_mode,
                 "resolved_data_root": data_root,
                 "crop_size": crop_size, "stage1_ckpt": stage1_ckpt,
                 "thr_rectum": args.thr_rectum, "thr_tumor": args.thr_tumor,
                 "small_cc_area": args.small_cc_area,
                 "small_cc_thr": args.small_cc_thr,
                 "consistency": args.consistency,
                 "n_labeled": len(labeled_files), "n_unlabeled": len(unlabeled_files),
                 "oversample_tumor": ov,
                 "history": history})
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"完成。最终 ckpt: {final_best}")


if __name__ == "__main__":
    main()
