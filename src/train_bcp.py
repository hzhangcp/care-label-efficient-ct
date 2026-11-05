"""BCP（Bidirectional Copy-Paste, Bai et al. CVPR 2023）半监督分割基线。

结构对齐 train_cps.py / train_mt.py：单网络 student + EMA teacher，
- 损失 = 标注切片有监督 loss（CE+Dice，含 class_weights，复用 common.compute_loss）
        + λ(t)·BCP 双向 copy-paste 一致性 loss；
- λ(t) 与 train_mt.py 相同的 sigmoid ramp-up（exp(-5(1-t/T)²)）。

双向 copy-paste（参照 BCP 原论文，逐对配对 labeled/unlabeled batch）：
- inward：把 labeled 前景（label>0）贴到 unlabeled 图上，混合标签 = 粘贴处用 GT、
  其余用 teacher 伪标签；
- outward：把 unlabeled 高置信伪标签前景（argmax>0 且 conf>=threshold）贴到
  labeled 图上，混合标签 = 粘贴处用伪标签、其余用 GT；
- 零均值化（zero-centering）：粘贴前把供体图的前景块减去供体图的逐图均值，
  消除供体/受体平均灰度差带来的边界捷径（BCP 论文 Sec.3.2 的 mean matching）。
混合图送入 student，与混合标签算一致性 loss（compute_loss，CE+Dice）。
注：混合图不再过 augment_batch 的强度抖动（其 clamp(0,1) 会破坏零均值化）；
labeled 视图已由 loader 增广，几何/强度多样性仍保留。

模型选择：BCP 推理用 student（EMA teacher 只产伪标签），按 student 验证集前景
平均 Dice 保存 best.pt（单模型格式，evaluate.py 兼容），teacher dice 仅入日志。

用法（级联 stage-2）：
    python src/train_bcp.py --config configs/label_ratio_10.json \
        --out_dir runs/bcp_10 --crop_size 256 --stage1_ckpt runs/s1_10/best.pt
"""

import argparse
import copy
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import dataset
from common import (compute_loss, evaluate_val_dice, load_config, load_model,
                    resolve_device, set_seed)
from dataset import labeled_subset, list_slices, parse_case_id, split_labeled_val, split_train_val
from models import LightUNet
from tensorstore import TensorStore
from train_mt import sigmoid_rampup, update_ema


def bidirectional_copy_paste(l_img, l_lab, u_img, u_pseudo, u_conf, threshold):
    """BCP 双向零均值化 copy-paste。返回 (mix_img, mix_lab)（两个方向拼接，2N）。

    l_img (N,1,H,W) 标注图（已增广），l_lab (N,H,W) GT；u_img 未标注图，
    u_pseudo/u_conf 为 teacher 在其上的 argmax/置信度。
    """
    n = min(l_img.shape[0], u_img.shape[0])
    l_img, l_lab = l_img[:n], l_lab[:n]
    u_img, u_pseudo, u_conf = u_img[:n], u_pseudo[:n], u_conf[:n]
    # 供体图逐图均值 -> 粘贴块零均值化（消除供/受体灰度差边界捷径）
    mean_l = l_img.mean(dim=(1, 2, 3), keepdim=True)
    mean_u = u_img.mean(dim=(1, 2, 3), keepdim=True)
    # inward：labeled 前景 -> unlabeled 图；标签 = 粘贴处 GT + 其余伪标签
    mask_l = l_lab > 0
    in_img = torch.where(mask_l.unsqueeze(1), l_img - mean_l, u_img)
    in_lab = torch.where(mask_l, l_lab, u_pseudo)
    # outward：unlabeled 高置信伪前景 -> labeled 图；标签 = 粘贴处伪标签 + 其余 GT
    mask_u = (u_pseudo > 0) & (u_conf >= threshold)
    out_img = torch.where(mask_u.unsqueeze(1), u_img - mean_u, l_img)
    out_lab = torch.where(mask_u, u_pseudo, l_lab)
    return torch.cat([in_img, out_img]), torch.cat([in_lab, out_lab])


def train_bcp(model, lab_loader, unlab_loader, val_loader, device, epochs, lr,
              weight_decay, out_dir, ema_decay, consistency_weight,
              rampup_epochs, bcp_threshold, final_as_best=False):
    """BCP 训练循环。每步：标注 batch 有监督 loss + λ·双向 copy-paste 一致性 loss。

    teacher = EMA(student)，eval 模式在未标注图上产伪标签（同 train_mt）。
    按 student 验证集前景平均 Dice 保存 best.pt（student 权重，evaluate.py 兼容），
    另存 last.pt（student + teacher）。final_as_best=True（val_mode=fixed）：不做
    验证选择，best.pt 存最终 epoch 的 student。返回 best student val dice。
    """
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    os.makedirs(out_dir, exist_ok=True)
    use_amp = str(device).startswith("cuda")
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = model.to(memory_format=torch.channels_last)
        teacher = teacher.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_dice = -1.0
    fg_names = dataset.fg_class_names()  # CARE: rectum/tumor；MSD: tumor
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "sup_loss", "bcp_loss", "lambda",
                         "val_dice_mean"]
                        + [f"val_dice_{n}" for n in fg_names.values()]
                        + ["teacher_val_dice_mean", "lr"])
        for epoch in range(1, epochs + 1):
            model.train()
            lam = consistency_weight * sigmoid_rampup(epoch, rampup_epochs)
            tot = sup_tot = bcp_tot = 0.0
            n_batch = 0
            lab_iter = iter(lab_loader)
            for ubatch in unlab_loader:
                try:
                    lbatch = next(lab_iter)
                except StopIteration:
                    lab_iter = iter(lab_loader)
                    lbatch = next(lab_iter)
                l_img = lbatch["image"].to(device, non_blocking=True)
                l_lab = lbatch["label"].to(device, non_blocking=True)
                u_img = ubatch["image"].to(device, non_blocking=True)
                if use_amp:
                    l_img = l_img.to(memory_format=torch.channels_last)
                    u_img = u_img.to(memory_format=torch.channels_last)
                optimizer.zero_grad()
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        t_probs = torch.softmax(teacher(u_img).float(), dim=1)
                u_conf, u_pseudo = t_probs.max(1)
                mix_img, mix_lab = bidirectional_copy_paste(
                    l_img, l_lab, u_img, u_pseudo, u_conf, bcp_threshold)
                if use_amp:
                    mix_img = mix_img.to(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    sup_loss = compute_loss(model(l_img), l_lab)
                    bcp_loss = compute_loss(model(mix_img), mix_lab)
                    loss = sup_loss + lam * bcp_loss
                loss.backward()
                optimizer.step()
                update_ema(model, teacher, ema_decay)
                tot += loss.item()
                sup_tot += sup_loss.item()
                bcp_tot += bcp_loss.item()
                n_batch += 1
            scheduler.step()
            s_mean, s_per = evaluate_val_dice(model, val_loader, device)
            t_mean, _ = evaluate_val_dice(teacher, val_loader, device)
            writer.writerow([epoch, f"{tot / max(n_batch,1):.4f}",
                             f"{sup_tot / max(n_batch,1):.4f}",
                             f"{bcp_tot / max(n_batch,1):.4f}", f"{lam:.4f}",
                             f"{s_mean:.4f}"]
                            + [f"{s_per[c]:.4f}" for c in fg_names]
                            + [f"{t_mean:.4f}", f"{scheduler.get_last_lr()[0]:.2e}"])
            f.flush()
            if s_mean > best_dice:
                best_dice = s_mean
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_dice": s_mean,
                            "num_classes": dataset.NUM_CLASSES},
                           os.path.join(out_dir, "best.pt"))
            per_str = ", ".join(f"{n}={s_per[c]:.4f}" for c, n in fg_names.items())
            print(f"[epoch {epoch}/{epochs}] loss={tot/max(n_batch,1):.4f} "
                  f"(sup={sup_tot/max(n_batch,1):.4f} bcp={bcp_tot/max(n_batch,1):.4f} "
                  f"λ={lam:.3f}) student_val_dice={s_mean:.4f} "
                  f"({per_str}) teacher={t_mean:.4f}")
    torch.save({"model": model.state_dict(), "teacher": teacher.state_dict(),
                "epoch": epochs, "val_dice": best_dice,
                "num_classes": dataset.NUM_CLASSES},
               os.path.join(out_dir, "last.pt"))
    if final_as_best:
        torch.save({"model": model.state_dict(), "epoch": epochs,
                    "val_dice": s_mean,
                    "num_classes": dataset.NUM_CLASSES},
                   os.path.join(out_dir, "best.pt"))
        print(f"[final_as_best] best.pt = 最终 epoch {epochs} 的 student"
              f"（val_dice={s_mean:.4f} 仅日志参考）")
        return s_mean
    return best_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default=None, help="覆盖 config 中的 data_root")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 config（冒烟测试用）")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--ema_decay", type=float, default=None, help="teacher EMA 衰减")
    ap.add_argument("--consistency_weight", type=float, default=None, help="λ_max")
    ap.add_argument("--rampup_epochs", type=int, default=None, help="λ ramp-up 长度 T")
    ap.add_argument("--bcp_threshold", type=float, default=None,
                    help="outward 方向伪标签置信度阈值（默认 0.9）")
    ap.add_argument("--crop_size", type=int, default=None,
                    help="级联 stage-2：裁剪边长（需同时给 --stage1_ckpt）")
    ap.add_argument("--stage1_ckpt", default=None,
                    help="级联 stage-1 ckpt（未标注池/验证集的裁剪中心来源）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config 中的 seed（多 seed 重复实验；标注子集随 seed 重抽）")
    ap.add_argument("--val_mode", choices=["pool", "labeled", "fixed"], default="pool",
                    help="模型选择验证集来源：pool=全训练池 15%%（默认，原行为）；"
                         "labeled=仅从标注病例后缀划 15%%（标注预算一致，<4 例回退 fixed）；"
                         "fixed=不做验证选择，最终 epoch 的 student 存 best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg["data_root"]
    bcp_cfg = cfg.get("bcp", {})
    epochs = args.epochs or bcp_cfg.get("epochs", cfg["epochs"])
    batch_size = args.batch_size or cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)
    ema_decay = args.ema_decay if args.ema_decay is not None else bcp_cfg.get("ema_decay", 0.99)
    consistency_weight = (args.consistency_weight if args.consistency_weight is not None
                          else bcp_cfg.get("consistency_weight", 1.0))
    rampup_epochs = (args.rampup_epochs if args.rampup_epochs is not None
                     else bcp_cfg.get("rampup_epochs", 40))
    bcp_threshold = (args.bcp_threshold if args.bcp_threshold is not None
                     else bcp_cfg.get("bcp_threshold", 0.9))
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
    if not unlabeled_files:
        raise SystemExit("未标注池为空（label_ratio=100% 时 BCP 无意义），请用 train_supervised.py")
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
        print("val_mode=fixed：不做验证集模型选择，最终 epoch 的 student 保存为 best.pt；"
              "log.csv 的 val_dice 在训练标注切片上计算，仅作参考")
    else:  # pool：原行为
        train_labeled_files = labeled_files

    print(f"device={device} seed={seed} val_mode={val_mode} "
          f"标注切片 {len(labeled_files)} / 未标注 {len(unlabeled_files)} "
          f"/ 验证 {len(val_files)}；epochs={epochs} ema_decay={ema_decay} "
          f"λ_max={consistency_weight} rampup={rampup_epochs} "
          f"bcp_threshold={bcp_threshold}")

    if not cfg.get("preload", False):
        raise SystemExit("train_bcp 需要 preload=true（TensorStore 管线）")
    crop_size = args.crop_size or cfg.get("crop_size")
    stage1_ckpt = args.stage1_ckpt or cfg.get("stage1_ckpt")
    if crop_size and not stage1_ckpt:
        raise SystemExit("--crop_size 需要 --stage1_ckpt")

    cache_dir = cfg.get("cache_dir")
    ov = cfg.get("oversample_tumor", 1)  # 肿瘤切片过采样（MSD 用；仅标注训练列表）
    train_labeled = train_labeled_files
    if ov > 1:
        from dataset import oversample_positive_files
        train_labeled = oversample_positive_files(train_labeled_files, ov)
        print(f"过采样标注肿瘤切片 x{ov}: {len(train_labeled_files)} -> {len(train_labeled)}")
    lab_store = TensorStore.from_files(train_labeled, device, cache_dir)
    unlab_store = TensorStore.from_files(unlabeled_files, device, cache_dir)
    val_store = TensorStore.from_files(val_files, device, cache_dir)
    if crop_size:
        from cascade import batch_argmax, fg_centers
        m1 = load_model(stage1_ckpt, device)
        # 标注切片按 GT 前景质心裁剪；未标注/验证切片按 stage-1 预测质心
        # （无 GT 泄漏，与 train_mt/train_cps 级联模式一致）
        lab_store.crop_in_place(fg_centers(lab_store.labels), crop_size)
        unlab_pred1 = batch_argmax(m1, unlab_store.images, batch_size)
        unlab_store.crop_in_place(fg_centers(unlab_pred1), crop_size)
        val_pred1 = batch_argmax(m1, val_store.images, batch_size)
        val_store.crop_in_place(fg_centers(val_pred1), crop_size)
        del m1, unlab_pred1, val_pred1
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    lab_loader = lab_store.loader(batch_size, shuffle=True, augment=True,
                                  drop_last=len(lab_store) >= batch_size)
    unlab_loader = unlab_store.loader(batch_size, shuffle=True, augment=False,
                                      drop_last=True)
    val_loader = val_store.loader(batch_size, shuffle=False)

    common._LOSS_CLASS_WEIGHTS = (  # 与 train_model 相同的有监督损失设置
        torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
        if cfg.get("class_weights") else None)
    model = LightUNet(num_classes=n_cls, dropout=cfg.get("dropout", 0.0)).to(device)
    best = train_bcp(model, lab_loader, unlab_loader, val_loader, device, epochs,
                     cfg["lr"], cfg.get("weight_decay", 1e-5), args.out_dir,
                     ema_decay, consistency_weight, rampup_epochs, bcp_threshold,
                     final_as_best=final_as_best)

    meta = dict(cfg)
    meta.update({"method": "bcp", "resolved_data_root": data_root,
                 "resolved_epochs": epochs, "ema_decay": ema_decay,
                 "seed": seed, "val_mode": val_mode,
                 "val_mode_requested": args.val_mode,
                 "consistency_weight": consistency_weight,
                 "rampup_epochs": rampup_epochs, "bcp_threshold": bcp_threshold,
                 "crop_size": crop_size, "stage1_ckpt": stage1_ckpt,
                 "n_labeled": len(labeled_files), "n_unlabeled": len(unlabeled_files),
                 "n_val_slices": len(val_files), "best_val_dice": best,
                 "note": "best.pt 为 student 权重（evaluate.py 兼容）；"
                         "last.pt 含 student(model)+teacher(EMA)"})
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"完成。best student val dice={best:.4f}，ckpt 保存在 {args.out_dir}")


if __name__ == "__main__":
    main()
