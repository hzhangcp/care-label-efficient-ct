"""CPS（Cross Pseudo Supervision, Chen et al. CVPR 2021）半监督分割基线。

与 Mean Teacher（train_mt.py）的单模型 EMA 不同，CPS 联合训练两个独立初始化的
LightUNet，用彼此的伪标签交叉监督：
- 损失 = net1/net2 各自的有监督 loss（标注切片，CE+Dice，含 class_weights，
  复用 common.compute_loss）+ λ(t)·交叉伪监督 loss（未标注切片）；
- 交叉伪监督：pseudo1 = argmax(net1(u)) 以 CE 监督 net2，pseudo2 = argmax(net2(u))
  以 CE 监督 net1。原始 CPS 不加置信度阈值（--cps_threshold 可选，默认 None）；
- λ(t) 与 train_mt.py 相同的 sigmoid ramp-up；
- 扰动差异 = 独立初始化（不同 seed）+ 两个网络共享的增广视图 + 各自 dropout。

模型选择：每 epoch 分别评估两个网络的验证集前景平均 Dice，当 max(dice1, dice2)
刷新历史最优时，把较优网络存 best.pt（单模型格式，与 evaluate.py 兼容），
另一个存 best_aux.pt。

用法（级联 stage-2）：
    python src/train_cps.py --config configs/label_ratio_10.json \
        --out_dir runs/cps_10 --crop_size 256 --stage1_ckpt runs/s1_10/best.pt
"""

import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import dataset
from common import (compute_loss, evaluate_val_dice, load_config, load_model,
                    resolve_device, set_seed)
from dataset import IGNORE_INDEX, labeled_subset, list_slices, parse_case_id, split_labeled_val, split_train_val
from models import LightUNet
from tensorstore import TensorStore, augment_batch
from train_mt import sigmoid_rampup


def cps_loss_pair(logits1, logits2, threshold=None):
    """交叉伪监督：net1 的伪标签监督 net2，net2 的伪标签监督 net1。

    threshold 为 None 时保持原始 CPS 行为（无置信度过滤）；否则低置信像素
    置 IGNORE_INDEX 不参与交叉监督。返回标量 loss。
    """
    probs1 = torch.softmax(logits1.float(), dim=1)
    probs2 = torch.softmax(logits2.float(), dim=1)
    conf1, pseudo1 = probs1.max(1)
    conf2, pseudo2 = probs2.max(1)
    if threshold is not None:
        pseudo1 = pseudo1.clone()
        pseudo2 = pseudo2.clone()
        pseudo1[conf1 < threshold] = IGNORE_INDEX
        pseudo2[conf2 < threshold] = IGNORE_INDEX
    loss1 = F.cross_entropy(logits2, pseudo1, ignore_index=IGNORE_INDEX)
    loss2 = F.cross_entropy(logits1, pseudo2, ignore_index=IGNORE_INDEX)
    return loss1 + loss2


def train_cps(net1, net2, lab_loader, unlab_loader, val_loader, device, epochs,
              lr, weight_decay, out_dir, consistency_weight, rampup_epochs,
              cps_threshold, final_as_best=False):
    """CPS 训练循环。每步：两网络各自有监督 loss + λ·交叉伪监督 loss。

    按两网络验证集前景平均 Dice 的较大者保存 best.pt（优）/best_aux.pt（次），
    另存 last.pt（含两个网络权重）。final_as_best=True（val_mode=fixed）：不做
    验证选择，best.pt/best_aux.pt 存最终 epoch 的两网络，返回最后一轮 lead dice
    （仅日志参考）。默认 False，行为不变。返回 best val dice。
    """
    os.makedirs(out_dir, exist_ok=True)
    use_amp = str(device).startswith("cuda")
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        net1 = net1.to(memory_format=torch.channels_last)
        net2 = net2.to(memory_format=torch.channels_last)
    opt1 = torch.optim.AdamW(net1.parameters(), lr=lr, weight_decay=weight_decay)
    opt2 = torch.optim.AdamW(net2.parameters(), lr=lr, weight_decay=weight_decay)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs)
    best_dice = -1.0
    fg_names = dataset.fg_class_names()  # CARE: rectum/tumor；MSD: tumor
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "sup_loss", "cps_loss", "lambda",
                         "val_dice_mean"]
                        + [f"val_dice_{n}" for n in fg_names.values()]
                        + ["net1_val_dice_mean", "net2_val_dice_mean", "lr"])
        for epoch in range(1, epochs + 1):
            net1.train()
            net2.train()
            lam = consistency_weight * sigmoid_rampup(epoch, rampup_epochs)
            tot = sup_tot = cps_tot = 0.0
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
                # 两个网络共享同一增广视图；差异来自独立初始化 + 各自 dropout
                u_img_aug, _ = augment_batch(u_img)
                opt1.zero_grad()
                opt2.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    sup_loss = (compute_loss(net1(l_img), l_lab)
                                + compute_loss(net2(l_img), l_lab))
                    cps_loss = cps_loss_pair(net1(u_img_aug), net2(u_img_aug),
                                             cps_threshold)
                    loss = sup_loss + lam * cps_loss
                loss.backward()
                opt1.step()
                opt2.step()
                tot += loss.item()
                sup_tot += sup_loss.item()
                cps_tot += cps_loss.item()
                n_batch += 1
            sch1.step()
            sch2.step()
            d1, p1 = evaluate_val_dice(net1, val_loader, device)
            d2, p2 = evaluate_val_dice(net2, val_loader, device)
            # 选 val 更优的网络为当轮代表；历史最优时落盘 best.pt / best_aux.pt
            lead, lead_per = (d1, p1) if d1 >= d2 else (d2, p2)
            writer.writerow([epoch, f"{tot / max(n_batch,1):.4f}",
                             f"{sup_tot / max(n_batch,1):.4f}",
                             f"{cps_tot / max(n_batch,1):.4f}", f"{lam:.4f}",
                             f"{lead:.4f}"]
                            + [f"{lead_per[c]:.4f}" for c in fg_names]
                            + [f"{d1:.4f}", f"{d2:.4f}",
                               f"{sch1.get_last_lr()[0]:.2e}"])
            f.flush()
            if lead > best_dice:
                best_dice = lead
                lead_net, aux_net = (net1, net2) if d1 >= d2 else (net2, net1)
                torch.save({"model": lead_net.state_dict(), "epoch": epoch,
                            "val_dice": lead,
                            "num_classes": dataset.NUM_CLASSES},
                           os.path.join(out_dir, "best.pt"))
                torch.save({"model": aux_net.state_dict(), "epoch": epoch,
                            "val_dice": min(d1, d2),
                            "num_classes": dataset.NUM_CLASSES},
                           os.path.join(out_dir, "best_aux.pt"))
            print(f"[epoch {epoch}/{epochs}] loss={tot/max(n_batch,1):.4f} "
                  f"(sup={sup_tot/max(n_batch,1):.4f} cps={cps_tot/max(n_batch,1):.4f} "
                  f"λ={lam:.3f}) val_dice net1={d1:.4f} net2={d2:.4f} best={best_dice:.4f}")
    torch.save({"model1": net1.state_dict(), "model2": net2.state_dict(),
                "epoch": epochs, "val_dice": best_dice,
                "num_classes": dataset.NUM_CLASSES},
               os.path.join(out_dir, "last.pt"))
    if final_as_best:
        lead_net, aux_net = (net1, net2) if d1 >= d2 else (net2, net1)
        torch.save({"model": lead_net.state_dict(), "epoch": epochs,
                    "val_dice": lead, "num_classes": dataset.NUM_CLASSES},
                   os.path.join(out_dir, "best.pt"))
        torch.save({"model": aux_net.state_dict(), "epoch": epochs,
                    "val_dice": min(d1, d2), "num_classes": dataset.NUM_CLASSES},
                   os.path.join(out_dir, "best_aux.pt"))
        print(f"[final_as_best] best.pt = 最终 epoch {epochs} 的较优网络"
              f"（val_dice={lead:.4f} 仅日志参考）")
        return lead
    return best_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default=None, help="覆盖 config 中的 data_root")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 config（冒烟测试用）")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--consistency_weight", type=float, default=None, help="λ_max")
    ap.add_argument("--rampup_epochs", type=int, default=None, help="λ ramp-up 长度 T")
    ap.add_argument("--cps_threshold", type=float, default=None,
                    help="可选：交叉伪标签置信度阈值（默认 None=原始 CPS 不过滤）")
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
                         "fixed=不做验证选择，最终 epoch 两网络存 best.pt/best_aux.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg["data_root"]
    cps_cfg = cfg.get("cps", {})
    epochs = args.epochs or cps_cfg.get("epochs", cfg["epochs"])
    batch_size = args.batch_size or cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)
    consistency_weight = (args.consistency_weight if args.consistency_weight is not None
                          else cps_cfg.get("consistency_weight", 1.0))
    rampup_epochs = (args.rampup_epochs if args.rampup_epochs is not None
                     else cps_cfg.get("rampup_epochs", 40))
    cps_threshold = (args.cps_threshold if args.cps_threshold is not None
                     else cps_cfg.get("cps_threshold"))
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = resolve_device(args.device)
    set_seed(seed)
    dataset.set_num_classes(cfg.get("num_classes", 3))  # 默认 3=CARE；MSD Colon 配 2

    all_train = list_slices(data_root, "train")
    pool_files, val_files = split_train_val(all_train, cfg.get("val_ratio", 0.15), seed)
    labeled_files = labeled_subset(pool_files, cfg["label_ratio"], seed)
    labeled_set = set(labeled_files)
    unlabeled_files = [p for p in pool_files if p not in labeled_set]
    if not unlabeled_files:
        raise SystemExit("未标注池为空（label_ratio=100% 时 CPS 无意义），请用 train_supervised.py")
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
        print("val_mode=fixed：不做验证集模型选择，最终 epoch 两网络保存为 best.pt/best_aux.pt；"
              "log.csv 的 val_dice 在训练标注切片上计算，仅作参考")
    else:  # pool：原行为
        train_labeled_files = labeled_files

    print(f"device={device} seed={seed} val_mode={val_mode} "
          f"标注切片 {len(labeled_files)} / 未标注 {len(unlabeled_files)} "
          f"/ 验证 {len(val_files)}；epochs={epochs} λ_max={consistency_weight} "
          f"rampup={rampup_epochs} cps_threshold={cps_threshold}")

    if not cfg.get("preload", False):
        raise SystemExit("train_cps 需要 preload=true（TensorStore 管线）")
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
        # （无 GT 泄漏，与 train_mt/train_semi 级联模式一致）
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
    # 两个独立初始化：net1 用全局 seed，net2 换 seed 保证初始化不同
    n_cls = cfg.get("num_classes", 3)  # 默认 3=CARE；MSD Colon 配 2
    net1 = LightUNet(num_classes=n_cls, dropout=cfg.get("dropout", 0.0)).to(device)
    torch.manual_seed(seed + 1)
    net2 = LightUNet(num_classes=n_cls, dropout=cfg.get("dropout", 0.0)).to(device)
    best = train_cps(net1, net2, lab_loader, unlab_loader, val_loader, device,
                     epochs, cfg["lr"], cfg.get("weight_decay", 1e-5),
                     args.out_dir, consistency_weight, rampup_epochs,
                     cps_threshold, final_as_best=final_as_best)

    meta = dict(cfg)
    meta.update({"method": "cps", "resolved_data_root": data_root,
                 "resolved_epochs": epochs,
                 "seed": seed, "val_mode": val_mode,
                 "val_mode_requested": args.val_mode,
                 "consistency_weight": consistency_weight,
                 "rampup_epochs": rampup_epochs, "cps_threshold": cps_threshold,
                 "crop_size": crop_size, "stage1_ckpt": stage1_ckpt,
                 "n_labeled": len(labeled_files), "n_unlabeled": len(unlabeled_files),
                 "n_val_slices": len(val_files), "best_val_dice": best,
                 "note": "best.pt 为 val 更优网络（evaluate.py 兼容）；best_aux.pt 为另一个"})
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"完成。best val dice={best:.4f}，ckpt 保存在 {args.out_dir}")


if __name__ == "__main__":
    main()
