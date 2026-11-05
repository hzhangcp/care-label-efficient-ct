"""全监督基线训练。

用法：
    python src/train_supervised.py --config configs/label_ratio_100.json --out_dir runs/sup_100

标注子集按病人级抽取（见 dataset.labeled_subset），test 集全程不参与。
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
from dataset import labeled_subset, list_slices, parse_case_id, split_labeled_val, split_train_val
from models import LightUNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default=None, help="覆盖 config 中的 data_root")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 config（冒烟测试用）")
    ap.add_argument("--target_updates", type=int, default=None,
                    help="目标优化器更新数；按实际 steps/epoch 换算 epochs，覆盖 epochs")
    ap.add_argument("--val_interval", type=int, default=None,
                    help="每隔多少 epoch 验证一次；最后一个 epoch 总是验证")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--crop_size", type=int, default=None,
                    help="级联 stage-2：裁剪边长（需同时给 --stage1_ckpt）")
    ap.add_argument("--stage1_ckpt", default=None,
                    help="级联 stage-1 ckpt，用于验证集裁剪中心（模拟测试行为）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config 中的 seed（多 seed 重复实验；标注子集随 seed 重抽）")
    ap.add_argument("--val_mode", choices=["pool", "labeled", "fixed"], default="pool",
                    help="模型选择验证集来源：pool=全训练池 15%%（默认，原行为）；"
                         "labeled=仅从标注病例后缀划 15%%（标注预算一致，<4 例回退 fixed）；"
                         "fixed=不做验证选择，最终 epoch 模型存 best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg["data_root"]
    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    device = resolve_device(args.device)
    set_seed(seed)
    n_cls = cfg.get("num_classes", 3)  # 默认 3=CARE；MSD Colon 配 2
    dataset.set_num_classes(n_cls)

    all_train = list_slices(data_root, "train")
    pool_files, val_files = split_train_val(all_train, cfg.get("val_ratio", 0.15), seed)
    label_ratio = cfg["label_ratio"]
    labeled_files = labeled_subset(pool_files, label_ratio, seed)
    val_mode = args.val_mode
    if val_mode == "labeled":
        n_lab_cases = len({parse_case_id(p) for p in labeled_files})
        if n_lab_cases < 4:
            print(f"⚠️ val_mode=labeled 但标注病例仅 {n_lab_cases} 例（<4），"
                  "无法划出验证集，回退 val_mode=fixed")
            val_mode = "fixed"
    final_as_best = val_mode == "fixed"
    if val_mode == "labeled":
        train_files, val_files = split_labeled_val(labeled_files,
                                                   cfg.get("val_ratio", 0.15), seed)
    elif val_mode == "fixed":
        train_files = labeled_files
        val_files = labeled_files  # 仅日志用（训练标注切片上的 Dice），不做模型选择
        print("val_mode=fixed：不做验证集模型选择，最终 epoch 模型保存为 best.pt；"
              "log.csv 的 val_dice 在训练标注切片上计算，仅作参考")
    else:  # pool：原行为
        train_files = labeled_files
    ov = cfg.get("oversample_tumor", 1)  # 肿瘤切片过采样（MSD 用；CARE 默认 1=关闭）
    if ov > 1:
        from dataset import oversample_positive_files
        n0 = len(train_files)
        train_files = oversample_positive_files(train_files, ov)
        print(f"过采样肿瘤切片 x{ov}: {n0} -> {len(train_files)}")

    print(f"data_root={data_root} device={device} seed={seed} val_mode={val_mode}")
    print(f"训练池 {len(pool_files)} 切片 / 验证 {len(val_files)} 切片；"
          f"标注比例 {label_ratio:.0%} -> 标注训练切片 {len(train_files)}")

    preload = cfg.get("preload", False)
    crop_size = args.crop_size or cfg.get("crop_size")
    stage1_ckpt = args.stage1_ckpt or cfg.get("stage1_ckpt")
    if crop_size and not preload:
        raise SystemExit("--crop_size 需要 preload=true（TensorStore 管线）")
    if preload:
        # 设备驻留张量管线（快一个量级，见 tensorstore.py）
        from tensorstore import TensorStore
        cache_dir = cfg.get("cache_dir")
        train_store = TensorStore.from_files(train_files, device, cache_dir)
        val_store = TensorStore.from_files(val_files, device, cache_dir)
        if crop_size:
            from cascade import batch_argmax, fg_centers
            if not stage1_ckpt:
                raise SystemExit("--crop_size 需要 --stage1_ckpt")
            # 训练集按 GT 前景质心裁剪；验证集按 stage-1 预测质心（模拟测试行为）
            train_store.crop_in_place(fg_centers(train_store.labels), crop_size)
            m1 = load_model(stage1_ckpt, device)
            val_pred1 = batch_argmax(m1, val_store.images, batch_size)
            val_store.crop_in_place(fg_centers(val_pred1), crop_size)
            del m1, val_pred1
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        train_loader = train_store.loader(batch_size, shuffle=True, augment=True,
                                          drop_last=len(train_store) >= batch_size)
        val_loader = val_store.loader(batch_size, shuffle=False)
    else:
        train_loader, val_loader = build_loaders(train_files, val_files, batch_size,
                                                 num_workers, preload=False)
    target_updates = (args.target_updates if args.target_updates is not None
                      else cfg.get("target_updates"))
    if isinstance(target_updates, dict):
        target_updates = target_updates.get(str(seed))
    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise SystemExit("训练 loader 没有完整 batch，无法训练")
    if target_updates is not None:
        target_updates = int(target_updates)
        if target_updates < 1:
            raise SystemExit("target_updates 必须 >= 1")
        epochs = (target_updates + steps_per_epoch - 1) // steps_per_epoch
        print(f"目标更新数 {target_updates} / 每 epoch {steps_per_epoch} 步 -> "
              f"{epochs} epoch（实际 {epochs * steps_per_epoch} 步）")
    val_interval = (args.val_interval if args.val_interval is not None
                    else cfg.get("val_interval", 1))
    if val_interval < 1:
        raise SystemExit("val_interval 必须 >= 1")
    model = LightUNet(num_classes=n_cls, dropout=cfg.get("dropout", 0.0)).to(device)
    best = train_model(model, train_loader, val_loader, device, epochs,
                       cfg["lr"], cfg.get("weight_decay", 1e-5), args.out_dir,
                       class_weights=cfg.get("class_weights"),
                       final_as_best=final_as_best, val_interval=val_interval)

    meta = dict(cfg)
    meta.update({"resolved_data_root": data_root, "resolved_epochs": epochs,
                 "resolved_target_updates": target_updates,
                 "steps_per_epoch": steps_per_epoch,
                 "actual_optimizer_updates": epochs * steps_per_epoch,
                 "val_interval": val_interval,
                 "seed": seed, "val_mode": val_mode,
                 "val_mode_requested": args.val_mode,
                 "n_labeled_slices": len(train_files), "n_val_slices": len(val_files),
                 "crop_size": crop_size, "stage1_ckpt": stage1_ckpt,
                 "best_val_dice": best})
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"完成。best val dice={best:.4f}，ckpt 保存在 {args.out_dir}")


if __name__ == "__main__":
    main()
