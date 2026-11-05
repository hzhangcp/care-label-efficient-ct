"""训练/评估共享工具：配置加载、训练循环、验证集 Dice。"""

import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataset
from dataset import CAREDataset, IGNORE_INDEX
from models import LightUNet, dice_loss


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(train_files, val_files, batch_size, num_workers, pseudo_dir=None,
                  preload=False):
    train_ds = CAREDataset(train_files, augment=True, pseudo_dir=pseudo_dir,
                           preload=preload)
    val_ds = CAREDataset(val_files, augment=False, preload=preload)
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=len(train_ds) >= batch_size, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader


_LOSS_CLASS_WEIGHTS = None  # 由 train_model 按 config 设置，None=不加权


def compute_loss(logits, labels):
    ce = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX,
                         weight=_LOSS_CLASS_WEIGHTS)
    dl = dice_loss(logits, labels, dataset.NUM_CLASSES, ignore_index=IGNORE_INDEX)
    return ce + dl


@torch.no_grad()
def evaluate_val_dice(model, loader, device):
    """验证集前景类（1,2）平均 Dice，用于模型选择。"""
    model.eval()
    use_amp = str(device).startswith("cuda")
    fg = dataset.FG_CLASSES
    inter = {c: 0.0 for c in fg}
    denom = {c: 0.0 for c in fg}
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if use_amp:
            images = images.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            preds = model(images).argmax(1)
        for c in fg:
            p = preds == c
            g = labels == c
            inter[c] += (p & g).sum().item()
            denom[c] += p.sum().item() + g.sum().item()
    per_class = {c: (2 * inter[c] / denom[c] if denom[c] > 0 else 0.0) for c in fg}
    mean_dice = sum(per_class.values()) / len(per_class)
    return mean_dice, per_class


def train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay,
                out_dir, log_prefix="", compile=True, class_weights=None,
                final_as_best=False, val_interval=1):
    """标准训练循环。按验证集前景平均 Dice 保存 best.pt，另存 last.pt 与 log.csv。

    CUDA 上启用 bf16 AMP + TF32 + channels_last + torch.compile（A100 实测
    309 -> 101 ms/step，512x512 下 epoch 约 6min -> 1.5min）。
    compile 失败时自动回退 eager。class_weights: 可选 CE 类权重（如 [1,2,3]）。
    final_as_best=True（val_mode=fixed）：不做验证集模型选择，训练结束后把
    最终 epoch 模型覆写为 best.pt，返回最后一轮的 val dice（仅日志参考）。
    val_interval 控制验证与日志中的验证指标间隔，最后一个 epoch 总是验证。
    默认 False，行为不变。返回 best val dice。
    """
    global _LOSS_CLASS_WEIGHTS
    _LOSS_CLASS_WEIGHTS = (torch.tensor(class_weights, dtype=torch.float32, device=device)
                           if class_weights else None)
    if val_interval < 1:
        raise ValueError("val_interval 必须 >= 1")
    os.makedirs(out_dir, exist_ok=True)
    use_amp = str(device).startswith("cuda")
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model = model.to(memory_format=torch.channels_last)
    step_model = model
    if use_amp and compile:
        try:
            step_model = torch.compile(model)
        except Exception as e:
            print(f"⚠️ torch.compile 失败，回退 eager: {e}")
            step_model = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_dice = -1.0
    fg_names = dataset.fg_class_names()  # CARE: rectum/tumor；MSD: tumor
    log_path = os.path.join(out_dir, f"{log_prefix}log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice_mean"]
                        + [f"val_dice_{n}" for n in fg_names.values()] + ["lr"])
        for epoch in range(1, epochs + 1):
            step_model.train()
            total_loss, n_batch = 0.0, 0
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                if use_amp:
                    images = images.to(memory_format=torch.channels_last)
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = compute_loss(step_model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batch += 1
            scheduler.step()
            should_validate = epoch % val_interval == 0 or epoch == epochs
            if should_validate:
                val_mean, val_per = evaluate_val_dice(model, val_loader, device)
                val_cells = [f"{val_mean:.4f}"] + [f"{val_per[c]:.4f}" for c in fg_names]
                if val_mean > best_dice:
                    best_dice = val_mean
                    torch.save({"model": model.state_dict(), "epoch": epoch,
                                "val_dice": val_mean,
                                "num_classes": dataset.NUM_CLASSES},
                               os.path.join(out_dir, "best.pt"))
                per_str = ", ".join(f"{n}={val_per[c]:.4f}" for c, n in fg_names.items())
                val_str = f"{val_mean:.4f} ({per_str})"
            else:
                val_cells = [""] * (1 + len(fg_names))
                val_str = "skipped"
            writer.writerow([epoch, f"{total_loss / max(n_batch,1):.4f}"] + val_cells
                            + [f"{scheduler.get_last_lr()[0]:.2e}"])
            f.flush()
            print(f"[{log_prefix}epoch {epoch}/{epochs}] loss={total_loss/max(n_batch,1):.4f} "
                  f"val_dice={val_str}")
    torch.save({"model": model.state_dict(), "epoch": epochs, "val_dice": best_dice,
                "num_classes": dataset.NUM_CLASSES},
               os.path.join(out_dir, "last.pt"))
    if final_as_best:
        torch.save({"model": model.state_dict(), "epoch": epochs,
                    "val_dice": val_mean,
                    "num_classes": dataset.NUM_CLASSES},
                   os.path.join(out_dir, "best.pt"))
        print(f"[{log_prefix}final_as_best] best.pt = 最终 epoch {epochs} 模型"
              f"（val_dice={val_mean:.4f} 仅日志参考）")
        return val_mean
    return best_dice


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    n_cls = state.get("num_classes", 3)  # 旧 ckpt 无此键 -> CARE 三分类
    dataset.set_num_classes(n_cls)
    model = LightUNet(num_classes=n_cls)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
