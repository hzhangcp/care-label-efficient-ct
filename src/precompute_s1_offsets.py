"""M8 辅助：预计算 stage-1 级联裁剪偏移缓存（CPU 加速手段，不改变任何指标定义）。

50% 档 6 个方法共用同一个 stage-1（s1_50_s{s}），100% 档共用 s1_100_s{s}；
原 cascade_predict 每配置都要重跑 stage-1 全图 512² 前向。本脚本把 stage-1
对 test 集的推理结果（裁剪偏移 + 空预测标记）落盘一次，eval_hd95.py 用
--offsets_npz 直接复用，只对 256² 裁剪做 stage-2 前向，省 ~3/4 推理量。

输出：{out_dir}/{tag}_offsets.npz
    names:   (N,) str          切片文件名（与 list_slices(test) 对齐）
    offsets: (N,2) int64       裁剪窗左上角 (y0, x0)（= crop_offsets(fg_centers(pred1))）
    empty:   (N,) bool         stage-1 空预测标记（触发 tile fallback）
    crop_size / ckpt 记入 attrs.json 同级文件 {tag}_offsets.meta.json

用法：
    python src/precompute_s1_offsets.py --ckpt runs_v2/s1_50_s42/best.pt \
        --tag s1_50_s42 --out_dir results/v2/hd95/s1_cache
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cascade import batch_argmax, crop_offsets, fg_centers
from common import load_model, resolve_device
from dataset import list_slices
from tensorstore import TensorStore


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data_root", default="data/CARE")
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", default="results/v2/hd95/s1_cache")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    files = list_slices(args.data_root, "test")
    if args.limit:
        files = files[:args.limit]
    store = TensorStore.from_files(files, device,
                                   cache_dir=args.cache_dir or None)
    loader = store.loader(args.batch_size, shuffle=False)
    model1 = load_model(args.ckpt, device)

    names, offs, empt = [], [], []
    for bi, batch in enumerate(loader):
        imgs = batch["image"].to(device)
        pred1 = batch_argmax(model1, imgs, args.batch_size)
        n, h, w = pred1.shape
        empty = ~(pred1 > 0).flatten(1).any(dim=1)
        o = crop_offsets(fg_centers(pred1), args.crop_size, h, w)
        names.extend(batch["name"])
        offs.append(o.cpu().numpy())
        empt.append(empty.cpu().numpy())
        if (bi + 1) % 25 == 0:
            print(f"  [{args.tag}] {len(names)}/{len(store)}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.tag}_offsets.npz")
    np.savez(path, names=np.array(names),
             offsets=np.concatenate(offs), empty=np.concatenate(empt))
    with open(os.path.join(args.out_dir, f"{args.tag}_offsets.meta.json"),
              "w") as f:
        json.dump({"ckpt": os.path.abspath(args.ckpt),
                   "crop_size": args.crop_size, "n": len(names),
                   "data_root": os.path.abspath(args.data_root)}, f, indent=2)
    print(f"已保存: {path}（{len(names)} 切片）")


if __name__ == "__main__":
    main()
