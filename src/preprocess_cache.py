"""把 CARE npz 转成 .npy 缓存（float32 images + uint8 labels），加速后续所有实验。

python src/preprocess_cache.py --data_root data/CARE --out data/cache

train: 26,656 切片 -> images ~26GB(float32) + labels ~6.6GB(uint8)
只需运行一次；TensorStore.from_files(cache_dir=...) 会自动使用。
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import list_slices, remap_label
from tensorstore import cache_paths


def convert(data_root, split, out_dir):
    files = list_slices(data_root, split)
    with np.load(files[0]) as z:
        h, w = z["image"].shape
    img_p, lab_p, names_p = cache_paths(out_dir, split)
    images = np.lib.format.open_memmap(img_p, mode="w+", dtype=np.float32,
                                       shape=(len(files), h, w))
    labels = np.lib.format.open_memmap(lab_p, mode="w+", dtype=np.uint8,
                                       shape=(len(files), h, w))
    names = []
    for i, p in enumerate(files):
        with np.load(p) as z:
            images[i] = z["image"]
            labels[i] = remap_label(z["label"])
        names.append(os.path.basename(p))
        if (i + 1) % 2000 == 0:
            print(f"  {split}: {i + 1}/{len(files)}")
    images.flush()
    labels.flush()
    with open(names_p, "w") as f:
        json.dump(names, f)
    print(f"{split}: {len(files)} 切片 -> {img_p} / {lab_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    convert(args.data_root, "train", args.out)
    convert(args.data_root, "test", args.out)


if __name__ == "__main__":
    main()
