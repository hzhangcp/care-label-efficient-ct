"""生成假 CARE 数据（冒烟测试用），目录结构与真实数据一致。

python src/make_dummy.py --out data_dummy --n_train_cases 8 --n_test_cases 3 --slices_per_case 4
"""

import argparse
import csv
import os

import numpy as np


def make_case(rng, case_idx, n_slices, binary=False):
    files = []
    h = w = 224
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = rng.integers(80, 144, size=2)  # 直肠中心
    for s in range(n_slices):
        image = rng.normal(0.3, 0.1, (h, w)).astype(np.float64).clip(0, 1)
        label = np.zeros((h, w), dtype=np.uint8)
        r_rect = rng.integers(20, 35) + s  # 正常直肠圆
        rect = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_rect ** 2
        image[rect] = 0.7
        if not binary:
            label[rect] = 1
        if rng.random() < 0.7:  # 70% 切片带肿瘤
            r_tum = rng.integers(5, 10)
            tx, ty = cx + rng.integers(-8, 8), cy + rng.integers(-8, 8)
            tum = (yy - ty) ** 2 + (xx - tx) ** 2 <= r_tum ** 2
            image[tum] = 0.9
            label[tum] = 1 if binary else 2
        name = f"case{case_idx:03d}_{s:04d}.npz"
        files.append((name, image, label))
    return files


def write_split(root, split, rng, n_cases, slices_per_case, binary=False):
    npz_dir = os.path.join(root, split, f"{split}_npz")
    os.makedirs(npz_dir, exist_ok=True)
    rows = []
    for c in range(n_cases):
        for name, image, label in make_case(rng, c, slices_per_case, binary):
            np.savez(os.path.join(npz_dir, name), image=image, label=label)
            ys, xs = np.where(label > 0)
            bbox = f"{xs.min()},{ys.min()},{xs.max()},{ys.max()}" if len(xs) else ""
            rows.append([name, bbox])
    with open(os.path.join(root, split, f"{split}_bbox.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_dummy")
    ap.add_argument("--n_train_cases", type=int, default=8)
    ap.add_argument("--n_test_cases", type=int, default=3)
    ap.add_argument("--slices_per_case", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--binary", action="store_true",
                    help="二分类假数据（标签 {0,1}，仅肿瘤为前景；MSD Colon 冒烟用）")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    n_tr = write_split(args.out, "train", rng, args.n_train_cases,
                       args.slices_per_case, args.binary)
    n_te = write_split(args.out, "test", rng, args.n_test_cases,
                       args.slices_per_case, args.binary)
    print(f"假数据已生成: {args.out} (train {n_tr} 切片 / test {n_te} 切片)")


if __name__ == "__main__":
    main()
