"""Compute the CARE 50%-versus-100% annotation-content budget.

The denominator is the same-seed 100% labeled subset of the 270-case development
pool, not the original 318-case release. This makes the slice/foreground-pixel
saving used in the manuscript exactly reproducible.

Production:
    python src/compute_annotation_budget.py \
        --out results/v2/annotation_budget_50_vs_100.json
Self-test (no real data):
    python src/compute_annotation_budget.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (labeled_subset, list_slices, parse_case_id,  # noqa: E402
                     split_train_val)

SEEDS = (42, 43, 44)


def count_files(files, labels, name_to_index):
    ids = np.array([name_to_index[os.path.basename(f)] for f in files])
    selected = labels[ids]
    fg = (selected > 0).sum(axis=(1, 2), dtype=np.int64)
    return {
        "n_cases": len({parse_case_id(f) for f in files}),
        "n_slices": len(files),
        "foreground_pixels": int(fg.sum()),
        "rectum_pixels": int((selected == 1).sum()),
        "tumor_pixels": int((selected == 2).sum()),
    }


def compute(data_root, cache_dir):
    all_train = list_slices(data_root, "train")
    labels = np.load(os.path.join(cache_dir, "train_labels.npy"), mmap_mode="r")
    with open(os.path.join(cache_dir, "train_names.json")) as f:
        names = json.load(f)
    if isinstance(names, dict):
        names = list(names)
    name_to_index = {os.path.basename(n): i for i, n in enumerate(names)}
    out = {
        "definition": "Ratios use the current 270-case development pool; 50% is "
                      "compared with the same-seed 100% labeled subset, not the "
                      "original 318-case release. Foreground means label > 0.",
        "seeds": {},
    }
    for seed in SEEDS:
        pool, _ = split_train_val(all_train, 0.15, seed)
        row = {}
        for ratio in (0.5, 1.0):
            files = labeled_subset(pool, ratio, seed)
            row[str(int(ratio * 100))] = count_files(files, labels, name_to_index)
        keys = ("n_cases", "n_slices", "foreground_pixels",
                "rectum_pixels", "tumor_pixels")
        row["50_vs_100_percent"] = {
            k: 100 * row["50"][k] / row["100"][k] for k in keys}
        out["seeds"][str(seed)] = row
        print(seed, json.dumps(row["50_vs_100_percent"], indent=2), flush=True)
    keys = ("n_cases", "n_slices", "foreground_pixels",
            "rectum_pixels", "tumor_pixels")
    out["mean_50_vs_100_percent"] = {
        k: float(np.mean([out["seeds"][str(s)]["50_vs_100_percent"][k]
                          for s in SEEDS])) for k in keys}
    print("mean", json.dumps(out["mean_50_vs_100_percent"], indent=2))
    return out


def selftest():
    labels = np.array([
        [[0, 1], [2, 0]], [[1, 1], [0, 0]],
        [[2, 2], [0, 0]], [[0, 0], [0, 0]],
    ], dtype=np.uint8)
    names = ["case1_slice000.npz", "case1_slice001.npz",
             "case2_slice000.npz", "case2_slice001.npz"]
    index = {n: i for i, n in enumerate(names)}
    a = count_files(names[:2], labels, index)
    b = count_files(names, labels, index)
    assert a["n_slices"] == 2 and a["n_cases"] == 1
    assert a["foreground_pixels"] == 4 and a["rectum_pixels"] == 3
    assert a["tumor_pixels"] == 1
    assert b["foreground_pixels"] == 6 and b["tumor_pixels"] == 3
    assert np.isclose(100 * a["foreground_pixels"] / b["foreground_pixels"],
                      100 * 4 / 6)
    print("compute_annotation_budget selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/CARE")
    ap.add_argument("--cache_dir", default="data/cache")
    ap.add_argument("--out", default=os.path.join(
        "results", "v2", "annotation_budget_50_vs_100.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    out = compute(args.data_root, args.cache_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
