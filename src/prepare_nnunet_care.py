"""把 CARE 转成 nnU-Net v2 raw 格式：Dataset501_CARE100 / Dataset502_CARE50。

用法：
    python src/prepare_nnunet_care.py --data_root data/CARE --out data/nnunet

设计要点（与主仓库实验严格对齐，证据见注释）：
- 方向约定：主仓库训练/评估用的是 npz 里【原样存储】的数组——
  preprocess_cache.py 逐切片 `images[i] = z["image"]` / `labels[i] = remap_label(z["label"])`，
  tensorstore.py 回退路径同样不做任何转置（仅 visualize_predictions.py 显示时转 90°）。
  因此本脚本把同一病例的切片按文件名排序后直接堆成 (z, y, x) 体数据写 nii.gz，
  不做任何转置/翻转，保证 nnU-Net 与 LightUNet 看到逐像素一致的数据。
- 病人级防泄漏：训练/验证病例列表直接调用 dataset.split_train_val / labeled_subset
  （seed=42, val_ratio=0.15），与 train_supervised.py 完全一致：
    Dataset501_CARE100: train fold = 全部标注池病例（270），val = 项目 15% 验证病例（48）
    Dataset502_CARE50:  train fold = labeled_subset(pool, 0.5)（135），val 同上（48）
  同一病例的所有切片堆进同一个 nii.gz "病例"，nnU-Net 的划分天然按病例进行。
- 交叉验证：不用 nnU-Net 默认 5-fold CV。plan_and_preprocess 后用本脚本生成的
  out/splits/Dataset50X_*_splits.json 覆盖 nnUNet_preprocessed/.../splits_final.json
  （单 fold：train=标注病例, val=项目验证病例），训练只跑 fold 0。
  这样训练集与 LightUNet 基线逐病例一致，且 val 用于 best checkpoint 选择。
- 测试集写入 imagesTs（病例级 nii.gz），供 nnUNetv2_predict 推理；
  评估用 src/eval_nnunet.py（复用 evaluate.slice_metrics，指标定义不变）。
- Dataset502 的文件全部为 Dataset501 对应文件的硬链接（同内容只写一次，省 ~15GB）。

输出：
    {out}/raw/Dataset501_CARE100/{imagesTr,labelsTr,imagesTs,dataset.json}
    {out}/raw/Dataset502_CARE50/{imagesTr,labelsTr,imagesTs,dataset.json}
    {out}/splits/Dataset501_CARE100_splits.json / Dataset502_CARE50_splits.json
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (labeled_subset, list_slices, parse_case_id, remap_label,
                     split_train_val)

DATASET_100 = "Dataset501_CARE100"
DATASET_50 = "Dataset502_CARE50"
CHANNEL_NAMES = {"0": "CT"}
LABELS = {"background": 0, "rectum": 1, "tumor": 2}


def group_by_case(files):
    """npz 路径 -> {case_id: [排序后的路径]}（同病例切片按文件名排序，即堆叠 z 顺序）。"""
    groups = {}
    for p in files:
        groups.setdefault(parse_case_id(p), []).append(p)
    return {c: sorted(fs) for c, fs in groups.items()}


def _write_case(job):
    """worker：读一个病例的全部 npz，堆成 (z,y,x) 写 image/label nii.gz。

    返回 (case_id, n_slices, label_hist(3,)) 供主进程统计。
    """
    import SimpleITK as sitk
    case_id, files, img_path, lab_path = job
    imgs, labs = [], []
    for p in files:
        with np.load(p) as z:
            imgs.append(z["image"].astype(np.float32))
            labs.append(remap_label(z["label"]).astype(np.uint8))
    img_vol = np.stack(imgs)   # (z,y,x)，原样存储方向
    lab_vol = np.stack(labs)
    si = sitk.GetImageFromArray(img_vol)
    si.SetSpacing((1.0, 1.0, 1.0))  # CARE 无 spacing 元数据，假设 1mm/px（同 NSD 近似）
    sitk.WriteImage(si, img_path)
    hist = np.bincount(lab_vol.ravel(), minlength=3)
    if lab_path is not None:
        sl = sitk.GetImageFromArray(lab_vol)
        sl.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(sl, lab_path)
    return case_id, len(files), hist


def _write_dataset(out_raw, name, train_groups, val_groups, test_groups,
                   num_workers, link_from=None):
    """写一个 Dataset50X 目录。link_from 给出 {case_id: (img_src, lab_src)} 时
    用硬链接复用已写文件（Dataset502 复用 Dataset501）。"""
    ds_dir = os.path.join(out_raw, name)
    img_tr, lab_tr, img_ts = (os.path.join(ds_dir, d)
                              for d in ("imagesTr", "labelsTr", "imagesTs"))
    for d in (img_tr, lab_tr, img_ts):
        os.makedirs(d, exist_ok=True)

    hist = {}
    jobs, links = [], []
    for split_groups, kind in ((train_groups, "tr"), (val_groups, "tr"),
                               (test_groups, "ts")):
        link_map = (link_from or {}).get(kind, {})
        for case_id in sorted(split_groups):
            img_dst = os.path.join(img_tr if kind == "tr" else img_ts,
                                   f"{case_id}_0000.nii.gz")
            lab_dst = os.path.join(lab_tr, f"{case_id}.nii.gz") if kind == "tr" else None
            if case_id in link_map:
                links.append((link_map[case_id][0], img_dst))
                if lab_dst is not None:
                    links.append((link_map[case_id][1], lab_dst))
            else:
                jobs.append((case_id, split_groups[case_id], img_dst, lab_dst))

    for src, dst in links:
        if not os.path.exists(dst):
            os.link(src, dst)

    with Pool(num_workers) as pool:
        for i, (case_id, n_slices, h) in enumerate(
                pool.imap_unordered(_write_case, jobs)):
            hist[case_id] = (n_slices, h)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"  {name}: 写入 {i + 1}/{len(jobs)} 病例"
                      f"（另有 {len(links)} 个硬链接复用）")

    n_train = len(train_groups) + len(val_groups)
    dataset_json = {
        "channel_names": CHANNEL_NAMES,
        "labels": LABELS,
        "numTraining": n_train,
        "file_ending": ".nii.gz",
        "name": name,
        "description": ("CARE rectal cancer CT; 2D slices stacked per case (z,y,x) "
                        "as stored in source npz (no transpose); labels 3->2 merged; "
                        "patient-level single fold via custom splits_final.json"),
    }
    with open(os.path.join(ds_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    return hist


def convert(data_root, out, val_ratio, label_ratio_50, seed, num_workers):
    out_raw, out_splits = os.path.join(out, "raw"), os.path.join(out, "splits")
    os.makedirs(out_splits, exist_ok=True)

    # 与 train_supervised.py 完全相同的划分路径（同函数、同 seed）
    all_train = list_slices(data_root, "train")
    pool_files, val_files = split_train_val(all_train, val_ratio, seed)
    files_100 = labeled_subset(pool_files, 1.0, seed)
    files_50 = labeled_subset(pool_files, label_ratio_50, seed)
    test_files = list_slices(data_root, "test")

    g_pool, g_val = group_by_case(pool_files), group_by_case(val_files)
    g_100, g_50 = group_by_case(files_100), group_by_case(files_50)
    g_test = group_by_case(test_files)
    assert set(g_50) <= set(g_100) == set(g_pool), "50% 子集应是标注池子集"
    assert not (set(g_pool) & set(g_val)), "train/val 病例重叠（泄漏！）"
    print(f"标注池 {len(g_pool)} 例 / 验证 {len(g_val)} 例 / 测试 {len(g_test)} 例；"
          f"100% -> {len(g_100)} 例，50% -> {len(g_50)} 例")

    # Dataset501：全量写出（train fold=270 + val=48 + test=81）
    print(f"[1/2] {DATASET_100}")
    hist = _write_dataset(out_raw, DATASET_100, g_100, g_val, g_test, num_workers)

    # 记录 501 已写文件路径，供 502 硬链接（tr/ts 分开 keyed：假数据 train/test
    # 可能复用病例 id，真实 CARE 不相交，但分开存更稳妥）
    link_tr, link_ts = {}, {}
    ds501 = os.path.join(out_raw, DATASET_100)
    for case_id in set(g_100) | set(g_val):
        link_tr[case_id] = (os.path.join(ds501, "imagesTr", f"{case_id}_0000.nii.gz"),
                            os.path.join(ds501, "labelsTr", f"{case_id}.nii.gz"))
    for case_id in g_test:
        link_ts[case_id] = (os.path.join(ds501, "imagesTs", f"{case_id}_0000.nii.gz"),
                            None)
    overlap = (set(g_100) | set(g_val)) & set(g_test)
    if overlap:
        print(f"⚠️ train/test 病例 id 重叠 {len(overlap)} 个（假数据属预期）")

    # Dataset502：50% 标注病例是 501 的子集，全部硬链接
    print(f"[2/2] {DATASET_50}（全部硬链接自 {DATASET_100}）")
    _write_dataset(out_raw, DATASET_50, g_50, g_val, g_test, num_workers,
                   link_from={"tr": link_tr, "ts": link_ts})

    # 单 fold 划分文件（plan_and_preprocess 后覆盖 splits_final.json）
    for name, g_tr in ((DATASET_100, g_100), (DATASET_50, g_50)):
        splits = [{"train": sorted(g_tr), "val": sorted(g_val)}]
        path = os.path.join(out_splits, f"{name}_splits.json")
        with open(path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"单 fold 划分: {path} (train={len(g_tr)}, val={len(g_val)})")
    return hist


def validate(out, data_root):
    """回读校验：随机各取一个训练/测试病例，与源 npz 逐像素比对 + 打印统计。"""
    import SimpleITK as sitk
    out_raw = os.path.join(out, "raw")

    def check_case(img_path, lab_path, files):
        img_back = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        imgs = np.stack([np.load(p)["image"].astype(np.float32) for p in files])
        assert img_back.shape == imgs.shape, f"形状不符 {img_back.shape} vs {imgs.shape}"
        assert np.array_equal(img_back, imgs), f"图像像素不一致: {img_path}"
        if lab_path is not None:
            lab_back = sitk.GetArrayFromImage(sitk.ReadImage(lab_path))
            labs = np.stack([remap_label(np.load(p)["label"]).astype(np.uint8)
                             for p in files])
            assert np.array_equal(lab_back, labs), f"标签像素不一致: {lab_path}"
        return imgs.shape

    for name in (DATASET_100, DATASET_50):
        ds = os.path.join(out_raw, name)
        with open(os.path.join(ds, "dataset.json")) as f:
            dj = json.load(f)
        img_tr = sorted(os.listdir(os.path.join(ds, "imagesTr")))
        lab_tr = sorted(os.listdir(os.path.join(ds, "labelsTr")))
        img_ts = sorted(os.listdir(os.path.join(ds, "imagesTs")))
        assert len(img_tr) == dj["numTraining"] == len(lab_tr), \
            f"{name}: imagesTr/labelsTr/numTraining 不一致"
        # 训练病例回读校验（第一个病例）
        case_id = img_tr[0][: -len("_0000.nii.gz")]
        split = "train"
        files = group_by_case(list_slices(data_root, split))[case_id]
        shape = check_case(os.path.join(ds, "imagesTr", img_tr[0]),
                           os.path.join(ds, "labelsTr", f"{case_id}.nii.gz"), files)
        print(f"{name}: 回读校验 OK {case_id} 形状{shape} 像素与源 npz 完全一致")
        # 标签直方图（从 nii 回读累计）
        hist = np.zeros(3, dtype=np.int64)
        n_slices = 0
        for lf in lab_tr:
            v = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ds, "labelsTr", lf)))
            hist += np.bincount(v.ravel(), minlength=3)
            n_slices += v.shape[0]
        frac = hist / hist.sum()
        print(f"{name}: 训练 {dj['numTraining']} 例 / {n_slices} 切片, 测试 {len(img_ts)} 例; "
              f"标签直方图 bg/rectum/tumor = {hist.tolist()} "
              f"({frac[0]:.3%}/{frac[1]:.3%}/{frac[2]:.3%})")
        # 测试病例回读校验（第一个）
        ts_id = img_ts[0][: -len("_0000.nii.gz")]
        ts_files = group_by_case(list_slices(data_root, "test"))[ts_id]
        shape = check_case(os.path.join(ds, "imagesTs", img_ts[0]), None, ts_files)
        print(f"{name}: imagesTs 回读校验 OK {ts_id} 形状{shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True, help="nnunet 输出根目录（含 raw/splits）")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--label_ratio_50", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--only_validate", action="store_true",
                    help="跳过转换，只回读校验已有输出")
    args = ap.parse_args()

    if not args.only_validate:
        convert(args.data_root, args.out, args.val_ratio, args.label_ratio_50,
                args.seed, args.num_workers)
    print("回读校验：")
    validate(args.out, args.data_root)
    print("PREPARE NNUNET CARE OK")


if __name__ == "__main__":
    main()
