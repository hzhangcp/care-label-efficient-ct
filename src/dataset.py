"""CARE 数据集加载与病人级划分（防泄漏命门）。

数据格式（P0 已核实，2026-09-15）：
    {split}/{split}_bbox.csv   第1列=npz文件名, 第2列=bbox
    {split}/{split}_npz/*.npz  键 image(512x512 float64, [0,1]) + label({0,1,2,3})
    文件名形如 `case{id}_slice{nnn}.npz`（parse_case_id 假设已验证）
    标签 3 并入 2（肿瘤），依据 U-SAM 官方代码 rectum_dataloader.py: mask[mask>2]=2
"""

import os
import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 3          # 0=背景, 1=正常直肠, 2=肿瘤（原始 3 已并入 2）
FG_CLASSES = (1, 2)      # 参与指标计算的前景类
IGNORE_INDEX = 255       # 半监督伪标签中被过滤的低置信像素


def set_num_classes(n):
    """切换类别数（默认 3=CARE；MSD Colon 二分类用 2，前景仅肿瘤类 1）。

    会同步更新 NUM_CLASSES / FG_CLASSES；引用方须通过 dataset 模块动态访问
    （common/evaluate 均如此），在训练/评估开始前调用一次即可。
    """
    global NUM_CLASSES, FG_CLASSES
    NUM_CLASSES = n
    FG_CLASSES = tuple(range(1, n))


def fg_class_names():
    """前景类名：CARE 3 类 -> {1:rectum, 2:tumor}（保持不变）；
    二分类（MSD Colon）-> {1:tumor}。日志/评估列名统一从这里取。"""
    if FG_CLASSES == (1, 2):
        return {1: "rectum", 2: "tumor"}
    return {c: ("tumor" if c == 1 else f"class{c}") for c in FG_CLASSES}


def parse_case_id(fname):
    """从 npz 文件名解析病例 id：`case{id}_slice{nnn}.npz` -> `case{id}`。"""
    base = os.path.basename(fname)
    if base.endswith(".npz"):
        base = base[:-4]
    if "_" in base:
        return base.rsplit("_", 1)[0]
    return base  # 无下划线时整个文件名视为病例 id（每例一切片，仍安全）


def list_slices(data_root, split):
    """列出某个 split ('train'/'test') 下全部 npz 路径，按文件名排序。"""
    npz_dir = os.path.join(data_root, split, f"{split}_npz")
    files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]
    return sorted(files)


def read_bbox_csv(data_root, split):
    """读取 bbox csv -> {npz文件名: bbox字符串}。csv 缺失时返回空 dict。"""
    path = os.path.join(data_root, split, f"{split}_bbox.csv")
    table = {}
    if not os.path.exists(path):
        return table
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                table[row[0]] = row[1]
    return table


def _group_by_case(files):
    groups = {}
    for p in files:
        groups.setdefault(parse_case_id(p), []).append(p)
    return groups


def split_train_val(files, val_ratio=0.15, seed=42):
    """病人级 train/val 划分。val 从全部训练病例中抽取（与标注比例无关）。"""
    groups = _group_by_case(files)
    cases = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_val = max(1, int(round(len(cases) * val_ratio))) if len(cases) > 1 else 0
    val_cases = set(cases[:n_val])
    train_files = [p for c in cases[n_val:] for p in groups[c]]
    val_files = [p for c in sorted(val_cases) for p in groups[c]]
    return train_files, val_files


def labeled_subset(files, label_ratio, seed=42):
    """病人级标注子集：按病例抽取 label_ratio 比例的病例。

    同一 (seed, ratio) 永远得到同一子集；小比例子集是大比例子集的近似前缀
    （病例顺序由同一 seed 的 shuffle 决定），保证标注效率曲线单调可比。
    """
    if not (0 < label_ratio <= 1.0):
        raise ValueError(f"label_ratio 必须在 (0, 1]，收到 {label_ratio}")
    groups = _group_by_case(files)
    cases = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_keep = len(cases) if label_ratio >= 1.0 else max(1, int(round(len(cases) * label_ratio)))
    kept = set(cases[:n_keep])
    return [p for c in sorted(kept) for p in groups[c]]


def split_labeled_val(labeled_files, val_ratio=0.15, seed=42):
    """val_mode=labeled：仅从标注病例中划验证集（标注预算与训练一致）。

    病例顺序由同一 seed 的 shuffle 决定（与 labeled_subset 同款），取后缀
    max(1, round(val_ratio)) 例为 val，前缀为训练文件（训练文件不含 val 病例）。
    仅当标注病例数 >= 4 时调用（调用方负责更小时的 fixed 回退）。
    """
    groups = _group_by_case(labeled_files)
    cases = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_val = max(1, int(round(len(cases) * val_ratio)))
    n_val = min(n_val, len(cases) - 1)  # 至少留 1 例训练
    val_cases = set(cases[-n_val:])
    train_files = [p for c in cases[:-n_val] for p in groups[c]]
    val_files = [p for c in sorted(val_cases) for p in groups[c]]
    return train_files, val_files


def oversample_positive_files(files, k):
    """训练集肿瘤（前景）切片过采样：含前景像素的切片在列表中复制 k 份。

    仅用于构建训练文件列表（val/test 不经过此函数）；k<=1 原样返回。
    按 npz label 是否含前景像素（0 < label != 255）判定。
    """
    files = list(files)
    if k <= 1:
        return files
    pos, neg = [], []
    for p in files:
        with np.load(p) as z:
            lab = z["label"]
        if ((lab > 0) & (lab != IGNORE_INDEX)).any():
            pos.append(p)
        else:
            neg.append(p)
    return neg + pos * k


def remap_label(label):
    """原始标签值域 {0,1,2,3}，其中 3 须并入 2（肿瘤）。

    依据 U-SAM 官方代码 dataset/rectum_dataloader.py: `mask[mask > 2] = 2`。
    """
    label = label.astype(np.int64)
    label[label > 2] = 2
    return label


class CAREDataset(Dataset):
    """CARE npz 切片数据集。

    files:      npz 路径列表
    augment:    训练期随机翻转增广
    pseudo_dir: 可选，半监督伪标签 .npz 目录（键 'pseudo'，取值 {0,1,2,255=ignore}），
                设置后忽略原 label，返回伪标签；可用 set_pseudo_dir 更换（自训练每轮）
    preload:    True 时初始化即把全部数据读入内存（image float32 / label uint8）。
                真实 CARE 为 512x512 float64 npz，逐切片读盘每 epoch 需扫 ~110GB，
                预载后训练快一个量级；26k 切片约占 33GB 内存。
    """

    def __init__(self, files, augment=False, pseudo_dir=None, preload=False):
        self.files = list(files)
        self.augment = augment
        self.preload = preload
        self.pseudo_dir = None
        self._pseudo_cache = None
        self._cache = None
        if preload:
            self._cache = []
            for i, path in enumerate(self.files):
                self._cache.append(self._load_image(path))
                if (i + 1) % 5000 == 0:
                    print(f"  preload {i + 1}/{len(self.files)}")
        if pseudo_dir is not None:
            self.set_pseudo_dir(pseudo_dir)

    def _load_image(self, path):
        with np.load(path) as z:
            image = z["image"].astype(np.float32)
            label = remap_label(z["label"]).astype(np.uint8)
        return image, label

    def _load_pseudo(self, path):
        pseudo_path = os.path.join(self.pseudo_dir, os.path.basename(path))
        with np.load(pseudo_path) as z:
            return z["pseudo"].astype(np.uint8)

    def set_pseudo_dir(self, pseudo_dir):
        """切换伪标签目录；preload 模式下重读伪标签（自训练每轮调用）。"""
        self.pseudo_dir = pseudo_dir
        if self.preload:
            self._pseudo_cache = [self._load_pseudo(p) for p in self.files]

    def __len__(self):
        return len(self.files)

    def _augment(self, image, label, rng):
        if rng.random() < 0.5:
            image = image[:, ::-1]
            label = label[:, ::-1]
        if rng.random() < 0.5:
            image = image[::-1, :]
            label = label[::-1, :]
        k = rng.randrange(4)
        if k:
            image = np.rot90(image, k)
            label = np.rot90(label, k)
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def __getitem__(self, idx):
        if self._cache is not None:
            image, label = self._cache[idx]
        else:
            image, label = self._load_image(self.files[idx])
        if self.pseudo_dir is not None:
            if self._pseudo_cache is not None:
                label = self._pseudo_cache[idx]
            else:
                label = self._load_pseudo(self.files[idx])
        if self.augment:
            seed = (idx * 2654435761 + os.getpid()) % (2**32)
            image, label = self._augment(image, label, random.Random(seed))
        # 缓存已是 float32/uint8，仅必要时转换，避免每切片 ~3MB 冗余拷贝
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if label.dtype != np.int64:
            label = label.astype(np.int64)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),  # (1,H,W)
            "label": torch.from_numpy(label),               # (H,W)
            "name": os.path.basename(self.files[idx]),
        }
