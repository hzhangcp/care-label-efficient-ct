"""设备驻留张量数据管线：消除 CPU DataLoader 瓶颈。

真实 CARE（512x512 npz）下，CPU DataLoader（augment + 拷贝 + collate + pinned 传输）
使每 epoch 约 6 分钟，GPU 大量空转；把全部切片一次性转为 device 上的大张量后，
batch 只是 GPU 上的 index_select，epoch 时间降到接近纯计算。

配合 src/preprocess_cache.py 生成的 .npy 缓存（float32 images + uint8 labels），
可避免每次运行重复解压 26k 个 npz（110GB -> 30GB 顺序读）。
"""

import json
import os

import numpy as np
import torch

from dataset import remap_label


def cache_paths(cache_dir, split):
    return (os.path.join(cache_dir, f"{split}_images.npy"),
            os.path.join(cache_dir, f"{split}_labels.npy"),
            os.path.join(cache_dir, f"{split}_names.json"))


def load_split_cache(cache_dir, split):
    img_p, lab_p, names_p = cache_paths(cache_dir, split)
    images = np.load(img_p, mmap_mode="r")
    labels = np.load(lab_p, mmap_mode="r")
    with open(names_p) as f:
        names = json.load(f)
    return images, labels, names


def _sample_geom(n, dev):
    """几何增广参数：batch 级 hflip/vflip（p=0.5）+ 逐样本仿射 theta
    （旋转±15°、缩放0.85-1.15、平移±10%）。采样顺序与旧版 augment_batch 一致。"""
    do_hflip = bool(torch.rand(()) < 0.5)
    do_vflip = bool(torch.rand(()) < 0.5)
    ang = (torch.rand(n, device=dev) * 2 - 1) * (15 * 3.14159265 / 180)
    scale = 0.85 + 0.3 * torch.rand(n, device=dev)
    trans = (torch.rand(n, 2, device=dev) * 2 - 1) * 0.1
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin], 1),
                         torch.stack([sin, cos], 1)], 1)  # (N,2,2)
    theta = torch.cat([theta, trans.unsqueeze(2)], dim=2)  # (N,2,3)
    return do_hflip, do_vflip, theta


def _apply_geom(img, lab, geom):
    """按给定几何参数变换图像（双线性）与标签（nearest，边界填 0=背景；
    标签含 255(IGNORE) 时 nearest 保持原值）。"""
    import torch.nn.functional as F
    do_hflip, do_vflip, theta = geom
    if do_hflip:
        img = torch.flip(img, [-1])
        if lab is not None:
            lab = torch.flip(lab, [-1])
    if do_vflip:
        img = torch.flip(img, [-2])
        if lab is not None:
            lab = torch.flip(lab, [-2])
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    if lab is not None:
        lab = F.grid_sample(lab.unsqueeze(1).float(), grid, mode="nearest",
                            padding_mode="zeros", align_corners=False).squeeze(1).long()
    return img, lab


def _apply_intensity(img):
    """逐样本亮度/对比度抖动（不改变几何坐标）。"""
    n, dev = img.shape[0], img.device
    gain = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, device=dev)
    bias = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * 0.1
    return (img * gain + bias).clamp(0, 1)


def augment_batch(img, lab=None):
    """batch 增广：hflip/vflip（p=0.5）+ 逐样本仿射（旋转±15°、缩放0.85-1.15、
    平移±10%）+ 逐样本亮度/对比度抖动。仿射用 grid_sample 逐样本进行。

    lab=None 时只增广图像，返回 (img, None)。
    """
    geom = _sample_geom(img.shape[0], img.device)
    img, lab = _apply_geom(img, lab, geom)
    img = _apply_intensity(img)
    return img, lab


def augment_pair(img):
    """Mean Teacher 一致性双视图（2026-10-13 坐标对齐修复）：
    student 与 teacher 视图共享同一组几何变换参数（翻转+逐样本仿射），
    逐像素一致性 loss 因此在同一坐标系计算；student 额外加仅强度的
    亮度/对比度抖动。返回 (student_view, teacher_view)。
    """
    geom = _sample_geom(img.shape[0], img.device)
    s_view, _ = _apply_geom(img, None, geom)
    t_view, _ = _apply_geom(img, None, geom)
    s_view = _apply_intensity(s_view)
    return s_view, t_view


class TensorStore:
    """(N,1,H,W) float32 图像 + (N,H,W) uint8 标签，整体驻留 device。"""

    def __init__(self, images, labels, names, device):
        self.images = images
        self.labels = labels
        self.names = list(names)
        self.device = device

    def __len__(self):
        return len(self.names)

    @classmethod
    def from_files(cls, files, device, cache_dir=None, verbose=True):
        names = [os.path.basename(p) for p in files]
        images = labels = None
        if cache_dir and files:
            # 从路径推断 split（.../train_npz/x.npz -> train）
            split = os.path.basename(os.path.dirname(os.path.dirname(files[0])))
            img_p, lab_p, names_p = cache_paths(cache_dir, split)
            if os.path.exists(img_p) and os.path.exists(names_p):
                all_img, all_lab, all_names = load_split_cache(cache_dir, split)
                pos = {n: i for i, n in enumerate(all_names)}
                if all(n in pos for n in names):
                    sel = [pos[n] for n in names]
                    if verbose:
                        print(f"从缓存加载 {len(sel)} 切片 ({split}, {cache_dir})")
                    if len(sel) > len(all_names) // 3:
                        # 选中比例高时顺序读整个 split 再在内存中子集化：
                        # memmap 花式索引是随机读，HDD 寻道会让加载慢 3-4 倍
                        base_img = np.asarray(all_img)
                        base_lab = np.asarray(all_lab)
                        images = torch.from_numpy(base_img[sel])
                        labels = torch.from_numpy(base_lab[sel])
                    else:
                        images = torch.from_numpy(np.ascontiguousarray(all_img[sel]))
                        labels = torch.from_numpy(np.ascontiguousarray(all_lab[sel]))
                else:
                    print(f"⚠️ 缓存 {cache_dir} 与请求的文件不匹配，回退逐文件读 npz")
        if images is None:
            imgs, labs = [], []
            for i, p in enumerate(files):
                with np.load(p) as z:
                    imgs.append(z["image"].astype(np.float32))
                    labs.append(remap_label(z["label"]).astype(np.uint8))
                if verbose and (i + 1) % 5000 == 0:
                    print(f"  读取 npz {i + 1}/{len(files)}")
            images = torch.from_numpy(np.stack(imgs))
            labels = torch.from_numpy(np.stack(labs))
        images = images.unsqueeze(1).to(device)   # (N,1,H,W)
        labels = labels.to(device)                # (N,H,W) uint8
        return cls(images, labels, names, device)

    def loader(self, batch_size, shuffle, augment=False, drop_last=False):
        return TensorLoader(self, batch_size, shuffle, augment, drop_last)

    def crop_in_place(self, centers, size):
        """按 (N,2)(cy,cx) 中心裁剪为 size x size（级联 stage-2 用）。"""
        from cascade import crop_batch, crop_offsets
        h, w = self.images.shape[2], self.images.shape[3]
        offsets = crop_offsets(centers, size, h, w)
        new_img, new_lab = [], []
        bs = 1024
        for off in range(0, len(self), bs):
            end = min(off + bs, len(self))
            im, lb = crop_batch(self.images[off:end], offsets[off:end], size,
                                labels=self.labels[off:end])
            new_img.append(im)
            new_lab.append(lb)
        self.images = torch.cat(new_img)
        self.labels = torch.cat(new_lab)
        return self


class TensorLoader:
    """可重复迭代的 batch 生成器，接口对齐 common.train_model 的 dict batch。"""

    def __init__(self, store, batch_size, shuffle, augment=False, drop_last=False):
        self.store = store
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.store)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        store = self.store
        n = len(store)
        if self.shuffle:
            idx = torch.randperm(n, device=store.device)
        else:
            idx = torch.arange(n, device=store.device)
        if self.drop_last:
            idx = idx[: n - n % self.batch_size]
        for chunk in idx.split(self.batch_size):
            if len(chunk) == 0:
                break
            img = store.images.index_select(0, chunk)
            lab = store.labels.index_select(0, chunk).long()
            if self.augment:
                img, lab = augment_batch(img, lab)
            yield {"image": img, "label": lab,
                   "name": [store.names[i] for i in chunk.tolist()]}
