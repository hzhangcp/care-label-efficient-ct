"""两阶段级联：stage1 全图粗定位 -> 按预测前景质心裁剪 -> stage2 精分割 -> 贴回全画布。

设计要点：
- 裁剪不缩放（native 分辨率），预测贴回 512x512 全画布后计算指标，指标定义不变；
- stage-1 空预测的切片默认做重叠平铺兜底（tile_fallback）：3x3 网格、stride=size/2，
  概率平均融合后 argmax，覆盖全图，避免 stage-1 漏检导致整块肿瘤丢失；
- 训练 stage-2 时用 GT 前景质心，推理/验证时用 stage-1 预测质心（无 GT 泄漏）。
"""

import torch

from dataset import IGNORE_INDEX


def fg_centers(labels, crop_size=None, chunk=512):
    """逐切片前景（0<label<255）质心 -> (N,2) long，列为 (cy, cx)。

    空前景切片返回图像中心。labels: (N,H,W) uint8/long tensor（任意 device）。
    分块计算避免 (N,H,W) float 广播中间量撑爆显存（N=22k 时约需 50GB）。
    """
    n, h, w = labels.shape
    dev = labels.device
    ys = torch.arange(h, device=dev, dtype=torch.float32).view(1, h, 1)
    xs = torch.arange(w, device=dev, dtype=torch.float32).view(1, 1, w)
    outs = []
    for off in range(0, n, chunk):
        mask = (labels[off:off + chunk] > 0) & (labels[off:off + chunk] != IGNORE_INDEX)
        cnt = mask.sum(dim=(1, 2))
        cy = (mask * ys).sum(dim=(1, 2)) / cnt.clamp(min=1)
        cx = (mask * xs).sum(dim=(1, 2)) / cnt.clamp(min=1)
        cy = torch.where(cnt > 0, cy, torch.full_like(cy, h // 2))
        cx = torch.where(cnt > 0, cx, torch.full_like(cx, w // 2))
        outs.append(torch.stack([cy, cx], dim=1))
    return torch.cat(outs).round().long()


def crop_offsets(centers, size, h, w):
    """由中心点算左上角偏移 (N,2) (y0,x0)，裁剪窗 clamp 在图内。"""
    y0 = (centers[:, 0] - size // 2).clamp(0, h - size)
    x0 = (centers[:, 1] - size // 2).clamp(0, w - size)
    return torch.stack([y0, x0], dim=1)


def crop_batch(images, offsets, size, labels=None):
    """按 offsets 裁剪。images (N,1,H,W) -> (N,1,S,S)；labels (N,H,W) -> (N,S,S)。"""
    n = images.shape[0]
    img_out = torch.empty((n, images.shape[1], size, size),
                          dtype=images.dtype, device=images.device)
    lab_out = None
    if labels is not None:
        lab_out = torch.empty((n, size, size), dtype=labels.dtype,
                              device=labels.device)
    for i in range(n):
        y0, x0 = int(offsets[i, 0]), int(offsets[i, 1])
        img_out[i] = images[i, :, y0:y0 + size, x0:x0 + size]
        if labels is not None:
            lab_out[i] = labels[i, y0:y0 + size, x0:x0 + size]
    return img_out, lab_out


def paste_batch(crop_vals, offsets, size, h, w, fill=0):
    """把 (N,S,S) 的预测贴回 (N,H,W) 画布（其余填 fill）。"""
    n = crop_vals.shape[0]
    canvas = torch.full((n, h, w), fill, dtype=crop_vals.dtype,
                        device=crop_vals.device)
    for i in range(n):
        y0, x0 = int(offsets[i, 0]), int(offsets[i, 1])
        canvas[i, y0:y0 + size, x0:x0 + size] = crop_vals[i]
    return canvas


@torch.no_grad()
def batch_argmax(model, images, batch_size=32, amp=True):
    """对 (N,1,H,W) device 张量分批推理 argmax -> (N,H,W) uint8。"""
    model.eval()
    dev = images.device
    preds = torch.empty((images.shape[0], images.shape[2], images.shape[3]),
                        dtype=torch.uint8, device=dev)
    for off in range(0, images.shape[0], batch_size):
        x = images[off:off + batch_size]
        if amp:
            x = x.to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=str(dev).startswith("cuda")):
                p = model(x).argmax(1)
        else:
            p = model(x).argmax(1)
        preds[off:off + batch_size] = p.to(torch.uint8)
    return preds


def _tile_grid(h, w, size, stride=None):
    """重叠平铺的左上角偏移网格，覆盖 (h,w) 整图。"""
    stride = stride or max(1, size // 2)

    def axis(length):
        stop = max(length - size, 0)
        pts = list(range(0, stop + 1, stride))
        if not pts or pts[-1] != stop:
            pts.append(stop)
        return pts

    return [(y, x) for y in axis(h) for x in axis(w)]


@torch.no_grad()
def _tiled_predict(model2, images, crop_size, batch_size=32):
    """对 (N,1,H,W) 做重叠平铺推理，概率平均融合 -> (pred, conf) 全画布。

    外圈按 256 切片分块，限制概率累加器 (N,3,H,W) 的显存占用。
    """
    n, _, h, w = images.shape
    dev = images.device
    grid = _tile_grid(h, w, crop_size)
    pred = torch.zeros((n, h, w), dtype=torch.uint8, device=dev)
    conf = torch.zeros((n, h, w), dtype=torch.float32, device=dev)
    for coff in range(0, n, 256):
        chunk = images[coff:coff + 256]
        cn = chunk.shape[0]
        acc = cnt = None  # 类别数由首个前向的输出决定（CARE 3 类 / MSD 2 类）
        for y0, x0 in grid:
            for off in range(0, cn, batch_size):
                x = chunk[off:off + batch_size, :,
                          y0:y0 + crop_size, x0:x0 + crop_size]
                if str(dev).startswith("cuda"):
                    x = x.to(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=str(dev).startswith("cuda")):
                    probs = torch.softmax(model2(x).float(), dim=1)
                if acc is None:
                    acc = torch.zeros((cn, probs.shape[1], h, w),
                                      dtype=torch.float32, device=dev)
                    cnt = torch.zeros((cn, 1, h, w), dtype=torch.float32,
                                      device=dev)
                acc[off:off + batch_size, :,
                    y0:y0 + crop_size, x0:x0 + crop_size] += probs
                cnt[off:off + batch_size, :,
                    y0:y0 + crop_size, x0:x0 + crop_size] += 1
        probs = acc / cnt.clamp(min=1)
        c, p = probs.max(1)
        pred[coff:coff + cn] = p.to(torch.uint8)
        conf[coff:coff + cn] = c
    return pred, conf


@torch.no_grad()
def cascade_predict(model1, model2, images, crop_size, batch_size=32,
                    tile_fallback=True, return_stage1=False):
    """级联推理：返回全画布 argmax 预测 (N,H,W) uint8 及 stage2 置信度图。

    images: (N,1,H,W) device 张量。
    tile_fallback=True 时，stage-1 空预测的切片改用重叠平铺覆盖全图。
    return_stage1=True 时额外返回 stage-1 全图预测（跨阶段一致性用）。
    返回 (pred_canvas, conf_canvas)：conf 为 stage-2 softmax 最大概率，
    裁剪窗外 conf=0（半监督里窗外像素会被阈值滤为 IGNORE）。
    """
    n, _, h, w = images.shape
    dev = images.device
    pred1 = batch_argmax(model1, images, batch_size)
    empty = torch.zeros((n,), dtype=torch.bool, device=dev)
    for off in range(0, n, 512):
        empty[off:off + 512] = \
            ~(pred1[off:off + 512] > 0).flatten(1).any(dim=1)
    offsets = crop_offsets(fg_centers(pred1), crop_size, h, w)
    pred_canvas = torch.zeros((n, h, w), dtype=torch.uint8, device=dev)
    conf_canvas = torch.zeros((n, h, w), dtype=torch.float32, device=dev)
    for off in range(0, n, batch_size):
        crops, _ = crop_batch(images[off:off + batch_size],
                              offsets[off:off + batch_size], crop_size)
        if str(dev).startswith("cuda"):
            crops = crops.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(dev).startswith("cuda")):
            logits = model2(crops)
        probs = torch.softmax(logits.float(), dim=1)
        conf, pred = probs.max(1)
        pred_canvas[off:off + batch_size] = paste_batch(
            pred.to(torch.uint8), offsets[off:off + batch_size], crop_size, h, w)
        conf_canvas[off:off + batch_size] = paste_batch(
            conf, offsets[off:off + batch_size], crop_size, h, w)
    if tile_fallback and empty.any():
        idx = empty.nonzero(as_tuple=True)[0]
        tp, tc = _tiled_predict(model2, images[idx], crop_size, batch_size)
        pred_canvas[idx] = tp
        conf_canvas[idx] = tc
    if return_stage1:
        return pred_canvas, conf_canvas, pred1
    return pred_canvas, conf_canvas
