"""fig_mechanism_v2：同一 teacher 输出上的逐级过滤差异图（审稿意见 B）。

旧 fig_mechanism 被指出"FP 在 stage-2 输出中已消失，未展示一致性过滤相比
stage-2 精修额外移除了什么"。本图在同一 teacher（级联）输出上对比：
  col1  CT + GT 轮廓（绿=直肠，红=肿瘤）+ 256² 裁剪框（白色虚线）
  col2  teacher stage-2 置信度图（hot 叠加）+ 裁剪框
  col3  仅置信度过滤（conf>=0.9）后的伪标签；黄色 = 将被一致性过滤删除的像素
  col4  置信度+跨阶段一致性过滤后的伪标签（Ours 实际用于训练的监督信号）
  col5  一致性过滤丢弃的像素（ignore mask）按类型着色：
        红=背景上的假阳性，橙=前景类间混淆，蓝=被误删的真前景，
        紫=前景被标为背景，浅灰=被误删的正确背景

选片规则（caption 用）：在 runs_v2/ours_10_s42 第 1 轮 teacher（=sup_10_s42）
的未标注池 stride-16 确定性子采样中，保留 GT 含瘤 >=100px、stage-1 非空预测、
级联肿瘤 Dice>=0.5 的切片，按"一致性过滤删除的假阳性像素数"降序，每病人限
1 张，取前 --n_slices 张。GT 仅用于选片与背景轮廓展示，不参与训练。
输出：paper/fig_mechanism_v2.png + paper/manuscript/fig_mechanism_v2.png
（不覆盖旧图）；选片规则与图注草稿写 paper/fig_mechanism_v2_caption.txt。

用法：
  python src/make_fig_mechanism_v2.py --device cpu            # 正式图
  python src/make_fig_mechanism_v2.py --smoke                  # 冒烟（假数据）
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import dataset
from cascade import cascade_predict, crop_offsets, fg_centers
from common import load_model, resolve_device
from quantify_mechanism import unlabeled_pool
from tensorstore import TensorStore

PAPER_DIRS = ["paper", os.path.join("paper", "manuscript")]
# 被一致性过滤删除像素的类型配色
REM_COLORS = {"fp_bg": (1.00, 0.15, 0.15), "wrong_cls": (1.00, 0.60, 0.00),
              "true_fg": (0.20, 0.45, 1.00), "bg_on_fg": (0.70, 0.20, 1.00),
              "true_bg": (0.75, 0.75, 0.75)}
REM_LABEL = {"fp_bg": "FP on background", "wrong_cls": "wrong foreground class",
             "true_fg": "correct foreground (lost)",
             "bg_on_fg": "foreground mislabeled as background",
             "true_bg": "correct background"}


def disp(a):
    """项目显示约定：顺时针旋转 90°（见 visualize_predictions.disp）。"""
    return np.rot90(a, -1)


def overlay(image, mask):
    """与 src/visualize_predictions.py 一致：绿=直肠，红=肿瘤。"""
    base = np.stack([image] * 3, axis=-1).astype(np.float32)
    base /= max(base.max(), 1e-6)
    out = base.copy()
    out[mask == 1] = [0, 1, 0]
    out[mask == 2] = [1, 0, 0]
    return out


def crop_frame(shape, y0, x0, size):
    """裁剪窗边框 mask（仅边框像素为 1），用于显示坐标下画框。"""
    frame = np.zeros(shape, dtype=np.uint8)
    frame[y0:y0 + size, x0] = 1
    frame[y0:y0 + size, x0 + size - 1] = 1
    frame[y0, x0:x0 + size] = 1
    frame[y0 + size - 1, x0:x0 + size] = 1
    return frame


@torch.no_grad()
def scan_candidates(model1, model2, store, crop_size, conf_threshold,
                    batch_size, args):
    """扫描子采样池，返回按分数降序的候选切片记录列表。"""
    cands = []
    n = len(store)
    for off in range(0, n, batch_size):
        end = min(off + batch_size, n)
        pred, conf, pred1 = cascade_predict(model1, model2,
                                            store.images[off:end], crop_size,
                                            end - off, return_stage1=True)
        gt = store.labels[off:end]
        keep_conf = conf >= conf_threshold
        keep_full = keep_conf & (pred1 == pred)
        rem = keep_conf & ~keep_full
        for j in range(end - off):
            p1, pc, g = pred1[j], pred[j], gt[j]
            if not bool((p1 > 0).any()):      # fallback 切片无单一裁剪框
                continue
            gtm = g == 2
            if int(gtm.sum()) < args.min_tumor_px:
                continue
            inter = int(((pc == 2) & gtm).sum())
            dice = 2 * inter / max(int((pc == 2).sum()) + int(gtm.sum()), 1)
            if dice < args.min_dice:
                continue
            r = rem[j]
            fp_bg = int((r & (pc > 0) & (g == 0)).sum())
            wrong = int((r & (pc > 0) & (g > 0) & (pc != g)).sum())
            if fp_bg + wrong < args.min_fp_px:
                continue
            name = store.names[off + j]
            cands.append({"idx": off + j, "name": name,
                          "case": dataset.parse_case_id(name),
                          "score": fp_bg + wrong, "fp_bg": fp_bg,
                          "wrong_cls": wrong, "removed": int(r.sum()),
                          "dice": dice})
    cands.sort(key=lambda c: -c["score"])
    picked, seen_cases = [], set()
    for c in cands:
        if c["case"] in seen_cases:
            continue
        seen_cases.add(c["case"])
        picked.append(c)
        if len(picked) >= args.n_slices:
            break
    return picked


@torch.no_grad()
def render_slice(model1, model2, store, idx, crop_size, conf_threshold):
    """对选中切片重算全部中间产物（CPU 少量切片可接受）。"""
    img = store.images[idx:idx + 1]
    pred, conf, pred1 = cascade_predict(model1, model2, img, crop_size, 1,
                                        return_stage1=True)
    offs = crop_offsets(fg_centers(pred1), crop_size,
                        img.shape[2], img.shape[3])
    return {"image": img[0, 0].cpu().numpy(),
            "gt": store.labels[idx].cpu().numpy(),
            "pred1": pred1[0].cpu().numpy(),
            "pred": pred[0].cpu().numpy(),
            "conf": conf[0].cpu().numpy(),
            "offset": (int(offs[0, 0]), int(offs[0, 1])),
            "conf_threshold": conf_threshold}


def make_figure(datas, names, crop_size, out_paths, suptitle):
    """多病例为 n×5；单病例主文图为 3+2 大面板，第六格放图例。"""
    n = len(datas)
    if n == 1:
        fig, grid = plt.subplots(2, 3, figsize=(12.6, 7.2))
        axes = np.array([[grid[0, 0], grid[0, 1], grid[0, 2],
                          grid[1, 0], grid[1, 1]]])
        legend_ax = grid[1, 2]
        title_fs, label_fs, sup_fs, legend_fs = 15, 14, 17, 13
    else:
        fig, axes = plt.subplots(n, 5, figsize=(14.8, 3.05 * n))
        legend_ax = None
        title_fs, label_fs, sup_fs, legend_fs = 9, 7, 10, 7
    titles = ["CT + GT contours", "Teacher confidence (stage-2)",
              "Confidence filter only",
              "Confidence + agreement",
              "Pixels discarded by agreement\n(ignore mask)"]
    for i, data in enumerate(datas):
        image, gt = data["image"], data["gt"]
        pred, conf, pred1 = data["pred"], data["conf"], data["pred1"]
        thr = data["conf_threshold"]
        y0, x0 = data["offset"]
        frame = disp(crop_frame(image.shape, y0, x0, crop_size))
        keep_conf = conf >= thr
        keep_full = keep_conf & (pred1 == pred)
        rem = (keep_conf & ~keep_full)

        # col1: CT + GT 轮廓 + 裁剪框
        ax = axes[i][0]
        ax.imshow(disp(image), cmap="gray", interpolation="nearest")
        for c, color in ((1, "lime"), (2, "red")):
            m = disp((gt == c).astype(np.uint8))
            if m.any():
                ax.contour(m, levels=[0.5], colors=color, linewidths=0.9)
        ax.contour(frame, levels=[0.5], colors="white", linewidths=1.0,
                   linestyles="dashed")
        ax.text(-0.06, 0.5, names[i], transform=ax.transAxes, fontsize=label_fs,
                rotation=90, ha="center", va="center")

        # col2: 置信度图
        ax = axes[i][1]
        ax.imshow(disp(image), cmap="gray", interpolation="nearest")
        p = disp(conf)
        rgba = plt.get_cmap("hot")(np.clip(p, 0, 1))
        rgba[..., 3] = np.where(p > 0.02, 0.75 * np.clip(p, 0, 1), 0)
        ax.imshow(rgba, interpolation="nearest")
        ax.contour(frame, levels=[0.5], colors="white", linewidths=1.0,
                   linestyles="dashed")

        # col3: 仅置信度过滤（黄=将被一致性删除）
        ax = axes[i][2]
        pl_conf = np.where(keep_conf, pred, 0)
        vis = overlay(image, pl_conf)
        vis[rem] = [1.0, 1.0, 0.0]
        ax.imshow(disp(vis), interpolation="nearest")
        ax.contour(frame, levels=[0.5], colors="white", linewidths=1.0,
                   linestyles="dashed")

        # col4: 置信度+一致性
        ax = axes[i][3]
        pl_full = np.where(keep_full, pred, 0)
        ax.imshow(disp(overlay(image, pl_full)), interpolation="nearest")
        ax.contour(frame, levels=[0.5], colors="white", linewidths=1.0,
                   linestyles="dashed")

        # col5: ignore mask 分类着色
        ax = axes[i][4]
        base = np.stack([image] * 3, axis=-1).astype(np.float32)
        base /= max(base.max(), 1e-6)
        base = base * 0.45 + 0.55   # 淡化背景突出删除像素
        cat_counts = {}
        cat_masks = {
            "fp_bg": rem & (pred > 0) & (gt == 0),
            "wrong_cls": rem & (pred > 0) & (gt > 0) & (pred != gt),
            "true_fg": rem & (pred > 0) & (pred == gt),
            "bg_on_fg": rem & (pred == 0) & (gt > 0),
            "true_bg": rem & (pred == 0) & (gt == 0)}
        for k, m in cat_masks.items():
            cnt = int(m.sum())
            cat_counts[k] = cnt
            if cnt:
                base[m] = REM_COLORS[k]
        ax.imshow(disp(base), interpolation="nearest")
        ax.contour(frame, levels=[0.5], colors="white", linewidths=1.0,
                   linestyles="dashed")
        fp_share = ((cat_counts["fp_bg"] + cat_counts["wrong_cls"])
                    / max(int(rem.sum()), 1))
        ax.text(0.5, -0.035,
                f"{int(rem.sum())} px removed ({fp_share * 100:.0f}% false-positive)",
                transform=ax.transAxes, fontsize=label_fs,
                ha="center", va="top")
    for j, t in enumerate(titles):
        axes[0][j].set_title(t, fontsize=title_fs)
    for ax in axes.ravel():
        ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="s", color="none",
                          markerfacecolor=REM_COLORS[k], markersize=8,
                          label=REM_LABEL[k]) for k in REM_COLORS]
    handles += [plt.Line2D([0], [0], marker="s", color="none",
                           markerfacecolor=(1, 1, 0), markersize=8,
                           label="kept by confidence, removed by agreement")]
    if legend_ax is not None:
        legend_ax.axis("off")
        legend_ax.legend(handles=handles, loc="center", ncol=1,
                         fontsize=legend_fs, title="Mask legend",
                         title_fontsize=legend_fs, frameon=False)
        layout_rect = [0, 0.02, 1, 0.95]
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   fontsize=legend_fs, frameon=False)
        layout_rect = [0, 0.06, 1, 0.96]
    fig.suptitle(suptitle, fontsize=sup_fs)
    fig.tight_layout(rect=layout_rect)
    for out in out_paths:
        fig.savefig(out, dpi=300)
        print(f"已保存 {out}")
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=os.path.join("runs_v2", "ours_10_s42"),
                    help="Ours run（读 run_meta.json；teacher 取第 --round 轮）")
    ap.add_argument("--round", type=int, default=1,
                    help="用第 r 轮 teacher（1=teacher_init 的 sup 模型）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--scan_stride", type=int, default=16,
                    help="选片扫描子采样步长（在 stride-4 池基础上再抽）")
    ap.add_argument("--n_slices", type=int, default=3)
    ap.add_argument("--min_tumor_px", type=int, default=100)
    ap.add_argument("--min_fp_px", type=int, default=150,
                    help="一致性删除的假阳性像素数下限（选片）")
    ap.add_argument("--min_dice", type=float, default=0.5,
                    help="级联肿瘤 Dice 下限（选片）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：假数据 runs/smoke/semi_10_s2，输出 outputs/")
    ap.add_argument("--out", default=None,
                    help="覆盖输出路径（调试用的临时出图；缺省写 paper/ 两处）")
    ap.add_argument("--caption_out", default=None,
                    help="覆盖选片规则/caption 草稿输出路径")
    args = ap.parse_args()

    if args.smoke:
        args.run_dir = os.path.join("runs", "smoke", "semi_10_s2")
        args.round = 1
        args.scan_stride = 1
        args.n_slices = 1
        args.min_tumor_px = 1
        args.min_fp_px = 1
        args.min_dice = 0.0

    device = resolve_device(args.device)
    dataset.set_num_classes(3)
    with open(os.path.join(args.run_dir, "run_meta.json")) as f:
        meta = json.load(f)
    teacher = (meta["teacher_init"] if args.round == 1 else
               os.path.join(args.run_dir, f"teacher_r{args.round - 1}.pt"))
    crop_size = meta.get("crop_size", 256)
    thr = meta.get("conf_threshold", 0.9)
    pool = unlabeled_pool(meta, 4)[::args.scan_stride]
    print(f"选片扫描池 {len(pool)} 片（teacher={teacher}, crop={crop_size}, "
          f"thr={thr}）")
    store = TensorStore.from_files(pool, device, meta.get("cache_dir"))
    model1 = load_model(meta["stage1_ckpt"], device)
    model2 = load_model(teacher, device)

    picked = scan_candidates(model1, model2, store, crop_size, thr,
                             args.batch_size, args)
    if not picked and args.smoke:
        # 假数据欠拟合时 stage-1 常全空预测，硬过滤必然零候选；
        # 冒烟只验证管线跑通，放宽为取 GT 前景面积最大的切片
        print("⚠️ 冒烟模式无候选切片，放宽过滤：取 GT 前景最大切片")
        areas = [int((store.labels[i] > 0).sum()) for i in range(len(store))]
        idx = int(np.argmax(areas))
        name = store.names[idx]
        picked = [{"idx": idx, "name": name,
                   "case": dataset.parse_case_id(name), "score": 0,
                   "fp_bg": 0, "wrong_cls": 0, "removed": 0, "dice": 0.0}]
    if not picked:
        raise SystemExit("未找到满足条件的切片（可放宽 --min_fp_px 等阈值）")
    for c in picked:
        print(f"  选中 {c['name']}: 一致性删除 {c['removed']}px "
              f"(FP {c['fp_bg']}+{c['wrong_cls']} wrong-cls), "
              f"级联肿瘤 Dice={c['dice']:.3f}")
    datas = [render_slice(model1, model2, store, c["idx"], crop_size, thr)
             for c in picked]
    names = [c["name"].replace(".npz", "") for c in picked]

    if args.smoke:
        out_paths = [os.path.join("outputs", "fig_mechanism_v2_smoke.png")]
    elif args.out:
        out_paths = [args.out]
    else:
        out_paths = [os.path.join(d, "fig_mechanism_v2.png")
                     for d in PAPER_DIRS]
    ratio_pct = round(meta["label_ratio"] * 100)
    suptitle = (f"Cross-stage agreement filtering on the same teacher output "
                f"(round-{args.round} teacher, {ratio_pct}% labels, "
                f"seed {meta.get('seed', 42)})")
    make_figure(datas, names, crop_size, out_paths, suptitle)

    # 选片规则与图注草稿（caption 注释用；tex 正文不在本任务内修改）
    sep = "=" * 60
    note = (
        "fig_mechanism_v2 选片规则与图注草稿（生成时间见文件 mtime）\n"
        + sep + "\n"
        f"teacher: {teacher}（round {args.round}），stage-1: "
        f"{meta['stage1_ckpt']}\n"
        f"run: {args.run_dir}（label_ratio={meta['label_ratio']}, "
        f"seed={meta.get('seed', 42)}, crop_size={crop_size}, "
        f"conf_threshold={thr}, consistency={meta.get('consistency')}）\n"
        f"选片：未标注池 stride-4 子采样（同 quantify_mechanism）再 "
        f"stride-{args.scan_stride} 扫描；保留 GT 肿瘤 >= "
        f"{args.min_tumor_px}px、stage-1 非空预测、级联肿瘤 Dice >= "
        f"{args.min_dice} 的切片，按一致性过滤删除的假阳性像素数降序，"
        f"每病人限 1 张，取前 {args.n_slices} 张。GT 仅用于选片与背景"
        f"轮廓展示，不参与训练。\n"
        f"选中切片：{[c['name'] for c in picked]}\n"
        "\nDraft caption:\n"
        "Step-wise filtering of pseudo-labels on identical teacher outputs "
        "(CARE unlabeled training slices). From left to right: CT with "
        "ground-truth contours (green = rectum, red = tumour) and the "
        "stage-1-guided 256x256 crop window (dashed box); the stage-2 "
        "teacher confidence map; pseudo-labels kept by the confidence "
        "filter alone (conf >= 0.9), with pixels later discarded by "
        "cross-stage agreement highlighted in yellow; pseudo-labels after "
        "confidence + agreement filtering (the supervision signal used by "
        "Ours); and the discarded pixels (ignore mask) colour-coded by "
        "type (red = false positive on background, orange = wrong "
        "foreground class, blue = correct foreground lost, purple = "
        "foreground mislabelled as background, grey = correct background). "
        "Slices were selected from the stride-subsampled unlabeled pool as "
        "the top slices by the number of false-positive pixels removed by "
        "the agreement filter (one per patient, cascade tumour Dice >= "
        "0.5); ground truth was used only for slice selection and contour "
        "display, never for training.\n")
    if not args.smoke:
        cap_path = args.caption_out or os.path.join(
            "paper", "fig_mechanism_v2_caption.txt")
        with open(cap_path, "w") as f:
            f.write(note)
        print(f"已保存 {cap_path}")


if __name__ == "__main__":
    main()
