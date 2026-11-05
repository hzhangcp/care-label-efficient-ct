"""主文新增图表（fig5-fig8，Computers in Biology and Medicine）。

全部数字从 results/ 与 runs/ 现算，不手抄。产出同时落 paper/ 与
paper/manuscript/（同一份 PNG 复制两处，位图均 300 dpi）：
  fig_overview.png       方法流程示意图（两阶段级联 + 跨阶段一致性过滤自训练）
  fig_mechanism.png      跨阶段一致性过滤机制实例（真实切片：stage-1 假阳性被
                         stage-2 拒绝、被一致性过滤移除）
  fig_delta_heatmap.png  各半监督方法相对同比例全监督的 ΔDice 热图（tumor/rectum）
  fig_percase_box.png    CARE test 逐病例平均肿瘤 Dice 分布箱线图（10/25/50%）

用法：python src/make_figures_extra.py [--figures overview,mechanism,...]
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from make_figures import (RESULTS, PAPER, METHODS, METHOD_LABEL, METHOD_STYLE,
                          load_summary)

OUT_MANU = os.path.join(PAPER, "manuscript")
CACHE = "data/cache"

# 机制图专用模型与超参（与 runs/ours_* 配置一致）
S1_CKPT = "runs/s1_50/best.pt"
S2_CKPT = "runs/sup_50/best.pt"
CROP_SIZE = 256
CONF_THRESHOLD = 0.9
LABEL_RATIO = 0.5
SEED = 42


def save_both(fig, name):
    """保存到 paper/ 与 paper/manuscript/ 两处。"""
    out1 = os.path.join(PAPER, name)
    fig.savefig(out1, dpi=300)
    out2 = os.path.join(OUT_MANU, name)
    shutil.copyfile(out1, out2)
    plt.close(fig)
    print(f"已保存 {out1} 与 {out2}")


# ---------------------------------------------------------------- Figure A
def _box(ax, x, y, w, h, text, accent=False, fs=8):
    edge = "tab:red" if accent else "0.35"
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.004,rounding_size=0.012",
                         facecolor="#f2f2f2", edgecolor=edge,
                         lw=1.8 if accent else 1.1)
    ax.add_patch(box)
    # 文本宽度自适配：渲染测量最宽行，超出盒宽 94% 则逐档缩字号（下限 7pt）
    fig = ax.figure
    box_w_px = w * fig.bbox.width
    for f in range(fs, 6, -1):
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=f)
        fig.canvas.draw()
        tw = t.get_window_extent(fig.canvas.get_renderer()).width
        if tw <= box_w_px * 0.94 or f == 7:
            break
        t.remove()
    return box


def _arrow(ax, x0, x1, y, label=None, accent=False, label_dy=0.055):
    color = "tab:red" if accent else "0.25"
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.6 if accent else 1.3,
                                 color=color))
    if label:
        ax.text((x0 + x1) / 2, y + label_dy, label, ha="center", va="bottom",
                fontsize=9, color=color,
                style="italic" if accent else "normal")


@torch.no_grad()
def fig_overview(device="cuda"):
    """生成 Fig 1（TikZ 排版）所需的 5 张真实级联中间输出面板：
    输入 / stage-1 粗分割(含裁剪窗) / 256 裁剪块 / stage-2 精修 / 贴回输出。
    面板不含任何文字；标题、箭头、框图全部由 main.tex 的 tikzpicture 绘制。"""
    from matplotlib.patches import Rectangle
    from cascade import cascade_predict, crop_offsets, fg_centers
    from common import load_model
    from tensorstore import TensorStore

    # ---- 真实数据：一张 test 切片走级联（100% 模型，展示效果最好） ----
    name = "case17124951_slice100.npz"
    store = TensorStore.from_files([f"data/CARE/test/test_npz/{name}"],
                                   device, cache_dir=CACHE)
    x = store.images
    m1 = load_model("runs/s1_100/best.pt", device)
    m2 = load_model("runs/sup_100/best.pt", device)
    pred, _, pred1 = cascade_predict(m1, m2, x, CROP_SIZE, 1,
                                     return_stage1=True)
    # 先转到显示方向（旋转是刚体变换），再在显示坐标系重算裁剪框
    img = disp(x[0, 0].cpu().numpy())
    p1 = disp(pred1[0].cpu().numpy())
    pc = disp(pred[0].cpu().numpy())
    off_r = crop_offsets(
        fg_centers(torch.from_numpy(p1.copy()[None]).to(device)),
        CROP_SIZE, 512, 512)[0]
    ry0, rx0 = int(off_r[0]), int(off_r[1])
    crop_img = img[ry0:ry0 + CROP_SIZE, rx0:rx0 + CROP_SIZE]
    crop_pred = pc[ry0:ry0 + CROP_SIZE, rx0:rx0 + CROP_SIZE]

    panels = {
        "fig_ov_1_input.png": (img, "gray"),
        "fig_ov_2_stage1.png": (overlay(img, p1), None),
        "fig_ov_3_crop.png": (crop_img, "gray"),
        "fig_ov_4_stage2.png": (overlay(crop_img, crop_pred), None),
        "fig_ov_5_output.png": (overlay(img, pc), None),
    }
    for fname, (data, cmap) in panels.items():
        h, w = data.shape[:2]
        f, iax = plt.subplots(figsize=(w / 300, h / 300))
        iax.imshow(data, cmap=cmap)
        iax.axis("off")
        if fname == "fig_ov_2_stage1.png":
            iax.add_patch(Rectangle((rx0, ry0), CROP_SIZE, CROP_SIZE,
                                    fill=False, edgecolor="white", lw=2.0))
        f.subplots_adjust(left=0, right=1, top=1, bottom=0)
        save_both(f, fname)


# ---------------------------------------------------------------- Figure B
def overlay(image, mask):
    """与 src/visualize_predictions.py 一致：绿=直肠，红=肿瘤。"""
    base = np.stack([image] * 3, axis=-1).astype(np.float32)
    base /= max(base.max(), 1e-6)
    out = base.copy()
    out[mask == 1] = [0, 1, 0]
    out[mask == 2] = [1, 0, 0]
    return out


def disp(a):
    """项目显示约定：顺时针旋转 90°（见 visualize_predictions.disp）。"""
    return np.rot90(a, -1)


@torch.no_grad()
def _pick_mechanism_slice(device, max_scan=500):
    """在 50% 档未标注 train 切片中找一个 stage-1 假阳性被 stage-2 拒绝的实例。

    返回 (store, idx, info)。GT 仅用于挑选有教学价值的切片（文中说明）。
    """
    from scipy import ndimage
    from cascade import cascade_predict
    from common import load_model
    from dataset import labeled_subset, list_slices, split_train_val
    from tensorstore import TensorStore

    model1 = load_model(S1_CKPT, device)
    model2 = load_model(S2_CKPT, device)

    all_train = list_slices("data/CARE", "train")
    train_files, _ = split_train_val(all_train, val_ratio=0.15, seed=SEED)
    labeled = {os.path.basename(p)
               for p in labeled_subset(train_files, LABEL_RATIO, seed=SEED)}
    unlabeled = [p for p in train_files
                 if os.path.basename(p) not in labeled]
    print(f"未标注候选切片: {len(unlabeled)} (train={len(train_files)}, "
          f"labeled={len(labeled)})")
    store = TensorStore.from_files(unlabeled, device, cache_dir=CACHE)

    best = None  # (score, idx, info)
    n_tumor_seen = 0
    n = len(store)
    for off in range(0, n, 64):
        end = min(off + 64, n)
        pred, conf, pred1 = cascade_predict(model1, model2,
                                            store.images[off:end], CROP_SIZE,
                                            end - off, return_stage1=True)
        gt = store.labels[off:end].cpu().numpy()
        p1 = pred1.cpu().numpy()
        pc = pred.cpu().numpy()
        for j in range(gt.shape[0]):
            gtm = gt[j] == 2
            if gtm.sum() < 30:
                continue
            n_tumor_seen += 1
            if n_tumor_seen > max_scan:
                break
            # stage-1 肿瘤假阳性：与 GT 肿瘤（膨胀 5px）不重叠的预测肿瘤分量
            gt_dil = ndimage.binary_dilation(gtm, iterations=5)
            fp_zone = (p1[j] == 2) & ~gt_dil
            lab, ncc = ndimage.label(fp_zone)
            if ncc == 0:
                continue
            sizes = ndimage.sum_labels(fp_zone, lab, range(1, ncc + 1))
            k = int(np.argmax(sizes)) + 1
            blob = lab == k
            area = int(blob.sum())
            if area < 40:
                continue
            gy, gx = ndimage.center_of_mass(gtm)
            by, bx = ndimage.center_of_mass(blob)
            dist = float(np.hypot(gy - by, gx - bx))
            if dist < 60:
                continue
            reject = 1.0 - float((pc[j][blob] == 2).mean())  # stage-2 拒绝率
            if reject < 0.9:
                continue
            # 级联对真实肿瘤的分割质量（面板 4 要好看）
            inter = float(((pc[j] == 2) & gtm).sum())
            dice = 2 * inter / max((pc[j] == 2).sum() + gtm.sum(), 1)
            if dice < 0.5:
                continue
            score = area * reject * dice
            info = dict(blob=blob, area=area, dist=dist, reject=reject,
                        dice=dice, name=store.names[off + j])
            if best is None or score > best[0]:
                best = (score, off + j, info)
        if n_tumor_seen > max_scan:
            break
    if best is None:
        raise SystemExit("未找到满足条件的切片（可放宽阈值或增大 max_scan）")
    print(f"扫描含瘤切片 {min(n_tumor_seen, max_scan)} 例；选中 "
          f"{best[2]['name']}: FP面积={best[2]['area']}px, "
          f"距GT={best[2]['dist']:.0f}px, stage2拒绝率={best[2]['reject']:.2f}, "
          f"cascade肿瘤Dice={best[2]['dice']:.3f}, score={best[0]:.0f}")
    return model1, model2, store, best[1], best[2]


@torch.no_grad()
def fig_mechanism(device="cuda", max_scan=500):
    """5 联面板：CT | stage-1 肿瘤概率 | stage-1 预测 | 级联预测 | 过滤后伪标签。"""
    import torch  # noqa: F811
    from cascade import cascade_predict

    model1, model2, store, idx, info = _pick_mechanism_slice(device, max_scan)
    img = store.images[idx:idx + 1]
    pred, conf, pred1 = cascade_predict(model1, model2, img, CROP_SIZE, 1,
                                        return_stage1=True)
    with torch.autocast("cuda", dtype=torch.bfloat16,
                        enabled=str(device).startswith("cuda")):
        prob1 = torch.softmax(model1(img.to(
            memory_format=torch.channels_last)).float(), dim=1)
    image = img[0, 0].cpu().numpy()
    p1 = pred1[0].cpu().numpy()
    pc = pred[0].cpu().numpy()
    conf = conf[0].cpu().numpy()
    tprob = prob1[0, 2].cpu().numpy()
    blob = info["blob"]

    # 一致性过滤伪标签：conf>=0.9 且 stage-1 == stage-2，其余 IGNORE
    agree = (p1 == pc)
    keep = (conf >= CONF_THRESHOLD) & agree
    filtered = np.where(keep, pc, 255)
    fp_removed = int((blob & ~keep).sum())  # 假阳性块内被过滤掉的像素数
    print(f"假阳性块 {info['area']}px 中被过滤 {fp_removed}px "
          f"({fp_removed / info['area'] * 100:.1f}%)")

    # 假阳性块在显示坐标（rot90 顺时针）下的圆标注
    blob_d = disp(blob)
    ys, xs = np.nonzero(blob_d)
    cy, cx = ys.mean(), xs.mean()
    r = max(np.ptp(ys), np.ptp(xs)) / 2 + 12

    panels = [
        ("CT image", disp(image), dict(cmap="gray")),
        ("Stage-1\ntumor probability", None, None),
        ("Stage-1 prediction", disp(overlay(image, p1)), None),
        ("Stage-2 (cascade)\nprediction", disp(overlay(image, pc)), None),
        ("Filtered pseudo-label",
         disp(overlay(image, np.where(filtered == 255, 0, filtered))), None),
    ]
    # 3+2 居中排布：避免单排五联图在 A4 版心上把每张 CT 缩成邮票大小。
    fig = plt.figure(figsize=(10.8, 7.6))
    gs = fig.add_gridspec(2, 6, left=0.02, right=0.98, top=0.94,
                          bottom=0.03, wspace=0.35, hspace=0.28)
    slots = [(0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)),
             (1, slice(1, 3)), (1, slice(3, 5))]
    axes = [fig.add_subplot(gs[row, col]) for row, col in slots]
    for k, (ax, (title, arr, kw)) in enumerate(zip(axes, panels)):
        if k == 1:
            ax.imshow(disp(image), cmap="gray")
            p = disp(tprob)
            rgba = plt.get_cmap("hot")(np.clip(p, 0, 1))
            rgba[..., 3] = np.where(p > 0.05, 0.75 * np.clip(p * 1.5, 0, 1), 0)
            ax.imshow(rgba, interpolation="nearest")
        else:
            ax.imshow(arr, **(kw or {}), interpolation="nearest")
        ax.set_title(title, fontsize=13, pad=8)
        ax.axis("off")
        if k in (2, 4):
            circ = Circle((cx, cy), r, fill=False, edgecolor="red", lw=2.0,
                          ls="--" if k == 4 else "-")
            ax.add_patch(circ)
        if k == 4:
            ax.annotate("removed", xy=(cx, cy - r), xytext=(cx, cy - r - 26),
                        fontsize=11, color="red", ha="center",
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    save_both(fig, "fig_mechanism.png")
    print(f"机制图切片: {info['name']}（50% 档未标注切片；GT 仅用于挑选）")


# ---------------------------------------------------------------- Figure C
def fig_delta_heatmap():
    """4 方法 × 5 档 ΔDice（相对同比例 Supervised，百分点）热图，tumor/rectum。"""
    ratios = [1, 5, 10, 25, 50]
    methods = ["semi", "mt", "cps", "ours"]
    mats = {}
    for cls in ("tumor", "rectum"):
        sup = {r: load_summary(f"sup_{r}")["metrics"][cls]["dice"]["mean"] * 100
               for r in ratios}
        m = np.zeros((len(methods), len(ratios)))
        for i, meth in enumerate(methods):
            for j, r in enumerate(ratios):
                v = load_summary(f"{meth}_{r}")["metrics"][cls]["dice"]["mean"]
                m[i, j] = v * 100 - sup[r]
        mats[cls] = m
        print(f"ΔDice({cls}) 行={ [METHOD_LABEL[x] for x in methods] } 列={ratios}:\n"
              + np.array2string(m, precision=1, floatmode="fixed"))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5))
    fig.subplots_adjust(left=0.11, right=0.955, top=0.90, bottom=0.14,
                        wspace=0.55)
    for ai, (ax, cls, title) in enumerate(zip(axes, ("tumor", "rectum"),
                                              ("Tumor", "Normal rectum"))):
        m = mats[cls]
        vmax = float(np.ceil(abs(m).max()))  # 每面板独立发散刻度（0 居中）
        im = ax.imshow(m, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if abs(m[i, j]) > 0.62 * vmax else "black")
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([str(r) for r in ratios])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABEL[x] for x in methods], fontsize=9)
        ax.set_xlabel("Labeled ratio (%)")
        ax.set_title(title, fontsize=10)
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.014, pos.height])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("ΔDice vs Supervised (pp)", fontsize=8)
        cbar.ax.tick_params(labelsize=8)
    save_both(fig, "fig_delta_heatmap.png")


# ---------------------------------------------------------------- Figure D
def _gt_tumor_slices():
    """CARE test 中 GT 含肿瘤的切片名集合（与 make_figures.fig3 同约定）。"""
    labels = np.load(os.path.join(CACHE, "test_labels.npy"), mmap_mode="r")
    with open(os.path.join(CACHE, "test_names.json")) as f:
        names = json.load(f)
    out = set()
    for i, n in enumerate(names):
        if (np.asarray(labels[i]) == 2).any():
            out.add(n)
    return out


def fig_percase_box():
    """逐病例平均肿瘤 Dice 箱线图：ratio {10,25,50}% × 5 方法。"""
    import csv as _csv
    from collections import defaultdict

    from dataset import parse_case_id

    ratios = [10, 25, 50]
    tumor_slices = _gt_tumor_slices()
    data = {}  # (method, ratio) -> 病例均值数组
    med_table = {}
    for m in METHODS:
        for r in ratios:
            per_case = defaultdict(list)
            path = os.path.join(RESULTS, f"{m}_{r}_per_slice.csv")
            with open(path) as f:
                for row in _csv.DictReader(f):
                    if row["slice"] not in tumor_slices:
                        continue  # 只看 GT 含肿瘤切片
                    if not row["dice_tumor"]:
                        continue  # NaN 行（防御性跳过）
                    per_case[parse_case_id(row["slice"])].append(
                        float(row["dice_tumor"]))
            vals = np.array([np.mean(v) * 100 for v in per_case.values()])
            data[(m, r)] = vals
            med_table[(m, r)] = float(np.median(vals))
    print("逐病例肿瘤 Dice 中位数 (%)，行=方法 列=ratio：")
    print("method " + " ".join(f"{r:>6}" for r in ratios))
    for m in METHODS:
        print(f"{m:>6} " + " ".join(f"{med_table[(m, r)]:6.1f}" for r in ratios))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    offsets = np.linspace(-0.28, 0.28, len(METHODS))
    width = 0.11
    for gi, r in enumerate(ratios):
        for k, m in enumerate(METHODS):
            vals = data[(m, r)]
            bp = ax.boxplot([vals], positions=[gi + 1 + offsets[k]],
                            widths=width, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.0),
                            whiskerprops=dict(color=METHOD_STYLE[m]["color"]),
                            capprops=dict(color=METHOD_STYLE[m]["color"]))
            for patch in bp["boxes"]:
                patch.set_facecolor(METHOD_STYLE[m]["color"])
                patch.set_alpha(0.65)
                patch.set_edgecolor("0.3")
            med = med_table[(m, r)]
            ax.text(gi + 1 + offsets[k], med + 1.2, f"{med:.0f}",
                    ha="center", va="bottom", fontsize=6.5, color="0.25")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_xlabel("Labeled ratio (% of training cases)")
    ax.set_ylabel("Per-case mean tumor Dice (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Line2D([], [], color=METHOD_STYLE[m]["color"], lw=4,
                          alpha=0.65, label=METHOD_LABEL[m]) for m in METHODS]
    ax.legend(handles=handles, fontsize=8, loc="lower right", ncol=3)
    fig.tight_layout()
    save_both(fig, "fig_percase_box.png")


ALL = {
    "overview": fig_overview,
    "mechanism": fig_mechanism,
    "delta": fig_delta_heatmap,
    "percase": fig_percase_box,
}


@torch.no_grad()
def fig_qualitative_grid(device="cuda", n_per_tier=2):
    """定性对比整网格：6 行（good/mid/bad 各 2 例）× 5 列
    （Image | GT | Supervised 100% | ST 50% | Ours 50%）。

    直接由数据绘制，替代 visualize_predictions 先出条带再拼图的二级缩放
    （条带标题在 PDF 里缩到不可读）。选片规则与 visualize_predictions 一致：
    按 Supervised-100% 的肿瘤 Dice 升序，取最差/居中/最好各 n_per_tier 张。
    """
    import csv as _csv
    import matplotlib.patheffects as pe
    from cascade import cascade_predict
    from common import load_model
    from evaluate import slice_metrics

    # 1) 选片：复用 evaluate 的 per_slice csv（与可视化脚本同一级联推理口径）
    rows = []
    with open(os.path.join(RESULTS, "sup_100_per_slice.csv")) as f:
        for r in _csv.DictReader(f):
            if r["dice_tumor"] and r["dice_tumor"] != "nan":
                rows.append((r["slice"], float(r["dice_tumor"])))
    rows.sort(key=lambda t: t[1])
    L = len(rows)
    picks = ([("good", s) for s, _ in rows[L - n_per_tier:]] +
             [("mid", s) for s, _ in
              rows[L // 2 - n_per_tier // 2:
                   L // 2 - n_per_tier // 2 + n_per_tier]] +
             [("bad", s) for s, _ in rows[:n_per_tier]])
    print("选中切片:", [p[1] for p in picks])

    # 2) 读数据 + 三模型级联推理（必须走 TensorStore：缓存的数组方向与
    #    训练/评估一致，直接 np.load 原始 npz 会因轴序不同得到错误预测）
    from tensorstore import TensorStore
    files = [os.path.join("data/CARE/test/test_npz", name) for _, name in picks]
    store = TensorStore.from_files(files, device, cache_dir=CACHE)
    x = store.images
    gt = store.labels.cpu().numpy()
    imgs = x[:, 0].cpu().numpy()

    # stage-1 按各自标注比例配对（与 results/*_summary.json 的 stage1_ckpt 一致）
    s1 = {r: load_model(f"runs/s1_{r}/best.pt", device) for r in (50, 100)}
    specs = [("Supervised 100%", "runs/sup_100/best.pt", 100),
             ("ST 50%", "runs/semi_50/best.pt", 50),
             ("Ours 50%", "runs/ours_50/best.pt", 50)]
    preds, dices = [], []
    for label, ck, r in specs:
        m2 = load_model(ck, device)
        p, _ = cascade_predict(s1[r], m2, x, CROP_SIZE, len(picks))
        p = p.cpu().numpy()
        preds.append(p)
        dices.append([slice_metrics(p[i], gt[i], 2.0)[2]["dice"]
                      for i in range(len(picks))])

    # 3) 直接绘制 6×5 网格
    n_r, n_c = len(picks), 2 + len(specs)
    fig, axes = plt.subplots(n_r, n_c, figsize=(7.5, 8.9))
    col_titles = ["Image", "GT"] + [s for s, _, _ in specs]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10, pad=3)
    for r, (tier, name) in enumerate(picks):
        for c in range(n_c):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0:
                ax.imshow(disp(imgs[r]), cmap="gray")
            elif c == 1:
                ax.imshow(disp(overlay(imgs[r], gt[r])))
            else:
                k = c - 2
                ax.imshow(disp(overlay(imgs[r], preds[k][r])))
                ax.text(0.04, 0.05, f"{dices[k][r] * 100:.0f}",
                        transform=ax.transAxes, color="white", fontsize=8,
                        fontweight="bold", va="bottom",
                        path_effects=[pe.withStroke(linewidth=1.6,
                                                    foreground="black")])
        axes[r, 0].text(-0.06, 0.5, tier, transform=axes[r, 0].transAxes,
                        rotation=90, ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.965, bottom=0.005,
                        wspace=0.03, hspace=0.06)
    save_both(fig, "fig4_qualitative.png")
    print("各格肿瘤 Dice%:",
          {s: [f"{d * 100:.0f}" for d in ds]
           for (s, _, _), ds in zip(specs, dices)})


ALL["qualitative"] = fig_qualitative_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", default=",".join(ALL),
                    help="逗号分隔子集：" + ",".join(ALL))
    ap.add_argument("--max_scan", type=int, default=500,
                    help="mechanism 图最多扫描的含瘤候选切片数")
    args = ap.parse_args()

    os.makedirs(PAPER, exist_ok=True)
    os.makedirs(OUT_MANU, exist_ok=True)
    for name in args.figures.split(","):
        name = name.strip()
        if name == "mechanism":
            fig_mechanism(max_scan=args.max_scan)
        else:
            ALL[name]()


if __name__ == "__main__":
    main()
