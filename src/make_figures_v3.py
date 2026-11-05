"""v3 三种子结果图（CBM rebuttal；与 main.tex 当前引用文件名一致）。

数据源（权威，不手抄）：
  results/v2/agg_seeds.json              CARE 三种子 mean±std（患者级 v2 协议）
  results/v2/msd_*_summary_v2.json       MSD 单种子 v2（tumor-slice 协议，患者级）
  results_v2/{m}_{r}_s{seed}_per_slice.csv  逐切片（分层/箱线/定性选片用；
                             MT 箱线图遵循 resolve_mt_stems，可切换 mtfixed）
  data/cache/test_labels.npy + test_names.json  GT 面积/含瘤切片集合

产出同时落 paper/ 与 paper/manuscript/（300 dpi），文件名保持 main.tex 引用不变：
  fig1_label_efficiency.png  CARE 曲线：sup/semi/mt/cps/ours + bcp(全档，虚线)，
                             误差棒 = 三种子 std（均值点为患者级三种子均值）
  fig_delta_heatmap.png      ΔDice vs 同比例 sup（三种子均值，4 半监督 × 5 档）
  fig2_msd_external.png      MSD v2 肿瘤 Dice 曲线（单种子 seed42，患者级 + 95%CI）
  fig3_size_strata.png       病灶大小分层（sup_100 vs ours_50，三种子 mean±std）
  fig_percase_box.png        逐病例肿瘤 Dice 箱线（三种子逐病例平均，10/25/50%）
  fig4_qualitative.png       （--figures 加 qualitative 才跑；CPU 推理，v2 s42 ckpt）
  figS1_iou_nsd_curves.png   补充材料：IoU/NSD 曲线 2×2（agg_seeds 三种子 mean±std，
                             含 BCP 虚线五点；替代 make_supplement 的旧单种子版）
  figS2_pseudo_rounds.png    补充材料：逐轮伪标签覆盖率（runs_v2 三种子 mean±std）
  tabS3_pseudo_stats.tex     补充材料：逐轮覆盖率/验证 Dice 表（runs_v2 三种子
                             mean±std；替代 make_supplement 的旧单种子版）

用法：python src/make_figures_v3.py [--figures eff,delta,msd,strata,percase]
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import (RESULTS, PAPER, RATIOS, METHODS, METHOD_LABEL,
                          METHOD_STYLE, USAM)
from aggregate_v2_seeds import resolve_mt_stems

OUT_MANU = os.path.join(PAPER, "manuscript")
RESULTS_V2 = "results_v2"
RUNS_V2 = "runs_v2"
V2 = os.path.join(RESULTS, "v2")
CACHE = "data/cache"
SEEDS = [42, 43, 44]

METHODS_V3 = METHODS + ["bcp"]
METHOD_LABEL_V3 = dict(METHOD_LABEL, bcp="BCP")
METHOD_STYLE_V3 = dict(METHOD_STYLE,
                       bcp=dict(color="tab:purple", ls="--", marker="P",
                                lw=1.8, ms=8))
BCP_RATIOS = [1, 5, 10, 25, 50]  # bcp 全档（runs_v2/bcp_{1,5}_s* 2026-10-15 补齐）


def save_both(fig, name, tight=False):
    """保存到 paper/ 与 paper/manuscript/ 两处。tight=True 收编图外元素
    （fig2 底部注释、delta 热图右侧 colorbar 标签）。"""
    out1 = os.path.join(PAPER, name)
    fig.savefig(out1, dpi=300, bbox_inches="tight" if tight else None)
    out2 = os.path.join(OUT_MANU, name)
    shutil.copyfile(out1, out2)
    plt.close(fig)
    print(f"已保存 {out1} 与 {out2}")


def load_agg():
    with open(os.path.join(V2, "agg_seeds.json")) as f:
        return json.load(f)["experiments"]


def agg_dice(agg, tag, cls):
    """(mean%, std%)，患者级 v2 协议三种子汇总。"""
    d = agg[tag]["classes"][cls]["metrics"]["dice"]
    return d["mean"] * 100, d["std"] * 100


# ---------------------------------------------------------------- fig1
def fig1_efficiency():
    agg = load_agg()
    with plt.rc_context({"font.size": 12, "axes.titlesize": 16,
                         "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 11}):
        fig, axes = plt.subplots(2, 1, figsize=(9.2, 9.4), sharex=True)
        for ax, cls, title in zip(axes, ("tumor", "rectum"),
                                  ("Tumor", "Normal rectum")):
            for m in METHODS_V3:
                xs = BCP_RATIOS if m == "bcp" else \
                    RATIOS + ([100] if m == "sup" else [])
                ys, es = [], []
                for r in xs:
                    y, e = agg_dice(agg, f"{m}_{r}", cls)
                    ys.append(y)
                    es.append(e)
                style = dict(METHOD_STYLE_V3[m])
                style.update(lw=2.2 if m == "ours" else 1.8,
                             ms=9 if m == "ours" else 7)
                ax.errorbar(xs, ys, yerr=es, capsize=3.5, elinewidth=1.2,
                            label=METHOD_LABEL_V3[m], **style)
            ax.axhline(USAM[cls], color="purple", ls=":", lw=2.0,
                       label="U-SAM (SAM-based, bbox-prompted)")
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 10, 25, 50, 100])
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
            ax.set_ylabel("Test Dice (%)")
            ax.set_title(title, pad=8)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Labeled ratio (% of training cases)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   frameon=False, columnspacing=1.5, handlelength=2.4)
        fig.tight_layout(rect=[0.02, 0.075, 0.98, 0.99], h_pad=2.2)
        save_both(fig, "fig1_label_efficiency.png")


# ---------------------------------------------------------------- delta heatmap
def fig_delta_heatmap():
    """全档热图 + 5--50% 放大插图；两行分别共享发散色标。"""
    from matplotlib.patches import Rectangle

    agg = load_agg()
    methods = ["semi", "mt", "cps", "ours"]
    mats = {}
    for cls in ("tumor", "rectum"):
        sup = {r: agg_dice(agg, f"sup_{r}", cls)[0] for r in RATIOS}
        m = np.zeros((len(methods), len(RATIOS)))
        for i, meth in enumerate(methods):
            for j, r in enumerate(RATIOS):
                m[i, j] = agg_dice(agg, f"{meth}_{r}", cls)[0] - sup[r]
        mats[cls] = m
        print(f"ΔDice({cls}) 行={[METHOD_LABEL[x] for x in methods]} 列={RATIOS}:\n"
              + np.array2string(m, precision=1, floatmode="fixed"))

    vmax_all = float(np.ceil(max(np.abs(m).max() for m in mats.values())))
    vmax_zoom = float(np.ceil(max(np.abs(m[:, 1:]).max()
                                  for m in mats.values())))
    with plt.rc_context({"font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6))
        fig.subplots_adjust(left=0.13, right=0.85, top=0.95, bottom=0.08,
                            wspace=0.20, hspace=0.48)
        for row, (cols, ratios, vmax, suffix) in enumerate([
                (slice(None), RATIOS, vmax_all, "all ratios"),
                (slice(1, None), RATIOS[1:], vmax_zoom, "5–50% zoom")]):
            for col, (cls, title) in enumerate(
                    [("tumor", "Tumor"), ("rectum", "Normal rectum")]):
                ax = axes[row, col]
                m = mats[cls][:, cols]
                im = ax.imshow(m, cmap="RdBu", vmin=-vmax, vmax=vmax,
                               aspect="auto")
                for i in range(m.shape[0]):
                    for j in range(m.shape[1]):
                        ax.text(j, i, f"{m[i, j]:+.1f}", ha="center",
                                va="center", fontsize=10,
                                color="white" if abs(m[i, j]) > 0.62 * vmax
                                else "black")
                ax.set_xticks(range(len(ratios)))
                ax.set_xticklabels([str(r) for r in ratios])
                ax.set_yticks(range(len(methods)))
                if col == 0:
                    ax.set_yticklabels([METHOD_LABEL[x] for x in methods])
                else:
                    ax.set_yticklabels([])
                ax.set_title(f"{title} — {suffix}", pad=8)
                if row == 1:
                    ax.set_xlabel("Labeled ratio (%)")
                if row == 0:
                    ax.add_patch(Rectangle((0.5, -0.5), 4, 4, fill=False,
                                           edgecolor="black", lw=1.4, ls="--"))
            pos = axes[row, 0].get_position()
            cax = fig.add_axes([0.875, pos.y0, 0.018, pos.height])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("ΔDice vs Supervised (pp)", fontsize=11)
            cbar.ax.tick_params(labelsize=10)
        print(f"共享色标：全档 ±{vmax_all:.0f} pp；5--50% 放大 "
              f"±{vmax_zoom:.0f} pp")
        save_both(fig, "fig_delta_heatmap.png")


# ---------------------------------------------------------------- fig2 MSD
def fig2_msd():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for m in METHODS:
        ys, lo, hi = [], [], []
        for r in RATIOS:
            with open(os.path.join(V2, f"msd_{m}_{r}_summary_v2.json")) as f:
                d = json.load(f)["classes"]["tumor"]["metrics"]["dice"]
            y = d["patient_mean"] * 100
            ys.append(y)
            lo.append(y - d["ci95"][0] * 100)
            hi.append(d["ci95"][1] * 100 - y)
        ax.errorbar(RATIOS, ys, yerr=[lo, hi], capsize=2.5,
                    label=METHOD_LABEL[m], **METHOD_STYLE[m])
    ax.set_xscale("log")
    ax.set_xticks(RATIOS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Labeled ratio (% of training cases)")
    ax.set_ylabel("Tumor Dice (%)")
    ax.set_title("Cross-dataset replication on MSD Task10 Colon "
                 "(19 held-out cases)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    # 协议注记由 main.tex caption 承载（0.65\textwidth 下图内 8pt 注记不可读）
    fig.tight_layout()
    save_both(fig, "fig2_msd_external.png", tight=True)


# ---------------------------------------------------------------- helpers
def _gt_area_map():
    """CARE test 每切片 GT 肿瘤面积（px）；读 data/cache。"""
    labels = np.load(os.path.join(CACHE, "test_labels.npy"), mmap_mode="r")
    with open(os.path.join(CACHE, "test_names.json")) as f:
        names = json.load(f)
    return {n: int((np.asarray(labels[i]) == 2).sum())
            for i, n in enumerate(names)}


def _iter_tumor_dice(tag, seed):
    """results_v2/{tag}_s{seed}_per_slice.csv 的 (slice, dice_tumor) 有效行。"""
    import csv as _csv
    path = os.path.join(RESULTS_V2, f"{tag}_s{seed}_per_slice.csv")
    with open(path) as f:
        for row in _csv.DictReader(f):
            v = row["dice_tumor"]
            if not v or v == "nan":
                continue
            yield row["slice"], float(v)


# ---------------------------------------------------------------- fig3 strata
def fig3_size_strata():
    """GT 肿瘤面积分层；每种子先病例内平均再跨病例平均，与分层统计表一致。"""
    from dataset import parse_case_id

    area = _gt_area_map()
    strata = [("≤50", 0, 50), ("51–200", 50, 200),
              ("201–1000", 200, 1000), (">1000", 1000, 10**9)]
    series = {}
    for tag in ("sup_100", "ours_50"):
        per_seed = []
        ns = None
        for s in SEEDS:
            d = defaultdict(lambda: defaultdict(list))
            for name, v in _iter_tumor_dice(tag, s):
                a = area.get(name)
                if not a:
                    continue  # 只看 GT 含肿瘤切片
                for si, (_, lo, hi) in enumerate(strata):
                    if lo < a <= hi:
                        d[si][parse_case_id(name)].append(v)
                        break
            per_seed.append([
                float(np.mean([np.mean(case_vals) for case_vals in d[si].values()])) * 100
                for si in range(len(strata))])
            ns = [sum(len(case_vals) for case_vals in d[si].values())
                  for si in range(len(strata))]
        series[tag] = (np.array(per_seed), ns)
        print(f"{tag}: 各层逐种子患者级均值\n{np.array(per_seed) * 1.0}"
              f"\n  各层切片数 n={ns}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.36
    x = np.arange(len(strata))
    for k, (tag, label, color) in enumerate(
            [("sup_100", "Supervised 100%", "gray"),
             ("ours_50", "Ours 50%", "tab:red")]):
        mat, ns = series[tag]
        means = mat.mean(axis=0)
        stds = mat.std(axis=0, ddof=1)
        bars = ax.bar(x + (k - 0.5) * width, means, width, yerr=stds,
                      capsize=3, label=label, color=color,
                      alpha=0.85 if k else 0.6,
                      error_kw=dict(lw=1.0, ecolor="0.25"))
        for b, n_, sd in zip(bars, ns, stds):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + sd + 0.6,
                    f"n={n_}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in strata])
    ax.set_xlabel("GT tumor area (pixels, 512×512 slice)")
    ax.set_ylabel("Tumor Dice (%)")
    ax.set_title("Tumor Dice by tumor cross-sectional area (patient-level)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_both(fig, "fig3_size_strata.png")


# ---------------------------------------------------------------- percase box
def fig_percase_box():
    """逐病例平均肿瘤 Dice 箱线：ratio {10,25,50}% × 5 方法，三种子逐病例平均。"""
    from dataset import parse_case_id

    ratios = [10, 25, 50]
    use_mtfixed, mt_note = resolve_mt_stems({"mt": ratios}, RESULTS_V2)
    print(f"fig_percase MT 数据源：{mt_note}")
    area = _gt_area_map()
    tumor_slices = {n for n, a in area.items() if a > 0}
    data = {}
    med_table = {}
    for m in METHODS:
        for r in ratios:
            per_seed_case = []
            for s in SEEDS:
                d = defaultdict(list)
                base = "mtfixed" if (m == "mt" and use_mtfixed) else m
                for name, v in _iter_tumor_dice(f"{base}_{r}", s):
                    if name in tumor_slices:
                        d[parse_case_id(name)].append(v)
                per_seed_case.append({c: float(np.mean(v))
                                      for c, v in d.items()})
            cases = set.intersection(*[set(d) for d in per_seed_case])
            vals = np.array([np.mean([d[c] for d in per_seed_case])
                             for c in sorted(cases)]) * 100
            data[(m, r)] = vals
            med_table[(m, r)] = float(np.median(vals))
    print("逐病例肿瘤 Dice 中位数 (%)，行=方法 列=ratio（三种子逐病例平均）：")
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


# ---------------------------------------------------------------- qualitative
def fig4_qualitative(device="cpu", n_per_tier=2):
    """定性网格（6 行 × 5 列），v2 三种子协议的 seed42 ckpt：
    stage-1 = runs_v2/s1_{50,100}_s42，spec = sup_100/semi_50/ours_50 的 s42。
    选片按 results_v2/sup_100_s42_per_slice.csv 肿瘤 Dice 分层。"""
    import csv as _csv
    import matplotlib.patheffects as pe
    import torch
    from cascade import cascade_predict
    from common import load_model
    from evaluate import slice_metrics
    from make_figures_extra import overlay, disp
    from tensorstore import TensorStore

    crop_size = 256
    # 只在 GT 含瘤切片里分层选片：v2 csv 里 GT 空+预测非空的假阳性行
    # dice_tumor=0（非 nan），不过滤会把"bad"层选成无瘤 FP 切片
    area = _gt_area_map()
    rows = []
    with open(os.path.join(RESULTS_V2, "sup_100_s42_per_slice.csv")) as f:
        for r in _csv.DictReader(f):
            if r["dice_tumor"] and r["dice_tumor"] != "nan" \
                    and area.get(r["slice"], 0) > 0:
                rows.append((r["slice"], float(r["dice_tumor"])))
    rows.sort(key=lambda t: t[1])
    L = len(rows)
    picks = ([("good", s) for s, _ in rows[L - n_per_tier:]] +
             [("medium", s) for s, _ in
              rows[L // 2 - n_per_tier // 2:
                   L // 2 - n_per_tier // 2 + n_per_tier]] +
             [("poor", s) for s, _ in rows[:n_per_tier]])
    print("选中切片:", [p[1] for p in picks])

    files = [os.path.join("data/CARE/test/test_npz", name) for _, name in picks]
    store = TensorStore.from_files(files, device, cache_dir=CACHE)
    x = store.images
    gt = store.labels.cpu().numpy()
    imgs = x[:, 0].cpu().numpy()

    s1 = {r: load_model(f"runs_v2/s1_{r}_s42/best.pt", device)
          for r in (50, 100)}
    specs = [("Supervised 100%", "runs_v2/sup_100_s42/best.pt", 100),
             ("ST 50%", "runs_v2/semi_50_s42/best.pt", 50),
             ("Ours 50%", "runs_v2/ours_50_s42/best.pt", 50)]
    preds, dices = [], []
    with torch.no_grad():
        for label, ck, r in specs:
            m2 = load_model(ck, device)
            p, _ = cascade_predict(s1[r], m2, x, crop_size, len(picks))
            p = p.cpu().numpy()
            preds.append(p)
            dices.append([slice_metrics(p[i], gt[i], 2.0)[2]["dice"]
                          for i in range(len(picks))])

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


ALL = {
    "eff": fig1_efficiency,
    "delta": fig_delta_heatmap,
    "msd": fig2_msd,
    "strata": fig3_size_strata,
    "percase": fig_percase_box,
    "qualitative": fig4_qualitative,
}


# ---------------------------------------------------------------- figS1
def figS1_iou_nsd_curves():
    """补充材料 2x2：行=IoU/NSD，列=tumor/rectum；agg_seeds 三种子 mean±std，
    方法同 fig1（sup/semi/mt/cps/ours + bcp 虚线五点）。"""
    agg = load_agg()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharex=True)
    for row, metric in enumerate(("iou", "nsd")):
        for col, (cls, title) in enumerate(
                [("tumor", "Tumor"), ("rectum", "Normal rectum")]):
            ax = axes[row][col]
            for m in METHODS_V3:
                xs = BCP_RATIOS if m == "bcp" else \
                    RATIOS + ([100] if m == "sup" else [])
                ys, es = [], []
                for r in xs:
                    d = agg[f"{m}_{r}"]["classes"][cls]["metrics"][metric]
                    ys.append(d["mean"] * 100)
                    es.append(d["std"] * 100)
                ax.errorbar(xs, ys, yerr=es, capsize=2.5,
                            label=METHOD_LABEL_V3[m], **METHOD_STYLE_V3[m])
            ax.set_xscale("log")
            ax.set_xticks([1, 5, 10, 25, 50, 100])
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
            ax.set_title(title)
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("Labeled ratio (% of training cases)")
            if col == 0:
                ax.set_ylabel(f"Test {metric.upper()} (%)")
    axes[0][1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Label efficiency on CARE test (81 cases, 6,461 slices): "
                 "IoU (top) and NSD (bottom); mean \u00b1 std over three seeds")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_both(fig, "figS1_iou_nsd_curves.png")
    # 核对 supplement 正文论断：5% 起 Ours matches/exceeds sup，tumor NSD 边际最大
    for metric in ("iou", "nsd"):
        deltas = []
        for r in RATIOS:
            o = agg[f"ours_{r}"]["classes"]["tumor"]["metrics"][metric]["mean"]
            s = agg[f"sup_{r}"]["classes"]["tumor"]["metrics"][metric]["mean"]
            deltas.append((r, round((o - s) * 100, 1)))
        print(f"tumor {metric} ours-sup (pp): {deltas}")


# ---------------------------------------------------------------- figS2
def figS2_pseudo_rounds():
    """ST / Ours 逐轮伪标签覆盖率：runs_v2 三种子 mean±std。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    colors = {1: "tab:purple", 5: "tab:blue", 10: "tab:green",
              25: "tab:orange", 50: "tab:red"}
    for ax, (key, title) in zip(axes, [("semi", "Pseudo-label ST"),
                                       ("ours", "Ours")]):
        for r in RATIOS:
            covs = []
            for s in SEEDS:
                with open(os.path.join(
                        RUNS_V2, f"{key}_{r}_s{s}", "run_meta.json")) as f:
                    h = json.load(f)["history"]
                covs.append([e["pseudo_coverage"] * 100 for e in h])
            covs = np.array(covs)  # (n_seeds, n_rounds)
            xs = np.arange(1, covs.shape[1] + 1)
            ax.errorbar(xs, covs.mean(axis=0),
                        yerr=covs.std(axis=0, ddof=1),
                        marker="o", lw=1.6, capsize=2.5, color=colors[r],
                        label=f"{r}% labels")
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Self-training round")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pseudo-label coverage (% of unlabeled pixels)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Pseudo-label coverage across self-training rounds (CARE; "
                 "mean \u00b1 std over three seeds)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_both(fig, "figS2_pseudo_rounds.png")


ALL["figS1"] = figS1_iou_nsd_curves
ALL["figS2"] = figS2_pseudo_rounds


# ---------------------------------------------------------------- tabS3
def tabS3_pseudo_stats():
    """ST / Ours 各比例逐轮 pseudo_coverage(%) 与 best_val_dice(%)：
    runs_v2/{semi,ours}_{r}_s{42,43,44}/run_meta.json 三种子 mean±std。"""
    n_rounds = 3
    lines = [
        "% 逐轮伪标签统计，由 src/make_figures_v3.py 从 "
        "runs_v2/{semi,ours}_{r}_s{42,43,44}/run_meta.json 生成"
        "（三种子 mean$\\pm$std）",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Pseudo-label statistics per self-training round on CARE: "
        "coverage (fraction of unlabeled pixels passing the confidence "
        "threshold, \\%) and best validation Dice of the round (\\%). "
        "Values are the mean $\\pm$ std across the three training seeds "
        "(42/43/44), each with its own patient-level labeled/unlabeled "
        "split. At 1\\% labels the three labeled cases cannot yield a "
        "validation split, so the final-epoch Dice on the labeled training "
        "slices is reported for that row. ST = pseudo-label self-training.}",
        "\\label{tabS:pseudo}",
        "\\small",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{ll" + "c" * (2 * n_rounds) + "}",
        "\\toprule",
        " & & \\multicolumn{%d}{c}{Coverage (\\%%), round} & "
        "\\multicolumn{%d}{c}{Val Dice (\\%%), round} \\\\"
        % (n_rounds, n_rounds),
        "\\cmidrule(lr){3-%d} \\cmidrule(lr){%d-%d}"
        % (2 + n_rounds, 3 + n_rounds, 2 + 2 * n_rounds),
        "Method & Labeled & " + " & ".join(str(i + 1) for i in range(n_rounds))
        + " & " + " & ".join(str(i + 1) for i in range(n_rounds)) + " \\\\",
        "\\midrule",
    ]
    for key, label in [("semi", "Pseudo-label ST"), ("ours", "Ours")]:
        for i, r in enumerate(RATIOS):
            covs, vds = [], []
            for s in SEEDS:
                with open(os.path.join(
                        RUNS_V2, f"{key}_{r}_s{s}", "run_meta.json")) as f:
                    h = json.load(f)["history"]
                covs.append([e["pseudo_coverage"] * 100 for e in h])
                vds.append([e["best_val_dice"] * 100 for e in h])

            def ms(xs):
                a = np.array(xs)  # (n_seeds, n_rounds)
                return [f"{m:.2f}$\\pm${sd:.2f}" for m, sd in
                        zip(a.mean(axis=0), a.std(axis=0, ddof=1))]

            first = (f"\\multirow{{{len(RATIOS)}}}{{*}}{{{label}}}"
                     if i == 0 else "")
            lines.append(f"{first} & {r}\\% & "
                         + " & ".join(ms(covs) + ms(vds)) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1] + ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path = os.path.join(OUT_MANU, "tabS3_pseudo_stats.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"已保存 {path}")


ALL["tabS3"] = tabS3_pseudo_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures",
                    default="eff,delta,msd,strata,percase",
                    help="逗号分隔子集：" + ",".join(ALL))
    ap.add_argument("--device", default="cpu",
                    help="qualitative 图推理设备（默认 cpu）")
    args = ap.parse_args()

    os.makedirs(PAPER, exist_ok=True)
    os.makedirs(OUT_MANU, exist_ok=True)
    for name in args.figures.split(","):
        name = name.strip()
        if name == "qualitative":
            fig4_qualitative(device=args.device)
        else:
            ALL[name]()


if __name__ == "__main__":
    main()
