"""投稿级图表包生成（Computers in Biology and Medicine）。

全部数字从 results/ 的 summary json / per-slice csv 现算，不手抄。
产出（paper/ 下，位图均 300 dpi）：
  fig1_label_efficiency.png  CARE 标注效率曲线（tumor/rectum 双子图，5 方法 + U-SAM 参考线）
  fig2_msd_external.png      MSD Colon 外部验证肿瘤 Dice 曲线（5 方法 × 5 档）
  fig3_size_strata.png       CARE 肿瘤按 GT 面积分层的 Dice 柱状图（sup_100 vs ours_50）
  fig4_qualitative.png       定性面板拼图（读 paper/fig4/*.png，由 visualize_predictions.py 生成）
  stats_summary.md           41 个 compare json 的 Wilcoxon 分组表
  reproducibility.md         复现信息表（configs + runs/*/run_meta.json 现算）
  data_availability.md       数据可用性声明模板

用法：python src/make_figures.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"
PAPER = "paper"
RATIOS = [1, 5, 10, 25, 50]
METHODS = ["sup", "semi", "mt", "cps", "ours"]
METHOD_LABEL = {"sup": "Supervised", "semi": "Pseudo-label ST",
                "mt": "Mean Teacher", "cps": "CPS", "ours": "Ours"}
METHOD_STYLE = {
    "sup":  dict(color="gray", ls="--", marker="s", lw=1.4),
    "semi": dict(color="tab:blue", ls="-", marker="o", lw=1.4),
    "mt":   dict(color="tab:green", ls="-", marker="^", lw=1.4),
    "cps":  dict(color="tab:orange", ls="-", marker="D", lw=1.4),
    "ours": dict(color="tab:red", ls="-", marker="*", lw=2.6, ms=11),
}
USAM = {"tumor": 76.38, "rectum": 71.23}  # U-SAM (Commun Med 2025) 官方 test Dice%


def load_summary(name):
    with open(os.path.join(RESULTS, f"{name}_summary.json")) as f:
        return json.load(f)


def dice_point(summary, cls):
    m = summary["metrics"][cls]["dice"]
    return m["mean"] * 100, (m["mean"] - m["ci95"][0]) * 100, (m["ci95"][1] - m["mean"]) * 100


def fig1_label_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, cls, title in zip(axes, ("tumor", "rectum"),
                              ("Tumor", "Normal rectum")):
        for m in METHODS:
            xs = RATIOS + ([100] if m == "sup" else [])
            ys, lo, hi = [], [], []
            for r in xs:
                y, lo_, hi_ = dice_point(load_summary(f"{m}_{r}"), cls)
                ys.append(y)
                lo.append(lo_)
                hi.append(hi_)
            ax.errorbar(xs, ys, yerr=[lo, hi], capsize=2.5,
                        label=METHOD_LABEL[m], **METHOD_STYLE[m])
        ax.axhline(USAM[cls], color="purple", ls=":", lw=1.6,
                   label="U-SAM (SAM-based, bbox-prompted)")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 25, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Labeled ratio (% of training cases)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test Dice (%)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Label efficiency on CARE test (81 cases, 6,461 slices)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(PAPER, "fig1_label_efficiency.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"已保存 {out}")


def fig2_msd_external():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for m in METHODS:
        ys, lo, hi = [], [], []
        for r in RATIOS:
            y, lo_, hi_ = dice_point(load_summary(f"msd_{m}_{r}"), "tumor")
            ys.append(y)
            lo.append(lo_)
            hi.append(hi_)
        ax.errorbar(RATIOS, ys, yerr=[lo, hi], capsize=2.5,
                    label=METHOD_LABEL[m], **METHOD_STYLE[m])
    ax.set_xscale("log")
    ax.set_xticks(RATIOS)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Labeled ratio (% of training cases)")
    ax.set_ylabel("Tumor Dice (%)")
    ax.set_title("External validation on MSD Task10 Colon (19 held-out cases)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.text(0.5, -0.04, "Note: full-slice protocol — empty-GT slices with any "
                         "prediction score 0, lowering absolute Dice values.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    out = os.path.join(PAPER, "fig2_msd_external.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {out}")


def fig3_size_strata():
    """按 CARE test GT 肿瘤面积分层：sup_100 vs ours_50 的 dice_tumor 均值。"""
    import csv as _csv
    labels = np.load("data/cache/test_labels.npy", mmap_mode="r")
    with open("data/cache/test_names.json") as f:
        names = json.load(f)
    area = {}
    for i, n in enumerate(names):
        area[n] = int((np.asarray(labels[i]) == 2).sum())
    strata = [("≤50", 0, 50), ("51–200", 50, 200),
              ("201–1000", 200, 1000), (">1000", 1000, 10**9)]
    series = {}
    for tag in ("sup_100", "ours_50"):
        d = defaultdict(list)
        with open(os.path.join(RESULTS, f"{tag}_per_slice.csv")) as f:
            for row in _csv.DictReader(f):
                a = area.get(row["slice"])
                if a is None or a == 0:
                    continue  # 只看 GT 含肿瘤切片
                if not row["dice_tumor"]:
                    continue  # NaN 行（协议上 GT 有瘤时不会出现，防御性跳过）
                v = float(row["dice_tumor"])
                for si, (_, lo, hi) in enumerate(strata):
                    if lo < a <= hi:
                        d[si].append(v)
                        break
        series[tag] = d
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    width = 0.36
    x = np.arange(len(strata))
    for k, (tag, label, color) in enumerate(
            [("sup_100", "Supervised 100%", "gray"), ("ours_50", "Ours 50%", "tab:red")]):
        means = [np.mean(series[tag][si]) * 100 if series[tag][si] else 0
                 for si in range(len(strata))]
        ns = [len(series[tag][si]) for si in range(len(strata))]
        bars = ax.bar(x + (k - 0.5) * width, means, width, label=label, color=color,
                      alpha=0.85 if k else 0.6)
        for b, n_ in zip(bars, ns):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"n={n_}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in strata])
    ax.set_xlabel("GT tumor area (pixels, 512×512 slice)")
    ax.set_ylabel("Tumor Dice (%)")
    ax.set_title("Tumor Dice stratified by lesion size (CARE test)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PAPER, "fig3_size_strata.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"已保存 {out}；各层样本量 "
          f"{[len(series['ours_50'][si]) for si in range(len(strata))]}")


def fig4_qualitative():
    """⚠️ 已废弃：条带拼图经二次缩放，文字在 PDF 里不可读。
    定性图改用 make_figures_extra.fig_qualitative_grid（6×5 整网格直绘）。"""
    print("⚠️ fig4_qualitative 已废弃，请运行 "
          "python src/make_figures_extra.py --figures qualitative")


def _stars(p):
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def stats_summary():
    """把 compare_*.json 整理成分组表。A→B 按文件内记录方向呈现。"""
    groups = [  # (标题, [(A, B, ratios)])
        ("CARE: Ours vs Supervised", [("sup", "ours", RATIOS)]),
        ("CARE: Ours vs Pseudo-label ST", [("semi", "ours", RATIOS)]),
        ("CARE: Supervised vs Mean Teacher", [("sup", "mt", RATIOS)]),
        ("CARE: Supervised vs CPS", [("sup", "cps", RATIOS)]),
        ("CARE: Supervised vs Pseudo-label ST", [("sup", "semi", RATIOS)]),
        ("MSD: Supervised vs Ours", [("msd_sup", "msd_ours", RATIOS)]),
        ("MSD: Supervised vs Mean Teacher", [("msd_sup", "msd_mt", RATIOS)]),
        ("MSD: Supervised vs CPS", [("msd_sup", "msd_cps", RATIOS)]),
    ]
    lines = ["# 统计检验汇总（配对 Wilcoxon 符号秩，逐切片）",
             "",
             "显著性标注：\\* p<0.05，\\*\\* p<0.01，\\*\\*\\* p<0.001，ns=不显著。"
             "每格为 mean_A → mean_B（p 值）。",
             ""]
    n_files = 0
    for title, specs in groups:
        lines.append(f"## {title}")
        lines.append("")
        for a, b, ratios in specs:
            is_msd = a.startswith("msd_")
            metrics = ["dice_tumor"] if is_msd else ["dice_tumor", "dice_rectum"]
            header = "| ratio | " + " | ".join(
                f"{m.replace('dice_', '')} dice {a}→{b}" for m in metrics) + " |"
            lines.append(header)
            lines.append("|" + "---|" * (len(metrics) + 1))
            for r in ratios:
                path = os.path.join(RESULTS, f"compare_{a}_{r}_vs_{b}_{r}.json")
                if not os.path.exists(path):
                    continue
                n_files += 1
                with open(path) as f:
                    c = json.load(f)
                cells = []
                for m in metrics:
                    t = c["tests"][m]
                    p = t.get("p")
                    cells.append(f"{t['mean_a']:.4f} → {t['mean_b']:.4f} "
                                 f"({p:.2e} {_stars(p)})" if p is not None
                                 else f"{t['mean_a']:.4f} → {t['mean_b']:.4f} (n/a)")
                lines.append(f"| {r}% | " + " | ".join(cells) + " |")
            lines.append("")
    # 消融补充：sup_10 vs semi_10_rounds1
    extra = os.path.join(RESULTS, "compare_sup_10_vs_semi_10_rounds1.json")
    if os.path.exists(extra):
        n_files += 1
        with open(extra) as f:
            c = json.load(f)
        lines.append("## 附录：消融 sup_10 vs semi_10_rounds1")
        lines.append("")
        for m, t in c["tests"].items():
            p = t.get("p")
            lines.append(f"- {m}: {t['mean_a']:.4f} → {t['mean_b']:.4f} "
                         f"(p={p:.2e} {_stars(p)})" if p is not None else
                         f"- {m}: {t['mean_a']:.4f} → {t['mean_b']:.4f} (n/a)")
        lines.append("")
    out = os.path.join(PAPER, "stats_summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"已保存 {out}（收录 {n_files} 个 compare 文件）")


def reproducibility():
    import torch
    from models import LightUNet
    cfg = json.load(open("configs/label_ratio_10.json"))
    n_params = sum(p.numel() for p in LightUNet(num_classes=3).parameters())
    metas = {}
    for r in ("semi_50", "mt_50", "cps_50", "ours_50", "sup_100"):
        p = f"runs/{r}/run_meta.json"
        if os.path.exists(p):
            metas[r] = json.load(open(p))
    lines = [
        "# 复现信息表",
        "",
        "## 模型与训练（全方法共享）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 模型 | LightUNet（1ch→3类；MSD 2类），参数量 {n_params/1e6:.2f}M |",
        f"| 优化器 | AdamW，lr={cfg['lr']}，weight_decay={cfg['weight_decay']}，"
        f"CosineAnnealingLR |",
        f"| batch_size | {cfg['batch_size']} |",
        f"| dropout | {cfg['dropout']} |",
        f"| class_weights | {cfg['class_weights']}（MSD: [1,3]） |",
        f"| 损失 | CE + soft Dice |",
        f"| 级联 | stage1 全图 512² 粗定位 → 预测质心裁 256×256 → stage2 精分割 → 贴回全画布 |",
        f"| 数据划分 | 病人级，seed={cfg['seed']}，val_ratio={cfg['val_ratio']}；"
        f"标注子集同 seed 嵌套 |",
        f"| 混合精度 | bf16 AMP + channels_last + torch.compile |",
        f"| 软件 | Python，PyTorch {torch.__version__} |",
        "| 硬件 | NVIDIA A100 80GB |",
        "",
        "## 各方法特有超参（以 50% 档 run_meta 为例）",
        "",
        "| 方法 | 关键超参 |",
        "|---|---|",
    ]
    for r, desc in [
            ("sup_100", "全监督基线，epochs={epochs}"),
            ("semi_50", "伪标签自训练：rounds={rounds}，conf_threshold={conf_threshold}，"
                        "epochs_per_round=60"),
            ("mt_50", "Mean Teacher：ema_decay={ema_decay}，λ_max={consistency_weight}，"
                      "rampup_epochs={rampup_epochs}"),
            ("cps_50", "CPS：λ_max={consistency_weight}，rampup_epochs={rampup_epochs}，"
                       "双独立初始化"),
            ("ours_50", "Ours = 半监督 + 跨阶段一致性（consistency={consistency}，"
                        "rounds={rounds}，conf_threshold={conf_threshold}）")]:
        m = metas.get(r)
        if m:
            lines.append(f"| {r} | " + desc.format(**defaultdict(str, m)) + " |")
    lines += [
        "",
        "完整超参见 configs/*.json 与 runs/*/run_meta.json。",
    ]
    out = os.path.join(PAPER, "reproducibility.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"已保存 {out}")


def data_availability():
    lines = [
        "# Data Availability Statement（投稿模板）",
        "",
        "The CARE dataset used in this study is derived from the rectal cancer CT "
        "dataset released with the U-SAM study (Nature Communications Medicine, "
        "2025; DOI: 10.1038/s43856-025-00953-0) and is available under the "
        "CC BY-NC 4.0 license from the original authors "
        "(https://github.com/kanydao/U-SAM). The MSD Task10 Colon dataset is a "
        "publicly available dataset from the Medical Segmentation Decathlon "
        "(http://medicaldecathlon.com/).",
        "",
        "# Code Availability",
        "",
        "The source code used for training and evaluation will be made publicly "
        "available at [repository URL placeholder] upon acceptance.",
    ]
    out = os.path.join(PAPER, "data_availability.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"已保存 {out}")


def main():
    os.makedirs(PAPER, exist_ok=True)
    fig1_label_efficiency()
    fig2_msd_external()
    fig3_size_strata()
    fig4_qualitative()
    stats_summary()
    reproducibility()
    data_availability()


if __name__ == "__main__":
    main()
