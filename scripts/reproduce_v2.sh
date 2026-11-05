#!/bin/bash
# reproduce_v2.sh — CARE v2 三种子正式实验矩阵（论文主结果）完整复现命令。
# 与产出 runs_v2/ 与 results_v2/ 的队列脚本（queue_rebuttal_b1/b2/b3/bcp）命令逐条一致，
# 去掉机器相关的排队/看门逻辑，路径全部相对化。
#
# 前提：data/CARE（DataV6 解压）与 data/cache（python src/preprocess_cache.py）已就位；
#       训练需 GPU（512x512 全图 epoch ~1.5 min），评估可 CPU。
# 用法：bash scripts/reproduce_v2.sh [SEED...]   （默认 42 43 44）
set -e
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-data/CARE}
CACHE=${CACHE:-data/cache}
CROP=256
PY=${PYTHON:-python}
SEEDS=${*:-42 43 44}

mkdir -p runs_v2 results_v2

for s in $SEEDS; do
    echo "##### seed $s #####"
    for r in 1 5 10 25 50 100; do
        # stage-1 全图全监督（粗定位）
        $PY src/train_supervised.py --config configs/label_ratio_${r}.json \
            --data_root "$DATA_ROOT" --out_dir runs_v2/s1_${r}_s${s} \
            --seed $s --val_mode labeled
        $PY src/evaluate.py --ckpt runs_v2/s1_${r}_s${s}/best.pt \
            --data_root "$DATA_ROOT" --out results_v2/s1_${r}_s${s} \
            --cache_dir "$CACHE"
        # stage-2 全监督（256x256 裁剪精分割；验证/测试按 stage-1 预测质心，无 GT 泄漏）
        $PY src/train_supervised.py --config configs/label_ratio_${r}.json \
            --data_root "$DATA_ROOT" --out_dir runs_v2/sup_${r}_s${s} \
            --seed $s --val_mode labeled \
            --crop_size $CROP --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt
        $PY src/evaluate.py --ckpt runs_v2/sup_${r}_s${s}/best.pt \
            --data_root "$DATA_ROOT" --out results_v2/sup_${r}_s${s} \
            --cache_dir "$CACHE" \
            --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt --crop_size $CROP
        # Ours：stage-2 半监督自训练（3 轮 + 跨阶段一致性过滤）
        if [ "$r" != "100" ]; then
            $PY src/train_semi.py --config configs/label_ratio_${r}.json \
                --data_root "$DATA_ROOT" --teacher_ckpt runs_v2/sup_${r}_s${s}/best.pt \
                --out_dir runs_v2/ours_${r}_s${s} --rounds 3 --consistency \
                --seed $s --val_mode labeled \
                --crop_size $CROP --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt
            $PY src/evaluate.py --ckpt runs_v2/ours_${r}_s${s}/best.pt \
                --data_root "$DATA_ROOT" --out results_v2/ours_${r}_s${s} \
                --cache_dir "$CACHE" \
                --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt --crop_size $CROP
        fi
    done
    for r in 1 5 10 25 50; do
        # Mean Teacher（EMA teacher + 坐标对齐的一致性 MSE，100 epoch；mtfixed）
        $PY src/train_mt.py --config configs/label_ratio_${r}.json \
            --data_root "$DATA_ROOT" --out_dir runs_v2/mtfixed_${r}_s${s} \
            --seed $s --val_mode labeled \
            --crop_size $CROP --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt
        $PY src/evaluate.py --ckpt runs_v2/mtfixed_${r}_s${s}/best.pt \
            --data_root "$DATA_ROOT" --out results_v2/mtfixed_${r}_s${s} \
            --cache_dir "$CACHE" \
            --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt --crop_size $CROP
        # 纯伪标签自训练 ST（3 轮，无 consistency 过滤）
        $PY src/train_semi.py --config configs/label_ratio_${r}.json \
            --data_root "$DATA_ROOT" --teacher_ckpt runs_v2/sup_${r}_s${s}/best.pt \
            --out_dir runs_v2/semi_${r}_s${s} --rounds 3 \
            --seed $s --val_mode labeled \
            --crop_size $CROP --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt
        $PY src/evaluate.py --ckpt runs_v2/semi_${r}_s${s}/best.pt \
            --data_root "$DATA_ROOT" --out results_v2/semi_${r}_s${s} \
            --cache_dir "$CACHE" \
            --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt --crop_size $CROP
        # CPS 交叉伪监督（双独立初始化网络，100 epoch）
        $PY src/train_cps.py --config configs/label_ratio_${r}.json \
            --data_root "$DATA_ROOT" --out_dir runs_v2/cps_${r}_s${s} \
            --seed $s --val_mode labeled \
            --crop_size $CROP --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt
        $PY src/evaluate.py --ckpt runs_v2/cps_${r}_s${s}/best.pt \
            --data_root "$DATA_ROOT" --out results_v2/cps_${r}_s${s} \
            --cache_dir "$CACHE" \
            --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt --crop_size $CROP
    done
    for r in 1 5 10 25 50; do
        # BCP 双向 copy-paste（Bai et al. CVPR 2023，100 epoch，含低比例补充实验）
        $PY src/train_bcp.py --config configs/label_ratio_${r}.json \
            --data_root "$DATA_ROOT" --out_dir runs_v2/bcp_${r}_s${s} \
            --seed $s --val_mode labeled \
            --crop_size $CROP --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt
        $PY src/evaluate.py --ckpt runs_v2/bcp_${r}_s${s}/best.pt \
            --data_root "$DATA_ROOT" --out results_v2/bcp_${r}_s${s} \
            --cache_dir "$CACHE" \
            --stage1_ckpt runs_v2/s1_${r}_s${s}/best.pt --crop_size $CROP
    done
done

# 单种子参照（seed 42）：epoch-matched sup180（50%/100%）与修复版 MT+一致性过滤组合 mtcons（50%）
for r in 50 100; do
    $PY src/train_supervised.py --config configs/label_ratio_${r}.json \
        --data_root "$DATA_ROOT" --out_dir runs_v2/sup180_${r}_s42 \
        --epochs 180 --seed 42 --val_mode labeled \
        --crop_size $CROP --stage1_ckpt runs_v2/s1_${r}_s42/best.pt
    $PY src/evaluate.py --ckpt runs_v2/sup180_${r}_s42/best.pt \
        --data_root "$DATA_ROOT" --out results_v2/sup180_${r}_s42 \
        --cache_dir "$CACHE" \
        --stage1_ckpt runs_v2/s1_${r}_s42/best.pt --crop_size $CROP
done
$PY src/train_semi.py --config configs/label_ratio_50.json \
    --data_root "$DATA_ROOT" --teacher_ckpt runs_v2/mtfixed_50_s42/best.pt \
    --out_dir runs_v2/mtcons_50_s42 --rounds 1 --consistency \
    --seed 42 --val_mode labeled \
    --crop_size $CROP --stage1_ckpt runs_v2/s1_50_s42/best.pt
$PY src/evaluate.py --ckpt runs_v2/mtcons_50_s42/best.pt \
    --data_root "$DATA_ROOT" --out results_v2/mtcons_50_s42 \
    --cache_dir "$CACHE" \
    --stage1_ckpt runs_v2/s1_50_s42/best.pt --crop_size $CROP

# 相同优化器更新数监督对照（Table S27）：50%/1%，三种子，标注预算与主协议一致
PYTHON="$PY" bash scripts/run_update_matched_controls.sh
$PY src/make_table_update_matched.py

# 配对轮数消融（Table S24）：ST/Ours 在 10% 标注、同一教师、labeled 验证下
# 补跑 rounds={1,5}（rounds=3 即上面主矩阵的 semi_10_s*/ours_10_s*）
for s in $SEEDS; do
    for R in 1 5; do
        $PY src/train_semi.py --config configs/label_ratio_10.json \
            --data_root "$DATA_ROOT" --teacher_ckpt runs_v2/sup_10_s${s}/best.pt \
            --out_dir runs_v2/semi_10_r${R}_s${s} --rounds $R \
            --seed $s --val_mode labeled \
            --crop_size $CROP --stage1_ckpt runs_v2/s1_10_s${s}/best.pt
        $PY src/train_semi.py --config configs/label_ratio_10.json \
            --data_root "$DATA_ROOT" --teacher_ckpt runs_v2/sup_10_s${s}/best.pt \
            --out_dir runs_v2/ours_10_r${R}_s${s} --rounds $R --consistency \
            --seed $s --val_mode labeled \
            --crop_size $CROP --stage1_ckpt runs_v2/s1_10_s${s}/best.pt
        for name in semi_10_r${R}_s${s} ours_10_r${R}_s${s}; do
            $PY src/evaluate.py --ckpt runs_v2/${name}/best.pt \
                --data_root "$DATA_ROOT" --out results_v2/${name} \
                --cache_dir "$CACHE" \
                --stage1_ckpt runs_v2/s1_10_s${s}/best.pt --crop_size $CROP
        done
    done
done
$PY src/make_table_rounds_paired.py   # 生成 Table S24（另有 Table S23：
$PY src/make_table_1pct_perseed.py    # 1% 档逐种子配对差值表）

echo "=== 标注子集一致性核验（M9 证据）==="
$PY src/check_subset_consistency.py --data_root "$DATA_ROOT" --runs_dir runs_v2 \
    --report results/v2/subset_consistency.md

echo "=== 三种子聚合统计（mean±std + Wilcoxon + Holm）==="
$PY src/aggregate_v2_seeds.py
$PY src/aggregate_v4_rebuttal.py
$PY src/compute_annotation_budget.py

echo "REPRODUCE_V2 ALL DONE"
