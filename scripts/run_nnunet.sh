#!/bin/bash
# nnU-Net v2 参照实验（审稿人要求）：CARE 100% / 50% 标注比例，2d 配置。
# ⚠️ 本脚本需要 GPU，等当前 GPU 任务结束后再执行（数据转换已提前完成，见步骤 0 注释）。
# 用法: bash scripts/run_nnunet.sh
set -e
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
export nnUNet_raw="$PWD/data/nnunet/raw"
export nnUNet_preprocessed="$PWD/data/nnunet/preprocessed"
export nnUNet_results="$PWD/data/nnunet/results"

# ---------------------------------------------------------------------------
# 0) 数据转换（已用 CPU 跑过；如需重建取消注释，~15-30 min CPU，无需 GPU）：
#    $PY src/prepare_nnunet_care.py --data_root data/CARE --out data/nnunet
#    划分与 LightUNet 基线逐病例一致（split_train_val/labeled_subset, seed=42）：
#    Dataset501_CARE100: train fold=270 例标注池, val=48 例(项目 15% 验证集)
#    Dataset502_CARE50:  train fold=135 例(labeled_subset(pool,0.5)), val 同上
# ---------------------------------------------------------------------------

echo "=== [1/4] plan & preprocess（仅 2d 配置；CPU，约 1-3 h）==="
nnUNetv2_plan_and_preprocess -d 501 502 -c 2d -np 16 --verify_dataset_integrity

echo "=== [2/4] 覆盖 splits_final.json：单 fold、病人级（禁用默认 5-fold CV）==="
# nnUNetv2_train 固定读取 preprocessed 目录下的 splits_final.json，
# 用 prepare_nnunet_care.py 生成的单 fold 文件覆盖（train=标注病例, val=项目验证病例，
# 与 LightUNet 基线完全相同的病例列表）
cp data/nnunet/splits/Dataset501_CARE100_splits.json \
   "$nnUNet_preprocessed/Dataset501_CARE100/splits_final.json"
cp data/nnunet/splits/Dataset502_CARE50_splits.json \
   "$nnUNet_preprocessed/Dataset502_CARE50/splits_final.json"

echo "=== [3/4] 训练（GPU；默认 1000 epoch，每个数据集约 1-1.5 天/张现代 GPU）==="
# 如需缩短可换 -tr nnUNetTrainer_500epochs（nnunetv2 自带变体）
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer
nnUNetv2_train 502 2d 0 -tr nnUNetTrainer

echo "=== [4/4] 测试集推理 + 项目指标评估（与 evaluate.py 相同指标定义）==="
# checkpoint_best 由自定义 val（=项目验证集）选出；若想用 final 去掉 -chk 即可
nnUNetv2_predict -i "$nnUNet_raw/Dataset501_CARE100/imagesTs" \
    -o data/nnunet/predictions/Dataset501_CARE100 \
    -d 501 -c 2d -f 0 -tr nnUNetTrainer -chk checkpoint_best.pth
nnUNetv2_predict -i "$nnUNet_raw/Dataset502_CARE50/imagesTs" \
    -o data/nnunet/predictions/Dataset502_CARE50 \
    -d 502 -c 2d -f 0 -tr nnUNetTrainer -chk checkpoint_best.pth

$PY src/eval_nnunet.py --pred_dir data/nnunet/predictions/Dataset501_CARE100 \
    --data_root data/CARE --out results_v2/nnunet_sup_100 \
    --save_pred_npz data/nnunet/predictions_per_slice/Dataset501_CARE100
$PY src/eval_nnunet.py --pred_dir data/nnunet/predictions/Dataset502_CARE50 \
    --data_root data/CARE --out results_v2/nnunet_sup_50 \
    --save_pred_npz data/nnunet/predictions_per_slice/Dataset502_CARE50

echo "=== （可选）与 LightUNet 全监督基线配对 Wilcoxon ==="
$PY src/evaluate.py --compare results/sup_100_summary.json results_v2/nnunet_sup_100_summary.json || true
$PY src/evaluate.py --compare results/sup_50_summary.json  results_v2/nnunet_sup_50_summary.json || true

echo "NNUNET PIPELINE DONE"
