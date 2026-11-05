#!/usr/bin/env bash
# Aggregate the completed labeled-budget MSD matrix and refresh every MSD-derived table/figure.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python}
[ -f results_msd_lb/QUEUE_COMPLETE ] || exit 0
# 主队列完成时已用 Python 校验全部 ratio/method 的 ckpt、val_mode 与评估文件；
# 5%/10% 现由主队列顺序执行，不要求单独 worker 标记。
[ ! -f results_msd_lb/ANALYSIS_COMPLETE ] || exit 0
mkdir -p logs/msd_labeled_budget results/v2/hd95

$PY src/reanalyze_v2.py --msd_only --msd_results_dir results_msd_lb \
    --stats_md paper/v2_stats_msd_labeled.md \
    > logs/msd_labeled_budget/reanalyze.log 2>&1
$PY src/make_figures_v3.py --figures msd \
    > logs/msd_labeled_budget/figure_msd.log 2>&1
cp paper/fig2_msd_external.png paper/sr/figures/fig2_msd_external.png

for spec in msd_sup_50 msd_ours_50 msd_sup_25; do
    ratio=${spec##*_}
    OMP_NUM_THREADS=32 $PY src/eval_hd95.py --mode 3d \
        --ckpt "runs_msd_lb/${spec}/best.pt" \
        --stage1_ckpt "runs_msd_lb/msd_s1_${ratio}/best.pt" --crop_size 256 \
        --data_root data/MSD/colon_npz --nii_dir data/MSD/Task10_Colon/imagesTr \
        --out "results/v2/hd95/${spec}" --batch_size 64 --device cpu \
        > "logs/msd_labeled_budget/hd95_${spec}.log" 2>&1
done
$PY src/make_hd95_table.py > logs/msd_labeled_budget/hd95_table.log 2>&1

$PY - <<'PY'
import json, os
for r in (1, 5, 10, 25, 50):
    for m in ('sup', 'semi', 'mt', 'cps', 'ours'):
        tag = f'msd_{m}_{r}'
        assert os.path.isfile(f'results/v2/{tag}_summary_v2.json'), tag
        s = json.load(open(f'results_msd_lb/{tag}_summary.json'))
        assert 'runs_msd_lb/' in s['ckpt'], (tag, s['ckpt'])
for tag in ('msd_sup_50', 'msd_ours_50', 'msd_sup_25'):
    s = json.load(open(f'results/v2/hd95/{tag}_summary.json'))
    assert 'runs_msd_lb/' in s['ckpt'], (tag, s['ckpt'])
with open('results_msd_lb/ANALYSIS_COMPLETE', 'w') as f:
    f.write('MSD tables, Wilcoxon, curve and physical-unit boundary metrics refreshed.\n')
print('MSD_ANALYSIS_COMPLETE')
PY
