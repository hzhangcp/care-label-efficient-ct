#!/bin/bash
set -u
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
DATA_ROOT=data/CARE
CACHE=data/cache
CROP=256
MAX_ATTEMPTS=3
DRY=${DRY_RUN:-0}
mkdir -p runs_v2 results_v2 logs

run_cmd() {
    if [ "$DRY" = "1" ]; then
        printf 'DRY_RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

train_valid() {
    local tag=$1
    $PY - "$tag" <<'PY'
import json, os, sys
tag = sys.argv[1]
path = f'runs_v2/{tag}/run_meta.json'
if not os.path.isfile(path) or not os.path.isfile(f'runs_v2/{tag}/best.pt'):
    raise SystemExit(1)
meta = json.load(open(path))
assert meta['actual_optimizer_updates'] >= meta['resolved_target_updates']
assert meta['resolved_epochs'] >= 1
PY
}

summary_valid() {
    local tag=$1
    $PY - "$tag" <<'PY'
import json, os, sys
tag = sys.argv[1]
path = f'results_v2/{tag}_summary.json'
if not os.path.isfile(path) or not os.path.isfile(f'results_v2/{tag}_per_slice.csv'):
    raise SystemExit(1)
metrics = json.load(open(path))['metrics']
for cls in ('rectum', 'tumor'):
    for metric in ('dice', 'iou', 'nsd'):
        item = metrics[cls][metric]
        assert 'mean' in item and len(item['ci95']) == 2
PY
}

train_unit() {
    local ratio=$1 seed=$2 tag=supupd_${1}_s${2}
    local val_mode=labeled
    if [ "$ratio" = "1" ]; then
        val_mode=fixed
    fi
    if train_valid "$tag" 2>/dev/null; then
        echo "[skip-train] $tag"
        return 0
    fi
    local attempt=1
    while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
        echo "[train] $tag attempt=$attempt $(date)"
        rm -rf "runs_v2/$tag"
        if run_cmd "$PY" src/train_supervised.py \
            --config "configs/update_matched_sup_${ratio}.json" \
            --data_root "$DATA_ROOT" --out_dir "runs_v2/$tag" \
            --seed "$seed" --val_mode "$val_mode" \
            --crop_size "$CROP" --stage1_ckpt "runs_v2/s1_${ratio}_s${seed}/best.pt" \
            >> "logs/update_matched_${tag}.log" 2>&1 && train_valid "$tag"; then
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 60
    done
    return 1
}

eval_unit() {
    local ratio=$1 seed=$2 tag=supupd_${1}_s${2}
    if summary_valid "$tag" 2>/dev/null; then
        echo "[skip-eval] $tag"
        return 0
    fi
    local attempt=1
    while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
        echo "[eval] $tag attempt=$attempt $(date)"
        if run_cmd "$PY" src/evaluate.py --ckpt "runs_v2/$tag/best.pt" \
            --data_root "$DATA_ROOT" --out "results_v2/$tag" --cache_dir "$CACHE" \
            --stage1_ckpt "runs_v2/s1_${ratio}_s${seed}/best.pt" --crop_size "$CROP" \
            >> "logs/update_matched_${tag}.log" 2>&1 && summary_valid "$tag"; then
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 30
    done
    return 1
}

for ratio in 50 1; do
    for seed in 42 43 44; do
        test -f "runs_v2/s1_${ratio}_s${seed}/best.pt" || {
            echo "missing stage-1 checkpoint for ratio=$ratio seed=$seed"
            exit 1
        }
        tag=supupd_${ratio}_s${seed}
        if summary_valid "$tag" 2>/dev/null; then
            echo "[skip-complete] $tag"
            continue
        fi
        if ! train_unit "$ratio" "$seed"; then
            echo "TRAINING_FAILED $tag"
            exit 1
        fi
        if ! eval_unit "$ratio" "$seed"; then
            echo "EVALUATION_FAILED $tag"
            exit 1
        fi
        echo "[complete] $tag $(date)"
    done
done

echo "UPDATE_MATCHED_CONTROLS_COMPLETE $(date)"
