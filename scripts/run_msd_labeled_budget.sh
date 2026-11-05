#!/usr/bin/env bash
# MSD secondary replication: labeled-only validation, fixed-epoch fallback at 1%,
# and the coordinate-aligned Mean Teacher implementation. Never overwrites runs/ or results/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=${PYTHON:-python}
DATA=data/MSD/colon_npz
RUN=runs_msd_lb
RES=results_msd_lb
LOG=logs/msd_labeled_budget
CROP=256
mkdir -p "$RUN" "$RES" "$LOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/queue.log"; }

run_logged() {
    local name=$1
    shift
    local attempt
    for attempt in 1 2 3 4 5; do
        log "START $name (attempt $attempt)"
        if "$@" > "$LOG/${name}.log" 2>&1; then
            log "DONE  $name"
            return 0
        fi
        log "RETRY $name failed; see $LOG/${name}.log"
        sleep $((attempt * 30))
    done
    log "FATAL $name failed after 5 attempts"
    return 1
}

train_if_needed() {
    local tag=$1
    shift
    if [ -f "$RUN/$tag/best.pt" ] && [ -f "$RUN/$tag/run_meta.json" ]; then
        log "SKIP  $tag (checkpoint exists)"
        return 0
    fi
    run_logged "$tag" "$@"
}

eval_stage2() {
    local method=$1
    local ratio=$2
    local tag="msd_${method}_${ratio}"
    if [ ! -f "$RES/${tag}_summary.json" ] || [ ! -f "$RES/${tag}_per_slice.csv" ]; then
        run_logged "eval_${tag}" "$PY" src/evaluate.py \
            --ckpt "$RUN/$tag/best.pt" --data_root "$DATA" --out "$RES/$tag" \
            --stage1_ckpt "$RUN/msd_s1_${ratio}/best.pt" --crop_size "$CROP"
    else
        log "SKIP  eval_$tag (summary exists)"
    fi
    run_logged "compare_${method}_${ratio}" "$PY" src/evaluate.py --compare \
        "$RES/msd_sup_${ratio}_summary.json" "$RES/${tag}_summary.json"
}

for r in 1 5 10 25 50; do
    CFG="configs/msd_colon_${r}.json"
    S1="msd_s1_${r}"
    SUP="msd_sup_${r}"

    train_if_needed "$S1" "$PY" src/train_supervised.py --config "$CFG" \
        --data_root "$DATA" --out_dir "$RUN/$S1" --seed 42 --val_mode labeled
    train_if_needed "$SUP" "$PY" src/train_supervised.py --config "$CFG" \
        --data_root "$DATA" --out_dir "$RUN/$SUP" --seed 42 --val_mode labeled \
        --crop_size "$CROP" --stage1_ckpt "$RUN/$S1/best.pt"
    eval_stage2 sup "$r"

    train_if_needed "msd_semi_${r}" "$PY" src/train_semi.py --config "$CFG" \
        --data_root "$DATA" --out_dir "$RUN/msd_semi_${r}" \
        --teacher_ckpt "$RUN/$SUP/best.pt" --rounds 3 --seed 42 --val_mode labeled \
        --crop_size "$CROP" --stage1_ckpt "$RUN/$S1/best.pt"
    eval_stage2 semi "$r"

    train_if_needed "msd_ours_${r}" "$PY" src/train_semi.py --config "$CFG" \
        --data_root "$DATA" --out_dir "$RUN/msd_ours_${r}" \
        --teacher_ckpt "$RUN/$SUP/best.pt" --rounds 3 --consistency \
        --seed 42 --val_mode labeled --crop_size "$CROP" \
        --stage1_ckpt "$RUN/$S1/best.pt"
    eval_stage2 ours "$r"

    for method in mt cps; do
        train_if_needed "msd_${method}_${r}" "$PY" "src/train_${method}.py" \
            --config "$CFG" --data_root "$DATA" --out_dir "$RUN/msd_${method}_${r}" \
            --seed 42 --val_mode labeled --crop_size "$CROP" \
            --stage1_ckpt "$RUN/$S1/best.pt"
        eval_stage2 "$method" "$r"
    done
done

"$PY" - <<'PY'
import json, os, torch
ratios = (1, 5, 10, 25, 50)
methods = ('s1', 'sup', 'semi', 'ours', 'mt', 'cps')
for r in ratios:
    for m in methods:
        tag = f'msd_{m}_{r}'
        meta_path = f'runs_msd_lb/{tag}/run_meta.json'
        ckpt_path = f'runs_msd_lb/{tag}/best.pt'
        assert os.path.isfile(ckpt_path), ckpt_path
        meta = json.load(open(meta_path))
        assert meta['val_mode_requested'] == 'labeled', (tag, meta.get('val_mode_requested'))
        expected = 'fixed' if r == 1 else 'labeled'
        assert meta['val_mode'] == expected, (tag, meta.get('val_mode'), expected)
        if m == 'mt':
            assert meta.get('aug_aligned') is True, (tag, meta.get('aug_aligned'))
        if r == 1:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            expected_epoch = 60 if m in ('semi', 'ours') else 100
            assert ckpt.get('epoch') == expected_epoch, (tag, ckpt.get('epoch'))
    for m in ('sup', 'semi', 'ours', 'mt', 'cps'):
        tag = f'msd_{m}_{r}'
        summary = json.load(open(f'results_msd_lb/{tag}_summary.json'))
        assert 'runs_msd_lb/' in summary['ckpt'], (tag, summary['ckpt'])
        assert os.path.isfile(f'results_msd_lb/{tag}_per_slice.csv')
with open('results_msd_lb/QUEUE_COMPLETE', 'w') as f:
    f.write('30 training units and 25 evaluations validated: labeled-only validation; fixed at 1%; MT aligned.\n')
print('MSD_LABELED_BUDGET_ALL_VALIDATED')
PY
log "ALL DONE"
