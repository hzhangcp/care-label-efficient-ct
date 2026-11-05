#!/bin/sh
# 冒烟测试：假数据 + 极小训练量，验证全链路代码可跑通。
# ⚠️ 指标全 0 是预期（欠拟合），只验证流程不验证效果。
# 改任何代码后必须重跑本脚本保持全绿（AGENTS.md 红线 1）。
set -e
cd "$(dirname "$0")/.."

DUMMY=data_dummy
SMOKE_RUNS=runs/smoke
SMOKE_RESULTS=results/smoke
rm -rf "$SMOKE_RUNS" "$SMOKE_RESULTS"

echo "=== [1/6] 生成假数据 ==="
python src/make_dummy.py --out "$DUMMY" --n_train_cases 8 --n_test_cases 3 --slices_per_case 4

echo "=== [2/6] 数据校验 ==="
python src/verify_data.py --data_root "$DUMMY" --out_dir outputs/verify_smoke || true
# 注：假数据病例数与官方不符属预期，verify_data 会打警告但不失败
python src/audit_care_release.py --data_root "$DUMMY" \
    --out "$SMOKE_RESULTS/care_release_manifest.json" > /dev/null
python -c "
import json
m = json.load(open('$SMOKE_RESULTS/care_release_manifest.json'))
assert m['total_cases'] == 11 and m['total_slices'] == 44, m
assert all(len(x['file_list_sha256']) == 64 for x in m['splits'].values())
print('CARE 发布清单审计检查 OK')"

echo "=== [3/6] 全监督训练（10% 标注，2 epoch） ==="
python src/train_supervised.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --out_dir "$SMOKE_RUNS/sup_10" \
    --epochs 2 --batch_size 4 --device cpu

echo "=== [4/6] 半监督自训练（1 轮 × 1 epoch） ==="
python src/train_semi.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --teacher_ckpt "$SMOKE_RUNS/sup_10/best.pt" \
    --out_dir "$SMOKE_RUNS/semi_10" --rounds 1 --epochs 1 \
    --batch_size 4 --device cpu

echo "=== [5/6] 评估 + 配对检验 ==="
python src/evaluate.py --ckpt "$SMOKE_RUNS/sup_10/best.pt" --data_root "$DUMMY" \
    --out "$SMOKE_RESULTS/sup_10" --batch_size 4 --device cpu
python src/evaluate.py --ckpt "$SMOKE_RUNS/semi_10/best.pt" --data_root "$DUMMY" \
    --out "$SMOKE_RESULTS/semi_10" --batch_size 4 --device cpu
python src/evaluate.py --compare "$SMOKE_RESULTS/sup_10_summary.json" "$SMOKE_RESULTS/semi_10_summary.json"

echo "=== [6/6] 结果汇总 ==="
python src/aggregate_results.py --results_dir "$SMOKE_RESULTS" --out paper/smoke_table.md

echo "=== [附加] 定性可视化 ==="
python src/visualize_predictions.py --data_root "$DUMMY" \
    --ckpt sup="$SMOKE_RUNS/sup_10/best.pt" --ckpt semi="$SMOKE_RUNS/semi_10/best.pt" \
    --out_dir outputs/qualitative_smoke --n_per_tier 1 --device cpu

echo "=== [附加2] 级联 stage-2（sup + semi + 级联评估） ==="
python src/train_supervised.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --out_dir "$SMOKE_RUNS/sup_10_s2" \
    --epochs 1 --batch_size 4 --device cpu \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt"
python src/evaluate.py --ckpt "$SMOKE_RUNS/sup_10_s2/best.pt" --data_root "$DUMMY" \
    --out "$SMOKE_RESULTS/sup_10_s2" --batch_size 4 --device cpu \
    --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt" --crop_size 64
python src/train_semi.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --teacher_ckpt "$SMOKE_RUNS/sup_10_s2/best.pt" \
    --out_dir "$SMOKE_RUNS/semi_10_s2" --rounds 1 --epochs 1 \
    --batch_size 4 --device cpu \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt" \
    --thr_rectum 0.95 --thr_tumor 0.8 \
    --small_cc_area 50 --small_cc_thr 0.5 --consistency
python src/evaluate.py --ckpt "$SMOKE_RUNS/semi_10_s2/best.pt" --data_root "$DUMMY" \
    --out "$SMOKE_RESULTS/semi_10_s2" --batch_size 4 --device cpu \
    --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt" --crop_size 64

echo "=== [附加3] Mean Teacher（级联 stage-2，1 epoch） ==="
python src/train_mt.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --out_dir "$SMOKE_RUNS/mt_10_s2" \
    --epochs 1 --batch_size 4 --device cpu --rampup_epochs 2 \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt"
test -f "$SMOKE_RUNS/mt_10_s2/best.pt"
python src/evaluate.py --ckpt "$SMOKE_RUNS/mt_10_s2/best.pt" --data_root "$DUMMY" \
    --out "$SMOKE_RESULTS/mt_10_s2" --batch_size 4 --device cpu \
    --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt" --crop_size 64
python -c "
import json, sys, torch
sys.path.insert(0, 'src')
import tensorstore as ts
# 2026-10-13 MT 坐标对齐修复：augment_pair 双视图共享同一组几何参数
img = torch.rand(4, 1, 32, 32)
torch.manual_seed(0)
s, t = ts.augment_pair(img)
assert s.shape == t.shape == img.shape
assert not torch.equal(s, t), 'student 应额外有强度抖动'
_orig = ts._apply_intensity
ts._apply_intensity = lambda x: x
torch.manual_seed(0)
s2, t2 = ts.augment_pair(img)
ts._apply_intensity = _orig
assert torch.equal(s2, t2), '去除强度抖动后两视图应逐像素一致（共享几何参数）'
# augment_batch 原行为不变：图像与标签同步几何变换
lab = torch.zeros(4, 32, 32, dtype=torch.long); lab[:, 8:24, 8:24] = 1
a_img, a_lab = ts.augment_batch(img.clone(), lab.clone())
assert a_img.shape == img.shape and a_lab.shape == lab.shape
assert set(a_lab.unique().tolist()) <= {0, 1}, 'nearest 插值不应产生新标签值'
m = json.load(open('$SMOKE_RUNS/mt_10_s2/run_meta.json'))
assert m.get('aug_aligned') is True, 'run_meta.json 缺 aug_aligned 标记'
print('MT 坐标对齐（augment_pair）检查 OK')"

echo "=== [附加4] CPS（级联 stage-2，1 epoch） ==="
python src/train_cps.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --out_dir "$SMOKE_RUNS/cps_10_s2" \
    --epochs 1 --batch_size 4 --device cpu --rampup_epochs 2 \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt"
test -f "$SMOKE_RUNS/cps_10_s2/best.pt"
python src/evaluate.py --ckpt "$SMOKE_RUNS/cps_10_s2/best.pt" --data_root "$DUMMY" \
    --out "$SMOKE_RESULTS/cps_10_s2" --batch_size 4 --device cpu \
    --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt" --crop_size 64

echo "=== [附加4b] BCP（级联 stage-2，1 epoch） ==="
python src/train_bcp.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --out_dir "$SMOKE_RUNS/bcp_10_s2" \
    --epochs 1 --batch_size 4 --device cpu --rampup_epochs 2 \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt"
test -f "$SMOKE_RUNS/bcp_10_s2/best.pt"
python src/evaluate.py --ckpt "$SMOKE_RUNS/bcp_10_s2/best.pt" --data_root "$DUMMY" \
    --out "$SMOKE_RESULTS/bcp_10_s2" --batch_size 4 --device cpu \
    --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt" --crop_size 64

echo "=== [附加5] MSD 二分类路径（迷你假数据，sup/semi/mt/cps + 级联评估） ==="
DUMMY_MSD=data_dummy_msd
rm -rf "$DUMMY_MSD"
python src/make_dummy.py --out "$DUMMY_MSD" --binary \
    --n_train_cases 8 --n_test_cases 3 --slices_per_case 4
python src/train_supervised.py --config configs/msd_colon_10.json \
    --data_root "$DUMMY_MSD" --out_dir "$SMOKE_RUNS/msd_sup_10" \
    --epochs 1 --batch_size 4 --device cpu
python src/train_supervised.py --config configs/msd_colon_10.json \
    --data_root "$DUMMY_MSD" --out_dir "$SMOKE_RUNS/msd_sup_10_s2" \
    --epochs 1 --batch_size 4 --device cpu \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/msd_sup_10/best.pt"
python src/train_semi.py --config configs/msd_colon_10.json \
    --data_root "$DUMMY_MSD" --teacher_ckpt "$SMOKE_RUNS/msd_sup_10_s2/best.pt" \
    --out_dir "$SMOKE_RUNS/msd_semi_10_s2" --rounds 1 --epochs 1 \
    --batch_size 4 --device cpu --consistency \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/msd_sup_10/best.pt"
python src/train_mt.py --config configs/msd_colon_10.json \
    --data_root "$DUMMY_MSD" --out_dir "$SMOKE_RUNS/msd_mt_10_s2" \
    --epochs 1 --batch_size 4 --device cpu --rampup_epochs 2 \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/msd_sup_10/best.pt"
python src/train_cps.py --config configs/msd_colon_10.json \
    --data_root "$DUMMY_MSD" --out_dir "$SMOKE_RUNS/msd_cps_10_s2" \
    --epochs 1 --batch_size 4 --device cpu --rampup_epochs 2 \
    --crop_size 64 --stage1_ckpt "$SMOKE_RUNS/msd_sup_10/best.pt"
test -f "$SMOKE_RUNS/msd_semi_10_s2/best.pt"
test -f "$SMOKE_RUNS/msd_mt_10_s2/best.pt"
test -f "$SMOKE_RUNS/msd_cps_10_s2/best.pt"
python src/evaluate.py --ckpt "$SMOKE_RUNS/msd_sup_10_s2/best.pt" \
    --data_root "$DUMMY_MSD" --out "$SMOKE_RESULTS/msd_sup_10_s2" \
    --batch_size 4 --device cpu \
    --stage1_ckpt "$SMOKE_RUNS/msd_sup_10/best.pt" --crop_size 64
python -c "
import json
m = json.load(open('$SMOKE_RESULTS/msd_sup_10_s2_summary.json'))['metrics']
assert list(m) == ['tumor'], f'二分类评估应只报 tumor 类，实际: {list(m)}'
print('二分类评估类别检查 OK:', list(m))"
python -c "
import sys; sys.path.insert(0, 'src')
from dataset import list_slices, oversample_positive_files
fs = list_slices('$DUMMY_MSD', 'train')
out = oversample_positive_files(fs, 10)
assert len(out) > len(fs), '过采样未生效'
assert oversample_positive_files(fs, 1) == fs, 'k=1 应原样返回'
print('过采样检查 OK:', len(fs), '->', len(out))"

echo "=== [附加6] nnU-Net 数据转换 + 评估适配器（假数据，CPU） ==="
SMOKE_NNUNET=outputs/nnunet_smoke
rm -rf "$SMOKE_NNUNET" results_v2/smoke
python src/prepare_nnunet_care.py --data_root "$DUMMY" --out "$SMOKE_NNUNET" --num_workers 4
# 用 GT 当预测验证 eval_nnunet 的病例->切片映射（所有指标应全 1.0）
python - "$DUMMY" <<'EOF'
import os, sys
import numpy as np
import SimpleITK as sitk
sys.path.insert(0, 'src')
from dataset import list_slices, remap_label
from prepare_nnunet_care import group_by_case
pred_dir = 'outputs/nnunet_smoke/fake_pred'
os.makedirs(pred_dir, exist_ok=True)
for cid, files in group_by_case(list_slices(sys.argv[1], 'test')).items():
    vol = np.stack([remap_label(np.load(p)['label']).astype(np.uint8) for p in files])
    sitk.WriteImage(sitk.GetImageFromArray(vol), os.path.join(pred_dir, f'{cid}.nii.gz'))
print('假预测（=GT）已写入', pred_dir)
EOF
python src/eval_nnunet.py --pred_dir "$SMOKE_NNUNET/fake_pred" --data_root "$DUMMY" \
    --out results_v2/smoke/nnunet_fake
python -c "
import json
m = json.load(open('results_v2/smoke/nnunet_fake_summary.json'))['metrics']
for cls, d in m.items():
    for k, v in d.items():
        assert v['n'] == 0 or abs(v['mean'] - 1.0) < 1e-9, (cls, k, v)
print('eval_nnunet GT-as-pred 检查 OK（指标全 1.0）')"

echo "=== [附加6] --seed 覆盖 / --val_mode（labeled / fixed / 自动回退） ==="
# labeled：100% 档假数据有 7 个标注病例（>=4），应正常走 labeled 划分
python src/train_supervised.py --config configs/label_ratio_100.json \
    --data_root "$DUMMY" --out_dir "$SMOKE_RUNS/sup_100_labeled" \
    --epochs 1 --batch_size 4 --device cpu --seed 43 --val_mode labeled
# 回退：10% 档假数据仅 1 个标注病例（<4），请求 labeled 应回退 fixed
python src/train_supervised.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --out_dir "$SMOKE_RUNS/sup_10_fallback" \
    --epochs 1 --batch_size 4 --device cpu --seed 44 --val_mode labeled
# fixed：最终 epoch 模型存 best.pt；同时覆盖 target_updates 与 val_interval
python src/train_supervised.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --out_dir "$SMOKE_RUNS/sup_10_fixed" \
    --target_updates 2 --val_interval 2 --batch_size 4 --device cpu --seed 44 --val_mode fixed
python src/train_semi.py --config configs/label_ratio_10.json \
    --data_root "$DUMMY" --teacher_ckpt "$SMOKE_RUNS/sup_10/best.pt" \
    --out_dir "$SMOKE_RUNS/semi_10_fixed" --rounds 1 --epochs 1 \
    --batch_size 4 --device cpu --seed 44 --val_mode fixed
python -c "
import csv, json, sys, torch
sys.path.insert(0, 'src')
from dataset import labeled_subset, list_slices, split_train_val
m = json.load(open('$SMOKE_RUNS/sup_100_labeled/run_meta.json'))
assert m['val_mode'] == 'labeled' and m['seed'] == 43, m
assert m['n_val_slices'] > 0 and m['n_labeled_slices'] > 0
m = json.load(open('$SMOKE_RUNS/sup_10_fallback/run_meta.json'))
assert m['val_mode'] == 'fixed' and m['val_mode_requested'] == 'labeled', m
m = json.load(open('$SMOKE_RUNS/sup_10_fixed/run_meta.json'))
assert m['val_mode'] == 'fixed', m
assert m['resolved_target_updates'] == 2 and m['actual_optimizer_updates'] == 2, m
rows = list(csv.reader(open('$SMOKE_RUNS/sup_10_fixed/log.csv')))
assert rows[1][2] == '' and rows[-1][2] != '', 'val_interval 应跳过中间验证并在末轮验证'
ck = torch.load('$SMOKE_RUNS/sup_10_fixed/best.pt', map_location='cpu', weights_only=False)
assert ck['epoch'] == 2, 'fixed 模式 best.pt 应为 target_updates 对应的最终 epoch 模型'
m = json.load(open('$SMOKE_RUNS/semi_10_fixed/run_meta.json'))
assert m['val_mode'] == 'fixed' and m['seed'] == 44, m
# 标注子集随 seed 变化（多 seed 实验的前提）
pool, _ = split_train_val(list_slices('$DUMMY', 'train'), 0.15, 42)
a = set(labeled_subset(pool, 0.5, 42)); b = set(labeled_subset(pool, 0.5, 43))
assert a != b, '不同 seed 应抽到不同标注子集'
print('val_mode/seed 检查 OK')"
test -f "$SMOKE_RUNS/semi_10_fixed/best.pt"

echo "=== [附加7] analyze_fp_3d（FP 负担 + 3D Dice/HD95，假数据 CPU） ==="
python src/analyze_fp_3d.py --tasks AB --data_root "$DUMMY" --cache_dir "" \
    --mask_dir "$SMOKE_RESULTS/masks" --out_dir "$SMOKE_RESULTS/v2" \
    --model smoke_sup="$SMOKE_RUNS/sup_10_s2/best.pt":"$SMOKE_RUNS/sup_10/best.pt" \
    --crop_size 64 --batch_size 4 --device cpu --n_boot 200
# 第二次运行应直接加载已保存 masks（不重复推理）
python src/analyze_fp_3d.py --tasks A --data_root "$DUMMY" --cache_dir "" \
    --mask_dir "$SMOKE_RESULTS/masks" --out_dir "$SMOKE_RESULTS/v2" \
    --model smoke_sup="$SMOKE_RUNS/sup_10_s2/best.pt":"$SMOKE_RUNS/sup_10/best.pt" \
    --crop_size 64 --batch_size 4 --device cpu --n_boot 200
python -c "
import json
a = json.load(open('$SMOKE_RESULTS/v2/fp_burden.json'))
b = json.load(open('$SMOKE_RESULTS/v2/metrics3d.json'))
assert 'smoke_sup' in a and 'smoke_sup' in b
for cls in ('rectum', 'tumor'):
    assert a['smoke_sup'] and b['smoke_sup'][cls]['dice']['n'] > 0, cls
print('analyze_fp_3d 冒烟检查 OK')"

echo "=== [附加8] 分层统计 + 耗时表（假数据，CPU） ==="
python src/stratified_stats.py --results_dir "$SMOKE_RESULTS" \
    --tags sup_10,semi_10 --baseline sup_10 --compare semi_10 --extra_pair "" \
    --cache_dir "" --data_root "$DUMMY" \
    --out_json "$SMOKE_RESULTS/v2/stratified_stats.json" \
    --out_tex "$SMOKE_RESULTS/tab_stratified.tex" \
    --out_tex_quartiles "" --n_boot 200
python src/timing_table.py --runs_dir "$SMOKE_RUNS" \
    --out_tex "$SMOKE_RESULTS/tab_timing.tex" \
    --out_json "$SMOKE_RESULTS/timing.json" \
    --bench_s1 "$SMOKE_RUNS/sup_10/best.pt" \
    --bench_s2 "$SMOKE_RUNS/sup_10_s2/best.pt" \
    --crop_size 64 --bench_cache_dir "" --bench_data_root "$DUMMY" \
    --bench_n 4 --device cpu
python -c "
import json
s = json.load(open('$SMOKE_RESULTS/v2/stratified_stats.json'))
assert 'paper' in s['stratifications'], '缺 paper 分层'
assert 'boundary_analysis' in s, '缺边界分析'
t = json.load(open('$SMOKE_RESULTS/timing.json'))
assert t['runs'] and t['inference_bench']['cascade_ms_per_slice_mean'] > 0
print('分层统计/耗时表冒烟检查 OK')"

echo "=== [附加9] 三种子聚合统计（aggregate_v2_seeds，合成 CSV，CPU） ==="
AGG_SMOKE=results_v2/smoke/agg_seeds
rm -rf "$AGG_SMOKE" results/smoke/v2agg
python - "$AGG_SMOKE" <<'EOF'
import json, os, sys
import numpy as np
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
# 6 病例 x 4 切片；rectum 出现于 j>=1，tumor 出现于 j>=2
names = [f"case{2000 + i}_slice{j:03d}.npz" for i in range(6) for j in range(4)]
gt = {n: [int(n.split("slice")[1][:3]) >= 1, int(n.split("slice")[1][:3]) >= 2] for n in names}
json.dump(gt, open(os.path.join(out, "gt.json"), "w"))
base = {"sup": (0.50, 0.60), "ours": (0.55, 0.65), "semi": (0.45, 0.55),
        "mt": (0.48, 0.58), "refsup": (0.52, 0.62), "mtfixed": (0.60, 0.70)}
stems = [f"{m}_1_s{s}" for m in ("sup", "ours", "semi", "mt") for s in (42, 43, 44)] \
    + ["refsup_1_s42"] + [f"mtfixed_1_s{s}" for s in (42, 43, 44)]
for stem in stems:
    method = stem.split("_")[0]
    sd = int(stem.rsplit("_s", 1)[1])
    rows = ["slice,dice_rectum,iou_rectum,nsd_rectum,dice_tumor,iou_tumor,nsd_tumor"]
    for n in names:
        hr, ht = gt[n]
        vals = []
        for ci, has in enumerate((hr, ht)):
            if has:
                v = base[method][ci] + 0.01 * (sd - 43) + rng.uniform(-0.05, 0.05)
                vals += [v, v * 0.8, v * 0.9]
            else:
                vals += ([0.0, 0.0, 0.0] if rng.random() < 0.5 else [np.nan] * 3)
        rows.append(n + "," + ",".join("nan" if np.isnan(v) else f"{v:.6f}" for v in vals))
    open(os.path.join(out, f"{stem}_per_slice.csv"), "w").write("\n".join(rows) + "\n")
    # mtfixed 替换 gate 要求 summary+json 与 csv 双全（内容从略，gate 只看存在性）
    if method == "mtfixed":
        json.dump({"tag": stem}, open(os.path.join(out, f"{stem}_summary.json"), "w"))
print("合成聚合冒烟数据 OK:", len(stems), "stems,", len(names), "slices")
EOF
python src/aggregate_v2_seeds.py --smoke --results_dir "$AGG_SMOKE" \
    --gt_json "$AGG_SMOKE/gt.json" --out_dir results/smoke/v2agg \
    --man_dir results/smoke/v2agg --sr_dir results/smoke/v2agg/sr \
    --stats_md results/smoke/v2agg/v3_stats.md
python - <<'EOF'
import json, os
a = json.load(open('results/smoke/v2agg/agg_seeds.json'))
assert set(a['experiments']) >= {'sup_1', 'ours_1', 'refsup_1_s42'}
g = a['experiments']['ours_1']['classes']['tumor']['metrics']['dice']
assert g['std'] is not None and len(g['seed_means']) == 3
t = json.load(open('results/smoke/v2agg/agg_tests.json'))
assert t['headline']['result']['tumor/dice']['n_cases'] == 6
fam = t['families']['tumor/dice']
assert '1%/sup vs ours' in fam and 'holm_p' in fam['1%/sup vs ours']
# sup_vs_others + sup50_vs_sup100 补充检验
fam2 = t['sup_vs_others']['families']['tumor/dice']
assert '1%/sup vs semi' in fam2 and '1%/sup vs mt' in fam2
assert 'holm_p' in fam2['1%/sup vs semi']
assert t['sup50_vs_sup100']['result']['tumor/dice']['p_max'] > 0
tex = open('results/smoke/v2agg/table1_care_v3.tex').read()
assert 'RefSup' in tex and '\\dagger' in tex
tex8 = open('results/smoke/v2agg/tabS8b_sup_vs_others.tex').read()
assert 'Supervised vs ST' in tex8 and 'sup\\_1 vs refsup\\_1\\_s42' in tex8
tex5 = open('results/smoke/v2agg/tabS5_wilcoxon_care_v3.tex').read()
assert 'Mean Teacher vs Ours' in tex5 and 'tabS:wilcoxon_care_v3' in tex5
assert os.path.exists('results/smoke/v2agg/sr/tabS5_wilcoxon_care_v3.tex')
md = open('results/smoke/v2agg/v3_stats.md').read()
assert 'ours_1 vs sup_1' in md and 'Supervised vs 各半监督' in md
# mtfixed 替换规则：gate 开（3/3 齐全）→ mt_1 stems 应切换到 mtfixed
assert a['experiments']['mt_1']['stems'] == [f'mtfixed_1_s{s}' for s in (42, 43, 44)]
gmt = a['experiments']['mt_1']['classes']['tumor']['metrics']['dice']
assert abs(gmt['mean'] - 0.70) < 0.02, f"mt_1 应取 mtfixed 值≈0.70，实际 {gmt['mean']}"
assert 'mtfixed' in a['mt_source']
# gate 关（缺文件）→ 保留旧版
import sys, tempfile
sys.path.insert(0, 'src')
from aggregate_v2_seeds import resolve_mt_stems
with tempfile.TemporaryDirectory() as td:
    ok, note = resolve_mt_stems({'mt': [1]}, td)
    assert not ok and '旧版' in note
print('aggregate_v2_seeds 冒烟检查 OK')
EOF

echo "=== [附加9b] 1% 逐种子表（make_table_1pct_perseed，复用附加9聚合 json，CPU） ==="
python src/make_table_1pct_perseed.py --smoke \
    --seeds_json results/smoke/v2agg/agg_seeds.json \
    --tests_json results/smoke/v2agg/agg_tests.json \
    --out results/smoke/v2agg/tabS_1pct_perseed.tex
python - <<'EOF'
tex = open('results/smoke/v2agg/tabS_1pct_perseed.tex').read()
assert 'Panel A: tumor' in tex and 'Panel B: normal rectum' in tex
assert 'Supervised' in tex and 'ST ' in tex and 'Ours' in tex
assert 'tabS:perseed_1pct' in tex
# 合成数据 sup 均值高于半监督 -> 差值应为负、无方向反转 dag
import re
panel_a = tex.split('Panel A: tumor')[1].split('Panel B')[0]
assert '\\dagger' not in panel_a or 'exceeds' in tex
print('make_table_1pct_perseed 冒烟检查 OK')
EOF

echo "=== [附加9c] 配对轮数消融表（make_table_rounds_paired，合成 CSV，CPU） ==="
RP_SMOKE=results/smoke/rounds_paired
rm -rf "$RP_SMOKE"
python - "$RP_SMOKE" <<'EOF'
import json, os, sys
import numpy as np
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
names = [f"case{3000 + i}_slice{j:03d}.npz" for i in range(6) for j in range(4)]
gt = {n: [int(n.split("slice")[1][:3]) >= 1, int(n.split("slice")[1][:3]) >= 2] for n in names}
json.dump(gt, open(os.path.join(out, "gt.json"), "w"))
base = {("semi",1):(0.50,0.60), ("semi",3):(0.47,0.57), ("semi",5):(0.44,0.54),
        ("ours",1):(0.51,0.61), ("ours",3):(0.49,0.60), ("ours",5):(0.48,0.59)}
def stem(m, r, s): return f"{m}_10_s{s}" if r == 3 else f"{m}_10_r{r}_s{s}"
for (m, r), (br, bt) in base.items():
    for s in (42, 43, 44):
        rows = ["slice,dice_rectum,iou_rectum,nsd_rectum,dice_tumor,iou_tumor,nsd_tumor"]
        for n in names:
            hr, ht = gt[n]; vals = []
            for ci, has in enumerate((hr, ht)):
                if has:
                    v = (br, bt)[ci] + 0.005 * (s - 43) + rng.uniform(-0.03, 0.03)
                    vals += [v, v * 0.8, v * 0.9]
                else:
                    vals += ([0.0, 0.0, 0.0] if rng.random() < 0.5 else [np.nan] * 3)
            rows.append(n + "," + ",".join("nan" if np.isnan(v) else f"{v:.6f}" for v in vals))
        open(os.path.join(out, f"{stem(m, r, s)}_per_slice.csv"), "w").write("\n".join(rows) + "\n")
print("合成配对轮数冒烟数据 OK")
EOF
python src/make_table_rounds_paired.py --smoke --results_dir "$RP_SMOKE" \
    --gt_json "$RP_SMOKE/gt.json" \
    --out_json "$RP_SMOKE/rounds_paired.json" --out_tex "$RP_SMOKE/tabS_rounds_paired.tex"
python - <<'EOF'
import json
res = json.load(open('results/smoke/rounds_paired/rounds_paired.json'))
assert 'semi_r1' in res['cells'] and 'ours_r5' in res['cells']
assert 'ours_r1_vs_semi_r1/tumor' in res['tests']
assert 'semi_r1_vs_r5/tumor' in res['tests']
tex = open('results/smoke/rounds_paired/tabS_rounds_paired.tex').read()
assert 'Panel A: tumor' in tex and 'r1 vs r3' in tex
print('make_table_rounds_paired 冒烟检查 OK')
EOF
python src/make_table_update_matched.py --selftest \
    --out_json results/smoke/update_matched/selftest.json \
    --out_tex results/smoke/update_matched/selftest.tex

echo "=== [附加10] v4 rebuttal 统计（aggregate_v4_rebuttal，合成 CSV，CPU） ==="
# 在附加9的合成数据上补充 cps_1 + sup_100 三种子 stem（bcp 故意缺失 → NA 路径）
python - "$AGG_SMOKE" <<'EOF'
import json, os, sys
import numpy as np
out = sys.argv[1]
gt = json.load(open(os.path.join(out, "gt.json")))
names = sorted(gt)
rng = np.random.default_rng(1)
base = {"cps": (0.47, 0.57), "sup_100": (0.53, 0.63)}
stems = [f"{m}_s{s}" for m in ("cps_1", "sup_100") for s in (42, 43, 44)]
for stem in stems:
    key = "sup_100" if stem.startswith("sup_100") else "cps"
    sd = int(stem.rsplit("_s", 1)[1])
    rows = ["slice,dice_rectum,iou_rectum,nsd_rectum,dice_tumor,iou_tumor,nsd_tumor"]
    for n in names:
        hr, ht = gt[n]
        vals = []
        for ci, has in enumerate((hr, ht)):
            if has:
                v = base[key][ci] + 0.01 * (sd - 43) + rng.uniform(-0.05, 0.05)
                vals += [v, v * 0.8, v * 0.9]
            else:
                vals += ([0.0, 0.0, 0.0] if rng.random() < 0.5 else [np.nan] * 3)
        rows.append(n + "," + ",".join("nan" if np.isnan(v) else f"{v:.6f}" for v in vals))
    open(os.path.join(out, f"{stem}_per_slice.csv"), "w").write("\n".join(rows) + "\n")
print("v4 冒烟补充 stems OK:", stems)
EOF
python src/aggregate_v4_rebuttal.py --smoke --n_boot 200 \
    --results_dir "$AGG_SMOKE" --gt_json "$AGG_SMOKE/gt.json" \
    --out_dir results/smoke/v4 --man_dir results/smoke/v4 \
    --sr_dir results/smoke/v4/sr --stats_md results/smoke/v4/v4_stats.md \
    --snippet results/smoke/v4/snippet_stats_methods.txt
python - <<'EOF'
import json, os
cr = json.load(open('results/smoke/v4/cross_ratio_tests.json'))
assert 'F3' in cr['family_definition'] and cr['reference'] == 'sup_100'
d = cr['results']['dice']
assert d['tumor/ours_1']['holm_p'] is not None
assert d['tumor/bcp_1'] is None, 'bcp 缺失应记 NA'
assert d['rectum/cps_1']['n_cases'] == 6
eq = json.load(open('results/smoke/v4/equivalence_tost.json'))
prim = eq['results']['primary']['ours_1 vs sup_100']
for cls in ('rectum', 'tumor'):
    e = prim[cls]
    assert e['n_cases'] == 6 and len(e['per_seed']) == 3
    for k in ('1pp', '2pp', '3pp'):
        assert 'equivalent' in e['tost'][k] and 'tost_p' in e['tost'][k]
    assert len(e['pooled']['ci90']) == 2 and len(e['pooled']['ci95']) == 2
    assert len(e['pooled_twoway']['ci90']) == 2
    assert len(e['pooled_twoway']['ci95']) == 2
assert 'post hoc' in eq['protocol'] and 'pooled_twoway' in eq['protocol']
assert eq['results']['secondary']['bcp_1 vs sup_100'] is None
assert eq['results']['secondary']['semi_1 vs sup_100'] is not None
es = json.load(open('results/smoke/v4/effect_sizes.json'))
g = es['groups']
assert g['ours_vs_st_same_ratio']['tumor']['1%']['ci95_twoway_boot']
assert g['sup_vs_semis_same_ratio']['rectum']['1%/cps']['n_seeds'] == 3
assert g['cross_ratio_vs_sup100']['tumor']['1%/ours']['diff_mean'] is not None
for f in ('tabS7b_cross_ratio.tex', 'tab_equiv_tost.tex', 'tab_effect_sizes.tex'):
    t = open(f'results/smoke/v4/{f}').read()
    assert '\\begin{table}' in t and 'NA' in t or f != 'tabS7b_cross_ratio.tex'
    assert os.path.exists(f'results/smoke/v4/sr/{f}'), f'sr 副本缺失: {f}'
md = open('results/smoke/v4/v4_stats.md').read()
assert 'TOST' in md and 'F3' in md and '头条结论' in md
assert 'MT 对齐修复前后对比' in md and '翻转' in md
mt_section = md.split('## MT 对齐修复前后对比', 1)[1]
mt_row = next(l for l in mt_section.splitlines() if l.startswith('| 1% |'))
mt_vals = [float(x.split('±')[0]) for x in mt_row.split('|')[2:6]]
# 列序必须是 旧tumor/新tumor/旧rectum/新rectum
# （合成基元组为 (rectum,tumor)，表格百分数基准为 58/70/48/60）
for v, target in zip(mt_vals, (58.0, 70.0, 48.0, 60.0)):
    assert abs(v - target) < 6.0, (mt_row, mt_vals)
sn = open('results/smoke/v4/snippet_stats_methods.txt').read()
assert 'F1' in sn and 'F3' in sn and 'two-way cluster' in sn
print('aggregate_v4_rebuttal 冒烟检查 OK')
EOF

echo "=== [附加12] 全切片口径指标（eval_allslice，假数据 CPU） ==="
ALLSLICE_SMOKE=results/smoke/allslice
rm -rf "$ALLSLICE_SMOKE"
# 级联 ckpt 推理 -> 逐切片 tp/fp/fn 计数
python src/eval_allslice.py --infer_one "$SMOKE_RUNS/sup_10_s2/best.pt:$SMOKE_RUNS/sup_10/best.pt" \
    --crop_size 64 --data_root "$DUMMY" --cache_dir "" \
    --out_csv "$ALLSLICE_SMOKE/smoke_sup_s42_counts.csv" --batch_size 4 --device cpu
# nnU-Net 路径（GT 当预测 -> pooled/micro Dice 应全 1.0，FP=0）
python src/eval_allslice.py --infer_nnunet --pred_dir "$SMOKE_NNUNET/fake_pred" \
    --data_root "$DUMMY" --out_csv "$ALLSLICE_SMOKE/nnunet_fake_counts.csv"
python src/eval_allslice.py --aggregate --counts_dir "$ALLSLICE_SMOKE" \
    --out_json "$ALLSLICE_SMOKE/allslice_metrics.json" \
    --out_tex "$ALLSLICE_SMOKE/tab_allslice.tex" --no_sr_copy
python -c "
import json
import pandas as pd
d = json.load(open('$ALLSLICE_SMOKE/allslice_metrics.json'))
fake = d['experiments']['nnunet_fake']['classes']
for cls in ('rectum', 'tumor'):
    c = fake[cls]
    assert abs(c['patient_pooled_dice']['mean'] - 1.0) < 1e-9, cls
    assert abs(c['micro_dice']['mean'] - 1.0) < 1e-9, cls
    assert c['n_fp_slices'] == 0, cls
    # 假数据每切片都含 rectum（无空 GT 切片）-> rectum FP 率未定义为 nan，属预期
    if cls == 'tumor':
        assert c['fp_slice_rate']['mean'] == 0.0, cls
sup = json.load(open('$ALLSLICE_SMOKE/allslice_metrics.json'))['experiments']['smoke_sup']
assert sup['single_seed'] and sup['classes']['tumor']['n_cases_gt_positive'] > 0
n_test = len(pd.read_csv('$ALLSLICE_SMOKE/smoke_sup_s42_counts.csv'))
assert n_test == 12, n_test  # 3 病例 x 4 切片
print('eval_allslice 冒烟检查 OK')"

echo "=== [附加10] 标注子集一致性核验自测（check_subset_consistency --selftest） ==="
python src/check_subset_consistency.py --selftest

echo "=== [附加11] 分层统计三种子模式（stratified_stats --seeds，合成 CSV，CPU） ==="
STRAT3_SMOKE=results_v2/smoke/strat3
rm -rf "$STRAT3_SMOKE" results/smoke/strat3
python - "$STRAT3_SMOKE" <<'EOF'
import json, os, sys
import numpy as np
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
# 8 病例 x 8 切片；肿瘤面积跨 4 个层（0 = 空 GT，剔除）
areas = {}
names = []
for i in range(8):
    for j, a in enumerate([0, 10, 30, 100, 150, 500, 800, 3000]):
        n = f"case{3000 + i}_slice{j:03d}.npz"
        names.append(n)
        areas[n] = a
json.dump(areas, open(os.path.join(out, "areas.json"), "w"))
base = {"sup": 0.55, "semi": 0.56, "mt": 0.57, "cps": 0.56, "bcp": 0.58,
        "ours": 0.62}
for tag in ["sup_100", "sup_50", "semi_50", "mt_50", "cps_50", "bcp_50",
            "ours_50"]:
    method = tag.split("_")[0]
    for sd in (42, 43, 44):
        rows = ["slice,dice_rectum,iou_rectum,nsd_rectum,"
                "dice_tumor,iou_tumor,nsd_tumor"]
        for n in names:
            if areas[n] > 0:
                v = base[method] + 0.01 * (sd - 43) + rng.uniform(-0.03, 0.03)
                t = f"{v:.6f},{v * 0.8:.6f},{v * 0.9:.6f}"
            else:
                t = "nan,nan,nan"
            rows.append(f"{n},0.5,0.4,0.3,{t}")
        open(os.path.join(out, f"{tag}_s{sd}_per_slice.csv"), "w").write(
            "\n".join(rows) + "\n")
print("合成分层三种子冒烟数据 OK:", len(names), "slices")
EOF
python src/stratified_stats.py --seeds 42,43,44 --results_dir_v2 "$STRAT3_SMOKE" \
    --areas_json "$STRAT3_SMOKE/areas.json" \
    --out_json_3seed results/smoke/strat3/stratified_stats_3seed.json \
    --out_tex_3seed results/smoke/strat3/tab_stratified_3seed.tex \
    --sr_dir "" --n_boot 200
python - <<'EOF'
import json
d = json.load(open('results/smoke/strat3/stratified_stats_3seed.json'))
r = d['results']
assert r['seeds'] == [42, 43, 44] and len(r['strata']) == 4
for s in r['strata']:
    h = s['headline']
    assert h['comparison'] == 'sup_100 vs ours_50' and 'holm_p' in h
    assert len(h['p_per_seed']) == 3
    e = h['effect']
    assert e['diff_ci95'][0] <= e['diff_mean'] <= e['diff_ci95'][1]
    assert e['diff_ci95_twoway'][0] <= e['diff_mean'] <= e['diff_ci95_twoway'][1]
    assert len(s['tests_vs_ours']) == 4
    assert all('holm_p' in t for t in s['tests_vs_ours'])
    assert s['methods']['ours_50']['std'] is not None
tex = open('results/smoke/strat3/tab_stratified_3seed.tex').read()
assert 'three training seeds 42/43/44' in tex and 'BCP' in tex
assert '$\\leq$50 px' in tex and '51--200 px' in tex
print('分层统计三种子冒烟检查 OK')
EOF

echo "=== [附加10] eval_hd95（2D HD95 px + NSD 多 tau；3D HD95 mm 假 spacing，CPU） ==="
python src/eval_hd95.py --ckpt "$SMOKE_RUNS/sup_10_s2/best.pt" \
    --data_root "$DUMMY" --out "$SMOKE_RESULTS/hd95_sup_10_s2" \
    --batch_size 4 --device cpu --n_boot 50 \
    --stage1_ckpt "$SMOKE_RUNS/sup_10/best.pt" --crop_size 64
python src/eval_hd95.py --mode 3d --ckpt "$SMOKE_RUNS/msd_sup_10_s2/best.pt" \
    --data_root "$DUMMY_MSD" --out "$SMOKE_RESULTS/hd95_msd_sup_10_s2" \
    --batch_size 4 --device cpu --n_boot 50 --spacing 0.8 0.8 5.0 \
    --stage1_ckpt "$SMOKE_RUNS/msd_sup_10/best.pt" --crop_size 64
python -c "
import json
s = json.load(open('$SMOKE_RESULTS/hd95_sup_10_s2_summary.json'))
assert s['mode'] == '2d' and s['unit'] == 'pixel' and s['n_cases'] == 3
for cls in ('rectum', 'tumor'):
    m = s['metrics'][cls]
    assert 'hd95_px_case_p95' in m and 'nsd_tau1' in m and 'nsd_tau3' in m
s3 = json.load(open('$SMOKE_RESULTS/hd95_msd_sup_10_s2_summary.json'))
assert s3['mode'] == '3d' and s3['unit'] == 'mm' and s3['n_cases'] == 3
m = s3['metrics']['tumor']
assert 'hd95_mm' in m and 'nsd3d_tau2mm' in m and 'dice3d' in m
print('eval_hd95 冒烟检查 OK')"
echo "=== [附加10b] stage-1 偏移缓存路径（precompute_s1_offsets + --offsets_npz） ==="
python src/precompute_s1_offsets.py --ckpt "$SMOKE_RUNS/sup_10/best.pt" \
    --tag smoke_s1 --data_root "$DUMMY" --cache_dir "" \
    --out_dir "$SMOKE_RESULTS/s1_cache" --crop_size 64 --batch_size 4 --device cpu
python src/eval_hd95.py --ckpt "$SMOKE_RUNS/sup_10_s2/best.pt" \
    --offsets_npz "$SMOKE_RESULTS/s1_cache/smoke_s1_offsets.npz" --crop_size 64 \
    --data_root "$DUMMY" --out "$SMOKE_RESULTS/hd95_sup_10_s2_cached" \
    --batch_size 4 --device cpu --n_boot 50
python -c "
import json
a = json.load(open('$SMOKE_RESULTS/hd95_sup_10_s2_summary.json'))['metrics']
b = json.load(open('$SMOKE_RESULTS/hd95_sup_10_s2_cached_summary.json'))['metrics']
for cls in ('rectum', 'tumor'):
    for k in ('hd95_px_case_p95', 'nsd_tau2'):
        va, vb = a[cls][k]['mean'], b[cls][k]['mean']
        if va == va and vb == vb:  # nan（全剔除）跳过
            assert abs(va - vb) < 1e-9, (cls, k, va, vb)
print('偏移缓存 == 全级联 冒烟检查 OK')"

echo "=== [附加12] 覆盖率分解 + fig_mechanism_v2（假数据，CPU） ==="
python src/quantify_mechanism.py --coverage_breakdown --smoke --device cpu --batch_size 4 \
    --runs_dir "$SMOKE_RUNS"
python src/make_fig_mechanism_v2.py --smoke --device cpu --batch_size 4
python -c "
import json
d = json.load(open('results/smoke/coverage_breakdown_smoke.json'))
assert d['crop_area'], '缺 crop_area'
ca = next(iter(d['crop_area'].values()))
assert ca['crop_pixels'] > 0 and 0 < ca['crop_area_frac'] <= 1
ext = d['extended_breakdown_seed']
assert ext, '缺 extended 口径'
e = next(iter(ext.values()))
for v in ('raw', 'conf', 'full'):
    vv = e['variants'][v]
    assert 0 <= vv['coverage_in_crop'] <= 1.0001, v
    assert 'retention' in vv['per_class']['tumor'], v
import os
assert os.path.exists('outputs/fig_mechanism_v2_smoke.png')
assert os.path.exists('results/smoke/snippet_coverage_smoke.txt')
print('覆盖率分解/机制图 v2 冒烟检查 OK')"

echo "=== [附加13] 标注内容预算（compute_annotation_budget 自测） ==="
python src/compute_annotation_budget.py --selftest

echo ""
echo "SMOKE TEST PASSED"
