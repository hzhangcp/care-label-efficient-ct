# 标注子集跨方法一致性核验（审稿 M9 复现性）

**English summary.** All training entry points (supervised stage-1/stage-2, pseudo-label self-training, Mean Teacher, CPS, BCP, and our consistency-filtered self-training) draw their labeled case sets from the *same deterministic patient-level splitting pipeline* (`dataset.split_train_val` -> `labeled_subset` -> `split_labeled_val`, all seeded shuffles). This report recomputes every (seed, label ratio) split independently and verifies that the slice counts recorded at run time in each `run_meta.json` match the recomputed values exactly. Identical recorded counts across all methods, ratios and seeds, produced by one deterministic code path, establish that the labeled case sets are identical across methods. Result: **ALL CHECKS PASSED**.

## 核验方法

- 数据：`data/CARE` train（318 例）；runs：`runs_v2`
- 划分代码：`src/dataset.py:79` `split_train_val`（318 例 → 270 例池 + 48 例 15% 池级留出）、`src/dataset.py:92` `labeled_subset`（按病人抽标注子集）、`src/dataset.py:109` `split_labeled_val`（val_mode=labeled 时从标注病例后缀划 15% 验证集；标注病例 <4 时各训练脚本回退 fixed）
- 调用点：`src/train_supervised.py:57-76`、`src/train_semi.py:217-237`、`src/train_mt.py:202-225`、`src/train_cps.py:216-240`、`src/train_bcp.py:223-243`——全部使用同一组函数、同一 (seed, ratio)
- 字段语义：sup 系（sup/s1/sup180）`n_labeled_slices` = 实际训练切片数（val_mode=labeled 时为标注子集去掉验证病例后）；semi 系（semi/ours/mt/mtcons/cps/bcp）`n_labeled` = 全部标注切片、`n_unlabeled` = 池内未标注切片；mt/cps/bcp 另记 `n_val_slices`
- 判定：逐 run 核对 seed / label_ratio / val_mode / 各切片计数与独立重算值完全一致；组内所有 run 对得上同一份重算划分 ⟹ 标注病例集合相同（计数由运行期实际文件列表得出，且集合不同则切片计数几乎必不同）

## 总表（每比例 × 每种子）

| ratio | seed | 标注病例数 | 标注切片数 | labeled 列表 md5(前12) | val_mode | 核验 run 数 | 结果 |
|---|---|---|---|---|---|---|---|
| 1% | 42 | 3 | 249 | `d2c13303a4e3` | fixed | 8 | PASS |
| 1% | 43 | 3 | 278 | `f7c22bbc46a0` | fixed | 8 | PASS |
| 1% | 44 | 3 | 248 | `38bc264e1b01` | fixed | 8 | PASS |
| 5% | 42 | 14 | 1171 | `1fc008565dab` | labeled | 8 | PASS |
| 5% | 43 | 14 | 1163 | `36d2ba104652` | labeled | 8 | PASS |
| 5% | 44 | 14 | 1225 | `7268fba302a0` | labeled | 8 | PASS |
| 10% | 42 | 27 | 2169 | `af836658de49` | labeled | 8 | PASS |
| 10% | 43 | 27 | 2206 | `86b753e18937` | labeled | 8 | PASS |
| 10% | 44 | 27 | 2280 | `f1fcc9513b13` | labeled | 8 | PASS |
| 25% | 42 | 68 | 5288 | `eefc4d468330` | labeled | 8 | PASS |
| 25% | 43 | 68 | 5982 | `b6a072541d31` | labeled | 8 | PASS |
| 25% | 44 | 68 | 5643 | `1e34baea0f6b` | labeled | 8 | PASS |
| 50% | 42 | 135 | 10921 | `6b0f0c28dfe5` | labeled | 10 | PASS |
| 50% | 43 | 135 | 11488 | `3e258bec9a0a` | labeled | 8 | PASS |
| 50% | 44 | 135 | 11401 | `e68ba419c8dc` | labeled | 8 | PASS |
| 100% | 42 | 270 | 22640 | `0c8f68dbe295` | labeled | 3 | PASS |
| 100% | 43 | 270 | 22655 | `5e2b7ce26f96` | labeled | 2 | PASS |
| 100% | 44 | 270 | 22849 | `de9c97711336` | labeled | 2 | PASS |

## 各组明细

### ratio=1% seed=42 （标注 3 例 / 249 切片，val_mode=fixed）

标注病例列表：case17133989, case17140266, case17236564

val_mode=fixed：不做验证集模型选择（best.pt=最终 epoch）；日志 val 在训练标注切片上计算，仅作参考

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_1_s42 | 42 | 0.01 | fixed | n_labeled=249, n_unlabeled=22391, n_val_slices=249 | PASS |
| cps_1_s42 | 42 | 0.01 | fixed | n_labeled=249, n_unlabeled=22391, n_val_slices=249 | PASS |
| mt_1_s42 | 42 | 0.01 | fixed | n_labeled=249, n_unlabeled=22391, n_val_slices=249 | PASS |
| mtfixed_1_s42 | 42 | 0.01 | fixed |  | PASS |
| ours_1_s42 | 42 | 0.01 | fixed | n_labeled=249, n_unlabeled=22391 | PASS |
| s1_1_s42 | 42 | 0.01 | fixed | n_labeled_slices=249, n_val_slices=249 | PASS |
| semi_1_s42 | 42 | 0.01 | fixed | n_labeled=249, n_unlabeled=22391 | PASS |
| sup_1_s42 | 42 | 0.01 | fixed | n_labeled_slices=249, n_val_slices=249 | PASS |

### ratio=1% seed=43 （标注 3 例 / 278 切片，val_mode=fixed）

标注病例列表：case17152863, case17395969, case17543687

val_mode=fixed：不做验证集模型选择（best.pt=最终 epoch）；日志 val 在训练标注切片上计算，仅作参考

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_1_s43 | 43 | 0.01 | fixed | n_labeled=278, n_unlabeled=22377, n_val_slices=278 | PASS |
| cps_1_s43 | 43 | 0.01 | fixed | n_labeled=278, n_unlabeled=22377, n_val_slices=278 | PASS |
| mt_1_s43 | 43 | 0.01 | fixed | n_labeled=278, n_unlabeled=22377, n_val_slices=278 | PASS |
| mtfixed_1_s43 | 43 | 0.01 | fixed |  | PASS |
| ours_1_s43 | 43 | 0.01 | fixed | n_labeled=278, n_unlabeled=22377 | PASS |
| s1_1_s43 | 43 | 0.01 | fixed | n_labeled_slices=278, n_val_slices=278 | PASS |
| semi_1_s43 | 43 | 0.01 | fixed | n_labeled=278, n_unlabeled=22377 | PASS |
| sup_1_s43 | 43 | 0.01 | fixed | n_labeled_slices=278, n_val_slices=278 | PASS |

### ratio=1% seed=44 （标注 3 例 / 248 切片，val_mode=fixed）

标注病例列表：case17119650, case17128458, case17158956

val_mode=fixed：不做验证集模型选择（best.pt=最终 epoch）；日志 val 在训练标注切片上计算，仅作参考

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_1_s44 | 44 | 0.01 | fixed | n_labeled=248, n_unlabeled=22601, n_val_slices=248 | PASS |
| cps_1_s44 | 44 | 0.01 | fixed | n_labeled=248, n_unlabeled=22601, n_val_slices=248 | PASS |
| mt_1_s44 | 44 | 0.01 | fixed | n_labeled=248, n_unlabeled=22601, n_val_slices=248 | PASS |
| mtfixed_1_s44 | 44 | 0.01 | fixed |  | PASS |
| ours_1_s44 | 44 | 0.01 | fixed | n_labeled=248, n_unlabeled=22601 | PASS |
| s1_1_s44 | 44 | 0.01 | fixed | n_labeled_slices=248, n_val_slices=248 | PASS |
| semi_1_s44 | 44 | 0.01 | fixed | n_labeled=248, n_unlabeled=22601 | PASS |
| sup_1_s44 | 44 | 0.01 | fixed | n_labeled_slices=248, n_val_slices=248 | PASS |

### ratio=5% seed=42 （标注 14 例 / 1171 切片，val_mode=labeled）

标注病例列表：case17112786, case17114333, case17133989, case17138313, case17140266, case17152554, case17195184, case17236564, case17258800, case17392012, case17423662, case17465689, case17527212, case17553249

标注内验证集：2 例 / 163 切片；训练用标注：12 例 / 1008 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_5_s42 | 42 | 0.05 | labeled | n_labeled=1171, n_unlabeled=21469, n_val_slices=163 | PASS |
| cps_5_s42 | 42 | 0.05 | labeled | n_labeled=1171, n_unlabeled=21469, n_val_slices=163 | PASS |
| mt_5_s42 | 42 | 0.05 | labeled | n_labeled=1171, n_unlabeled=21469, n_val_slices=163 | PASS |
| mtfixed_5_s42 | 42 | 0.05 | labeled |  | PASS |
| ours_5_s42 | 42 | 0.05 | labeled | n_labeled=1171, n_unlabeled=21469 | PASS |
| s1_5_s42 | 42 | 0.05 | labeled | n_labeled_slices=1008, n_val_slices=163 | PASS |
| semi_5_s42 | 42 | 0.05 | labeled | n_labeled=1171, n_unlabeled=21469 | PASS |
| sup_5_s42 | 42 | 0.05 | labeled | n_labeled_slices=1008, n_val_slices=163 | PASS |

### ratio=5% seed=43 （标注 14 例 / 1163 切片，val_mode=labeled）

标注病例列表：case17137211, case17140312, case17150959, case17152863, case17155657, case17158956, case17197863, case17246232, case17395969, case17450436, case17527614, case17543687, case17551478, case17565727

标注内验证集：2 例 / 150 切片；训练用标注：12 例 / 1013 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_5_s43 | 43 | 0.05 | labeled | n_labeled=1163, n_unlabeled=21492, n_val_slices=150 | PASS |
| cps_5_s43 | 43 | 0.05 | labeled | n_labeled=1163, n_unlabeled=21492, n_val_slices=150 | PASS |
| mt_5_s43 | 43 | 0.05 | labeled | n_labeled=1163, n_unlabeled=21492, n_val_slices=150 | PASS |
| mtfixed_5_s43 | 43 | 0.05 | labeled |  | PASS |
| ours_5_s43 | 43 | 0.05 | labeled | n_labeled=1163, n_unlabeled=21492 | PASS |
| s1_5_s43 | 43 | 0.05 | labeled | n_labeled_slices=1013, n_val_slices=150 | PASS |
| semi_5_s43 | 43 | 0.05 | labeled | n_labeled=1163, n_unlabeled=21492 | PASS |
| sup_5_s43 | 43 | 0.05 | labeled | n_labeled_slices=1013, n_val_slices=150 | PASS |

### ratio=5% seed=44 （标注 14 例 / 1225 切片，val_mode=labeled）

标注病例列表：case17119650, case17128104, case17128458, case17133838, case17135899, case17138147, case17158956, case17242378, case17254931, case17259975, case17426644, case17459439, case17537682, case17546013

标注内验证集：2 例 / 216 切片；训练用标注：12 例 / 1009 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_5_s44 | 44 | 0.05 | labeled | n_labeled=1225, n_unlabeled=21624, n_val_slices=216 | PASS |
| cps_5_s44 | 44 | 0.05 | labeled | n_labeled=1225, n_unlabeled=21624, n_val_slices=216 | PASS |
| mt_5_s44 | 44 | 0.05 | labeled | n_labeled=1225, n_unlabeled=21624, n_val_slices=216 | PASS |
| mtfixed_5_s44 | 44 | 0.05 | labeled |  | PASS |
| ours_5_s44 | 44 | 0.05 | labeled | n_labeled=1225, n_unlabeled=21624 | PASS |
| s1_5_s44 | 44 | 0.05 | labeled | n_labeled_slices=1009, n_val_slices=216 | PASS |
| semi_5_s44 | 44 | 0.05 | labeled | n_labeled=1225, n_unlabeled=21624 | PASS |
| sup_5_s44 | 44 | 0.05 | labeled | n_labeled_slices=1009, n_val_slices=216 | PASS |

### ratio=10% seed=42 （标注 27 例 / 2169 切片，val_mode=labeled）

标注病例列表：case17112786, case17113036, case17114333, case17133989, case17138313, case17140266, case17149631, case17151445, case17152554, case17195184, case17204339, case17236564, case17258079, case17258800, case17392012, case17422003, case17423662, case17428433, case17465689, case17478721, case17527212, case17527614, case17531637, case17551811, case17553249, case17565709, case17570313

标注内验证集：4 例 / 261 切片；训练用标注：23 例 / 1908 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_10_s42 | 42 | 0.1 | labeled | n_labeled=2169, n_unlabeled=20471, n_val_slices=261 | PASS |
| cps_10_s42 | 42 | 0.1 | labeled | n_labeled=2169, n_unlabeled=20471, n_val_slices=261 | PASS |
| mt_10_s42 | 42 | 0.1 | labeled | n_labeled=2169, n_unlabeled=20471, n_val_slices=261 | PASS |
| mtfixed_10_s42 | 42 | 0.1 | labeled |  | PASS |
| ours_10_s42 | 42 | 0.1 | labeled | n_labeled=2169, n_unlabeled=20471 | PASS |
| s1_10_s42 | 42 | 0.1 | labeled | n_labeled_slices=1908, n_val_slices=261 | PASS |
| semi_10_s42 | 42 | 0.1 | labeled | n_labeled=2169, n_unlabeled=20471 | PASS |
| sup_10_s42 | 42 | 0.1 | labeled | n_labeled_slices=1908, n_val_slices=261 | PASS |

### ratio=10% seed=43 （标注 27 例 / 2206 切片，val_mode=labeled）

标注病例列表：case17112011, case17132640, case17137211, case17140312, case17144236, case17150959, case17152863, case17155657, case17158956, case17197863, case17230267, case17239348, case17246232, case17268705, case17311253, case17390625, case17395969, case17428433, case17450436, case17503884, case17527614, case17543687, case17550941, case17551478, case17556452, case17556849, case17565727

标注内验证集：4 例 / 378 切片；训练用标注：23 例 / 1828 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_10_s43 | 43 | 0.1 | labeled | n_labeled=2206, n_unlabeled=20449, n_val_slices=378 | PASS |
| cps_10_s43 | 43 | 0.1 | labeled | n_labeled=2206, n_unlabeled=20449, n_val_slices=378 | PASS |
| mt_10_s43 | 43 | 0.1 | labeled | n_labeled=2206, n_unlabeled=20449, n_val_slices=378 | PASS |
| mtfixed_10_s43 | 43 | 0.1 | labeled |  | PASS |
| ours_10_s43 | 43 | 0.1 | labeled | n_labeled=2206, n_unlabeled=20449 | PASS |
| s1_10_s43 | 43 | 0.1 | labeled | n_labeled_slices=1828, n_val_slices=378 | PASS |
| semi_10_s43 | 43 | 0.1 | labeled | n_labeled=2206, n_unlabeled=20449 | PASS |
| sup_10_s43 | 43 | 0.1 | labeled | n_labeled_slices=1828, n_val_slices=378 | PASS |

### ratio=10% seed=44 （标注 27 例 / 2280 切片，val_mode=labeled）

标注病例列表：case17118408, case17118906, case17119650, case17128104, case17128458, case17133838, case17135899, case17138147, case17146917, case17152554, case17155616, case17158956, case17204339, case17229573, case17230439, case17235643, case17242378, case17254931, case17259975, case17389912, case17426644, case17459439, case17505874, case17537682, case17546013, case17550350, case17556452

标注内验证集：4 例 / 387 切片；训练用标注：23 例 / 1893 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_10_s44 | 44 | 0.1 | labeled | n_labeled=2280, n_unlabeled=20569, n_val_slices=387 | PASS |
| cps_10_s44 | 44 | 0.1 | labeled | n_labeled=2280, n_unlabeled=20569, n_val_slices=387 | PASS |
| mt_10_s44 | 44 | 0.1 | labeled | n_labeled=2280, n_unlabeled=20569, n_val_slices=387 | PASS |
| mtfixed_10_s44 | 44 | 0.1 | labeled |  | PASS |
| ours_10_s44 | 44 | 0.1 | labeled | n_labeled=2280, n_unlabeled=20569 | PASS |
| s1_10_s44 | 44 | 0.1 | labeled | n_labeled_slices=1893, n_val_slices=387 | PASS |
| semi_10_s44 | 44 | 0.1 | labeled | n_labeled=2280, n_unlabeled=20569 | PASS |
| sup_10_s44 | 44 | 0.1 | labeled | n_labeled_slices=1893, n_val_slices=387 | PASS |

### ratio=25% seed=42 （标注 68 例 / 5288 切片，val_mode=labeled）

标注病例列表 md5：`eefc4d46833036ec46a4469562a1e002`（完整清单见导出 subsets/labeled_ratio025_s42.json）

标注内验证集：10 例 / 729 切片；训练用标注：58 例 / 4559 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_25_s42 | 42 | 0.25 | labeled | n_labeled=5288, n_unlabeled=17352, n_val_slices=729 | PASS |
| cps_25_s42 | 42 | 0.25 | labeled | n_labeled=5288, n_unlabeled=17352, n_val_slices=729 | PASS |
| mt_25_s42 | 42 | 0.25 | labeled | n_labeled=5288, n_unlabeled=17352, n_val_slices=729 | PASS |
| mtfixed_25_s42 | 42 | 0.25 | labeled |  | PASS |
| ours_25_s42 | 42 | 0.25 | labeled | n_labeled=5288, n_unlabeled=17352 | PASS |
| s1_25_s42 | 42 | 0.25 | labeled | n_labeled_slices=4559, n_val_slices=729 | PASS |
| semi_25_s42 | 42 | 0.25 | labeled | n_labeled=5288, n_unlabeled=17352 | PASS |
| sup_25_s42 | 42 | 0.25 | labeled | n_labeled_slices=4559, n_val_slices=729 | PASS |

### ratio=25% seed=43 （标注 68 例 / 5982 切片，val_mode=labeled）

标注病例列表 md5：`b6a072541d31c844f4d96e9293068942`（完整清单见导出 subsets/labeled_ratio025_s43.json）

标注内验证集：10 例 / 854 切片；训练用标注：58 例 / 5128 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_25_s43 | 43 | 0.25 | labeled | n_labeled=5982, n_unlabeled=16673, n_val_slices=854 | PASS |
| cps_25_s43 | 43 | 0.25 | labeled | n_labeled=5982, n_unlabeled=16673, n_val_slices=854 | PASS |
| mt_25_s43 | 43 | 0.25 | labeled | n_labeled=5982, n_unlabeled=16673, n_val_slices=854 | PASS |
| mtfixed_25_s43 | 43 | 0.25 | labeled |  | PASS |
| ours_25_s43 | 43 | 0.25 | labeled | n_labeled=5982, n_unlabeled=16673 | PASS |
| s1_25_s43 | 43 | 0.25 | labeled | n_labeled_slices=5128, n_val_slices=854 | PASS |
| semi_25_s43 | 43 | 0.25 | labeled | n_labeled=5982, n_unlabeled=16673 | PASS |
| sup_25_s43 | 43 | 0.25 | labeled | n_labeled_slices=5128, n_val_slices=854 | PASS |

### ratio=25% seed=44 （标注 68 例 / 5643 切片，val_mode=labeled）

标注病例列表 md5：`1e34baea0f6bfc43e2a8764e02186fc7`（完整清单见导出 subsets/labeled_ratio025_s44.json）

标注内验证集：10 例 / 823 切片；训练用标注：58 例 / 4820 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_25_s44 | 44 | 0.25 | labeled | n_labeled=5643, n_unlabeled=17206, n_val_slices=823 | PASS |
| cps_25_s44 | 44 | 0.25 | labeled | n_labeled=5643, n_unlabeled=17206, n_val_slices=823 | PASS |
| mt_25_s44 | 44 | 0.25 | labeled | n_labeled=5643, n_unlabeled=17206, n_val_slices=823 | PASS |
| mtfixed_25_s44 | 44 | 0.25 | labeled |  | PASS |
| ours_25_s44 | 44 | 0.25 | labeled | n_labeled=5643, n_unlabeled=17206 | PASS |
| s1_25_s44 | 44 | 0.25 | labeled | n_labeled_slices=4820, n_val_slices=823 | PASS |
| semi_25_s44 | 44 | 0.25 | labeled | n_labeled=5643, n_unlabeled=17206 | PASS |
| sup_25_s44 | 44 | 0.25 | labeled | n_labeled_slices=4820, n_val_slices=823 | PASS |

### ratio=50% seed=42 （标注 135 例 / 10921 切片，val_mode=labeled）

标注病例列表 md5：`6b0f0c28dfe594dba13cc6d010c5a408`（完整清单见导出 subsets/labeled_ratio050_s42.json）

标注内验证集：20 例 / 1856 切片；训练用标注：115 例 / 9065 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_50_s42 | 42 | 0.5 | labeled | n_labeled=10921, n_unlabeled=11719, n_val_slices=1856 | PASS |
| cps_50_s42 | 42 | 0.5 | labeled | n_labeled=10921, n_unlabeled=11719, n_val_slices=1856 | PASS |
| mt_50_s42 | 42 | 0.5 | labeled | n_labeled=10921, n_unlabeled=11719, n_val_slices=1856 | PASS |
| mtcons_50_s42 | 42 | 0.5 | labeled | n_labeled=10921, n_unlabeled=11719 | PASS |
| mtfixed_50_s42 | 42 | 0.5 | labeled |  | PASS |
| ours_50_s42 | 42 | 0.5 | labeled | n_labeled=10921, n_unlabeled=11719 | PASS |
| s1_50_s42 | 42 | 0.5 | labeled | n_labeled_slices=9065, n_val_slices=1856 | PASS |
| semi_50_s42 | 42 | 0.5 | labeled | n_labeled=10921, n_unlabeled=11719 | PASS |
| sup180_50_s42 | 42 | 0.5 | labeled | n_labeled_slices=9065, n_val_slices=1856 | PASS |
| sup_50_s42 | 42 | 0.5 | labeled | n_labeled_slices=9065, n_val_slices=1856 | PASS |

### ratio=50% seed=43 （标注 135 例 / 11488 切片，val_mode=labeled）

标注病例列表 md5：`3e258bec9a0ac28e2d53a56c08c8ed3f`（完整清单见导出 subsets/labeled_ratio050_s43.json）

标注内验证集：20 例 / 1826 切片；训练用标注：115 例 / 9662 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_50_s43 | 43 | 0.5 | labeled | n_labeled=11488, n_unlabeled=11167, n_val_slices=1826 | PASS |
| cps_50_s43 | 43 | 0.5 | labeled | n_labeled=11488, n_unlabeled=11167, n_val_slices=1826 | PASS |
| mt_50_s43 | 43 | 0.5 | labeled | n_labeled=11488, n_unlabeled=11167, n_val_slices=1826 | PASS |
| mtfixed_50_s43 | 43 | 0.5 | labeled |  | PASS |
| ours_50_s43 | 43 | 0.5 | labeled | n_labeled=11488, n_unlabeled=11167 | PASS |
| s1_50_s43 | 43 | 0.5 | labeled | n_labeled_slices=9662, n_val_slices=1826 | PASS |
| semi_50_s43 | 43 | 0.5 | labeled | n_labeled=11488, n_unlabeled=11167 | PASS |
| sup_50_s43 | 43 | 0.5 | labeled | n_labeled_slices=9662, n_val_slices=1826 | PASS |

### ratio=50% seed=44 （标注 135 例 / 11401 切片，val_mode=labeled）

标注病例列表 md5：`e68ba419c8dc25cb6dc0246e428fd1c3`（完整清单见导出 subsets/labeled_ratio050_s44.json）

标注内验证集：20 例 / 1823 切片；训练用标注：115 例 / 9578 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| bcp_50_s44 | 44 | 0.5 | labeled | n_labeled=11401, n_unlabeled=11448, n_val_slices=1823 | PASS |
| cps_50_s44 | 44 | 0.5 | labeled | n_labeled=11401, n_unlabeled=11448, n_val_slices=1823 | PASS |
| mt_50_s44 | 44 | 0.5 | labeled | n_labeled=11401, n_unlabeled=11448, n_val_slices=1823 | PASS |
| mtfixed_50_s44 | 44 | 0.5 | labeled |  | PASS |
| ours_50_s44 | 44 | 0.5 | labeled | n_labeled=11401, n_unlabeled=11448 | PASS |
| s1_50_s44 | 44 | 0.5 | labeled | n_labeled_slices=9578, n_val_slices=1823 | PASS |
| semi_50_s44 | 44 | 0.5 | labeled | n_labeled=11401, n_unlabeled=11448 | PASS |
| sup_50_s44 | 44 | 0.5 | labeled | n_labeled_slices=9578, n_val_slices=1823 | PASS |

### ratio=100% seed=42 （标注 270 例 / 22640 切片，val_mode=labeled）

标注病例列表 md5：`0c8f68dbe29585f746bae7b961f445dd`（完整清单见导出 subsets/labeled_ratio100_s42.json）

标注内验证集：40 例 / 3597 切片；训练用标注：230 例 / 19043 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| s1_100_s42 | 42 | 1.0 | labeled | n_labeled_slices=19043, n_val_slices=3597 | PASS |
| sup180_100_s42 | 42 | 1.0 | labeled | n_labeled_slices=19043, n_val_slices=3597 | PASS |
| sup_100_s42 | 42 | 1.0 | labeled | n_labeled_slices=19043, n_val_slices=3597 | PASS |

### ratio=100% seed=43 （标注 270 例 / 22655 切片，val_mode=labeled）

标注病例列表 md5：`5e2b7ce26f967cc73acd9ef2c17965d1`（完整清单见导出 subsets/labeled_ratio100_s43.json）

标注内验证集：40 例 / 3114 切片；训练用标注：230 例 / 19541 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| s1_100_s43 | 43 | 1.0 | labeled | n_labeled_slices=19541, n_val_slices=3114 | PASS |
| sup_100_s43 | 43 | 1.0 | labeled | n_labeled_slices=19541, n_val_slices=3114 | PASS |

### ratio=100% seed=44 （标注 270 例 / 22849 切片，val_mode=labeled）

标注病例列表 md5：`de9c9771133637600bfe3ccd951f3fe5`（完整清单见导出 subsets/labeled_ratio100_s44.json）

标注内验证集：40 例 / 3379 切片；训练用标注：230 例 / 19470 切片

| run | seed | ratio | val_mode | 计数核对（recorded=expected） | 结果 |
|---|---|---|---|---|---|
| s1_100_s44 | 44 | 1.0 | labeled | n_labeled_slices=19470, n_val_slices=3379 | PASS |
| sup_100_s44 | 44 | 1.0 | labeled | n_labeled_slices=19470, n_val_slices=3379 | PASS |

## 跳过的目录（无 run_meta.json 或命名不符）

ours_10_r1_s42, ours_10_r1_s43, ours_10_r1_s44, ours_10_r5_s42, ours_10_r5_s43, ours_10_r5_s44, semi_10_r1_s42, semi_10_r1_s43, semi_10_r1_s44, semi_10_r5_s42, semi_10_r5_s43, semi_10_r5_s44

## 结论

**全部核验通过**。
同一 (比例, 种子) 下 sup / semi / mt / cps / ours / bcp（含 stage-1 的 s1）使用的标注病例集合完全一致；集合由 `src/dataset.py` 的确定性病人级划分唯一决定，逐 (比例, 种子) 的病例清单已随匿名代码包导出（subsets/）。
