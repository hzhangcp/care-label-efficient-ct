# Label-Efficient Semi-Supervised Segmentation for Rectal Cancer CT — Reproducibility Package

Anonymous code release accompanying the manuscript. Contains all code needed to
reproduce the training/evaluation pipeline and the labeled-subset definitions used
in every experiment. No author-identifying information is included.

## 1. Environment

```bash
pip install -r requirements.txt   # torch, numpy, pandas, scipy, matplotlib, nibabel, SimpleITK
```

Reference environment (used for all reported experiments): see `ENVIRONMENT.md`
(Python 3.10, PyTorch 2.9 + CUDA 12.8). Training requires a CUDA GPU; evaluation,
statistics and the smoke test run on CPU.

## 2. Data

**CARE** (main dataset, 318 train cases / 81 test cases, contrast-enhanced CT with
pixel-level rectum/tumor labels): download from the dataset authors' release
(Google Drive: https://drive.google.com/file/d/1X_JTfD8Ch-IxmG5VHtKk_xGZT336Fl1Q/view,
top-level folder `DataV6/`). License: CC BY-NC 4.0. Source: U-SAM,
Communications Medicine 2025 (DOI 10.1038/s43856-025-00953-0). Direct enumeration
of this release gives 26,656 training and 6,461 test slices. The source article's
398-patient and 26,563-training-slice descriptions differ from the downloaded
files; the released files are authoritative for this package. The original authors
removed slices not containing the rectum before release, so prompt-free inference
here refers to this prefiltered slice collection rather than unfiltered whole-CT
series. Expected layout:

```
data/CARE/train/train_bbox.csv
data/CARE/train/train_npz/case{id}_slice{nnn}.npz   # keys: image (512x512, [0,1]), label ({0,1,2,3})
data/CARE/test/...                                   # same structure
```

The DataV6 files used here were downloaded on 2026-09-15. The included
`care_release_manifest.json` records the direct case/slice counts and SHA-256 hashes
of the sorted split-relative `.npz` path lists. Regenerate them with:

```bash
python src/audit_care_release.py --data_root data/CARE \
    --out results/v2/care_release_manifest.json --downloaded 2026-09-15
```

Labels are remapped `3 -> 2` at load time (`src/dataset.py: remap_label`), following
the U-SAM official dataloader. Then build the disk cache once:

```bash
python src/preprocess_cache.py            # npz -> data/cache/*.npy (train+test)
```

**MSD Task10 Colon** (cross-dataset replication): download from the Medical
Segmentation Decathlon (http://medicaldecathlon.com/) and convert with
`src/preprocess_msd.py` (HU [-100, 400] -> [0,1], binary labels); configs
`configs/msd_colon_*.json`.

## 3. Reproduce the main experiments (three seeds: 42/43/44)

```bash
bash scripts/reproduce_v2.sh            # full matrix: s1/sup/ours/semi/mt/cps x {1..100}%,
                                        # bcp x {1,5,10,25,50}%, single-seed references,
                                        # update-matched controls, subset audit, aggregation
python src/aggregate_v4_rebuttal.py     # cross-ratio tests, effect sizes, exploratory TOST
python src/compute_annotation_budget.py # exact 50%-vs-100% slice/pixel accounting
```

The optimizer-update-matched controls can also be run separately. They preserve the
main labeled subsets and validation rules while extending the supervised cosine
schedule to the Ours update count at 1% and 50% labels:

```bash
bash scripts/run_update_matched_controls.sh
python src/make_table_update_matched.py  # Table S27 + JSON; patient/seed bootstrap CIs
```

Per-run metadata (hyper-parameters, seed, resolved validation mode, slice counts)
are written to `runs_v2/<run>/run_meta.json`; evaluation summaries go to
`results_v2/`. CARE aggregation: `python src/aggregate_v2_seeds.py`.

The secondary MSD matrix uses independent `runs_msd_lb/` and `results_msd_lb/`
roots, labeled-subset validation only, fixed-schedule final checkpoints at 1%, and
the coordinate-aligned Mean Teacher implementation:

```bash
bash scripts/run_msd_labeled_budget.sh
bash scripts/finalize_msd_labeled_budget.sh   # tables, Wilcoxon, curve, 3D mm metrics
```

Optional nnU-Net v2 reference (reviewer-requested): `bash scripts/run_nnunet.sh`
(requires the `nnunetv2` package and a GPU; patient-level split identical to ours).

## 4. Labeled-subset definitions (annotation budgets)

All methods at the same (label ratio, seed) use **identical** patient-level labeled
case sets, produced by one deterministic pipeline
(`dataset.split_train_val` -> `dataset.labeled_subset` -> `dataset.split_labeled_val`).
The exact case lists are exported to `subsets/labeled_ratio{XXX}_s{SEED}.json`
(including the 15% patient-level holdout, labeled/validation split, and MD5 hashes),
and can be regenerated and audited against run metadata with:

```bash
python src/check_subset_consistency.py --data_root data/CARE --runs_dir runs_v2 \
    --report subset_consistency.md --export_subsets subsets/
```

The audit report produced on our runs is included as `SUBSET_CONSISTENCY.md`
(all checks passed: identical labeled sets across sup/semi/MT/CPS/ours/BCP).

Note on the 1% budget: 3 labeled cases only, so validation-based model selection is
impossible; every method falls back to a fixed-epoch protocol (no early stopping,
final-epoch checkpoint reported), see `val_mode=fixed` in the training scripts.

## 5. Included result artifacts

The package includes the aggregated evidence behind the main statistical analyses and
Table S27, without datasets, model checkpoints or training logs:

- `results/v2/agg_seeds.json` and `agg_tests.json`: three-seed CARE metrics and tests;
- `results/v2/cross_ratio_tests.json`, `effect_sizes.json` and `equivalence_tost.json`:
  cross-ratio comparisons, two-way bootstrap effects and exploratory TOST analyses;
- `results/v2/allslice_metrics.json`: false-positive-sensitive pooled-Dice analysis;
- `results/v2/update_matched_controls.json` and `tabS_update_matched.tex`: the complete
  three-seed update-matched analysis and generated Table S27;
- `results_v2/supupd_{1,50}_s{42,43,44}_per_slice.csv`: the six new controls'
  slice-level predictions/metrics used by that analysis.

Absolute workstation paths in the copied all-slice JSON were replaced by relative paths.
These files are evidence snapshots; rerunning the commands above regenerates them from
the released data and checkpoints.

## 6. Smoke test (self-contained, CPU, ~minutes)

```bash
sh scripts/smoke_test.sh    # synthetic dummy data; exercises the full pipeline
                            # (sup/semi/MT/CPS/BCP, cascade, MSD binary path,
                            #  nnU-Net converters, aggregation, subset-consistency selftest)
```

Near-zero metrics on dummy data are expected (deliberate underfitting); the test
only verifies that the whole pipeline runs end to end.

## 7. Code map

```
src/dataset.py            data loading; patient-level splits (leakage-safe); label remap 3->2
src/models.py             LightUNet (1ch -> 3 classes, fully convolutional)
src/common.py             training loop / validation / checkpointing
src/cascade.py            two-stage cascade (centroid crop / paste-back / cascade inference)
src/tensorstore.py        device-resident tensor pipeline
src/train_supervised.py   supervised baseline; --crop_size + --stage1_ckpt = stage-2
src/train_semi.py         pseudo-label self-training (also "ours" with --consistency etc.)
src/train_mt.py           Mean Teacher baseline
src/train_cps.py          Cross Pseudo Supervision baseline
src/train_bcp.py          Bidirectional Copy-Paste baseline (Bai et al., CVPR 2023)
src/evaluate.py           Dice/IoU/NSD + bootstrap 95% CI + paired Wilcoxon (--compare)
src/check_subset_consistency.py  subset audit + subsets/ export (this package's evidence)
src/aggregate_v2_seeds.py three-seed aggregation (mean +/- std, Wilcoxon max-p, Holm)
src/aggregate_v4_rebuttal.py cross-ratio tests, two-way effect sizes, exploratory TOST
src/reanalyze_v2.py       patient-level MSD metrics and automatically generated tables
src/compute_annotation_budget.py exact 50%-vs-100% slice/foreground-pixel budget
scripts/reproduce_v2.sh   exact command matrix for all reported runs
scripts/run_msd_labeled_budget.sh + finalize_msd_labeled_budget.sh  MSD replication
configs/                  all hyper-parameters (nothing hard-coded)
```
