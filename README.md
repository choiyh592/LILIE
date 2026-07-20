# LILIE — Learning-based Inference of Longitudinal Intra-patient EEGs

LILIE learns whether a patient's EEG has changed **between two visits** and, in
the downstream `analysis/` package, turns that learned change into
**covariate-adjusted trajectory phenotypes**. The repo contains two pipelines
that run back-to-back:

1. **Original pipeline** — build longitudinal session pairs, train a
   temporal-ordering model on segment embeddings, and (optionally) explain it
   with input-gradient saliency maps.
2. **Analysis pipeline** (`analysis/`) — take the *trained* model's
   per-progression change vectors and run PCA → a go/no-go scree gate →
   clustering/stability → QEEG → covariate-adjusted phenotype stats → report.

The two share one data contract (precomputed per-segment embeddings + a
metadata CSV) and one splitting utility (`create_data_splits`), so a patient's
sessions are grouped consistently everywhere.

---

## Repository layout

```
LILIE/
├── dataset/
│   ├── datasets.py                     # LongitudinalEEGDataset, create_train_test_splits
│   └── dataset_creation/
│       ├── create_data_splits.py       # session pairs + subject-wise CV folds
│       └── utils/write_logs.py
├── models/
│   ├── models.py                       # LILIE (LightningModule)
│   └── pool/poolers.py                 # Delta / AttentiveDelta / NNDelta (+ Pooler)
├── explain/
│   ├── LaBraM_Goes_Here/               # saliency for the LaBraM backbone
│   │   ├── saliency_map_LaBraM.py
│   │   └── saliency_map_importance_LaBraM.py
│   └── LUNA_Goes_Here/                 # saliency for the LUNA backbone
│       ├── saliency_map_LUNA.py
│       └── saliency_map_importance_LUNA.py
├── train.py                            # training entry point (argparse)
├── analysis/                           # ← downstream phenotyping pipeline (see Part 2)
│   ├── config.yaml  run_all.py  invariants.py
│   ├── assemble.py delta.py reduce.py  # modules 1-3 (implemented)
│   ├── cluster.py stability.py qeeg.py phenotype_stats.py report.py graded_score.py  # 4-8 (scaffold)
│   ├── config.py io.py
│   └── tests/
└── README.md                           # this file
```

---

## Data & embedding contract (shared by both pipelines)

An EEG **foundation-model backbone** (LaBraM or LUNA) is run *first, outside
this repo*, to turn preprocessed raw EEG into **per-segment embeddings**. Both
pipelines consume that output through two files:

- **`embeddings.npy`** — array indexed by `dataset_idx`; each row is one
  segment's embedding (a vector, or a `[num_tokens, dim]` token sequence for the
  attentive/NN poolers). Loaded memory-mapped.
- **`metadata.csv`** — one row per **segment**, with:
  - `group_name` — the **session** id, formatted `ID_YYYY_MM_DD…`
    (patient id + session date). Segments of the same session share it.
  - `dataset_idx` — the row of `embeddings.npy` for that segment.

Explainability (and the analysis QEEG module) additionally read **raw EEG from
HDF5**, where each `group_name` group holds an `eeg` dataset of shape
`[channels, time]`.

> The analysis pipeline also needs a small **clinical table** (`MMSE, age,
> APOE4, ARIA` per session) — see Part 2.

---

# Part 1 — Original pipeline

### Requirements

```bash
pip install torch lightning torchmetrics timm einops pandas numpy
# explainability also: h5py scipy matplotlib safetensors
#   + the backbone repos on PYTHONPATH: LaBraM (and/or BioFoundation for LUNA)
```

### Step 0 — Precompute embeddings (external)

Run your LaBraM/LUNA backbone over the preprocessed raw EEG to produce
`embeddings.npy` + `metadata.csv` in the contract above. This step lives outside
the repo; everything below assumes it is done.

### Step 1 — Create session pairs + subject-wise CV folds

`create_data_splits` sorts each patient's sessions by date, pairs consecutive
sessions (`t_i → t_{i+1}` via `shift(-1)`, keeping **both** pairs for a
3-session patient and dropping only the final unpaired session), then splits by
**patient** into `num_splits` folds so no patient spans two folds. It writes
`longitudinal_pairs_fold_{1..k}.csv`.

```bash
# the module's __main__ has hard-coded example paths; call the function directly:
python -c "from dataset.dataset_creation.create_data_splits import create_data_splits; \
create_data_splits('/path/to/Embeddings/metadata.csv', '/path/to/Embeddings/splits', 5)"
```

### Step 2 — Train the temporal-ordering model

`train.py` builds train/val loaders with `create_train_test_splits`
(`LongitudinalEEGDataset` draws `n_draws` random segment pairs per session pair
each epoch — the segment sampling is the augmentation, and time order is
randomly flipped so the model must *learn* ordering), then fits `LILIE` with a
binary cross-entropy ordering objective and logs `val_auroc`.

```bash
python train.py \
  --split_csv_dir  /path/to/Embeddings/splits \
  --metadata_csv   /path/to/Embeddings/metadata.csv \
  --embeddings_npy /path/to/Embeddings/embeddings.npy \
  --test_idx 1 \                 # which fold is validation (1-indexed)
  --pool_method Attentive \      # Attentive | NN | Linear | Raw  (see note)
  --clf_method  NN \             # NN | Linear
  --input_dim 256 --embedding_size 256 \
  --n_draws 5 --batch_size 32 \
  --max_epochs 1000 --start_saving_epoch 20 \
  --accelerator gpu --devices auto \
  --log_dir ./exp_pl --exp_name eeg_experiment
```

Key hyperparameters (defaults in `train.py`):

| arg | default | meaning |
|-----|---------|---------|
| `--test_idx` | 1 | validation fold (1-indexed); train = all other folds |
| `--pool_method` | Attentive | pooler; maps to `Delta`/`AttentiveDelta`/`NNDelta` |
| `--clf_method` | NN | classifier head (`Mlp` or `nn.Linear`) |
| `--input_dim` / `--embedding_size` | 256 / 256 | backbone dim / pooled dim |
| `--n_draws` | 5 | segment pairs drawn per session pair per epoch |
| `--max_epochs` | 1000 | with `DelayedCheckpoint` saving after `--start_saving_epoch` |
| `--accelerator` / `--devices` | gpu / auto | Lightning trainer target |

The best checkpoint (`monitor=val_auroc`) is written under
`--log_dir/--exp_name`. For cross-validation, run once per `--test_idx` and pool
held-out AUROC.

> **Pooler note.** `LILIE` accepts `pool_method ∈ {Attentive, NN, Linear, Raw}`
> (`Linear`/`Raw` → the plain `Delta = x₁−x₀` pooler). `train.py`'s argparse
> declares `choices=["Attentive","Mean","Max"]`, which do **not** match the
> model — a latent bug. Use `Attentive`/`NN`/`Linear`/`Raw`, and either edit the
> `choices` list or pass a valid value.

### Step 3 — Explainability (saliency maps)

The `explain/` scripts stitch the backbone and the trained LILIE head into an
`EndToEndLongitudinal` wrapper so gradients flow from the ordering logit back to
the **raw EEG**, producing Grad-CAM-style overlays per timepoint. There is a
LaBraM variant and a LUNA variant; each has a `saliency_map_*` (per-sample
heatmaps) and a `saliency_map_importance_*` (aggregated channel/segment
importance) script.

```bash
# LaBraM example (run from a location where LILIE + LaBraM are importable;
# the scripts sys.path.append a repo root — adjust that line to your layout)
python explain/LaBraM_Goes_Here/saliency_map_LaBraM.py \
  --hdf5_path   /path/to/raw_eeg.h5 \
  --labram_ckpt /path/to/labram.pth \
  --lilie_ckpt  /path/to/best-eeg-*.ckpt \
  --group_0 "123_2022_01_15" --group_1 "123_2022_07_15" \
  --target_class 1 --window_size 7600 --patch_size 200 --labram_embed_dim 768
# LUNA variant: explain/LUNA_Goes_Here/saliency_map_LUNA.py (needs BioFoundation.LUNA)
```

Outputs `cam_timepoint_0.png` / `cam_timepoint_1.png`.

> The scripts load LILIE with `embedding_size=128, pool_method="Attentive",
> clf_method="NN"` — match these to whatever you trained in Step 2, or edit the
> `LILIE.load_from_checkpoint(...)` call.

---

# Part 2 — Analysis pipeline (`analysis/`)

Turns the trained ordering model into **trajectory phenotypes**. It is a set of
ordered, independently runnable steps that write intermediate artifacts to disk,
so the **scree go/no-go gate can halt the run before clustering** — you inspect
the explained-variance curve first, then decide whether to cluster.

### Module → pre-registered-plan map

| # | Module | Plan section | Status |
|---|--------|--------------|--------|
| 1 | `assemble.py` | §1 progressions + metadata | **implemented** |
| 2 | `delta.py` | §2 temporal-order model → per-progression delta | **implemented** (needs torch) |
| 3 | `reduce.py` | §3 PCA + go/no-go scree gate | **implemented** |
| 4 | `cluster.py` | §4 clustering + k selection | scaffold |
| 5 | `stability.py` | §5 bootstrap stability | scaffold |
| 6 | `qeeg.py` + `connectivity.py` | §6 QEEG features | **FC implemented** (power/PAF TODO) |
| 7 | `phenotype_stats.py` | §7 covariate-adjusted comparison | **FC cluster comparison implemented** |
| 8 | `report.py` | §8 letter outputs | scaffold (scree = panel *a*) |
| — | `graded_score.py` | §3 gate STOP route (rank-1 fallback) | scaffold |
| — | `run_all.py` | orchestrator honoring the gate | **implemented** |

Modules 4, 5, 8 are deliberately scaffolded (they raise a clear
`NotImplementedError` and declare their I/O contract): the plan is to **stop at
the scree gate**, inspect the variance curve, then implement the branch the gate
selects (clustering, or the graded-score fallback). The **functional-connectivity
arm (modules 6–7) is implemented** ahead of the gate: VC-robust connectivity
(wPLI / imaginary coherence) + graph metrics per session (`connectivity.py`,
`qeeg.py`), compared across clusters with confound + patient-random-effect models
and BH-FDR (`phenotype_stats.py`). This is a non-circular validation — clusters
come from the learned delta embeddings, connectivity from the raw EEG.

### What each implemented step does

- **`assemble.py`** reuses `create_data_splits` to build progressions + folds,
  attaches clinical covariates, derives `dt` (inter-session days) and
  `baseline_severity`, and emits a disjoint `patient_id → fold` map. →
  `progressions.parquet|csv`, `patient_group_map.csv`.
- **`delta.py`** trains a LILIE ordering model **per fold** (subject-wise,
  reusing `create_train_test_splits`), reports pooled out-of-fold AUC + bootstrap
  CI, then for each progression pushes before×after **segment pairs** through the
  *trained* pooler and takes the **median** → one delta vector per progression.
  Orientation is forced earlier→later (never the training loader's random flip).
  → `deltas.npz`, `ordering_auc.json`. *(This pipeline trains its own CV models
  rather than consuming a single `train.py` checkpoint, so each progression's
  delta comes from a model that never saw that patient.)*
- **`reduce.py`** z-scores each dim, fits PCA on the deltas, saves the scree
  curve, and runs the **rank-1 gate**: if PC1 dominates with PC2+ at the noise
  floor (broken-stick), it HALTS clustering and routes to `graded_score`;
  otherwise PROCEED. → `scree.png`, `scree.csv`, `X_pca.npz`, `gate.json`.

### Requirements

```bash
pip install pandas numpy scikit-learn scipy pyyaml matplotlib   # modules 1 & 3
pip install torch lightning torchmetrics timm einops            # module 2 (training)
pip install statsmodels mne pyarrow                             # modules 6 & 7 (when enabled)
```

### Configure

Edit `analysis/config.yaml` — point `paths.metadata_csv`, `paths.embeddings_npy`,
`paths.clinical_csv`, and `paths.raw_eeg_dir` at your data; set `repo_root` to
the repo root so `dataset`/`models` import. The clinical column names are mapped
under `config.clinical`; `baseline_severity_from` chooses `before` (MMSE at the
progression's earlier session, default) or `patient_baseline` (first session).
`run.stop_at_gate: true` halts after module 3.

The clinical table (`clinical_csv`) has one row per `(patient, session)`:

| patient_id | session_date | MMSE | age | APOE4 | ARIA |
|-----------|--------------|------|-----|-------|------|

### Run

```bash
# from the repo root (so `dataset` / `models` import)
python -m analysis.run_all --config analysis/config.yaml     # 1→2→3, stops at gate
python -m analysis.run_all --config analysis/config.yaml --from reduce  # resume at step 3

# individual steps
python -m analysis.assemble --config analysis/config.yaml
python -m analysis.delta    --config analysis/config.yaml    # needs torch
python -m analysis.reduce   --config analysis/config.yaml    # writes scree.png + gate.json
```

Then open `scree.png` / `gate.json` in the output dir. If the gate says
**PROCEED**, implement/enable modules 4-8; if **STOP**, implement
`graded_score.py`.

### Correctness invariants (enforced in `analysis/invariants.py`, tested in `analysis/tests/`)

1. **No patient split across folds** — `assert_disjoint_groups`, `assert_resample_groups`.
2. **Deltas earlier→later oriented** — `oriented_pair`, `assert_earlier_to_later`.
3. **Clustering unit = progression, never segment** — `assert_progression_unit`.
4. **Phenotype stats carry confounds + patient random effect** — `validate_phenotype_model_spec`.
5. **Scree gate can halt before clustering** — `gate_decision`, `should_proceed`.

```bash
pytest analysis/tests/                              # if pytest is installed
python analysis/tests/test_invariants.py            # plain-script fallback (5 invariants)
ANALYSIS_REPO_ROOT=. python analysis/tests/test_pipeline_smoke.py   # module 1 + gate, synthetic
```

### Signature notes (where the repo differs from the analysis spec)

- The embeddings `metadata.csv` has only `group_name` + `dataset_idx`; clinical
  covariates come from a **separate** `clinical_csv` (mapped in `config.clinical`).
- `create_data_splits(csv_path, save_dir, num_splits, seed)` already **builds the
  progressions** (not just folds); `assemble.py` reuses it rather than re-pairing.
- `LILIE` selects the `Delta` pooler via `pool_method="Linear"/"Raw"` (not a
  literal `"Delta"`).
- `LongitudinalEEGDataset` randomly flips time order for augmentation; delta
  computation re-derives earlier→later from dates and never inherits the flip.
- `EndToEndLongitudinal` exists as a **local class inside the `explain/` scripts**
  (wrapping backbone + LILIE for end-to-end gradients), not as an importable
  library class; the analysis pipeline reuses its embedding contract
  (precomputed `.npy` + metadata), not the wrapper itself.

See `analysis/README.md` for the same detail scoped to the sub-package.
