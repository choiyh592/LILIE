# `analysis/` — EEG Trajectory Phenotype Pipeline

Ordered, independently runnable steps that turn **precomputed LaBraM segment
embeddings** + **preprocessed raw EEG** into covariate-adjusted phenotype
comparisons, implementing `ANALYSIS_PIPELINE_SPEC.md`. Each step writes
intermediate artifacts to disk so the **scree go/no-go gate can halt the run**
before clustering.

## Module → plan-section map

| # | Module | Plan section | Status |
|---|--------|--------------|--------|
| 1 | `assemble.py` | §1 progressions + metadata | **implemented** |
| 2 | `delta.py` | §2 temporal-order model → per-progression delta | **implemented** (needs torch at runtime) |
| 3 | `reduce.py` | §3 PCA + go/no-go gate | **implemented** |
| 4 | `cluster.py` | §4 clustering + k selection | scaffold |
| 5 | `stability.py` | §5 bootstrap stability | scaffold |
| 6 | `qeeg.py` + `connectivity.py` | §6 QEEG features | **FC implemented** (power/PAF TODO) |
| 7 | `phenotype_stats.py` | §7 covariate-adjusted comparison | **FC cluster comparison implemented** |
| 8 | `report.py` | §8 letter outputs | scaffold (panel *a* = scree.png) |
| — | `graded_score.py` | §3 gate STOP route (rank-1 fallback) | scaffold |
| — | `run_all.py` | orchestrator respecting the gate | **implemented** |
| — | `invariants.py` | the 5 correctness invariants | **implemented** |

Modules 4, 5, 8 are scaffolded on purpose: per the brief we **stop at the scree
gate** so the explained-variance curve can be inspected before committing to
clustering. Each scaffold declares its input/output contract and raises a clear
`NotImplementedError` (or halts if the gate routed elsewhere). The
functional-connectivity arm (modules 6–7) is implemented ahead of the gate on
request — see below.

## Functional connectivity across clusters (modules 6–7)

Compares connectivity between the delta-embedding clusters as an **independent,
non-circular validation**: clusters come from the learned change embeddings
(modules 2–4); connectivity is computed from the raw EEG, so a cluster
difference means the learned phenotype has an electrophysiological signature.

- **`connectivity.py`** — VC-robust spectral connectivity (**wPLI** primary,
  **imaginary coherence** sensitivity) via scipy cross-spectra (no hard
  `mne-connectivity` dependency), plus pure-numpy graph metrics (mean strength,
  global efficiency, characteristic path length, weighted clustering, Newman
  modularity). Raw coherence / PLV are intentionally excluded (volume conduction).
- **`qeeg.py`** — per session (identical pipeline), for each method × band:
  global + posterior-subset connectivity and the graph summaries; then **baseline
  value AND within-progression Δ**, keyed by `progression_id` (invariant 3).
  → `qeeg_connectivity.parquet|csv`. *(Spectral-power/PAF/entropy features are
  stubbed as TODO in the same loop.)*
- **`phenotype_stats.py`** — per feature, `feature ~ C(cluster) + dt +
  baseline_severity + APOE4 + ARIA + age` with a **patient random effect**
  (statsmodels mixedlm; GEE fallback) — invariant 4. **BH-FDR** across the
  confirmatory FC family (global + posterior alpha-band wPLI Δ); graph metrics
  reported as exploratory. → `phenotype_stats_fc.parquet|csv`.

Config lives under `qeeg` (channels, `posterior_channels`, `connectivity_methods`,
`connectivity_bands`, `graph_metrics`, epoching) and `phenotype_stats`
(`fc_confirmatory_features`, `fc_exploratory_pattern`, `contrast`). Runtime needs
`scipy` (connectivity), `statsmodels` (module 7), and an HDF5/`.npy` raw-EEG store
keyed by `group_name`. If the gate STOPs (no clusters), the analogue is
correlating ΔFC with the graded change score instead.

## Reused repo classes (confirmed signatures)

- `dataset.datasets.LongitudinalEEGDataset(pairs_df, metadata_csv_path, embeddings_npy_path, n_draws=5)`
  and `create_train_test_splits(...)` — segment sampling; the random time-flip
  is augmentation only.
- `models.models.LILIE(input_dim, embedding_size, num_classes, pool_method, clf_method)`
  with `get_embeddings(x_0, x_1)`; poolers `Delta` (`pool_method="Linear"/"Raw"`),
  `AttentiveDelta` (`"Attentive"`), `NNDelta` (`"NN"`).
- `dataset.dataset_creation.create_data_splits.create_data_splits(csv_path, save_dir, num_splits, seed)`
  — **also builds the consecutive-session progressions** and the subject-wise
  fold CSVs (`longitudinal_pairs_fold_{i}.csv`, 1-indexed).

See "Signature notes vs spec" below for where the repo differs from the brief.

## Correctness invariants (enforced in `invariants.py`, tested in `tests/`)

1. **No patient split across folds** — `assert_disjoint_groups`, `assert_resample_groups`.
2. **Deltas earlier→later oriented** — `oriented_pair`, `assert_earlier_to_later`.
3. **Clustering unit = progression, never segment** — `assert_progression_unit`.
4. **Phenotype stats carry confounds + patient random effect** — `validate_phenotype_model_spec`.
5. **Scree gate can halt before clustering** — `gate_decision`, `should_proceed`.

## Running

```bash
# from the repo root (so `dataset` / `models` import)
python -m analysis.run_all --config analysis/config.yaml     # stops at the gate
python -m analysis.assemble --config analysis/config.yaml    # step 1 only
python -m analysis.delta   --config analysis/config.yaml     # step 2 only (torch)
python -m analysis.reduce  --config analysis/config.yaml     # step 3 + gate

# tests (pytest, or plain scripts if pytest is absent)
pytest analysis/tests/
python analysis/tests/test_invariants.py
ANALYSIS_REPO_ROOT=. python analysis/tests/test_pipeline_smoke.py
```

Edit `config.yaml` first: point `paths.*` at your embeddings metadata/`.npy`,
clinical table, and preprocessed raw EEG. `run.stop_at_gate: true` halts after
module 3; the gate decision + `scree.png` + `gate.json` land in
`paths.output_dir`.

## Signature notes vs spec (differences to be aware of)

- The reused metadata CSV has only `group_name` (`ID_YYYY_MM_DD…`) + `dataset_idx`.
  The clinical covariates (`MMSE, age, APOE4, ARIA`) are **not** in it — module 1
  reads them from a separate `clinical_csv` (mapped in `config.clinical`).
- `create_data_splits` already **builds progressions**; `assemble.py` reuses it
  rather than re-pairing sessions.
- `LILIE` selects `Delta` via `pool_method="Linear"/"Raw"` (not a literal
  `"Delta"`). Also `train.py`'s argparse `choices=["Attentive","Mean","Max"]` do
  **not** match the model's accepted values — a latent repo bug; `delta.py` uses
  the model's real values.
- No `EndToEndLongitudinal` wrapper exists in the repo; embeddings are consumed
  as the precomputed `.npy` + metadata contract above.
