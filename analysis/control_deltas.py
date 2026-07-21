"""Control-cohort deltas - apply the TREATED-trained models to untreated patients.

To ask "are the directional phenotypes Leqembi-specific?", the control deltas
must be produced by the SAME pooler as the treated ones. This loads each fold's
best checkpoint (saved by delta.py early-stopping under out/_ckpt/foldN/best.ckpt),
computes the median segment-pair delta per control progression with each fold
model, and ENSEMBLES (averages) across folds. The control patients were never in
training, so every fold model is validly out-of-sample for them.

Reuses delta.py's segment indexing and per-progression delta. Needs torch.

Config (add a `control` block):
  paths.control_metadata_csv, paths.control_embeddings_npy, paths.control_splits_dir

Output: control_deltas.npz (same schema as deltas.npz).

Run:  python -m analysis.control_deltas --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .config import Config, load_config, add_arg
from . import delta as delta_mod


def _fold_date(row, which):
    return pd.Timestamp(int(row[f"Year_{which}"]), int(row[f"Month_{which}"]),
                        int(row[f"Day_{which}"]))


def _build_control_progressions(config: Config) -> pd.DataFrame:
    delta_mod._ensure_repo_on_path(config)
    from dataset.dataset_creation.create_data_splits import create_data_splits
    meta = config.path("paths", "control_metadata_csv")
    splits = config.path("paths", "control_splits_dir")
    os.makedirs(splits, exist_ok=True)
    create_data_splits(meta, splits, num_splits=None, seed=int(config["seed"]))
    pairs = pd.read_csv(os.path.join(splits, "longitudinal_pairs.csv"))
    pairs["patient_id"] = pairs["ID"].astype(int)
    pairs["before_date"] = pairs.apply(lambda r: _fold_date(r, "Before"), axis=1)
    pairs["after_date"] = pairs.apply(lambda r: _fold_date(r, "After"), axis=1)
    pairs["progression_id"] = (pairs["patient_id"].astype(str)
                               + "__" + pairs["before_date"].dt.strftime("%Y%m%d")
                               + "__" + pairs["after_date"].dt.strftime("%Y%m%d"))
    pairs["dt"] = (pairs["after_date"] - pairs["before_date"]).dt.days
    return pairs


def _load_fold_models(config: Config):
    import torch  # noqa: F401
    from models.models import LILIE
    d = config["delta"]
    n_folds = int(config["assemble"]["num_folds"])
    models = []
    for f in range(1, n_folds + 1):
        ck = os.path.join(config.out("_ckpt"), f"fold{f}", "best.ckpt")
        if not os.path.exists(ck):
            continue
        models.append(LILIE.load_from_checkpoint(
            ck, map_location="cpu",
            input_dim=int(d["input_dim"]), embedding_size=int(d["embedding_size"]),
            num_classes=2, pool_method=d["pool_method"], clf_method=d["clf_method"]).eval())
    if not models:
        raise SystemExit("[control_deltas] no fold checkpoints under "
                         f"{config.out('_ckpt')} - run delta.py with early_stopping first.")
    return models


def main(config: Config) -> str:
    delta_mod._ensure_repo_on_path(config)
    d = config["delta"]
    rng = np.random.default_rng(int(config["seed"]))
    prog = _build_control_progressions(config)
    seg_index = delta_mod._segment_index(config.path("paths", "control_metadata_csv"))
    embeddings = np.load(config.path("paths", "control_embeddings_npy"), mmap_mode="r")
    models = _load_fold_models(config)
    max_pairs = d["max_segment_pairs"]

    prog_ids, pids, dts, deltas, spread_std, spread_iqr, skipped = [], [], [], [], [], [], []
    for _, r in prog.iterrows():
        pid = int(r["patient_id"])
        bi = seg_index.get((pid, r["before_date"].normalize()))
        ai = seg_index.get((pid, r["after_date"].normalize()))
        if not bi or not ai:
            skipped.append(r["progression_id"]); continue
        pairs = delta_mod._pair_indices(bi, ai, max_pairs, rng)
        # ensemble the per-fold median deltas
        meds, s_std, s_iqr = [], [], []
        for m in models:
            med, ss, si = delta_mod._progression_delta(m, embeddings, bi, ai, pairs)
            meds.append(med); s_std.append(ss); s_iqr.append(si)
        deltas.append(np.mean(meds, axis=0))
        prog_ids.append(r["progression_id"]); pids.append(pid); dts.append(float(r["dt"]))
        spread_std.append(float(np.mean(s_std))); spread_iqr.append(float(np.mean(s_iqr)))

    if skipped:
        print(f"[control_deltas] skipped {len(skipped)} (no matching segments)")
    delta_mat = np.vstack(deltas)
    out = config.out("control_deltas.npz")
    np.savez(out, progression_id=np.array(prog_ids, dtype=object),
             patient_id=np.array(pids), fold=np.zeros(len(pids), int),
             delta=delta_mat, dt=np.array(dts),
             spread_std=np.array(spread_std), spread_iqr=np.array(spread_iqr))
    print(f"[control_deltas] {delta_mat.shape[0]} control progressions "
          f"(ensemble of {len(models)} fold models) -> {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control-cohort deltas (apply trained models)")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
