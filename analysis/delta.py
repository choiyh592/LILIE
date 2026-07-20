"""Module 2 - delta: temporal-order model -> per-progression delta vector.

Trains ``LILIE`` on the binary temporal-ordering task using
``LongitudinalEEGDataset`` segment sampling (segments kept separate = the
augmentation), with subject-wise folds from ``create_data_splits`` so no
patient leaks across a split (invariant 1). Reports ordering AUC + bootstrap CI.

Then, per progression, it pushes before x after SEGMENT pairs through the
*trained* pooler and takes the MEDIAN across pairs -> one delta vector per
progression. Orientation is forced earlier -> later (the training loader flips
labels at random for augmentation; we never inherit that flip) and asserted
(invariant 2). The output has exactly one row per progression (invariant 3).

Primary pooler = Delta (repo pool_method "Linear"/"Raw"); AttentiveDelta / NNDelta
are config-switchable sensitivity variants.

torch / lightning are imported lazily so this module only needs them at runtime.

Output (paths.output_dir): deltas.npz with arrays
  progression_id, patient_id, fold, delta (P x D), dt, spread_std, spread_iqr
  + ordering_auc, ordering_auc_ci  (saved alongside in ordering_auc.json)

Run:  python -m analysis.delta --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .config import Config, load_config, add_arg
from . import io
from . import invariants


# --- repo path + metadata helpers (no torch) --------------------------------
def _ensure_repo_on_path(config: Config) -> None:
    repo_root = config.path("repo_root")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _segment_index(metadata_csv: str) -> dict:
    """(patient_id, pd.Timestamp) -> list[dataset_idx], mirroring the dataset."""
    meta = pd.read_csv(metadata_csv)
    parts = meta["group_name"].str.split("_", expand=True)
    meta["patient_id"] = parts[0].astype(int)
    meta["date"] = pd.to_datetime(dict(
        year=parts[1].astype(int), month=parts[2].astype(int), day=parts[3].astype(int)))
    return (meta.groupby(["patient_id", "date"])["dataset_idx"]
            .apply(list).to_dict())


def _pair_indices(before_idx, after_idx, max_pairs, rng):
    """Cartesian before x after segment pairs, optionally subsampled."""
    pairs = [(b, a) for b in before_idx for a in after_idx]
    if max_pairs is not None and len(pairs) > max_pairs:
        sel = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in sel]
    return pairs


# --- training (torch) --------------------------------------------------------
def _train_fold(config: Config, test_idx: int):
    """Train LILIE with fold ``test_idx`` held out; return (model, val_probs, val_labels)."""
    import torch
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger

    from dataset.datasets import create_train_test_splits
    from models.models import LILIE

    d = config["delta"]
    train_loader, test_loader = create_train_test_splits(
        split_csv_dir=config.path("paths", "splits_dir"),
        metadata_csv_path=config.path("paths", "metadata_csv"),
        embeddings_npy_path=config.path("paths", "embeddings_npy"),
        batch_size=int(d["batch_size"]),
        num_workers=int(d["num_workers"]),
        test_idx=test_idx,
        n_draws=int(d["n_draws"]),
    )

    model = LILIE(
        input_dim=int(d["input_dim"]),
        embedding_size=int(d["embedding_size"]),
        num_classes=2,
        pool_method=d["pool_method"],
        clf_method=d["clf_method"],
    )

    # A100 tensor cores: trade a little float32 precision for speed (silences the
    # Lightning tip and speeds training).
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # SINGLE DEVICE ONLY. This pipeline is one orchestrator script (run_all);
    # multi-GPU DDP re-launches the whole script per process, which would re-run
    # assemble/reduce N times and tangle the run. We train the 5 folds
    # sequentially on one device (=> no DDP). Set delta.device_index to pin a
    # specific GPU (cuda:N) via Trainer devices=[N].
    accel = d["accelerator"]
    device_index = d.get("device_index", None)
    cuda_ok = torch.cuda.is_available()
    if device_index is not None and str(device_index).strip() != "":
        if cuda_ok:
            accel, devices = "gpu", [int(device_index)]      # pin cuda:N
            print(f"[delta] using cuda:{int(device_index)}")
        else:
            print(f"[delta] NOTE: device_index={device_index} set but no CUDA "
                  f"available; using CPU.")
            accel, devices = "cpu", 1
    else:
        devices = int(d["devices"]) if str(d["devices"]).isdigit() else 1
        if devices != 1:
            print(f"[delta] NOTE: forcing devices=1 (multi-GPU DDP unsupported for "
                  f"this single-script pipeline; requested {d['devices']}). "
                  f"Use delta.device_index to pick a specific GPU.")
            devices = 1

    logger = CSVLogger(config.out("_lightning_logs"), name=f"fold{test_idx}")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=int(d["max_epochs"]),
        accelerator=accel,
        devices=devices,
        strategy="auto",
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    # Move to CPU for all subsequent inference (held-out predictions + the
    # per-progression deltas in main), so the model and the CPU segment tensors
    # always agree on device regardless of which GPU trained the fold.
    model = model.to("cpu")
    # Collect held-out ordering predictions.
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x0, x1, y in test_loader:
            logits = model.clf(model.get_embeddings(x0, x1))
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p.cpu().numpy())
            labels.append(y.cpu().numpy())
    return model, np.concatenate(probs), np.concatenate(labels)


def _bootstrap_auc_ci(probs, labels, n_boot, seed, alpha=0.05):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    point = float(roc_auc_score(labels, probs))
    n = len(labels)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        boots.append(roc_auc_score(labels[idx], probs[idx]))
    lo, hi = (np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if boots else (np.nan, np.nan))
    return point, float(lo), float(hi)


def _progression_delta(model, embeddings, before_idx, after_idx, pairs):
    """Median segment-pair delta through the trained pooler (before -> after)."""
    import torch
    xb = torch.tensor(np.stack([np.asarray(embeddings[b]) for b, _ in pairs]),
                      dtype=torch.float32)
    xa = torch.tensor(np.stack([np.asarray(embeddings[a]) for _, a in pairs]),
                      dtype=torch.float32)
    with torch.no_grad():
        # Enforced orientation: before = x_0, after = x_1 (never the random flip).
        delta = model.get_embeddings(xb, xa)          # [n_pairs, ...]
    delta = delta.reshape(delta.shape[0], -1).cpu().numpy()  # [n_pairs, D]
    med = np.median(delta, axis=0)
    spread_std = float(np.mean(np.std(delta, axis=0)))
    q75, q25 = np.percentile(delta, [75, 25], axis=0)
    spread_iqr = float(np.mean(q75 - q25))
    return med, spread_std, spread_iqr


def main(config: Config) -> str:
    _ensure_repo_on_path(config)

    d = config["delta"]
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)

    prog = io.read_table(config.out("progressions"))
    prog["before_date"] = pd.to_datetime(prog["before_date"])
    prog["after_date"] = pd.to_datetime(prog["after_date"])
    # Orientation guaranteed at assembly; re-assert here at the delta step.
    invariants.assert_earlier_to_later(prog["before_date"], prog["after_date"])

    seg_index = _segment_index(config.path("paths", "metadata_csv"))
    embeddings = np.load(config.path("paths", "embeddings_npy"), mmap_mode="r")

    num_folds = int(config["assemble"]["num_folds"])
    max_pairs = d["max_segment_pairs"]
    out_of_fold = bool(d["out_of_fold_deltas"])

    # 1. Train per fold; keep models + pooled ordering predictions.
    models, all_probs, all_labels = {}, [], []
    for f in range(1, num_folds + 1):
        print(f"[delta] training LILIE (pool={d['pool_method']}) holding out fold {f}")
        model, probs, labels = _train_fold(config, test_idx=f)
        models[f] = model
        all_probs.append(probs)
        all_labels.append(labels)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc, lo, hi = _bootstrap_auc_ci(all_probs, all_labels, int(d["bootstrap_ci_n"]), seed)
    print(f"[delta] pooled out-of-fold ordering AUC = {auc:.3f} "
          f"[{lo:.3f}, {hi:.3f}] (95% bootstrap)")

    # 2. Per-progression median segment-pair delta (out-of-fold model).
    pids, prog_ids, folds, dts = [], [], [], []
    deltas, spread_std, spread_iqr = [], [], []
    skipped = []
    for _, r in prog.iterrows():
        pid, fold = int(r["patient_id"]), int(r["fold"])
        before_idx = seg_index.get((pid, r["before_date"].normalize()))
        after_idx = seg_index.get((pid, r["after_date"].normalize()))
        if not before_idx or not after_idx:
            skipped.append(r["progression_id"])
            continue
        # earlier -> later orientation is explicit here.
        assert r["before_date"] < r["after_date"], "orientation violated"
        pairs = _pair_indices(before_idx, after_idx, max_pairs, rng)
        model = models[fold] if out_of_fold else models[1]
        med, s_std, s_iqr = _progression_delta(model, embeddings, before_idx, after_idx, pairs)
        deltas.append(med)
        prog_ids.append(r["progression_id"])
        pids.append(pid)
        folds.append(fold)
        dts.append(float(r["dt"]))
        spread_std.append(s_std)
        spread_iqr.append(s_iqr)

    if skipped:
        print(f"[delta] WARNING: {len(skipped)} progression(s) had no matching "
              f"segments and were skipped: {skipped[:5]}")

    delta_mat = np.vstack(deltas)
    # Invariant 3: one row per progression (segments collapsed, not clustered).
    invariants.assert_progression_unit(delta_mat.shape[0], prog_ids)

    out_path = config.out("deltas.npz")
    np.savez(
        out_path,
        progression_id=np.array(prog_ids, dtype=object),
        patient_id=np.array(pids),
        fold=np.array(folds),
        delta=delta_mat,
        dt=np.array(dts),
        spread_std=np.array(spread_std),
        spread_iqr=np.array(spread_iqr),
    )
    io.write_json(
        {"pool_method": d["pool_method"], "ordering_auc": auc,
         "ordering_auc_ci95": [lo, hi], "n_progressions": int(delta_mat.shape[0]),
         "delta_dim": int(delta_mat.shape[1]), "n_skipped": len(skipped)},
        config.out("ordering_auc.json"),
    )
    print(f"[delta] wrote {out_path}  (deltas: {delta_mat.shape})")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 2 - temporal-order model -> deltas")
    add_arg(parser)
    parser.add_argument("--device", type=int, default=None,
                        help="CUDA device index (cuda:N) to train on; "
                             "overrides config delta.device_index")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.raw.setdefault("delta", {})["device_index"] = args.device
    main(cfg)
