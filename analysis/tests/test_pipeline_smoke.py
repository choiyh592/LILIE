"""Integration smoke test for modules 1 (assemble) and 3 (reduce/gate).

Exercises the real ``create_data_splits`` reuse path and the scree gate on
synthetic data (no torch). Proves: assemble builds progressions with a disjoint
patient->fold map (invariant 1), keeps both progressions for 3-session
patients, and the gate halts on a rank-1 delta space / proceeds on a
multi-component one (invariant 5).

Requires the LILIE repo importable; set ANALYSIS_REPO_ROOT to the repo root
(defaults to '.'). Runnable under pytest or as a plain script.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from analysis.config import Config
from analysis import assemble, reduce as reduce_mod, io, invariants

REPO_ROOT = os.environ.get("ANALYSIS_REPO_ROOT", ".")


def _make_synth(root: str):
    """Write synthetic metadata.csv, embeddings.npy, clinical.csv under root."""
    rng = np.random.default_rng(0)
    D = 16
    rows, clin_rows, embs = [], [], []
    idx = 0
    for pid in range(1, 13):
        n_sessions = 3 if pid % 3 == 0 else 2          # some 3-session patients
        base_year = 2022
        for s in range(n_sessions):
            month = 1 + s * 4
            date = pd.Timestamp(year=base_year, month=month, day=15)
            group = f"{pid}_{date.year}_{date.month:02d}_{date.day:02d}"
            for _seg in range(3):                       # 3 segments per session
                rows.append({"group_name": group, "dataset_idx": idx})
                embs.append(rng.normal(size=D))
                idx += 1
            clin_rows.append({"patient_id": pid, "session_date": date.date().isoformat(),
                              "MMSE": 26 - s, "age": 70 + pid % 5,
                              "APOE4": pid % 3, "ARIA": int(pid % 4 == 0)})
    pd.DataFrame(rows).to_csv(os.path.join(root, "metadata.csv"), index=False)
    np.save(os.path.join(root, "embeddings.npy"), np.vstack(embs))
    pd.DataFrame(clin_rows).to_csv(os.path.join(root, "clinical.csv"), index=False)
    return D


def _config(root: str, D: int) -> Config:
    raw = {
        "paths": {
            "metadata_csv": os.path.join(root, "metadata.csv"),
            "embeddings_npy": os.path.join(root, "embeddings.npy"),
            "clinical_csv": os.path.join(root, "clinical.csv"),
            "raw_eeg_dir": os.path.join(root, "raw"),
            "splits_dir": os.path.join(root, "splits"),
            "output_dir": os.path.join(root, "out"),
        },
        "repo_root": REPO_ROOT,
        "seed": 42,
        "clinical": {"patient_id_col": "patient_id", "session_date_col": "session_date",
                     "mmse_col": "MMSE", "age_col": "age", "apoe4_col": "APOE4",
                     "aria_col": "ARIA", "baseline_severity_from": "before"},
        "assemble": {"num_folds": 5},
        "reduce": {"zscore": True, "max_components": None,
                   "gate": {"min_components_above_floor": 2,
                            "pc1_dominance_threshold": 0.80, "noise_floor": 0.05,
                            "use_broken_stick": True}},
    }
    return Config(raw=raw, config_dir=root)


def test_assemble_builds_progressions_and_disjoint_folds():
    with tempfile.TemporaryDirectory() as root:
        D = _make_synth(root)
        cfg = _config(root, D)
        assemble.main(cfg)
        prog = io.read_table(cfg.out("progressions"))
        # every 2-session patient -> 1 progression; 3-session -> 2 (both kept)
        counts = prog.groupby("patient_id").size()
        three_sess = [pid for pid in range(1, 13) if pid % 3 == 0]
        assert all(counts[p] == 2 for p in three_sess), "3-session patients must give 2"
        # invariant 1: disjoint folds
        invariants.assert_disjoint_groups(prog, "patient_id", "fold")
        gmap = pd.read_csv(cfg.out("patient_group_map.csv"))
        assert gmap["patient_id"].is_unique
        # covariates present + earlier->later
        assert prog["dt"].gt(0).all()
        assert prog["baseline_severity"].notna().all()
        return cfg, prog


def _run_gate(cfg, delta_mat, prog):
    ids = prog["progression_id"].to_numpy()
    np.savez(cfg.out("deltas.npz"),
             progression_id=ids.astype(object),
             patient_id=prog["patient_id"].to_numpy(),
             fold=prog["fold"].to_numpy(),
             delta=delta_mat, dt=prog["dt"].to_numpy(),
             spread_std=np.zeros(len(ids)), spread_iqr=np.zeros(len(ids)))
    return reduce_mod.main(cfg)


def test_gate_halts_on_rank1_and_proceeds_on_multicomponent():
    with tempfile.TemporaryDirectory() as root:
        D = _make_synth(root)
        cfg = _config(root, D)
        assemble.main(cfg)
        prog = io.read_table(cfg.out("progressions"))
        P = len(prog)
        rng = np.random.default_rng(1)

        # rank-1: one direction + tiny isotropic noise
        direction = rng.normal(size=D)
        scores = rng.normal(size=P) * 5.0
        rank1 = np.outer(scores, direction) + rng.normal(size=(P, D)) * 0.01
        gate1 = _run_gate(cfg, rank1, prog)
        assert gate1["proceed"] is False, "rank-1 space must HALT clustering"

        # multi-component: three comparable directions
        d1, d2, d3 = rng.normal(size=(3, D))
        multi = (np.outer(rng.normal(size=P), d1)
                 + np.outer(rng.normal(size=P), d2) * 0.8
                 + np.outer(rng.normal(size=P), d3) * 0.6
                 + rng.normal(size=(P, D)) * 0.05)
        gate2 = _run_gate(cfg, multi, prog)
        assert gate2["proceed"] is True, "multi-component space must PROCEED"
        assert os.path.exists(cfg.out("scree.png"))


def _run():
    fns = {k: v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)}
    failed = 0
    for name, fn in fns.items():
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run())
