"""Integration test for modules 4 (cluster), 5 (stability), 8 (report).

Synthetic PC scores with 3 well-separated clusters, no clinical, no QEEG.
Verifies clustering recovers structure, stability is high, the report renders,
and the clinical/QEEG-absent paths degrade gracefully. sklearn/scipy/matplotlib
only. Runnable under pytest or as a plain script.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from analysis.config import Config
from analysis import cluster, stability, report, io


def _cfg(root):
    return Config(raw={
        "paths": {"output_dir": root, "progressions": None, "raw_eeg_dir": os.path.join(root, "noraw"),
                  "clinical_csv": None},
        "seed": 0,
        "cluster": {"algorithms": ["kmeans", "gmm"], "k_range": [2, 6]},
        "stability": {"n_bootstrap": 60, "jaccard_flag_below": 0.60},
        "report": {"fig_dpi": 100},
        "phenotype_stats": {"fc_confirmatory_features": ["wpli_alpha_global_delta"]},
    }, config_dir=root)


def _make_pca_and_progressions(cfg, root, seed=0):
    rng = np.random.default_rng(seed)
    D, per = 5, 20
    centers = np.array([[6, 0, 0, 0, 0], [-6, 5, 0, 0, 0], [0, -6, 4, 0, 0]], float)
    X, cl = [], []
    for ci, cen in enumerate(centers):
        X.append(cen + rng.normal(scale=0.7, size=(per, D)))
        cl += [ci] * per
    X = np.vstack(X)
    N = X.shape[0]
    prog_ids = np.array([f"p{i:03d}" for i in range(N)], dtype=object)
    # patients: mostly 1 progression, some share 2 (kept within a cluster)
    patient_id = np.arange(N)
    for i in range(0, 16, 2):
        patient_id[i + 1] = patient_id[i]                 # pair up -> same patient
    fold = (patient_id % 5) + 1
    np.savez(cfg.out("X_pca.npz"), X_pca=X,
             explained_variance_ratio=np.array([.3, .25, .2, .15, .1]),
             components=np.eye(5), progression_id=prog_ids,
             patient_id=patient_id, fold=fold, n_retained=5)
    io.write_json({"route": "clustering", "proceed": True,
                   "rationale": "synthetic"}, cfg.out("gate.json"))
    # progressions table with NaN clinical (no clinical available)
    df = pd.DataFrame({"progression_id": prog_ids, "patient_id": patient_id, "fold": fold,
                       "dt": rng.integers(120, 400, N).astype(float)})
    for c in ["baseline_severity", "age", "APOE4", "ARIA", "MMSE_delta"]:
        df[c] = np.nan
    io.write_table(df, cfg.out("progressions"))
    return np.array(cl)


def test_cluster_stability_report_no_clinical():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(root)
        true_cl = _make_pca_and_progressions(cfg, root)

        cluster.main(cfg)
        L = np.load(cfg.out("labels.npz"), allow_pickle=True)
        k = int(L["k"])
        assert 2 <= k <= 5
        # recover 3 clusters: adjusted rand vs truth should be high at chosen k=3
        from sklearn.metrics import adjusted_rand_score
        if k == 3:
            assert adjusted_rand_score(true_cl, L["cluster"]) > 0.9
        assert L["cluster"].shape[0] == 60

        stability.main(cfg)
        stab = io.read_json(cfg.out("stability.json"))
        assert stab["ari_mean"] > 0.7            # well-separated -> stable
        assert stab["k"] == k

        report.main(cfg)
        assert os.path.exists(cfg.out("report_figure.png"))
        assert os.path.exists(cfg.out("report_table.csv"))
        md = open(cfg.out("report.md")).read()
        assert "Clinical covariates absent" in md   # graceful clinical-skip note
        assert "QEEG/FC not computed" in md


def test_runall_skip_helpers():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(root)
        from analysis import run_all
        assert run_all._clinical_available(cfg) is False      # clinical_csv null
        assert run_all._raw_eeg_available(cfg) is False       # noraw dir absent
        # auto-resolve: run_phenotype defaults to clinical availability (False)
        assert run_all._resolve(cfg, "run_phenotype", False) is False
        # explicit override wins
        cfg.raw["run"] = {"run_qeeg": True}
        assert run_all._resolve(cfg, "run_qeeg", False) is True


def _run():
    fns = {k: v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)}
    failed = 0
    for name, fn in fns.items():
        try:
            fn(); print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            import traceback; failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run())
