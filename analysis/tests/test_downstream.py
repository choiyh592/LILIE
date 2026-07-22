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


def _make_pca_and_progressions(cfg, root, seed=0, n_out=0):
    rng = np.random.default_rng(seed)
    D, per = 5, 20
    centers = np.array([[6, 0, 0, 0, 0], [-6, 5, 0, 0, 0], [0, -6, 4, 0, 0]], float)
    X, cl = [], []
    for ci, cen in enumerate(centers):
        X.append(cen + rng.normal(scale=0.7, size=(per, D)))
        cl += [ci] * per
    X = np.vstack(X)
    # inject extreme outliers far from the three clusters
    for i in range(n_out):
        pt = np.zeros(D); pt[0] = (40 + 10 * i) * (-1) ** i; pt[1] = 30
        X = np.vstack([X, pt]); cl.append(-99)
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


def test_outlier_mask_flags_extremes():
    from analysis import outliers
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    X = np.vstack([X, [20, 20, 20], [-25, 0, 0]])       # 2 clear outliers
    # LOF (default): density-based, multimodal-safe
    mask, score, _ = outliers.outlier_mask(X, method="lof", seed=0)
    assert mask[-1] and mask[-2]                         # both extremes flagged
    # LOF must NOT wipe a whole minority cluster: on 3 balanced clusters + far
    # points, only the far points should be flagged.
    c = np.vstack([rng.normal([8, 0, 0], 0.5, (20, 3)),
                   rng.normal([-8, 6, 0], 0.5, (20, 3)),
                   rng.normal([0, -8, 5], 0.5, (20, 3)),
                   np.array([[40, 30, 0], [-45, 30, 0]])])
    m2, _, _ = outliers.outlier_mask(c, method="lof", seed=0)
    assert m2[-1] and m2[-2]                             # far points flagged
    assert m2[:60].sum() <= 6                            # clusters largely intact


def test_cluster_marks_outliers_and_clusters_core():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(root)
        cfg.raw["cluster"]["outlier_handling"] = "mark_separately"
        cfg.raw["cluster"]["outlier_quantile"] = 0.975
        _make_pca_and_progressions(cfg, root, n_out=3)   # 60 core + 3 outliers
        cluster.main(cfg)
        L = np.load(cfg.out("labels.npz"), allow_pickle=True)
        lab = L["cluster"]
        # the 3 injected extremes (last rows) are marked -1
        assert (lab[-3:] == -1).all()
        # core still clustered (>=2 real clusters), outliers excluded
        assert len(set(lab[lab >= 0])) >= 2
        # stability runs on core only without error
        stability.main(cfg)
        stab = io.read_json(cfg.out("stability.json"))
        assert stab["k"] == len(set(lab[lab >= 0]))


def test_directional_recovers_direction_groups():
    """Two change directions 30 deg apart with a wide magnitude spread: magnitude
    k-means splits by radius and misses the directions; directional recovers them."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from analysis.cluster import _unit
    rng = np.random.default_rng(0)
    X, true = [], []
    for i, a in enumerate([np.deg2rad(80), np.deg2rad(110)]):
        dv = np.array([np.cos(a), np.sin(a)])
        for _ in range(40):
            mag = rng.uniform(0.2, 12.0)                 # wide magnitude range
            X.append(dv * mag + rng.normal(0, 0.05, 2)); true.append(i)
    X, true = np.array(X), np.array(true)
    ari_eu = adjusted_rand_score(true, KMeans(2, n_init=10, random_state=0).fit_predict(X))
    ari_sp = adjusted_rand_score(true, KMeans(2, n_init=10, random_state=0).fit_predict(_unit(X)))
    assert ari_sp > 0.9                                  # directional recovers directions
    assert ari_sp > ari_eu + 0.3                         # and clearly beats magnitude


def test_clusterability_discrete_vs_continuum():
    from analysis.trajectory_eval import clusterability, _unit
    rng = np.random.default_rng(0)
    D = 8
    # discrete: 3 tight vMF-like blobs of directions
    dirs = _unit(rng.normal(size=(3, D)))
    disc = np.vstack([d + rng.normal(0, 0.05, (25, D)) for d in dirs])
    vd, _ = clusterability(_unit(disc)[:, :6], kmax=6, seed=0)
    assert vd["verdict"] in ("discrete", "weak")          # structure detected
    # continuum: directions spread smoothly around a great circle
    t = np.linspace(0, np.pi, 75)
    cont = np.zeros((75, D)); cont[:, 0] = np.cos(t); cont[:, 1] = np.sin(t)
    cont += rng.normal(0, 0.02, (75, D))
    vc, _ = clusterability(_unit(cont)[:, :6], kmax=6, seed=0)
    # a smooth arc should look less discrete than 3 tight blobs
    assert vc["votes_for_discrete"] <= vd["votes_for_discrete"]


def test_vmf_and_spherical_recover_directions():
    from analysis.directional_phenotype import vmf_select, spherical_select, _unit
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(0)
    D = 8
    mus = _unit(rng.normal(size=(3, D)))
    X, true = [], []
    for i, mu in enumerate(mus):
        pts = mu + rng.normal(0, 0.12, (20, D))          # concentrated around mu
        X.append(_unit(pts)); true += [i] * 20
    X = _unit(np.vstack(X)); true = np.array(true)
    vk, _, vfit = vmf_select(X, [2, 6], seed=0, n_init=4, max_iter=80)
    sk, _, slab = spherical_select(X, [2, 6], seed=0)
    assert vk == 3 and sk == 3                            # both recover 3 directions
    assert adjusted_rand_score(true, vfit["labels"]) > 0.9
    assert adjusted_rand_score(true, slab) > 0.9


def test_reliability_split_drops_low_magnitude():
    from analysis.directional_phenotype import reliability_split
    norm = np.concatenate([np.full(20, 0.1), np.full(20, 5.0)])   # low + high
    spread = np.zeros(40)
    reliable, cut = reliability_split(norm, spread, {"reliable_magnitude_percentile": 50})
    assert reliable[20:].all() and not reliable[:20].any()        # keeps only the high


def test_centroid_angles_and_autok():
    from analysis.phenotype_geometry import centroid_angles, auto_k_sweep, _unit
    rng = np.random.default_rng(0); D = 8
    mu = _unit(rng.normal(size=(1, D)))[0]
    U, lab = [], []
    for _ in range(12):
        U.append(mu + rng.normal(0, 0.05, D)); lab.append(0)
    for _ in range(12):
        U.append(-mu + rng.normal(0, 0.05, D)); lab.append(1)   # antipodal
    U = _unit(np.array(U)); lab = np.array(lab)
    cs, R, ang = centroid_angles(U, lab)
    assert ang["0-1"]["angle_deg"] > 150                        # detected antipodal
    assert R[0] > 0.9 and R[1] > 0.9                            # concentrated
    # auto-k sweep runs and returns per-threshold selections
    norm = np.full(len(U), 5.0)
    sweep = auto_k_sweep(U, norm, {"direction_dims": 6, "direction_var": 0.9,
                                   "vmf_n_init": 3, "vmf_max_iter": 40},
                         0, [50, 80], [2, 4])
    assert all("spherical_k" in r and "vmf_k" in r for r in sweep)


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


def test_spectral_features_inband_paf_and_guards():
    """Analysis-band restriction (rel_* tile to ~1), aperiodic-flattened PAF recovers
    a real peak while pure-1/f channels NaN out, and band-name guards raise."""
    from analysis import spectral as sp
    rng = np.random.default_rng(0)
    fs, N, C = 200, 200 * 30, 8
    freqs = np.fft.rfftfreq(N, 1 / fs)
    amp = np.zeros_like(freqs); amp[1:] = freqs[1:] ** (-1.2 / 2)
    eeg = np.zeros((C, N))
    for c in range(C):
        x = np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, len(freqs))), N)
        if c < 4:                                          # half get a real 10.5 Hz peak
            x = x / x.std() + 2.5 * np.sin(2 * np.pi * 10.5 * np.arange(N) / fs)
        eeg[c] = x
    bands = {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "alpha1": [8, 10],
             "alpha2": [10, 13], "beta1": [13, 20], "beta2": [20, 30], "gamma": [30, 45]}
    f = sp.spectral_features(eeg, fs, bands, [4, 5, 6, 7], analysis_band=(1, 45))
    tile = ["delta", "theta", "alpha", "beta1", "beta2", "gamma"]
    assert abs(sum(f[f"rel_{b}_global"] for b in tile) - 1.0) < 0.02     # rel_* tile to 1
    assert 1 < f["sef95_global"] <= 45                                   # SEF95 in band
    assert abs(f["paf_global"] - 10.5) < 1.0                             # PAF recovers peak
    assert abs(f["alpha_cog_global"] - 10.5) < 1.5                       # CoG robust primary
    assert 0.0 <= f["paf_nan_frac_global"] <= 1.0
    assert abs(f["aperiodic_exponent_global"] - 1.2) < 0.3               # 1/f recovered
    # guards: missing alpha / missing beta must raise, not silently degrade
    for bad in ({"delta": [1, 4], "theta": [4, 8]},
                {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13]}):
        try:
            sp.spectral_features(eeg, fs, bad, None); assert False, "guard did not raise"
        except ValueError:
            pass


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
