"""Tests for the functional-connectivity math + graph metrics + BH-FDR.

Runnable under pytest or as a plain script. Uses scipy/numpy only (no mne).
The load-bearing validity check: wPLI must IGNORE zero-lag coupling (volume
conduction) yet DETECT a genuine phase lag.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from analysis import connectivity as fc
from analysis.phenotype_stats import benjamini_hochberg, select_features
from analysis.config import Config


def _synth_epochs(E=60, L=256, fs=128.0, f0=10.0, seed=0):
    """3 channels: ch0, ch1 zero-lag with ch0, ch2 lags ch0 by pi/2."""
    rng = np.random.default_rng(seed)
    t = np.arange(L) / fs
    ep = np.zeros((E, 3, L))
    for e in range(E):
        phi = rng.uniform(0, 2 * np.pi)                 # random per epoch
        base = np.cos(2 * np.pi * f0 * t + phi)
        ep[e, 0] = base + 0.3 * rng.standard_normal(L)
        ep[e, 1] = base + 0.3 * rng.standard_normal(L)  # zero phase lag
        ep[e, 2] = np.cos(2 * np.pi * f0 * t + phi - np.pi / 2) + 0.3 * rng.standard_normal(L)
    return ep, fs


def test_wpli_ignores_zero_lag_detects_phase_lag():
    ep, fs = _synth_epochs()
    W = fc.spectral_connectivity(ep, fs, band=(8, 13), method="wpli")
    # symmetry, zero diagonal, range
    assert np.allclose(W, W.T, atol=1e-9)
    assert np.allclose(np.diag(W), 0.0)
    assert W.min() >= 0.0 and W.max() <= 1.0
    # zero-lag pair suppressed; lagged pair strong
    assert W[0, 1] < 0.3, f"zero-lag wPLI too high: {W[0,1]:.3f}"
    assert W[0, 2] > 0.6, f"phase-lag wPLI too low: {W[0,2]:.3f}"


def test_imcoh_runs_and_bounded():
    ep, fs = _synth_epochs()
    W = fc.spectral_connectivity(ep, fs, band=(8, 13), method="imcoh")
    assert np.allclose(W, W.T, atol=1e-9)
    assert W.min() >= 0.0 and W.max() <= 1.0
    assert W[0, 2] > W[0, 1]                             # lag > zero-lag


def test_make_epochs_shape():
    x = np.random.default_rng(0).standard_normal((5, 1000))
    ep = fc.make_epochs(x, fs=100.0, epoch_len_s=2.0, overlap=0.5)  # L=200, step=100
    assert ep.shape[1:] == (5, 200)
    assert ep.shape[0] == 9                              # (1000-200)/100 + 1


def test_graph_metrics_complete_graph():
    C = 4
    W = np.ones((C, C)) - np.eye(C)                      # complete, unit weights
    assert abs(fc.global_efficiency(W) - 1.0) < 1e-9
    assert abs(fc.char_path_length(W) - 1.0) < 1e-9
    assert abs(fc.mean_strength(W) - (C - 1)) < 1e-9
    assert fc.weighted_clustering(W) > 0.9
    assert fc.modularity(W) < 0.05                       # no community structure


def test_graph_modularity_two_cliques():
    # two weakly-connected cliques -> positive modularity
    W = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0.05, 0, 0],
        [0, 0, 0.05, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ], dtype=float)
    assert fc.modularity(W) > 0.2
    assert fc.global_efficiency(W) > 0
    assert fc.submatrix_mean(W, [0, 1, 2]) > fc.submatrix_mean(W, [0, 3])


def test_benjamini_hochberg():
    p = np.array([0.001, 0.20, 0.03, 0.50])
    rej, q = benjamini_hochberg(p, alpha=0.05)
    assert q.shape == p.shape
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)   # monotone in p-order
    assert rej[0] and not rej[1]                         # smallest rejected, large not
    assert np.all((q >= 0) & (q <= 1))


def test_select_features_families():
    cfg = Config(raw={"phenotype_stats": {
        "fc_confirmatory_features": ["wpli_alpha_global_delta"],
        "fc_exploratory_pattern": "graph_"}}, config_dir=".")
    fam = select_features(
        ["wpli_alpha_global_delta", "graph_wpli_alpha_modularity_delta",
         "imcoh_theta_posterior_baseline"], cfg).set_index("feature")["family"].to_dict()
    assert fam["wpli_alpha_global_delta"] == "confirmatory"
    assert fam["graph_wpli_alpha_modularity_delta"] == "exploratory_graph"
    assert fam["imcoh_theta_posterior_baseline"] == "exploratory"


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
