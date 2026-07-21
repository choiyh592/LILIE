"""Trajectory eval - clusterability test + direction x magnitude map.

The alternative to hard clustering when the delta directions look like a
continuum rather than discrete phenotypes (see cluster_selection.csv: gap rising
monotonically to the ceiling). Two parts:

1. CLUSTERABILITY - is there discrete structure at all? Three complementary
   tests on the directional (unit-delta PCA) space, combined into a verdict:
     - Hopkins statistic  : cluster tendency (~0.5 uniform, ->1 clustered).
     - GMM-BIC curve       : an interior BIC minimum => a natural #components;
                             monotone-decreasing => no natural k (continuum).
     - HDBSCAN             : density clusters vs points labelled noise.
   (A von Mises-Fisher mixture BIC would be the textbook directional version;
   GMM-BIC on the unit-delta PCs is used here as a dependency-free proxy.)

2. DIRECTION x MAGNITUDE MAP - characterize each progression on two continuous,
   interpretable axes instead of forcing boxes: the change DIRECTION (PCA of the
   unit-normalized deltas) and a tempered MAGNITUDE (standardized log-norm =
   "how much the patient moved"). Honors both notions at once.

Outputs (paths.output_dir):
  trajectory_clusterability.json  - the three tests + combined verdict
  trajectory_map.png              - direction x magnitude figure
  trajectory_coords.csv           - per-progression coordinates

Run:  python -m analysis.trajectory_eval --config analysis/config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from .config import Config, load_config, add_arg
from . import io


def _unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n


# ---------------------------------------------------------------------------
# Clusterability tests
# ---------------------------------------------------------------------------
def hopkins(X, seed=0):
    """Hopkins statistic: ~0.5 = no cluster tendency, ->1 = clustered."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = min(max(5, int(0.1 * n)), n - 1)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(n, m, replace=False)
    w = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]           # NN dist among real
    mins, maxs = X.min(0), X.max(0)
    U = rng.uniform(mins, maxs, size=(m, d))
    u = nn.kneighbors(U, n_neighbors=1)[0][:, 0]                # uniform->real
    denom = u.sum() + w.sum()
    return float(u.sum() / denom) if denom > 0 else 0.5


def gmm_bic_curve(X, kmax, seed=0):
    kmax = int(min(kmax, X.shape[0] - 1))
    bics = []
    for k in range(1, kmax + 1):
        try:
            g = GaussianMixture(k, n_init=3, random_state=seed).fit(X)
            bics.append(float(g.bic(X)))
        except Exception:
            bics.append(float("inf"))
    argmin = 1 + int(np.argmin(bics))
    return bics, argmin, kmax


def hdbscan_summary(X, min_cluster_size):
    try:
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(min_cluster_size=int(min_cluster_size)).fit_predict(X)
        n_clusters = len({l for l in lab if l >= 0})
        noise = float(np.mean(lab == -1))
        return {"available": True, "n_clusters": n_clusters,
                "noise_fraction": noise, "labels": lab}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e), "labels": None}


def clusterability(dir_scores, kmax, seed):
    n = dir_scores.shape[0]
    H = hopkins(dir_scores, seed=seed)
    bics, argmin, kmax_eff = gmm_bic_curve(dir_scores, kmax, seed)
    hdb = hdbscan_summary(dir_scores, max(5, int(0.1 * n)))

    votes = 0
    reasons = []
    if H > 0.75:
        votes += 1; reasons.append(f"Hopkins={H:.2f} (>0.75, cluster tendency)")
    else:
        reasons.append(f"Hopkins={H:.2f} (weak/uniform tendency)")
    if 1 < argmin < kmax_eff:
        votes += 1; reasons.append(f"GMM-BIC interior min at k={argmin}")
    else:
        reasons.append(f"GMM-BIC min at k={argmin} (boundary => no natural k)")
    if hdb["available"]:
        if hdb["n_clusters"] >= 2 and hdb["noise_fraction"] < 0.5:
            votes += 1
            reasons.append(f"HDBSCAN found {hdb['n_clusters']} clusters, "
                           f"{hdb['noise_fraction']:.0%} noise")
        else:
            reasons.append(f"HDBSCAN: {hdb['n_clusters']} clusters, "
                           f"{hdb['noise_fraction']:.0%} noise (sparse/continuum)")

    verdict = "discrete" if votes >= 2 else ("weak" if votes == 1 else "continuum")
    return {
        "verdict": verdict, "votes_for_discrete": votes,
        "hopkins": H, "gmm_bic": bics, "gmm_bic_argmin": argmin,
        "gmm_bic_kmax": kmax_eff,
        "hdbscan_n_clusters": hdb.get("n_clusters"),
        "hdbscan_noise_fraction": hdb.get("noise_fraction"),
        "hdbscan_available": hdb["available"],
        "reasons": reasons,
    }, hdb.get("labels")


# ---------------------------------------------------------------------------
# Direction x magnitude map
# ---------------------------------------------------------------------------
def _plot(dir_scores, mag_z, hdb_labels, verdict, path, dpi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    pc2 = dir_scores[:, 1] if dir_scores.shape[1] > 1 else np.zeros(len(dir_scores))
    sizes = 25 + 55 * (mag_z - mag_z.min()) / (np.ptp(mag_z) + 1e-9)

    sc = ax1.scatter(dir_scores[:, 0], pc2, c=mag_z, s=sizes, cmap="viridis",
                     edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("direction PC1"); ax1.set_ylabel("direction PC2")
    ax1.set_title("Change direction (size/color = magnitude)")
    fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, label="std log-magnitude")

    ax2.scatter(dir_scores[:, 0], mag_z, s=30, color="#3b6ea5",
                edgecolor="white", linewidth=0.4)
    ax2.set_xlabel("direction PC1 (type of change)")
    ax2.set_ylabel("std log-magnitude (amount of change)")
    ax2.set_title(f"Trajectory map | clusterability: {verdict}")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main(config: Config) -> str:
    seed = int(config["seed"])
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    delta = dz["delta"].astype(float)
    prog_ids = dz["progression_id"]
    patient_id = dz["patient_id"]

    # direction space: PCA of unit-normalized deltas
    U = _unit(delta)
    n_comp = int(min(10, U.shape[0] - 1, U.shape[1]))
    dir_scores = PCA(n_components=n_comp, random_state=seed).fit_transform(U)

    # tempered magnitude: standardized log-norm
    norm = np.linalg.norm(delta, axis=1)
    log_norm = np.log(norm + 1e-12)
    mag_z = (log_norm - log_norm.mean()) / (log_norm.std() + 1e-12)

    kmax = int(config["cluster"]["k_range"][1])
    verdict, hdb_labels = clusterability(dir_scores, kmax, seed)

    io.write_json(verdict, config.out("trajectory_clusterability.json"))
    coords = pd.DataFrame({
        "progression_id": prog_ids, "patient_id": patient_id,
        "dir_pc1": dir_scores[:, 0],
        "dir_pc2": dir_scores[:, 1] if dir_scores.shape[1] > 1 else 0.0,
        "log_norm": log_norm, "mag_z": mag_z,
        "hdbscan_label": hdb_labels if hdb_labels is not None else -1,
    })
    coords.to_csv(config.out("trajectory_coords.csv"), index=False)
    _plot(dir_scores, mag_z, hdb_labels, verdict["verdict"],
          config.out("trajectory_map.png"), int(config["report"]["fig_dpi"]))

    print(f"[trajectory_eval] clusterability verdict: {verdict['verdict'].upper()} "
          f"({verdict['votes_for_discrete']}/3 tests favor discrete)")
    for r in verdict["reasons"]:
        print(f"[trajectory_eval]   - {r}")
    print(f"[trajectory_eval] wrote trajectory_map.png, trajectory_coords.csv, "
          f"trajectory_clusterability.json to {config.output_dir}")
    return config.out("trajectory_map.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory eval - clusterability + map")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
