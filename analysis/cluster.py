"""Module 4 - cluster: clustering + k selection.

k-means (primary) and GMM (sensitivity) on the retained PCs from module 3.
k is selected by the **gap statistic** (Tibshirani 1-SE rule, which favors small
k) with the **silhouette** reported alongside. k is chosen from cluster geometry
only -- never from downstream phenotype separation.

The clustering unit is the progression (invariant 3): X_pca has one row per
progression, and labels are written back keyed by progression_id.

Output (paths.output_dir):
  labels.npz    - progression_id, patient_id, fold, cluster (primary, k-means)
  cluster_selection.csv - per-k silhouette + gap + s_k, and the chosen k
  cluster_gmm.npz - GMM labels at the chosen k (sensitivity)

Run:  python -m analysis.cluster --config analysis/config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from .config import Config, load_config, add_arg
from . import io
from . import invariants


def _within_dispersion(X, labels):
    """Sum of within-cluster squared distances to centroid (W_k)."""
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) > 0:
            total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return total


def gap_statistic(X, k, seed, n_ref=25):
    """Tibshirani gap statistic for a given k, with uniform bounding-box refs."""
    rng = np.random.default_rng(seed + k)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    logWk = np.log(_within_dispersion(X, km.labels_) + 1e-12)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    ref_logW = []
    for _ in range(n_ref):
        Xr = rng.uniform(mins, maxs, size=X.shape)
        kmr = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Xr)
        ref_logW.append(np.log(_within_dispersion(Xr, kmr.labels_) + 1e-12))
    ref_logW = np.array(ref_logW)
    gap = ref_logW.mean() - logWk
    sk = ref_logW.std() * np.sqrt(1.0 + 1.0 / n_ref)
    return gap, sk


def select_k(X, k_min, k_max, seed):
    """Return (chosen_k, table) using gap 1-SE rule; silhouette reported too."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil = silhouette_score(X, km.labels_) if k >= 2 and len(np.unique(km.labels_)) > 1 else np.nan
        gap, sk = gap_statistic(X, k, seed)
        rows.append({"k": k, "silhouette": sil, "gap": gap, "sk": sk})
    # Tibshirani: smallest k with Gap(k) >= Gap(k+1) - s_{k+1}. Favors small k.
    chosen = rows[0]["k"]
    picked = False
    for i in range(len(rows) - 1):
        if rows[i]["gap"] >= rows[i + 1]["gap"] - rows[i + 1]["sk"]:
            chosen = rows[i]["k"]
            picked = True
            break
    if not picked:                                   # fallback: max gap
        chosen = max(rows, key=lambda r: r["gap"])["k"]
    return chosen, rows


def main(config: Config) -> str:
    cc = config["cluster"]
    seed = int(config["seed"])
    z = np.load(config.out("X_pca.npz"), allow_pickle=True)
    X = z["X_pca"].astype(float)
    prog_ids = z["progression_id"]
    patient_id = z["patient_id"]
    fold = z["fold"]

    invariants.assert_progression_unit(X.shape[0], prog_ids)

    k_min, k_max = int(cc["k_range"][0]), int(cc["k_range"][1])
    k_max = min(k_max, X.shape[0] - 1)
    chosen_k, table = select_k(X, k_min, k_max, seed)

    # primary: k-means at chosen k
    km = KMeans(n_clusters=chosen_k, n_init=25, random_state=seed).fit(X)
    labels = km.labels_

    # sensitivity: GMM at chosen k
    gmm = GaussianMixture(n_components=chosen_k, n_init=5, random_state=seed).fit(X)
    gmm_labels = gmm.predict(X)

    # write selection table
    import csv
    sel_path = config.out("cluster_selection.csv")
    with open(sel_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "silhouette", "gap", "sk", "chosen"])
        for r in table:
            w.writerow([r["k"], r["silhouette"], r["gap"], r["sk"], int(r["k"] == chosen_k)])

    np.savez(config.out("labels.npz"),
             progression_id=prog_ids, patient_id=patient_id, fold=fold,
             cluster=labels, k=chosen_k, algorithm="kmeans")
    np.savez(config.out("cluster_gmm.npz"),
             progression_id=prog_ids, cluster=gmm_labels, k=chosen_k, algorithm="gmm")

    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    print(f"[cluster] chosen k={chosen_k} (gap 1-SE rule). k-means cluster sizes: {sizes}")
    print(f"[cluster] selection table -> {sel_path}; labels -> {config.out('labels.npz')}")
    return config.out("labels.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4 - clustering + k selection")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
