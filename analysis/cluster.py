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
from . import outliers


def _within_dispersion(X, labels):
    """Sum of within-cluster squared distances to centroid (W_k)."""
    total = 0.0
    for c in np.unique(labels):
        pts = X[labels == c]
        if len(pts) > 0:
            total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return total


def _unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n


def gap_statistic(X, k, seed, n_ref=25, spherical=False):
    """Tibshirani gap statistic for a given k, with uniform bounding-box refs.

    In spherical mode the uniform references are projected to the unit sphere so
    the null matches directional (unit-vector) data.
    """
    rng = np.random.default_rng(seed + k)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    logWk = np.log(_within_dispersion(X, km.labels_) + 1e-12)
    mins, maxs = X.min(axis=0), X.max(axis=0)
    ref_logW = []
    for _ in range(n_ref):
        Xr = rng.uniform(mins, maxs, size=X.shape)
        if spherical:
            Xr = _unit(Xr)
        kmr = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Xr)
        ref_logW.append(np.log(_within_dispersion(Xr, kmr.labels_) + 1e-12))
    ref_logW = np.array(ref_logW)
    gap = ref_logW.mean() - logWk
    sk = ref_logW.std() * np.sqrt(1.0 + 1.0 / n_ref)
    return gap, sk


def select_k(X, k_min, k_max, seed, spherical=False):
    """Return (chosen_k, table) using gap 1-SE rule; silhouette reported too."""
    sil_metric = "cosine" if spherical else "euclidean"
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        sil = (silhouette_score(X, km.labels_, metric=sil_metric)
               if k >= 2 and len(np.unique(km.labels_)) > 1 else np.nan)
        gap, sk = gap_statistic(X, k, seed, spherical=spherical)
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

    # Directional clustering: on the unit sphere, spherical k-means == Euclidean
    # k-means on L2-normalized vectors, so we normalize the PC scores and run the
    # usual machinery. Clusters then group progressions by DIRECTION of change.
    metric = str(cc.get("metric", "euclidean"))
    spherical = metric == "cosine"
    Xuse = _unit(X) if spherical else X
    if spherical:
        print("[cluster] directional (cosine/spherical) k-means on unit-normalized "
              "PC scores")

    # Outlier handling: cluster the CORE population, mark extremes as -1 so a few
    # aberrant deltas don't force tiny outlier-only clusters. In directional mode
    # this flags anomalous change *directions*, not large magnitudes.
    handling = str(cc.get("outlier_handling", "none"))
    if handling == "mark_separately":
        mask, _score, _cut = outliers.outlier_mask(
            Xuse, method=str(cc.get("outlier_method", "lof")),
            quantile=float(cc.get("outlier_quantile", 0.975)),
            n_neighbors=int(cc.get("outlier_n_neighbors", 20)), seed=seed)
    else:
        mask = np.zeros(X.shape[0], dtype=bool)
    core = ~mask
    Xc = Xuse[core]
    n_out = int(mask.sum())
    if n_out:
        print(f"[cluster] {n_out}/{X.shape[0]} progressions marked outliers "
              f"(cluster=-1); clustering the {int(core.sum())} core progressions.")

    k_min, k_max = int(cc["k_range"][0]), int(cc["k_range"][1])
    k_max = min(k_max, Xc.shape[0] - 1)
    chosen_k, table = select_k(Xc, k_min, k_max, seed, spherical=spherical)

    # primary: (spherical) k-means at chosen k (fit on core; outliers stay -1)
    km = KMeans(n_clusters=chosen_k, n_init=25, random_state=seed).fit(Xc)
    labels = np.full(X.shape[0], -1, dtype=int)
    labels[core] = km.labels_

    # sensitivity: GMM at chosen k (core)
    gmm = GaussianMixture(n_components=chosen_k, n_init=5, random_state=seed).fit(Xc)
    gmm_labels = np.full(X.shape[0], -1, dtype=int)
    gmm_labels[core] = gmm.predict(Xc)

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
             cluster=labels, is_outlier=mask, k=chosen_k, algorithm="kmeans")
    np.savez(config.out("cluster_gmm.npz"),
             progression_id=prog_ids, cluster=gmm_labels, k=chosen_k, algorithm="gmm")

    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    print(f"[cluster] chosen k={chosen_k} (gap 1-SE rule) on core. cluster sizes "
          f"(-1 = outliers): {sizes}")
    print(f"[cluster] selection table -> {sel_path}; labels -> {config.out('labels.npz')}")
    return config.out("labels.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4 - clustering + k selection")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
