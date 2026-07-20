"""Module 5 - stability: bootstrap cluster stability (load-bearing at this n).

Bootstrap resamples **patients** (not progressions), so a patient's
progressions move together and no patient is split across the resample
(invariant 1, via ``invariants.assert_resample_groups``). For each resample we
recluster and measure:

* **Clusterwise Jaccard** (Hennig clusterboot) per reference cluster; flag < 0.60.
* **ARI / NMI / Cohen's kappa** between the resample labelling and the reference.

Sensitivity: k +/- 1 (ARI vs reference) and k-means vs GMM (ARI). The pooler
variant is a separate axis (re-run module 2 with delta.sensitivity_pool_methods)
and is reported as a note rather than recomputed here.

Output (paths.output_dir):
  stability.json  - Jaccard per cluster, flags, ARI/NMI/kappa, sensitivity
  stability.csv   - per-cluster Jaccard + flag

Run:  python -m analysis.stability --config analysis/config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             cohen_kappa_score, silhouette_score)
from scipy.optimize import linear_sum_assignment

from .config import Config, load_config, add_arg
from . import io
from . import invariants


def _align(ref, other, k):
    """Relabel `other` to best match `ref` via Hungarian on the contingency."""
    C = np.zeros((k, k), dtype=int)
    for a, b in zip(ref, other):
        if a < k and b < k:
            C[a, b] += 1
    row, col = linear_sum_assignment(-C)
    mapping = {c: r for r, c in zip(row, col)}
    return np.array([mapping.get(b, b) for b in other])


def _clusterwise_jaccard(ref_labels, boot_full, present_mask, k):
    """Max Jaccard of each reference cluster vs bootstrap clusters (present pts)."""
    out = {}
    present = np.where(present_mask)[0]
    for i in range(k):
        Ri = set(np.where(ref_labels == i)[0]) & set(present.tolist())
        if not Ri:
            out[i] = np.nan
            continue
        best = 0.0
        for l in range(k):
            Bl = set(np.where((boot_full == l))[0]) & set(present.tolist())
            union = Ri | Bl
            if union:
                best = max(best, len(Ri & Bl) / len(union))
        out[i] = best
    return out


def main(config: Config) -> str:
    st = config["stability"]
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)

    z = np.load(config.out("X_pca.npz"), allow_pickle=True)
    X = z["X_pca"].astype(float)
    L = np.load(config.out("labels.npz"), allow_pickle=True)
    ref_all = L["cluster"].astype(int)

    # Match the clustering metric: in directional mode re-cluster on unit vectors.
    if str(config["cluster"].get("metric", "euclidean")) == "cosine":
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm

    # Stability is assessed on the CORE clusters only (outliers are cluster -1).
    core = ref_all >= 0
    n_out = int((~core).sum())
    X = X[core]
    ref = ref_all[core]
    patient_id = L["patient_id"][core]
    k = len(np.unique(ref))
    if n_out:
        print(f"[stability] excluding {n_out} outlier progression(s); "
              f"assessing {k} core cluster(s).")

    patients = np.unique(patient_id)
    B = int(st["n_bootstrap"])

    jac_acc = {i: [] for i in range(k)}
    aris, nmis, kappas = [], [], []
    for _ in range(B):
        samp_pat = rng.choice(patients, size=len(patients), replace=True)
        # progression indices for sampled patients (with duplication)
        idx = np.concatenate([np.where(patient_id == p)[0] for p in samp_pat])
        present_mask = np.zeros(len(ref), dtype=bool)
        present_mask[np.unique(idx)] = True
        # invariant 1: the held-out (not-present) patients share no id with present
        invariants.assert_resample_groups(
            patient_id[present_mask], patient_id[~present_mask])
        if np.unique(ref[present_mask]).size < 2:
            continue
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X[idx])
        boot_full = km.predict(X)                       # label every original point
        for i, v in _clusterwise_jaccard(ref, boot_full, present_mask, k).items():
            if not np.isnan(v):
                jac_acc[i].append(v)
        aligned = _align(ref, boot_full, k)
        aris.append(adjusted_rand_score(ref, boot_full))
        nmis.append(normalized_mutual_info_score(ref, boot_full))
        kappas.append(cohen_kappa_score(ref, aligned))

    jaccard = {int(i): (float(np.mean(v)) if v else float("nan")) for i, v in jac_acc.items()}
    flagged = [int(i) for i, v in jaccard.items() if not np.isnan(v) and v < float(st["jaccard_flag_below"])]

    def ci(a):
        a = np.array(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a.size else [np.nan, np.nan]

    # --- sensitivity ---------------------------------------------------------
    sens = {}
    for dk, tag in ((-1, "k_minus_1"), (1, "k_plus_1")):
        kk = k + dk
        if 2 <= kk <= X.shape[0] - 1:
            lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(X)
            sens[tag] = {"k": kk, "ari_vs_ref": float(adjusted_rand_score(ref, lab)),
                         "silhouette": float(silhouette_score(X, lab))}
    try:
        gmm = np.load(config.out("cluster_gmm.npz"), allow_pickle=True)["cluster"].astype(int)
        sens["kmeans_vs_gmm_ari"] = float(adjusted_rand_score(ref, gmm))
    except Exception:
        pass
    sens["pooler_variant"] = ("re-run module 2 with delta.sensitivity_pool_methods "
                              "and compare labels; not recomputed here")

    result = {
        "k": k, "n_bootstrap": B, "n_effective": len(aris),
        "jaccard_per_cluster": jaccard,
        "jaccard_flag_below": float(st["jaccard_flag_below"]),
        "flagged_clusters": flagged,
        "ari_mean": float(np.mean(aris)) if aris else np.nan, "ari_ci95": ci(aris),
        "nmi_mean": float(np.mean(nmis)) if nmis else np.nan,
        "kappa_mean": float(np.mean(kappas)) if kappas else np.nan,
        "sensitivity": sens,
    }
    io.write_json(result, config.out("stability.json"))

    import csv
    with open(config.out("stability.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster", "mean_jaccard", "flagged_below_%.2f" % float(st["jaccard_flag_below"])])
        for i in range(k):
            w.writerow([i, jaccard[i], int(i in flagged)])

    print(f"[stability] k={k} Jaccard/cluster: "
          + ", ".join(f"{i}:{jaccard[i]:.2f}" for i in range(k)))
    print(f"[stability] ARI={result['ari_mean']:.2f} {result['ari_ci95']}, "
          f"NMI={result['nmi_mean']:.2f}, kappa={result['kappa_mean']:.2f}; "
          f"flagged (<{st['jaccard_flag_below']}): {flagged or 'none'}")
    return config.out("stability.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 5 - bootstrap stability")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
