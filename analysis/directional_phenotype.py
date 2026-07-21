"""Directional phenotyping - cluster the *reliable* change directions.

Key idea: the direction of a near-zero delta is noise (dividing a tiny vector by
its tiny norm amplifies noise). So we FIRST split off the low-change / stable
group (small magnitude) as one phenotype, and cluster directions only on the
RELIABLE subset where direction is meaningful. This also removes the diffuse
noise-core that made global k-means/GMM hit the k ceiling.

Two directional clustering methods on the reliable subset:
  - von Mises-Fisher mixture (EM + BIC)  - the principled directional model:
    each cluster has a mean direction mu and concentration kappa.
  - spherical k-means (silhouette)        - the robust hard-assignment version.

Also: an angle rose (do directions form modes?) and a consecutive-delta
consistency check for 3-session patients (does step 2 continue step 1?).

Outputs (paths.output_dir):
  directional_phenotype.json    - split, k selection (both methods), per-cluster
                                   characterization, trajectory-consistency
  directional_phenotype.png     - scatter + rose + consistency + selection curves
  directional_phenotype_labels.csv

Run:  python -m analysis.directional_phenotype --config analysis/config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import ive, logsumexp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import Config, load_config, add_arg
from . import io

_PAL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]


def _unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n


# ---------------------------------------------------------------------------
# von Mises-Fisher mixture (EM)
# ---------------------------------------------------------------------------
def _log_vmf_norm(d, kappa):
    """log normalizing constant C_d(kappa); ive() keeps it stable for large kappa."""
    v = d / 2.0 - 1.0
    kappa = np.asarray(kappa, dtype=float)
    log_iv = np.log(ive(v, kappa) + 1e-300) + kappa       # log I_v(kappa)
    return v * np.log(kappa + 1e-300) - (d / 2.0) * np.log(2 * np.pi) - log_iv


def _kappa_from_rbar(rbar, d):
    rbar = np.clip(rbar, 1e-6, 1 - 1e-6)
    k = rbar * (d - rbar ** 2) / (1 - rbar ** 2)
    return np.clip(k, 1e-3, 1e5)


def _movmf_fit(X, k, seed, n_init=5, max_iter=100, tol=1e-5):
    n, d = X.shape
    best = None
    for init in range(n_init):
        rng = np.random.default_rng(seed + init)
        # init from spherical k-means
        km = KMeans(k, n_init=3, random_state=seed + init).fit(X)
        mu = _unit(np.vstack([X[km.labels_ == c].mean(0) if np.any(km.labels_ == c)
                              else X[rng.integers(n)] for c in range(k)]))
        kappa = np.full(k, 10.0)
        alpha = np.bincount(km.labels_, minlength=k).astype(float)
        alpha = np.clip(alpha, 1, None) / alpha.sum()
        prev_ll = -np.inf
        for _ in range(max_iter):
            # E-step (log space)
            log_norm = _log_vmf_norm(d, kappa)                     # [k]
            log_r = (np.log(alpha)[None, :] + log_norm[None, :]
                     + kappa[None, :] * (X @ mu.T))                # [n,k]
            ll = float(logsumexp(log_r, axis=1).sum())
            log_r = log_r - logsumexp(log_r, axis=1, keepdims=True)
            r = np.exp(log_r)                                      # [n,k]
            # M-step
            Nk = r.sum(0) + 1e-8
            alpha = Nk / n
            S = r.T @ X                                            # [k,d]
            norms = np.linalg.norm(S, axis=1)
            mu = S / (norms[:, None] + 1e-12)
            kappa = _kappa_from_rbar(norms / Nk, d)
            if ll - prev_ll < tol:
                break
            prev_ll = ll
        if best is None or ll > best["ll"]:
            labels = np.argmax(r, axis=1)
            best = {"ll": ll, "labels": labels, "mu": mu, "kappa": kappa, "alpha": alpha}
    # BIC: p = k*d (mu) + k (kappa) + (k-1) (weights)
    p = k * d + k + (k - 1)
    best["bic"] = -2 * best["ll"] + p * np.log(n)
    return best


def vmf_select(X, k_range, seed, n_init, max_iter):
    rows, fits = [], {}
    kmax = min(int(k_range[1]), X.shape[0] - 1)
    for k in range(int(k_range[0]), kmax + 1):
        f = _movmf_fit(X, k, seed, n_init, max_iter)
        fits[k] = f
        rows.append({"k": k, "bic": f["bic"], "loglik": f["ll"]})
    best_k = min(rows, key=lambda r: r["bic"])["k"]
    return best_k, rows, fits[best_k]


# ---------------------------------------------------------------------------
# spherical k-means
# ---------------------------------------------------------------------------
def spherical_select(X, k_range, seed):
    rows = {}
    best_k, best_sil = int(k_range[0]), -np.inf
    kmax = min(int(k_range[1]), X.shape[0] - 1)
    for k in range(int(k_range[0]), kmax + 1):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
        sil = silhouette_score(X, lab, metric="cosine") if len(set(lab)) > 1 else np.nan
        rows[k] = sil
        if np.isfinite(sil) and sil > best_sil:
            best_sil, best_k = sil, k
    labels = KMeans(best_k, n_init=25, random_state=seed).fit_predict(X)
    return best_k, rows, labels


# ---------------------------------------------------------------------------
# reliability split + trajectory consistency
# ---------------------------------------------------------------------------
def reliability_split(norm, spread, dp):
    pct = float(dp.get("reliable_magnitude_percentile", 50))
    cut = np.percentile(norm, pct)
    reliable = norm >= cut
    msp = dp.get("max_spread_percentile", None)
    if msp is not None:
        reliable &= spread <= np.percentile(spread, float(msp))
    return reliable, float(cut)


def _consecutive_consistency(prog_ids, patient_id, delta):
    """Per patient: cosine between consecutive deltas (t1->t2 vs t2->t3), tagged
    with each step's magnitude so persistence can be read against reliability."""
    norm = np.linalg.norm(delta, axis=1)
    df = pd.DataFrame({"pid": patient_id, "gid": prog_ids, "idx": np.arange(len(prog_ids))})
    parts = df["gid"].astype(str).str.split("__", expand=True)
    df["before"], df["after"] = parts[1], parts[2]
    pairs = []
    for pid, g in df.groupby("pid"):
        if len(g) < 2:
            continue
        g = g.sort_values("before")
        rows = g.to_dict("records")
        for a, b in zip(rows, rows[1:]):
            if a["after"] == b["before"]:                 # truly consecutive
                ia, ib = a["idx"], b["idx"]
                c = float(delta[ia] @ delta[ib] / (norm[ia] * norm[ib] + 1e-12))
                pairs.append({"patient": int(pid), "cosine": c,
                              "mag_step1": float(norm[ia]), "mag_step2": float(norm[ib])})
    return pairs


def robustness_sweep(U, norm, patient_id, dp, seed, percentiles):
    """Re-split at several magnitude thresholds; force k=2 (the two dominant
    modes) and report whether the two high-magnitude phenotypes persist."""
    cap0 = int(dp.get("direction_dims", 10))
    var_target = float(dp.get("direction_var", 0.9))
    rows = []
    for pct in percentiles:
        cut = np.percentile(norm, pct)
        rel = norm >= cut
        if rel.sum() < 6:
            continue
        cap = int(min(cap0, rel.sum() - 1, U.shape[1]))
        pca = PCA(n_components=cap, random_state=seed).fit(U[rel])
        cum = np.cumsum(pca.explained_variance_ratio_)
        r = int(max(2, min(cap, np.searchsorted(cum, var_target) + 1)))
        Xr = _unit(pca.transform(U[rel])[:, :r])
        lab = KMeans(2, n_init=10, random_state=seed).fit_predict(Xr)
        sil = float(silhouette_score(Xr, lab, metric="cosine")) if len(set(lab)) > 1 else np.nan
        idx = np.where(rel)[0]
        clusters = []
        for c in sorted(set(lab)):
            m = lab == c; gi = idx[m]
            clusters.append({"n": int(m.sum()),
                             "n_patients": int(len(set(patient_id[gi].tolist()))),
                             "mean_magnitude": float(norm[gi].mean()),
                             "resultant_length": float(np.linalg.norm(Xr[m].mean(0)))})
        rows.append({"percentile": int(pct), "cut": float(cut),
                     "n_reliable": int(rel.sum()), "k2_silhouette": sil,
                     "clusters": clusters})
    return rows


# ---------------------------------------------------------------------------
def _characterize(labels, reliable_idx, norm, spread, patient_id, mu=None, kappa=None):
    out = []
    for c in sorted(set(labels)):
        m = labels == c
        gi = reliable_idx[m]
        pats = patient_id[gi]
        rec = {"cluster": int(c), "n": int(m.sum()),
               "n_patients": int(len(set(pats.tolist()))),
               "single_patient_dominated": bool(pd.Series(pats).value_counts(normalize=True).iloc[0] > 0.6),
               "mean_magnitude": float(norm[gi].mean()),
               "mean_spread": float(spread[gi].mean())}
        if kappa is not None:
            rec["kappa"] = float(kappa[c])
        out.append(rec)
    return out


def _plot(emb2, reliable, labels_v, angles, cosines, vmf_rows, sph_rows, path, dpi):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    # (a) reliable-only direction scatter, colored by vMF; stable grey
    a = ax[0, 0]
    a.scatter(emb2[~reliable, 0], emb2[~reliable, 1], s=18, color="#CCCCCC",
              label="low-change (stable)")
    ridx = np.where(reliable)[0]
    for c in sorted(set(labels_v)):
        m = labels_v == c
        a.scatter(emb2[ridx[m], 0], emb2[ridx[m], 1], s=32, color=_PAL[c % len(_PAL)],
                  label=f"pheno {c} (n={int(m.sum())})", edgecolor="white", linewidth=0.4)
    a.set_title("(a) reliable directions by vMF phenotype")
    a.set_xlabel("dir PC1"); a.set_ylabel("dir PC2"); a.legend(fontsize=7, frameon=False)
    # (b) angle rose of reliable directions
    b = plt.subplot(2, 2, 2, projection="polar")
    b.hist(angles, bins=24, color="#3b6ea5")
    b.set_title("(b) reliable direction angles (modes?)", pad=18)
    # (c) consecutive-delta consistency
    c_ax = ax[1, 0]
    if cosines:
        c_ax.hist(cosines, bins=np.linspace(-1, 1, 13), color="#228833")
        c_ax.axvline(np.mean(cosines), color="#c0603a", ls="--",
                     label=f"mean={np.mean(cosines):.2f}")
        c_ax.legend(fontsize=8, frameon=False)
    else:
        c_ax.text(0.5, 0.5, "no consecutive pairs", ha="center", va="center")
    c_ax.set_title("(c) trajectory persistence: cos(step1, step2)")
    c_ax.set_xlabel("cosine"); c_ax.set_xlim(-1, 1)
    # (d) selection curves
    d = ax[1, 1]
    ks_v = [r["k"] for r in vmf_rows]; bic = [r["bic"] for r in vmf_rows]
    d.plot(ks_v, bic, "o-", color="#4477AA", label="vMF BIC (min=best)")
    d.set_xlabel("k"); d.set_ylabel("vMF BIC", color="#4477AA")
    d2 = d.twinx()
    ks_s = sorted(sph_rows); sil = [sph_rows[k] for k in ks_s]
    d2.plot(ks_s, sil, "s--", color="#EE6677", label="spherical silhouette (max=best)")
    d2.set_ylabel("silhouette", color="#EE6677")
    d.set_title("(d) k selection")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main(config: Config) -> str:
    seed = int(config["seed"])
    dp = config.get("directional_phenotype", {})
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    delta = dz["delta"].astype(float)
    prog_ids = dz["progression_id"]
    patient_id = dz["patient_id"]
    spread = dz["spread_std"] if "spread_std" in dz else np.zeros(len(delta))
    norm = np.linalg.norm(delta, axis=1)

    # 1. reliability split
    reliable, cut = reliability_split(norm, spread, dp)
    n_rel = int(reliable.sum())
    print(f"[dir_pheno] reliable (|delta|>=p{dp.get('reliable_magnitude_percentile',50)}"
          f"={cut:.3g}): {n_rel}/{len(delta)}; low-change/stable: {len(delta)-n_rel}")

    # 2. direction space: keep the low-rank directional subspace. Fixing a large
    # dim includes noise PCs that vMF-BIC overfits; instead keep enough PCs to
    # reach `direction_var` cumulative variance (capped by direction_dims).
    U = _unit(delta)
    cap = int(min(dp.get("direction_dims", 10), n_rel - 1, U.shape[1]))
    dir_pca = PCA(n_components=cap, random_state=seed).fit(U[reliable])
    cum = np.cumsum(dir_pca.explained_variance_ratio_)
    var_target = float(dp.get("direction_var", 0.9))
    r_dims = int(max(2, min(cap, np.searchsorted(cum, var_target) + 1)))
    Xr = _unit(dir_pca.transform(U[reliable])[:, :r_dims])   # sphere in reduced dims
    emb2 = PCA(n_components=2, random_state=seed).fit_transform(U)   # shared 2D view
    print(f"[dir_pheno] direction subspace: {r_dims} PCs "
          f"({cum[r_dims-1]:.0%} of directional variance)")

    kr = dp.get("k_range", config["cluster"]["k_range"])
    # 3a. vMF
    vk, vmf_rows, vfit = vmf_select(Xr, kr, seed,
                                    int(dp.get("vmf_n_init", 5)),
                                    int(dp.get("vmf_max_iter", 100)))
    # 3b. spherical
    sk, sph_rows, sph_labels = spherical_select(Xr, kr, seed)

    reliable_idx = np.where(reliable)[0]
    char_vmf = _characterize(vfit["labels"], reliable_idx, norm, spread, patient_id,
                             kappa=vfit["kappa"])
    char_sph = _characterize(sph_labels, reliable_idx, norm, spread, patient_id)

    # 4. trajectory consistency (tagged with step magnitudes) + angle rose
    pairs = _consecutive_consistency(prog_ids, patient_id, delta)
    cos_list = [p["cosine"] for p in pairs]
    med_mag = float(np.median(norm))
    hm_cos = [p["cosine"] for p in pairs
              if min(p["mag_step1"], p["mag_step2"]) >= med_mag]
    angles = np.arctan2(emb2[reliable, 1], emb2[reliable, 0])

    # 5. robustness: does the 2-phenotype structure survive higher thresholds?
    sweep_pcts = dp.get("robustness_percentiles", [40, 50, 60, 70, 80])
    sweep = robustness_sweep(U, norm, patient_id, dp, seed, sweep_pcts)

    result = {
        "n_total": int(len(delta)), "n_reliable": n_rel,
        "magnitude_cut": cut, "direction_dims": r_dims,
        "vmf": {"chosen_k": vk, "bic_curve": vmf_rows,
                "kappa": vfit["kappa"].tolist(), "characterization": char_vmf},
        "spherical": {"chosen_k": sk, "silhouette_curve": sph_rows,
                      "characterization": char_sph},
        "vmf_vs_spherical_ari": _ari(vfit["labels"], sph_labels),
        "trajectory_consistency": {
            "n_consecutive_pairs": len(pairs),
            "mean_cosine": float(np.mean(cos_list)) if cos_list else None,
            "mean_cosine_high_magnitude": float(np.mean(hm_cos)) if hm_cos else None,
            "n_high_magnitude_pairs": len(hm_cos),
            "pairs": pairs},
        "robustness_sweep": sweep,
    }
    io.write_json(result, config.out("directional_phenotype.json"))

    lab_full_v = np.full(len(delta), -1); lab_full_v[reliable] = vfit["labels"]
    lab_full_s = np.full(len(delta), -1); lab_full_s[reliable] = sph_labels
    pd.DataFrame({"progression_id": prog_ids, "patient_id": patient_id,
                  "magnitude": norm, "spread_std": spread, "is_reliable": reliable,
                  "vmf_label": lab_full_v, "spherical_label": lab_full_s}
                 ).to_csv(config.out("directional_phenotype_labels.csv"), index=False)

    _plot(emb2, reliable, vfit["labels"], angles, cos_list, vmf_rows, sph_rows,
          config.out("directional_phenotype.png"), int(config["report"]["fig_dpi"]))

    print(f"[dir_pheno] vMF chose k={vk} (BIC); spherical chose k={sk} (silhouette); "
          f"agreement ARI={result['vmf_vs_spherical_ari']:.2f}")
    if cos_list:
        print(f"[dir_pheno] trajectory persistence: mean cos={np.mean(cos_list):.2f} "
              f"(all {len(cos_list)} pairs); "
              f"{('mean cos=%.2f (%d high-mag pairs)' % (np.mean(hm_cos), len(hm_cos))) if hm_cos else 'no high-mag pairs'}")
    print("[dir_pheno] robustness sweep (k=2 forced):")
    for row in sweep:
        cl = ", ".join(f"n={c['n']}/pat{c['n_patients']}/mag{c['mean_magnitude']:.2f}/R{c['resultant_length']:.2f}"
                       for c in row["clusters"])
        print(f"[dir_pheno]   p{row['percentile']} (n={row['n_reliable']}, "
              f"sil={row['k2_silhouette']:.2f}): {cl}")
    print(f"[dir_pheno] wrote directional_phenotype.{{json,png,csv}} to {config.output_dir}")
    return config.out("directional_phenotype.png")


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directional phenotyping (vMF + spherical)")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
