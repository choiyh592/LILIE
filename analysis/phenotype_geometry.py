"""Phenotype geometry - auto-k sweep + centroid angles.

Two diagnostics on the directional phenotypes:

1. AUTO-K SWEEP - at several magnitude thresholds, let both selectors pick k
   automatically (spherical silhouette argmax, vMF BIC argmin) and report the
   full curves. If a stable small k emerges once the low-magnitude continuum is
   removed, that pins down the phenotype count.

2. CENTROID ANGLES - the mean direction of each phenotype and the pairwise
   angles between them. Antipodal (~180 deg) => opposite ends of ONE axis of
   change (e.g. improve vs decline); orthogonal (~90 deg) => independent
   processes. This decides the biological framing.

Reads deltas.npz + directional_phenotype_labels.csv. Outputs:
  phenotype_geometry.json, phenotype_geometry.png

Run:  python -m analysis.phenotype_geometry --config analysis/config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from .config import Config, load_config, add_arg
from . import io
from .directional_phenotype import _unit, spherical_select, vmf_select


def _direction_subspace(U_reliable, dp, seed):
    cap = int(min(dp.get("direction_dims", 10), U_reliable.shape[0] - 1, U_reliable.shape[1]))
    pca = PCA(n_components=cap, random_state=seed).fit(U_reliable)
    cum = np.cumsum(pca.explained_variance_ratio_)
    r = int(max(2, min(cap, np.searchsorted(cum, float(dp.get("direction_var", 0.9))) + 1)))
    return _unit(pca.transform(U_reliable)[:, :r])


def auto_k_sweep(U, norm, dp, seed, percentiles, kr):
    rows = []
    for pct in percentiles:
        rel = norm >= np.percentile(norm, pct)
        if rel.sum() < 6:
            continue
        Xr = _direction_subspace(U[rel], dp, seed)
        sk, sil_curve, _ = spherical_select(Xr, kr, seed)
        vk, vmf_rows, _ = vmf_select(Xr, kr, seed, int(dp.get("vmf_n_init", 5)),
                                     int(dp.get("vmf_max_iter", 100)))
        rows.append({"percentile": int(pct), "n_reliable": int(rel.sum()),
                     "spherical_k": sk, "silhouette_curve": sil_curve,
                     "vmf_k": vk, "vmf_bic_curve": {r["k"]: r["bic"] for r in vmf_rows}})
    return rows


def centroid_angles(U, labels):
    """Mean direction per phenotype + pairwise angles (degrees)."""
    cs = sorted(c for c in set(labels) if c >= 0)
    mus, R = {}, {}
    for c in cs:
        m = U[labels == c].mean(0)
        R[c] = float(np.linalg.norm(m))            # resultant length (concentration)
        mus[c] = m / (np.linalg.norm(m) + 1e-12)
    ang = {}
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            cos = float(np.clip(mus[a] @ mus[b], -1, 1))
            deg = float(np.degrees(np.arccos(cos)))
            rel = ("antipodal (one axis, opposite ends)" if deg > 150 else
                   "orthogonal (independent)" if 70 <= deg <= 110 else
                   "aligned" if deg < 30 else "oblique")
            ang[f"{a}-{b}"] = {"angle_deg": deg, "cosine": cos, "relationship": rel}
    return cs, R, ang


def _plot(sweep, cs, ang, path, dpi):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for row in sweep:
        ks = sorted(int(k) for k in row["silhouette_curve"])
        ax1.plot(ks, [row["silhouette_curve"][k] for k in ks], "o-",
                 label=f"p{row['percentile']} (n={row['n_reliable']}, k*={row['spherical_k']})")
    ax1.set_xlabel("k"); ax1.set_ylabel("silhouette"); ax1.legend(fontsize=7, frameon=False)
    ax1.set_title("(a) auto-k: silhouette curve per magnitude threshold")
    if len(cs) >= 2:
        M = np.full((len(cs), len(cs)), np.nan)
        for i, a in enumerate(cs):
            M[i, i] = 0.0
            for j, b in enumerate(cs):
                if f"{a}-{b}" in ang:
                    M[i, j] = M[j, i] = ang[f"{a}-{b}"]["angle_deg"]
        im = ax2.imshow(M, cmap="viridis", vmin=0, vmax=180)
        ax2.set_xticks(range(len(cs))); ax2.set_yticks(range(len(cs)))
        ax2.set_xticklabels(cs); ax2.set_yticklabels(cs)
        for i in range(len(cs)):
            for j in range(len(cs)):
                if np.isfinite(M[i, j]):
                    ax2.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center",
                             color="white", fontsize=9)
        fig.colorbar(im, ax=ax2, fraction=0.046, label="angle (deg)")
    ax2.set_title("(b) pairwise angle between phenotype directions")
    fig.tight_layout(); fig.savefig(path, dpi=dpi); plt.close(fig)


def main(config: Config) -> str:
    seed = int(config["seed"])
    dp = config.get("directional_phenotype", {})
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    delta = dz["delta"].astype(float)
    U = _unit(delta)
    norm = np.linalg.norm(delta, axis=1)

    lab = pd.read_csv(config.out("directional_phenotype_labels.csv"))
    which = dp.get("geometry_label", "spherical_label")
    labels = lab[which].to_numpy()

    kr = dp.get("k_range", config["cluster"]["k_range"])
    pcts = dp.get("robustness_percentiles", [40, 50, 60, 70, 80])
    sweep = auto_k_sweep(U, norm, dp, seed, pcts, kr)
    cs, R, ang = centroid_angles(U, labels)

    result = {"label_source": which, "auto_k_sweep": sweep,
              "phenotypes": cs, "resultant_length": {int(c): R[c] for c in cs},
              "pairwise_angles": ang}
    io.write_json(result, config.out("phenotype_geometry.json"))
    _plot(sweep, cs, ang, config.out("phenotype_geometry.png"),
          int(config["report"]["fig_dpi"]))

    print("[geometry] auto-k per threshold: " +
          ", ".join(f"p{r['percentile']}:sph_k={r['spherical_k']}/vmf_k={r['vmf_k']}"
                    for r in sweep))
    for pair, a in ang.items():
        print(f"[geometry] phenotypes {pair}: {a['angle_deg']:.0f} deg -> {a['relationship']}")
    print(f"[geometry] wrote phenotype_geometry.{{json,png}} to {config.output_dir}")
    return config.out("phenotype_geometry.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phenotype geometry - auto-k + angles")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
