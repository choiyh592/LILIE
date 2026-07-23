"""Are the OTHER protruding directions on the compass real, or projection/continuum?

The change compass shows the validated bipolar axis plus several apparent spokes
(e.g. an 'up' and a 'down-left' cluster). This asks, per apparent direction,
whether it is a genuine mode or an artifact of (a) the 2-D projection and (b)
small-n clustering of a continuum. Two independent criteria:

  1. CONCENTRATION vs a continuum null - is the cluster tighter (higher resultant
     length) than the same-rank cluster you get by running the SAME spherical
     k-means on structureless isotropic directions of matched n and dimension?
     (gap-statistic logic: a cluster of n=3 in high-D is trivially tight, so it
     only counts if it beats what a no-structure continuum produces at that size.)
  2. AXIS MEMBERSHIP - does the direction participate in a pair that clears the
     angle-null permutation test (phenotype_geometry.json)?

A direction is a SUPPORTED mode only if it passes (1) or (2); else it is reported
as 'continuum / projection'. Output is a per-direction table + a two-panel figure
(resultant length vs continuum null; annotated compass) you can drop in as the
supplementary that pre-empts the reviewer question.

Reads deltas.npz + directional_phenotype_labels.csv (+ phenotype_geometry.json).
Outputs: direction_modes.{json,csv,png}
Run:  python -m analysis.direction_modes --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import Config, load_config, add_arg
from . import io
from .directional_phenotype import _unit

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "gray": "#999999", "black": "#000000"}


def _resultant(Xu):
    return float(np.linalg.norm(Xu.mean(0))) if len(Xu) else 0.0


def _sph(Xu, k, seed):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(Xu)


def main(config: Config) -> str:
    seed = int(config["seed"])
    dp = config.get("directional_phenotype", {})
    which = dp.get("geometry_label", "spherical_label")
    B = int(dp.get("mode_null_perm", 500))
    dpi = int(config["report"]["fig_dpi"])

    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    delta = dz["delta"].astype(float)
    pid = np.array([str(x) for x in dz["progression_id"]])
    patient = np.array(dz["patient_id"])
    U = _unit(delta); norm = np.linalg.norm(delta, axis=1)

    lab = pd.read_csv(config.out("directional_phenotype_labels.csv"))
    lab["progression_id"] = lab["progression_id"].astype(str)
    lut = dict(zip(lab["progression_id"], lab[which]))
    rlut = dict(zip(lab["progression_id"], lab["is_reliable"].astype(bool)))
    labels = np.array([lut.get(p, -1) for p in pid], dtype=float)
    rel = np.array([bool(rlut.get(p, False)) for p in pid]) & (labels >= 0)

    Ur, labr, patr, nr = U[rel], labels[rel].astype(int), patient[rel], norm[rel]
    D = Ur.shape[1]
    r = int(min(dp.get("direction_dims", 10), max(2, rel.sum() - 1), D))
    pca = PCA(n_components=r, random_state=seed).fit(Ur)
    X = _unit(pca.transform(Ur))                       # direction subspace (matches phenotyping)
    P2 = pca.transform(Ur)[:, :2]                      # 2-D compass plane for reporting angles

    clusters = sorted(set(labr.tolist()))
    k = len(clusters)
    Robs = {c: _resultant(X[labr == c]) for c in clusters}

    # ---- continuum null: cluster isotropic directions into the same k ----
    rng = np.random.default_rng(seed)
    null = np.full((B, k), np.nan)
    for b in range(B):
        Rr = _unit(rng.normal(size=(len(X), r)))
        lb = _sph(Rr, k, seed + b + 1)
        vals = sorted((_resultant(Rr[lb == c]) for c in set(lb)), reverse=True)
        null[b, :len(vals)] = vals[:k]

    # ---- axis membership from the angle-null ----
    geo = None
    gp = config.out("phenotype_geometry.json")
    if os.path.exists(gp):
        try:
            geo = io.read_json(gp)
        except Exception:
            geo = None
    axis_members = set()
    axis_pairs = []
    if geo:
        for pair, info in geo.get("pairwise_angles", {}).items():
            if info.get("p_two_sided", 1.0) < 0.05 and info.get("angle_deg", 0) > geo.get("angle_null", {}).get("null_mean_deg", 90):
                a, b_ = pair.split("-"); axis_members |= {int(a), int(b_)}
                axis_pairs.append((pair, info["angle_deg"], info["p_two_sided"]))

    # ---- per-cluster verdict (rank-matched concentration p) ----
    order = sorted(clusters, key=lambda c: -Robs[c])   # tightest first
    rows = []
    for rank, c in enumerate(order):
        col = null[:, rank]; col = col[np.isfinite(col)]
        p_conc = float(np.mean(col >= Robs[c])) if len(col) else np.nan
        m = labr == c
        ang = float(np.degrees(np.arctan2(P2[m, 1].mean(), P2[m, 0].mean())) % 360)
        in_axis = int(c) in axis_members
        supported = (np.isfinite(p_conc) and p_conc < 0.05) or in_axis
        why = []
        if in_axis: why.append("validated axis (angle-null)")
        if np.isfinite(p_conc) and p_conc < 0.05: why.append("tighter than continuum")
        rows.append({"cluster": int(c), "n": int(m.sum()), "n_patients": int(np.unique(patr[m]).size),
                     "mean_magnitude": float(nr[m].mean()), "compass_angle_deg": round(ang, 0),
                     "resultant_length": round(Robs[c], 3),
                     "continuum_null_R95": round(float(np.nanpercentile(col, 95)), 3) if len(col) else None,
                     "concentration_p": round(p_conc, 3) if np.isfinite(p_conc) else None,
                     "in_validated_axis": in_axis,
                     "verdict": "SUPPORTED mode (" + " + ".join(why) + ")" if supported
                                else "continuum / projection (not distinguishable)"})

    n_supported = sum(1 for r_ in rows if r_["verdict"].startswith("SUPPORTED"))

    # ============================ figure ============================
    dcm = LinearSegmentedColormap.from_list("oi", [OI["vermillion"], "#F4F4F4", OI["blue"]], N=256)
    fig = plt.figure(figsize=(14, 5.8))

    # (A) resultant length vs continuum null, per cluster (sorted tightest-first)
    axA = fig.add_subplot(1, 2, 1)
    xs = np.arange(len(order))
    Rs = [Robs[c] for c in order]
    n95 = [np.nanpercentile(null[:, i][np.isfinite(null[:, i])], 95) if np.isfinite(null[:, i]).any() else np.nan
           for i in range(len(order))]
    cols = [OI["green"] if rows[i]["verdict"].startswith("SUPPORTED") else OI["gray"] for i in range(len(order))]
    axA.bar(xs, Rs, color=cols, edgecolor="white", zorder=3,
            label="observed cluster")
    axA.plot(xs, n95, "D--", color=OI["vermillion"], ms=7, zorder=4, label="continuum null (95th pct)")
    for i, c in enumerate(order):
        axA.text(i, Rs[i] + 0.02, f"c{c}\nn={rows[i]['n']}", ha="center", fontsize=8)
    axA.set_xticks(xs); axA.set_xticklabels([f"rank {i+1}" for i in xs], fontsize=8)
    axA.set_ylabel("resultant length (concentration)")
    axA.set_ylim(0, 1.08)
    axA.set_title("(A) Is each direction tighter than a structureless continuum?", fontsize=10.5)
    axA.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]:
        axA.spines[s].set_visible(False)

    # (B) annotated compass: centroids as arrows, coloured by verdict
    axB = fig.add_subplot(1, 2, 2, projection="polar")
    axB.set_theta_zero_location("E"); axB.set_theta_direction(1)
    # orient PC1 so the largest cluster is positive (cosmetic)
    big = max(clusters, key=lambda c: (labr == c).sum())
    flip = -1.0 if P2[labr == big, 0].mean() < 0 else 1.0
    ang_all = np.arctan2(P2[:, 1], P2[:, 0] * flip)
    axB.scatter(ang_all, nr, s=26, color=OI["gray"], alpha=0.5, zorder=2)
    for r_ in rows:
        c = r_["cluster"]; m = labr == c
        ca = np.arctan2(np.sin(ang_all[m]).sum(), np.cos(ang_all[m]).sum())
        col = OI["green"] if r_["verdict"].startswith("SUPPORTED") else OI["black"]
        axB.plot([ca, ca], [0, nr[m].mean()], color=col, lw=3 if col == OI["green"] else 1.6,
                 alpha=0.8, zorder=4, solid_capstyle="round")
        axB.text(ca, nr[m].mean() * 1.06, f"c{c}", color=col, fontsize=8, ha="center")
    axB.set_rlim(0, float(nr.max()) * 1.12); axB.set_yticklabels([])
    axB.tick_params(axis="x", labelsize=8); axB.grid(color="#dfe3e7", lw=0.7)
    axB.set_title("(B) directions: green = supported mode, black = continuum/projection", fontsize=10)

    fig.suptitle(f"What are the other directions?  {n_supported}/{k} survive as real modes "
                 f"(only the validated axis + anything tighter than continuum)", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = config.out("direction_modes.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    pd.DataFrame(rows).to_csv(config.out("direction_modes.csv"), index=False)
    io.write_json({"label_source": which, "subspace_dims": r, "n_clusters": k,
                   "n_supported_modes": n_supported, "validated_axis_pairs": axis_pairs,
                   "per_direction": rows,
                   "note": "A direction is SUPPORTED only if it clears the angle-null (part of a "
                           "validated antipodal axis) OR is more concentrated than the same-rank "
                           "cluster from structureless isotropic directions (continuum null). "
                           "Otherwise it is the continuum viewed through the 2-D projection.",
                   "interpretation": ("Only the bipolar axis is a supported direction; the other "
                                      "protrusions are continuum/projection." if n_supported <= 2 else
                                      "More than the axis survived - inspect the supported clusters.")},
                  config.out("direction_modes.json"))

    print(f"[modes] {k} apparent directions; {n_supported} SUPPORTED as real modes.")
    for r_ in rows:
        print(f"[modes]   c{r_['cluster']} @~{r_['compass_angle_deg']:.0f}deg n={r_['n']} "
              f"R={r_['resultant_length']} (null95={r_['continuum_null_R95']}) "
              f"conc_p={r_['concentration_p']} axis={r_['in_validated_axis']} -> {r_['verdict']}")
    print(f"[modes] wrote direction_modes.{{json,csv,png}} to {config.output_dir}")
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Are the other compass directions real modes?")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
