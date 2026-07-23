"""Are the OTHER protruding directions on the compass real, or projection/continuum?

The change compass shows the validated bipolar axis plus several apparent spokes
(e.g. an 'up' and a 'down-left' cluster). This asks, per apparent direction,
whether it is a genuine mode or an artifact of (a) the 2-D projection and (b)
small-n clustering of a continuum. A direction has to clear THREE things, not one:

  1. CONCENTRATION vs a FAIR continuum null - is the cluster tighter (higher
     resultant length) than the same-rank cluster you get by running the
     *identical* clustering procedure (same k, same n_init, same subspace
     dimension the labels were derived in) on structureless isotropic
     directions of matched n? The earlier version clustered the null more
     weakly and in a fresh 10-D space, so a hand-picked n=3 cluster beat it
     trivially. Here the null optimises exactly as hard as the observed labels
     did - apples to apples - which is what a gap statistic requires.
  2. STABILITY - does the cluster survive resampling? A patient-clustered
     Hennig-style Jaccard bootstrap re-derives the clustering on resampled
     progressions; a real mode keeps its members together (mean Jaccard high),
     an optimiser-selected micro-cluster dissolves.
  3. AXIS MEMBERSHIP - does the direction participate in a pair that clears the
     angle-null permutation test (phenotype_geometry.json)?

Verdict: a direction is a SUPPORTED mode only if it is part of a validated axis
(3), OR it is BOTH tighter than the fair continuum (1) AND stable under the
bootstrap (2). Concentration alone no longer suffices - that was the loophole
that let tiny clusters through. Everything else is reported as
'continuum / projection'.

Reads deltas.npz + directional_phenotype_labels.csv (+ directional_phenotype.json
for the subspace dimension, + phenotype_geometry.json for the axis).
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
from matplotlib.colors import LinearSegmentedColormap

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .config import Config, load_config, add_arg
from . import io
from .directional_phenotype import _unit

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "gray": "#999999", "black": "#000000"}

# match the FINAL spherical k-means in directional_phenotype (n_init=25) so the
# null clusters isotropic data exactly as hard as the observed labels were fit.
NINIT = 25


def _resultant(Xu):
    return float(np.linalg.norm(Xu.mean(0))) if len(Xu) else 0.0


def _cluster(Xu, k, seed):
    """The identical clustering procedure used for the observed labels."""
    return KMeans(k, n_init=NINIT, random_state=seed).fit_predict(Xu)


def _label_dims(config: Config, fallback: int) -> int:
    """Recover the subspace dimension the observed labels were actually derived
    in (directional_phenotype fits PCA to `direction_var` cumulative variance,
    capped by direction_dims - usually far fewer than 10). Measuring
    concentration in the SAME space is what makes observed R and the labels
    consistent. Falls back to `fallback` if the json is absent."""
    p = config.out("directional_phenotype.json")
    if os.path.exists(p):
        try:
            j = io.read_json(p)
            d = int(j.get("direction_dims", 0))
            if d >= 2:
                return d
        except Exception:
            pass
    return fallback


def _jaccard_bootstrap(X, ref_labels, k, patient, seed, B):
    """Patient-clustered Hennig clusterboot. For each reference cluster, the
    mean (over bootstraps) of the best Jaccard overlap with any bootstrap
    cluster. >=0.75 stable, 0.6-0.75 borderline, <0.6 dissolves. Resampling is
    at the PATIENT level so a mode carried by one patient cannot look stable."""
    rng = np.random.default_rng(seed + 777)
    uniq_pat = np.unique(patient)
    clusters = sorted(set(ref_labels.tolist()))
    ref_sets = {c: set(np.where(ref_labels == c)[0].tolist()) for c in clusters}
    jacc = {c: [] for c in clusters}
    used = 0
    for b in range(B):
        samp_pat = rng.choice(uniq_pat, size=len(uniq_pat), replace=True)
        idx = np.concatenate([np.where(patient == p)[0] for p in samp_pat])
        present = set(np.unique(idx).tolist())
        if len(present) < k + 1:
            continue
        used += 1
        lb = _cluster(X[idx], k, seed + b + 1)
        # bootstrap clusters as sets of ORIGINAL indices (dedup resample copies)
        boot_sets = [set(np.unique(idx[lb == d]).tolist()) for d in sorted(set(lb.tolist()))]
        for c in clusters:
            C = ref_sets[c] & present          # reference members present in this resample
            if not C:
                continue
            best = 0.0
            for D in boot_sets:
                u = len(C | D)
                if u:
                    best = max(best, len(C & D) / u)
            jacc[c].append(best)
    return {c: (float(np.mean(jacc[c])) if jacc[c] else np.nan) for c in clusters}, used


def main(config: Config) -> str:
    seed = int(config["seed"])
    dp = config.get("directional_phenotype", {})
    which = dp.get("geometry_label", "spherical_label")
    B = int(dp.get("mode_null_perm", 500))
    B_boot = int(dp.get("mode_boot", 500))
    stab_min = float(dp.get("mode_stability_min", 0.60))
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
    # measure concentration in the SAME subspace the labels were derived in
    r_default = int(min(dp.get("direction_dims", 10), max(2, rel.sum() - 1), D))
    r = int(min(_label_dims(config, r_default), max(2, rel.sum() - 1), D))
    pca = PCA(n_components=max(r, 2), random_state=seed).fit(Ur)
    X = _unit(pca.transform(Ur)[:, :r])                # direction subspace (matches phenotyping)
    P2 = pca.transform(Ur)[:, :2]                      # 2-D compass plane for reporting angles

    clusters = sorted(set(labr.tolist()))
    k = len(clusters)
    Robs = {c: _resultant(X[labr == c]) for c in clusters}

    # ---- FAIR continuum null: identical procedure on isotropic directions ----
    # (same k, same n_init, same dimension r). Rank-matched: the tightest
    # observed cluster is compared to the tightest null cluster, etc.
    rng = np.random.default_rng(seed)
    null = np.full((B, k), np.nan)
    for b in range(B):
        Rr = _unit(rng.normal(size=(len(X), r)))
        lb = _cluster(Rr, k, seed + b + 1)
        vals = sorted((_resultant(Rr[lb == c]) for c in set(lb.tolist())), reverse=True)
        null[b, :len(vals)] = vals[:k]

    # ---- STABILITY: patient-clustered Jaccard bootstrap ----
    stability, n_boot_used = _jaccard_bootstrap(X, labr, k, patr, seed, B_boot)

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
        nmean = geo.get("angle_null", {}).get("null_mean_deg", 90)
        for pair, info in geo.get("pairwise_angles", {}).items():
            if info.get("p_two_sided", 1.0) < 0.05 and info.get("angle_deg", 0) > nmean:
                a, b_ = pair.split("-"); axis_members |= {int(a), int(b_)}
                axis_pairs.append((pair, info["angle_deg"], info["p_two_sided"]))

    # ---- per-cluster verdict (rank-matched concentration + stability + axis) ----
    order = sorted(clusters, key=lambda c: -Robs[c])   # tightest first
    rows = []
    for rank, c in enumerate(order):
        col = null[:, rank]; col = col[np.isfinite(col)]
        p_conc = float(np.mean(col >= Robs[c])) if len(col) else np.nan
        m = labr == c
        ang = float(np.degrees(np.arctan2(P2[m, 1].mean(), P2[m, 0].mean())) % 360)
        in_axis = int(c) in axis_members
        stab = stability.get(c, np.nan)
        conc_ok = np.isfinite(p_conc) and p_conc < 0.05
        stab_ok = np.isfinite(stab) and stab >= stab_min
        supported = in_axis or (conc_ok and stab_ok)
        # A GEOMETRIC mode = a bundle of near-parallel directions that is both
        # tighter than the fair continuum AND reproducible. That is necessary
        # but NOT sufficient to call it a clinical PHENOTYPE / AXIS: only a
        # direction that also clears the angle-null (an antipodal, validated
        # axis) is interpretable as an axis, and a phenotype claim additionally
        # needs a distinct QEEG signature (tested in phenotype_qeeg, NOT here).
        geom_mode = conc_ok and stab_ok
        why = []
        if geom_mode:
            why.append("tighter than fair continuum AND bootstrap-stable")
        elif conc_ok and not stab_ok:
            why.append("tighter than continuum but NOT stable (dissolves under resampling)")
        elif not conc_ok:
            why.append("not tighter than a fair, equally-hard continuum")
        rows.append({"cluster": int(c), "n": int(m.sum()),
                     "n_patients": int(np.unique(patr[m]).size),
                     "mean_magnitude": float(nr[m].mean()),
                     "compass_angle_deg": round(ang, 0),
                     "resultant_length": round(Robs[c], 3),
                     "continuum_null_R95": round(float(np.nanpercentile(col, 95)), 3) if len(col) else None,
                     "concentration_p": round(p_conc, 3) if np.isfinite(p_conc) else None,
                     "bootstrap_jaccard": round(float(stab), 3) if np.isfinite(stab) else None,
                     "stable": bool(stab_ok),
                     "is_geometric_mode": bool(geom_mode),
                     "in_validated_axis": in_axis,
                     "interpretable_as_axis": bool(in_axis),
                     "geometry_verdict": ("concentrated embedding mode (" + " + ".join(why) + ")"
                                          if geom_mode else "continuum / projection (" + " + ".join(why) + ")"),
                     "axis_verdict": ("VALIDATED axis (angle-null antipodal pair)" if in_axis
                                      else "not an angle-null axis (isolated spoke, not a bipolar contrast)")})

    n_geometric = sum(1 for r_ in rows if r_["is_geometric_mode"])
    n_axes = sum(1 for r_ in rows if r_["in_validated_axis"])
    n_supported = n_geometric + sum(1 for r_ in rows if r_["in_validated_axis"] and not r_["is_geometric_mode"])

    # ============================ figure ============================
    fig = plt.figure(figsize=(18, 5.6))

    # (A) resultant length vs FAIR continuum null, per cluster (tightest-first)
    axA = fig.add_subplot(1, 3, 1)
    xs = np.arange(len(order))
    Rs = [Robs[c] for c in order]
    n95 = [np.nanpercentile(null[:, i][np.isfinite(null[:, i])], 95) if np.isfinite(null[:, i]).any() else np.nan
           for i in range(len(order))]
    cols = [OI["green"] if rows[i]["is_geometric_mode"] else OI["gray"] for i in range(len(order))]
    axA.bar(xs, Rs, color=cols, edgecolor="white", zorder=3, label="observed cluster")
    axA.plot(xs, n95, "D--", color=OI["vermillion"], ms=7, zorder=4, label="fair continuum null (95th pct)")
    for i, c in enumerate(order):
        axA.text(i, Rs[i] + 0.02, f"c{c}\nn={rows[i]['n']}", ha="center", fontsize=8)
    axA.set_xticks(xs); axA.set_xticklabels([f"rank {i+1}" for i in xs], fontsize=8)
    axA.set_ylabel("resultant length (concentration)")
    axA.set_ylim(0, 1.08)
    axA.set_title("(A) Tighter than a fair, equally-hard continuum?", fontsize=10.5)
    axA.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]:
        axA.spines[s].set_visible(False)

    # (B) bootstrap stability per cluster (same tightest-first order)
    axB = fig.add_subplot(1, 3, 2)
    stabs = [rows[i]["bootstrap_jaccard"] if rows[i]["bootstrap_jaccard"] is not None else 0.0
             for i in range(len(order))]
    bcols = [OI["green"] if rows[i]["stable"] else OI["gray"] for i in range(len(order))]
    axB.bar(xs, stabs, color=bcols, edgecolor="white", zorder=3)
    axB.axhline(stab_min, color=OI["vermillion"], ls="--", lw=1.4, zorder=4,
                label=f"stability threshold ({stab_min:.2f})")
    for i, c in enumerate(order):
        axB.text(i, stabs[i] + 0.02, f"c{c}", ha="center", fontsize=8)
    axB.set_xticks(xs); axB.set_xticklabels([f"rank {i+1}" for i in xs], fontsize=8)
    axB.set_ylabel("mean bootstrap Jaccard (co-assignment)")
    axB.set_ylim(0, 1.05)
    axB.set_title("(B) Does the cluster survive resampling?", fontsize=10.5)
    axB.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]:
        axB.spines[s].set_visible(False)

    # (C) annotated compass: blue = validated axis (interpretable as a phenotype
    # axis), green = concentrated embedding mode only (NOT a validated axis),
    # black = continuum/projection.
    axC = fig.add_subplot(1, 3, 3, projection="polar")
    axC.set_theta_zero_location("E"); axC.set_theta_direction(1)
    big = max(clusters, key=lambda c: (labr == c).sum())
    flip = -1.0 if P2[labr == big, 0].mean() < 0 else 1.0
    ang_all = np.arctan2(P2[:, 1], P2[:, 0] * flip)
    axC.scatter(ang_all, nr, s=26, color=OI["gray"], alpha=0.5, zorder=2)
    for r_ in rows:
        c = r_["cluster"]; m = labr == c
        ca = np.arctan2(np.sin(ang_all[m]).sum(), np.cos(ang_all[m]).sum())
        if r_["in_validated_axis"]:
            col, lw = OI["blue"], 3.2
        elif r_["is_geometric_mode"]:
            col, lw = OI["green"], 2.4
        else:
            col, lw = OI["black"], 1.6
        axC.plot([ca, ca], [0, nr[m].mean()], color=col, lw=lw,
                 alpha=0.85, zorder=4, solid_capstyle="round")
        axC.text(ca, nr[m].mean() * 1.06, f"c{c}", color=col, fontsize=8, ha="center")
    axC.set_rlim(0, float(nr.max()) * 1.12); axC.set_yticklabels([])
    axC.tick_params(axis="x", labelsize=8); axC.grid(color="#dfe3e7", lw=0.7)
    axC.set_title("(C) blue = validated axis (phenotype-eligible),\n"
                  "green = embedding mode only, black = continuum", fontsize=9.5)

    fig.suptitle(f"What are the other directions?  {n_axes} validated axis direction(s) + "
                 f"{n_geometric} concentrated embedding mode(s) of {k}.  "
                 f"Geometric tightness ≠ phenotype: only angle-null axes are axis-eligible; "
                 f"distinct QEEG meaning is tested separately (phenotype_qeeg).",
                 fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = config.out("direction_modes.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    pd.DataFrame(rows).to_csv(config.out("direction_modes.csv"), index=False)
    io.write_json({"label_source": which, "subspace_dims": r, "n_clusters": k,
                   "n_geometric_modes": n_geometric, "n_validated_axes": n_axes,
                   "n_supported_modes": n_supported, "validated_axis_pairs": axis_pairs,
                   "null_perms": B, "bootstrap_resamples_used": n_boot_used,
                   "stability_threshold": stab_min, "cluster_n_init": NINIT,
                   "per_direction": rows,
                   "note": "TWO different questions, do not conflate them. (1) GEOMETRIC MODE "
                           "(is_geometric_mode): the bundle is more concentrated than the same-rank "
                           "cluster from an IDENTICAL clustering of structureless isotropic "
                           "directions (fair continuum null: same k, n_init and subspace dimension) "
                           "AND survives a patient-clustered Jaccard bootstrap. This certifies the "
                           "directions are genuinely near-parallel and reproducible IN THE EMBEDDING "
                           "- it does NOT make them a clinical phenotype. (2) VALIDATED AXIS "
                           "(in_validated_axis): the direction is part of an antipodal pair that "
                           "cleared the angle-null permutation test (phenotype_geometry). Only these "
                           "are interpretable as a bipolar phenotype AXIS. A PHENOTYPE claim needs a "
                           "third thing this module does not test: a distinct QEEG signature after "
                           "FDR (see phenotype_qeeg). Small-n geometric modes with no angle-null axis "
                           "and no corrected QEEG signature should be reported as embedding "
                           "sub-structure, not as separate phenotypes.",
                   "interpretation": (
                       f"{n_axes} direction(s) are validated bipolar axes (angle-null); "
                       f"{n_geometric} are concentrated embedding modes. Concentrated embedding "
                       f"modes that are NOT angle-null axes are tight, reproducible bundles of "
                       f"near-parallel deltas but are NOT established phenotypes - confirm/deny each "
                       f"against QEEG (phenotype_qeeg) and nuisance covariates before interpreting.")},
                  config.out("direction_modes.json"))

    print(f"[modes] {k} apparent directions: {n_axes} validated axis, {n_geometric} concentrated "
          f"embedding mode(s) (fair null + {n_boot_used} bootstraps).")
    for r_ in rows:
        print(f"[modes]   c{r_['cluster']} @~{r_['compass_angle_deg']:.0f}deg n={r_['n']} "
              f"R={r_['resultant_length']} (fairnull95={r_['continuum_null_R95']}) "
              f"conc_p={r_['concentration_p']} jaccard={r_['bootstrap_jaccard']} "
              f"| geom_mode={r_['is_geometric_mode']} axis={r_['in_validated_axis']} "
              f"-> {r_['axis_verdict'] if r_['in_validated_axis'] else r_['geometry_verdict']}")
    print(f"[modes] wrote direction_modes.{{json,csv,png}} to {config.output_dir}")
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Are the other compass directions real modes?")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
