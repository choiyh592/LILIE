"""Phenotype compass - the 'pretty' continuum + directional-phenotype figure.

Replaces the flat PC1xPC2 scatter with a view that puts BOTH variables that
define the story on axes:

  (A) CHANGE COMPASS (polar): angle = direction of change (top-2 PCs of the
      RELIABLE unit directions), radius = magnitude of change. The shaded central
      disk (|delta| < reliability cut) IS the low-magnitude continuum; reliable
      phenotypes fan out as colored spokes at the rim.

      Every point - including the low-magnitude core - is tinted by its NEAREST
      phenotype direction (full-D cosine, not the 2-D shadow), with opacity scaled
      by magnitude: faint = tentative near the centre, bold = reliable at the rim.
      This shows the continuum resolving into the phenotype spokes. BUT coloring
      the core is only honest if the low-magnitude directions really do lean toward
      the phenotypes more than chance - so we run a permutation test
      (`core_alignment`) and, in 'auto' mode, only color the core if it PASSES;
      otherwise the core stays gray (its direction is noise). Controls (black x)
      are projected into the same space and should sit in the core.

  (B) ANGULAR DENSITY (polar): a von Mises circular KDE of the reliable directions.

  (C) FAITHFULNESS CHECK (cartesian, auto-added if the 2-D angle is not faithful,
      i.e. PC1+PC2 below `compass_pc12_faithful_min`, or PC3 individually large):
      PC1 x PC3 of the reliable directions.

The reliable-set PCA explained-variance ratio is printed and written to JSON - the
empirical answer to 'is the 2-D angle faithful / would a 3rd PC help the picture?'.
This only affects the DRAWING; phenotype labels come from directional_phenotype.py
and are unchanged.

Reads deltas.npz + directional_phenotype_labels.csv (+ optional control_deltas.npz).
Outputs: phenotype_compass.png, phenotype_compass.json

Run:  python -m analysis.phenotype_compass --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from scipy.special import i0
from sklearn.decomposition import PCA

from .config import Config, load_config, add_arg
from . import io
from .directional_phenotype import _unit

# colorblind-friendly (Tol bright); index = phenotype label
_PAL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
        "#EE8866", "#77AADD"]
_STABLE = "#C4C8CC"
_CONTROL = "#111111"


def _color(lab):
    return _PAL[int(lab) % len(_PAL)]


def _circular_mean(angles):
    return float(np.arctan2(np.sin(angles).sum(), np.cos(angles).sum()))


def _vm_density(theta_grid, angles, kappa):
    """Von Mises kernel density on the circle, evaluated on theta_grid."""
    if len(angles) == 0:
        return np.zeros_like(theta_grid)
    d = np.exp(kappa * np.cos(theta_grid[:, None] - angles[None, :])).sum(1)
    return d / (2 * np.pi * i0(kappa) * len(angles))


def _centroids(U, labels, order):
    """Unit mean direction (full-D) per reliable phenotype label."""
    C = []
    for c in order:
        m = labels == c
        v = U[m].mean(0)
        C.append(v / (np.linalg.norm(v) + 1e-12))
    return np.array(C)                                # (k, D)


def _core_alignment_test(U_core, C, seed, n_perm):
    """Do the low-magnitude (core) directions lean toward the phenotype
    centroids MORE than random directions? Statistic = mean over core points of
    the max cosine to any centroid; null = same for random unit directions in the
    same dimension. Returns (p, observed, null_mean)."""
    if U_core.shape[0] == 0 or C.shape[0] == 0:
        return np.nan, np.nan, np.nan
    obs = float((U_core @ C.T).max(1).mean())
    rng = np.random.default_rng(seed)
    D = U_core.shape[1]
    null = np.empty(n_perm)
    for i in range(n_perm):
        R = _unit(rng.normal(size=(U_core.shape[0], D)))
        null[i] = (R @ C.T).max(1).mean()
    p = float((null >= obs).mean())
    return p, obs, float(null.mean())


def _merge_groups(C, order, deg):
    """Union reliable phenotypes whose centroids are within `deg` of each other
    (candidate single broad direction, e.g. a 'down' mode over-split by clustering)."""
    k = len(order)
    ang = np.full((k, k), np.nan)
    parent = list(range(k))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    for i in range(k):
        for j in range(i + 1, k):
            a = np.degrees(np.arccos(np.clip(C[i] @ C[j], -1, 1)))
            ang[i, j] = ang[j, i] = a
            if a <= deg:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(int(order[i]))
    return ang, [g for g in groups.values() if len(g) >= 2]


def main(config: Config) -> str:
    seed = int(config["seed"])
    dp = config.get("directional_phenotype", {})
    which = dp.get("compass_label", dp.get("geometry_label", "vmf_label"))
    kappa = float(dp.get("compass_kappa", 20.0))
    pc3_warn = float(dp.get("compass_pc3_evr_warn", 0.15))
    pc12_min = float(dp.get("compass_pc12_faithful_min", 0.60))
    color_core_cfg = dp.get("compass_color_core", "auto")     # "auto"|True|False
    a_min = float(dp.get("compass_core_alpha_min", 0.18))
    merge_deg = float(dp.get("compass_merge_below_deg", 45.0))
    n_perm = int(dp.get("compass_core_perm", 2000))
    dpi = int(config["report"]["fig_dpi"])

    # ---- treated deltas + labels ----
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    delta = dz["delta"].astype(float)
    pid = np.array([str(x) for x in dz["progression_id"]])
    norm = np.linalg.norm(delta, axis=1)
    U = _unit(delta)

    lab = pd.read_csv(config.out("directional_phenotype_labels.csv"))
    lab["progression_id"] = lab["progression_id"].astype(str)
    lut_lab = dict(zip(lab["progression_id"], lab[which]))
    lut_rel = dict(zip(lab["progression_id"], lab["is_reliable"].astype(bool)))
    labels = np.array([lut_lab.get(p, -1) for p in pid], dtype=float)
    reliable = np.array([bool(lut_rel.get(p, False)) for p in pid])
    is_pheno = reliable & (labels >= 0)
    st = ~is_pheno
    cut = float(np.min(norm[reliable])) if reliable.any() else float(np.median(norm))

    # ---- reliable-direction PCA (angle projection) ----
    if is_pheno.sum() >= 3:
        pca = PCA(n_components=min(3, U.shape[1], is_pheno.sum() - 1),
                  random_state=seed).fit(U[is_pheno])
    else:
        pca = PCA(n_components=min(3, U.shape[1]), random_state=seed).fit(U)
    evr = pca.explained_variance_ratio_
    P = pca.transform(U)
    ang = np.arctan2(P[:, 1], P[:, 0])
    cum2 = float(evr[:2].sum())
    pc3_big = len(evr) >= 3 and float(evr[2]) >= pc3_warn
    not_faithful = cum2 < pc12_min
    show_pc3 = pc3_big or not_faithful

    # ---- phenotype centroids (full-D) + nearest-direction membership ----
    order = sorted(np.unique(labels[is_pheno]).astype(int).tolist(),
                   key=lambda c: (labels == c).sum())          # weak -> strong
    C = _centroids(U, labels, order)                            # (k, D) unit
    cos_all = U @ C.T                                           # (N, k) full-D cosine
    nearest = np.array([order[j] for j in cos_all.argmax(1)])
    ang_mat, merge_sets = _merge_groups(C, order, merge_deg)

    # ---- is coloring the CORE by direction justified? ----
    p_core, obs_core, null_core = _core_alignment_test(U[st], C, seed, n_perm)
    if color_core_cfg == "auto":
        color_core = bool(np.isfinite(p_core) and p_core < 0.05)
    else:
        color_core = bool(color_core_cfg)

    # ---- controls ----
    ctrl = None
    cpath = config.out("control_deltas.npz")
    if os.path.exists(cpath):
        cz = np.load(cpath, allow_pickle=True)
        cdelta = cz["delta"].astype(float)
        cnorm = np.linalg.norm(cdelta, axis=1)
        cP = pca.transform(_unit(cdelta))
        cang = np.arctan2(cP[:, 1], cP[:, 0])
        ctrl = {"norm": cnorm, "ang": cang, "frac_reliable": float(np.mean(cnorm >= cut))}

    # ---- alpha (opacity) scaled by magnitude ----
    ref = max(cut * 1.5, float(np.percentile(norm, 90)))
    alpha = np.clip(a_min + (1 - a_min) * norm / ref, a_min, 1.0)

    # ============================ figure ============================
    n_panels = 3 if show_pc3 else 2
    fig = plt.figure(figsize=(7.2 * n_panels - 0.6, 7.4))
    rmax = norm.max() * 1.08

    # (A) change compass -------------------------------------------------------
    axA = fig.add_subplot(1, n_panels, 1, projection="polar")
    axA.set_theta_zero_location("E"); axA.set_theta_direction(1)
    tf = np.linspace(0, 2 * np.pi, 240)
    # graded core: concentric fills 0 -> cut (darker toward centre)
    for frac in np.linspace(1.0, 0.0, 6, endpoint=False):
        axA.fill_between(tf, 0, cut * frac, color="#E4E8EC", alpha=0.22, zorder=0, lw=0)
    axA.plot(tf, np.full_like(tf, cut), color="#9aa4ae", lw=1.2, ls=(0, (5, 4)), zorder=1)
    axA.text(np.deg2rad(268), cut * 0.42, "stable core\n(low-magnitude\ncontinuum)",
             ha="center", va="center", fontsize=8, color="#6b7580", zorder=2)

    # core / stable points: color by nearest direction (if justified) else gray
    if color_core:
        core_rgba = np.array([to_rgba(_color(nearest[i]), alpha[i]) for i in np.where(st)[0]])
    else:
        core_rgba = np.array([to_rgba(_STABLE, alpha[i]) for i in np.where(st)[0]])
    axA.scatter(ang[st], norm[st], s=30, facecolors=core_rgba,
                edgecolor="white", lw=0.35, zorder=3)
    # phenotype points + spokes (strong on top)
    for c in order:
        m = is_pheno & (labels == c)
        big = m.sum() >= 4
        rgba = np.array([to_rgba(_color(c), alpha[i]) for i in np.where(m)[0]])
        axA.scatter(ang[m], norm[m], s=82 if big else 54, facecolors=rgba,
                    edgecolor="white", lw=0.8, zorder=6 if big else 5,
                    label=f"pheno {int(c)} (n={int(m.sum())})")
        ca = _circular_mean(ang[m])
        axA.plot([ca, ca], [0, norm[m].mean()], color=_color(c),
                 lw=3.2 if big else 1.6, alpha=0.5, zorder=4, solid_capstyle="round")
    if ctrl is not None:
        axA.scatter(ctrl["ang"], ctrl["norm"], marker="x", s=70, color=_CONTROL,
                    lw=1.8, zorder=7, label=f"control untreated (n={len(ctrl['norm'])})")
    axA.set_rlim(0, rmax); axA.set_rlabel_position(112)
    axA.set_yticks([cut]); axA.set_yticklabels([f"cut={cut:.2f}"], fontsize=8, color="#6b7580")
    axA.tick_params(axis="x", labelsize=9); axA.grid(color="#dfe3e7", lw=0.7)
    core_note = (f"core colored by direction (p={p_core:.3f})" if color_core
                 else f"core gray: direction ~ noise (p={p_core:.2f})")
    axA.set_title(f"(A) change compass  -  angle = direction, radius = magnitude\n"
                  f"opacity = magnitude; {core_note}", fontsize=10.5, pad=16)
    axA.legend(loc="upper left", bbox_to_anchor=(-0.17, 1.15), fontsize=8, frameon=False)

    # (B) angular density ------------------------------------------------------
    axB = fig.add_subplot(1, n_panels, 2, projection="polar")
    axB.set_theta_zero_location("E"); axB.set_theta_direction(1)
    tg = np.linspace(0, 2 * np.pi, 361)
    dens = _vm_density(tg, ang[is_pheno], kappa)
    dmax = dens.max() if dens.max() > 0 else 1.0
    axB.plot(tg, dens / dmax, color="#33517a", lw=2.0, zorder=3)
    axB.fill(tg, dens / dmax, color="#4477AA", alpha=0.18, zorder=2)
    for c in order:
        m = is_pheno & (labels == c)
        for a in ang[m]:
            axB.plot([a, a], [1.02, 1.10], color=_color(c), lw=2.0, zorder=4)
    axB.set_rlim(0, 1.15); axB.set_yticklabels([]); axB.grid(color="#dfe3e7", lw=0.7)
    axB.tick_params(axis="x", labelsize=9)
    axB.set_title("(B) reliable-direction density (modes)  -  von Mises KDE\n"
                  f"(2-D projection, {cum2*100:.0f}% faithful - lobe sharpness may overstate)",
                  fontsize=10.5, pad=16)

    # (C) faithfulness: PC1 x PC3 ---------------------------------------------
    if show_pc3 and len(evr) >= 3:
        axC = fig.add_subplot(1, n_panels, 3)
        axC.scatter(P[st, 0], P[st, 2], s=22, color=_STABLE, edgecolor="white",
                    lw=0.3, alpha=0.6, zorder=1)
        for c in order:
            m = is_pheno & (labels == c)
            axC.scatter(P[m, 0], P[m, 2], s=68, color=_color(c), edgecolor="white",
                        lw=0.7, zorder=3, label=f"pheno {int(c)}")
        axC.axhline(0, color="#ccd2d8", lw=0.8); axC.axvline(0, color="#ccd2d8", lw=0.8)
        axC.set_xlabel(f"dir PC1 ({evr[0]*100:.0f}%)")
        axC.set_ylabel(f"dir PC3 ({evr[2]*100:.0f}%)")
        axC.set_title("(C) faithfulness check: structure hidden from the 2-D angle",
                      fontsize=10.5, pad=10)
        axC.legend(fontsize=7.5, frameon=False)
        for s in ["top", "right"]:
            axC.spines[s].set_visible(False)

    faith = "faithful" if not not_faithful else "a PROJECTION - read magnitude, not fine angle"
    fig.suptitle(f"Directional phenotypes: continuum core + reliable modes   "
                 f"(reliable PC1+PC2 = {cum2*100:.0f}% -> 2-D angle is {faith})",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = config.out("phenotype_compass.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    # ---- JSON ----
    pair_ang = {f"{order[i]}-{order[j]}": float(ang_mat[i, j])
                for i in range(len(order)) for j in range(i + 1, len(order))}
    result = {
        "label_source": which, "n_total": int(len(pid)),
        "n_stable": int(st.sum()), "n_phenotype": int(is_pheno.sum()),
        "magnitude_cut": cut,
        "reliable_pca_evr": [float(x) for x in evr],
        "reliable_pca_evr_cumulative": [float(x) for x in np.cumsum(evr)],
        "pc12_captures": cum2, "pc12_faithful_min": pc12_min,
        "angle_2d_faithful": bool(not not_faithful),
        "pc3_evr": float(evr[2]) if len(evr) >= 3 else None,
        "pc3_panel_shown": bool(show_pc3),
        "core_alignment": {"p_value": p_core, "observed_mean_maxcos": obs_core,
                           "null_mean_maxcos": null_core, "core_colored": color_core,
                           "interpretation": ("core directions lean toward phenotypes "
                                              "beyond chance -> continuum resolves into "
                                              "spokes" if color_core else
                                              "core directions ~ random -> low-magnitude "
                                              "direction is noise (kept gray)")},
        "phenotype_centroid_angles_deg": pair_ang,
        "candidate_merges_within_%.0fdeg" % merge_deg: merge_sets,
        "phenotype_counts": {int(c): int((is_pheno & (labels == c)).sum()) for c in order},
        "control": None if ctrl is None else {
            "n": int(len(ctrl["norm"])), "fraction_reliable": ctrl["frac_reliable"],
            "magnitudes": [float(x) for x in ctrl["norm"]]},
    }
    io.write_json(result, config.out("phenotype_compass.json"))

    print(f"[compass] reliable-set PCA EVR: " +
          ", ".join(f"PC{i+1}={e*100:.0f}%" for i, e in enumerate(evr)) +
          f"  (PC1+PC2={cum2*100:.0f}% -> {'faithful' if not not_faithful else 'PROJECTION only'})")
    print(f"[compass] core-alignment test: obs={obs_core:.3f} vs null={null_core:.3f} "
          f"p={p_core:.3f} -> core {'COLORED by direction' if color_core else 'kept GRAY (noise)'}")
    if merge_sets:
        print(f"[compass] phenotypes within {merge_deg:.0f} deg (likely one broad mode): {merge_sets}")
    else:
        print(f"[compass] no phenotype centroids within {merge_deg:.0f} deg -> directions distinct")
    if ctrl is not None:
        print(f"[compass] controls: {len(ctrl['norm'])} untreated, "
              f"{ctrl['frac_reliable']*100:.0f}% reach the treated cut.")
    print(f"[compass] wrote phenotype_compass.{{png,json}} to {config.output_dir}")
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phenotype compass (continuum + directional)")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
