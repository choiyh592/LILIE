"""Phenotype compass - the 'pretty' continuum + directional-phenotype figure.

Replaces the flat PC1xPC2 scatter with a view that puts BOTH variables that
define the story on axes:

  (A) CHANGE COMPASS (polar): angle = direction of change (top-2 PCs of the
      RELIABLE unit directions), radius = magnitude of change. The shaded central
      disk (|delta| < reliability cut) IS the low-magnitude continuum; reliable
      phenotypes fan out as colored spokes at the rim. Untreated CONTROLS are
      projected into the same direction space and overplotted (black x) - they
      should sit in the central stable core, which is the treatment-association
      story in one glance.

  (B) ANGULAR DENSITY (polar): a smooth von Mises circular KDE of the reliable
      directions - the 'do the directions form modes?' rose, upgraded to a curve
      so the dominant modes read as peaks. (Controls are NOT shown here: the
      direction of a low-magnitude delta is noise, so their angle is meaningless.)

  (C) FAITHFULNESS CHECK (cartesian, auto-added only if PC3 is load-bearing):
      PC1 x PC3 of the reliable directions. If two phenotypes that share an angle
      in (A) separate here, the 2-D compass is compressing real structure and you
      should report 3-D; if not, 2-D is faithful.

The reliable-set PCA explained-variance ratio (PC1/PC2/PC3 + cumulative) is
printed and written to JSON - that is the empirical answer to 'would a 3rd PC
help the picture?'. Note: this only affects the DRAWING; phenotype labels come
from directional_phenotype.py and are unchanged.

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


def main(config: Config) -> str:
    seed = int(config["seed"])
    dp = config.get("directional_phenotype", {})
    which = dp.get("compass_label", dp.get("geometry_label", "vmf_label"))
    kappa = float(dp.get("compass_kappa", 20.0))
    pc3_warn = float(dp.get("compass_pc3_evr_warn", 0.15))
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
    # stable = not reliable OR label -1; phenotypes = reliable with label >= 0
    is_pheno = reliable & (labels >= 0)
    cut = float(np.min(norm[reliable])) if reliable.any() else float(np.median(norm))

    # ---- reliable-direction PCA (this is what the angle is projected onto) ----
    if is_pheno.sum() >= 3:
        pca = PCA(n_components=min(3, U.shape[1], is_pheno.sum() - 1),
                  random_state=seed).fit(U[is_pheno])
    else:                                        # degenerate fallback
        pca = PCA(n_components=min(3, U.shape[1]), random_state=seed).fit(U)
    evr = pca.explained_variance_ratio_
    P = pca.transform(U)                          # all treated in reliable-PC space
    ang = np.arctan2(P[:, 1], P[:, 0])            # direction angle (top-2 PCs)
    pc3_loadbearing = len(evr) >= 3 and float(evr[2]) >= pc3_warn

    # ---- controls: project into the SAME direction space ----
    ctrl = None
    cpath = config.out("control_deltas.npz")
    if os.path.exists(cpath):
        cz = np.load(cpath, allow_pickle=True)
        cdelta = cz["delta"].astype(float)
        cnorm = np.linalg.norm(cdelta, axis=1)
        cP = pca.transform(_unit(cdelta))
        cang = np.arctan2(cP[:, 1], cP[:, 0])
        ctrl = {"norm": cnorm, "ang": cang, "P": cP,
                "frac_reliable": float(np.mean(cnorm >= cut))}

    # ---- figure ----
    n_panels = 3 if pc3_loadbearing else 2
    fig = plt.figure(figsize=(7.2 * n_panels - 0.6, 7.4))
    rmax = norm.max() * 1.08

    # (A) change compass -------------------------------------------------------
    axA = fig.add_subplot(1, n_panels, 1, projection="polar")
    axA.set_theta_zero_location("E"); axA.set_theta_direction(1)
    tf = np.linspace(0, 2 * np.pi, 240)
    axA.fill_between(tf, 0, cut, color="#E9ECEF", alpha=0.9, zorder=0)
    axA.plot(tf, np.full_like(tf, cut), color="#9aa4ae", lw=1.2, ls=(0, (5, 4)), zorder=1)
    axA.text(np.deg2rad(270), cut * 0.45, "stable core\n(low-magnitude\ncontinuum)",
             ha="center", va="center", fontsize=8.5, color="#6b7580", zorder=2)
    # stable points (gray, angle-agnostic story but drawn at their angle)
    st = ~is_pheno
    axA.scatter(ang[st], norm[st], s=30, color=_STABLE, edgecolor="white",
                lw=0.4, alpha=0.85, zorder=3, label="low-change (stable)")
    # phenotype points + centroid spokes, strong phenotypes on top
    order = sorted(np.unique(labels[is_pheno]),
                   key=lambda c: (labels[is_pheno] == c).sum())
    for c in order:
        m = is_pheno & (labels == c)
        big = m.sum() >= 4
        axA.scatter(ang[m], norm[m], s=80 if big else 52, color=_color(c),
                    edgecolor="white", lw=0.8, alpha=0.95,
                    zorder=6 if big else 5,
                    label=f"pheno {int(c)} (n={int(m.sum())})")
        ca = _circular_mean(ang[m])
        axA.plot([ca, ca], [0, norm[m].mean()], color=_color(c),
                 lw=3.2 if big else 1.6, alpha=0.5, zorder=4, solid_capstyle="round")
    # controls
    if ctrl is not None:
        axA.scatter(ctrl["ang"], ctrl["norm"], marker="x", s=70,
                    color=_CONTROL, lw=1.8, zorder=7,
                    label=f"control untreated (n={len(ctrl['norm'])})")
    axA.set_rlim(0, rmax); axA.set_rlabel_position(112)
    axA.set_yticks([cut]); axA.set_yticklabels([f"cut={cut:.2f}"], fontsize=8, color="#6b7580")
    axA.tick_params(axis="x", labelsize=9); axA.grid(color="#dfe3e7", lw=0.7)
    axA.set_title("(A) change compass  -  angle = direction, radius = magnitude",
                  fontsize=11, pad=18)
    axA.legend(loc="upper left", bbox_to_anchor=(-0.17, 1.14), fontsize=8, frameon=False)

    # (B) angular density ------------------------------------------------------
    axB = fig.add_subplot(1, n_panels, 2, projection="polar")
    axB.set_theta_zero_location("E"); axB.set_theta_direction(1)
    tg = np.linspace(0, 2 * np.pi, 361)
    dens = _vm_density(tg, ang[is_pheno], kappa)
    dmax = dens.max() if dens.max() > 0 else 1.0
    axB.plot(tg, dens / dmax, color="#33517a", lw=2.0, zorder=3)
    axB.fill(tg, dens / dmax, color="#4477AA", alpha=0.18, zorder=2)
    # colored rim ticks per phenotype direction
    for c in np.unique(labels[is_pheno]):
        m = is_pheno & (labels == c)
        for a in ang[m]:
            axB.plot([a, a], [1.02, 1.10], color=_color(c), lw=2.0, zorder=4)
    axB.set_rlim(0, 1.15); axB.set_yticklabels([]); axB.grid(color="#dfe3e7", lw=0.7)
    axB.tick_params(axis="x", labelsize=9)
    axB.set_title("(B) reliable-direction density (modes)  -  von Mises KDE",
                  fontsize=11, pad=18)

    # (C) faithfulness: PC1 x PC3 ---------------------------------------------
    if pc3_loadbearing:
        axC = fig.add_subplot(1, n_panels, 3)
        axC.scatter(P[st, 0], P[st, 2], s=24, color=_STABLE, edgecolor="white",
                    lw=0.3, alpha=0.7, zorder=1)
        for c in np.unique(labels[is_pheno]):
            m = is_pheno & (labels == c)
            axC.scatter(P[m, 0], P[m, 2], s=70, color=_color(c), edgecolor="white",
                        lw=0.7, zorder=3, label=f"pheno {int(c)}")
        axC.axhline(0, color="#ccd2d8", lw=0.8); axC.axvline(0, color="#ccd2d8", lw=0.8)
        axC.set_xlabel(f"dir PC1 ({evr[0]*100:.0f}%)")
        axC.set_ylabel(f"dir PC3 ({evr[2]*100:.0f}%)")
        axC.set_title("(C) faithfulness check: PC3 load-bearing", fontsize=11, pad=10)
        axC.legend(fontsize=7.5, frameon=False)
        for s in ["top", "right"]:
            axC.spines[s].set_visible(False)

    cum2 = float(evr[:2].sum())
    fig.suptitle(f"Directional phenotypes: continuum core + reliable modes   "
                 f"(reliable-set PC1+PC2 = {cum2*100:.0f}% of directional variance)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = config.out("phenotype_compass.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    # ---- JSON: the numbers behind the picture ----
    result = {
        "label_source": which, "n_total": int(len(pid)),
        "n_stable": int(st.sum()), "n_phenotype": int(is_pheno.sum()),
        "magnitude_cut": cut,
        "reliable_pca_evr": [float(x) for x in evr],
        "reliable_pca_evr_cumulative": [float(x) for x in np.cumsum(evr)],
        "pc12_captures": cum2,
        "pc3_evr": float(evr[2]) if len(evr) >= 3 else None,
        "pc3_load_bearing": bool(pc3_loadbearing),
        "pc3_warn_threshold": pc3_warn,
        "phenotype_counts": {int(c): int((is_pheno & (labels == c)).sum())
                             for c in np.unique(labels[is_pheno])},
        "control": None if ctrl is None else {
            "n": int(len(ctrl["norm"])),
            "fraction_reliable": ctrl["frac_reliable"],
            "magnitudes": [float(x) for x in ctrl["norm"]]},
        "viz_dim_note": ("2-D compass is faithful (PC1+PC2 dominate)."
                         if not pc3_loadbearing else
                         f"PC3 carries {evr[2]*100:.0f}% (>= {pc3_warn*100:.0f}%): "
                         "2-D angle compresses some structure -> PC1xPC3 panel added."),
    }
    io.write_json(result, config.out("phenotype_compass.json"))

    print(f"[compass] reliable-set PCA EVR: " +
          ", ".join(f"PC{i+1}={e*100:.0f}%" for i, e in enumerate(evr)) +
          f"  (PC1+PC2={cum2*100:.0f}%, cum3={np.cumsum(evr)[-1]*100:.0f}%)")
    print(f"[compass] {'PC3 is load-bearing -> added PC1xPC3 panel; consider 3-D.' if pc3_loadbearing else 'PC3 minor -> 2-D compass is faithful; a 3rd PC would add clutter, not signal.'}")
    if ctrl is not None:
        print(f"[compass] controls: {len(ctrl['norm'])} untreated, "
              f"{ctrl['frac_reliable']*100:.0f}% reach the treated cut "
              f"(expect ~0% -> they sit in the stable core).")
    print(f"[compass] wrote phenotype_compass.{{png,json}} to {config.output_dir}")
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phenotype compass (continuum + directional)")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
