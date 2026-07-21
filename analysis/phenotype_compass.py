"""Phenotype compass - continuum + directional-phenotype figure (bipolar-axis).

The data (angle-null test in phenotype_geometry) shows ONE robust structure: a
single dominant axis of change with two significantly-antipodal poles (pheno0 vs
pheno1), sitting on a low-magnitude continuum. The other 'phenotypes' are mutually
orthogonal, tiny, and consistent with the high-D null - not robust modes. So this
figure adopts an AXIS framing rather than N categorical clusters:

  (A) CHANGE COMPASS (polar): angle = direction (2-D projection, context only),
      radius = magnitude, and COLOR = position on the validated axis (PC1 of the
      reliable directions) on an Okabe-Ito diverging scale (blue pole <-> vermillion
      pole, neutral centre). Opacity = magnitude (faint core -> bold rim). The
      validated axis is drawn as a diameter and annotated with its antipodal p.
      Controls (black x) are projected in and should sit in the neutral centre.

  (B) MAGNITUDE DISTRIBUTION (cartesian, 100% faithful - no projection): |delta|
      for treated-stable vs treated-reliable vs control, with the reliability cut.
      Shows the continuum, the split, and treated >> control in one honest panel.

  (C) FAITHFULNESS CHECK (cartesian): PC1 x PC3 of the reliable directions,
      colored by the same diverging axis. Confirms PC1 IS the axis (poles separate
      on x) and shows the residual structure the 2-D angle cannot.

Colors are Okabe-Ito (colorblind-safe). Axis detection reads
phenotype_geometry.json (validated antipodal pair) when present; falls back to
PC1 as the axis otherwise. `compass_axis_mode` = auto|diverging|categorical.

Reads deltas.npz + directional_phenotype_labels.csv (+ optional
phenotype_geometry.json, control_deltas.npz). Outputs:
  phenotype_compass.png, phenotype_compass.json

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
from matplotlib.colors import to_rgba, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

from scipy.special import i0
from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.decomposition import PCA

from .config import Config, load_config, add_arg
from . import io
from .directional_phenotype import _unit

# Okabe-Ito (colorblind-safe)
OI = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
      "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#999999"}
_CAT = [OI["blue"], OI["vermillion"], OI["green"], OI["orange"], OI["skyblue"],
        OI["purple"], OI["yellow"], OI["black"]]         # categorical fallback
_STABLE = OI["gray"]
_CONTROL = OI["black"]


def _cat_color(lab):
    return _CAT[int(lab) % len(_CAT)]


def _diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "oi_div", [OI["vermillion"], "#F4F4F4", OI["blue"]], N=256)


def _circular_mean(angles):
    return float(np.arctan2(np.sin(angles).sum(), np.cos(angles).sum()))


def _vm_density(theta_grid, angles, kappa):
    if len(angles) == 0:
        return np.zeros_like(theta_grid)
    d = np.exp(kappa * np.cos(theta_grid[:, None] - angles[None, :])).sum(1)
    return d / (2 * np.pi * i0(kappa) * len(angles))


def _centroids(U, labels, order):
    C = []
    for c in order:
        v = U[labels == c].mean(0)
        C.append(v / (np.linalg.norm(v) + 1e-12))
    return np.array(C)


def _core_alignment_test(U_core, C, seed, n_perm):
    if U_core.shape[0] == 0 or C.shape[0] == 0:
        return np.nan, np.nan, np.nan
    obs = float((U_core @ C.T).max(1).mean())
    rng = np.random.default_rng(seed)
    D = U_core.shape[1]
    null = np.empty(n_perm)
    for i in range(n_perm):
        R = _unit(rng.normal(size=(U_core.shape[0], D)))
        null[i] = (R @ C.T).max(1).mean()
    return float((null >= obs).mean()), obs, float(null.mean())


def _validated_axis(config):
    """Read phenotype_geometry.json (if present) for the most-antipodal
    significant phenotype pair -> that is the validated axis."""
    p = config.out("phenotype_geometry.json")
    if not os.path.exists(p):
        return None
    try:
        g = io.read_json(p)
    except Exception:
        return None
    nm = g.get("angle_null", {}).get("null_mean_deg", 90.0)
    best = None
    for pair, info in g.get("pairwise_angles", {}).items():
        if (info.get("p_two_sided", 1.0) < 0.05 and info.get("angle_deg", 0) > nm):
            if best is None or info["angle_deg"] > best[1]:
                best = (pair, info["angle_deg"], info["p_two_sided"])
    if best is None:
        return None
    return {"pair": best[0], "angle_deg": best[1], "p": best[2],
            "label_source": g.get("label_source")}


def _kde_curve(x, grid):
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    try:
        return gaussian_kde(x)(grid)
    except Exception:
        return None


def main(config: Config) -> str:
    seed = int(config["seed"])
    dp = config.get("directional_phenotype", {})
    axis_json = _validated_axis(config)
    which = dp.get("compass_label",
                   (axis_json or {}).get("label_source") or dp.get("geometry_label", "spherical_label"))
    kappa = float(dp.get("compass_kappa", 20.0))
    pc12_min = float(dp.get("compass_pc12_faithful_min", 0.60))
    axis_mode = dp.get("compass_axis_mode", "auto")           # auto|diverging|categorical
    a_min = float(dp.get("compass_core_alpha_min", 0.20))
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

    # ---- reliable-direction PCA ----
    ncomp = min(3, U.shape[1], max(2, (is_pheno.sum() - 1) if is_pheno.sum() >= 3 else 2))
    pca = PCA(n_components=ncomp, random_state=seed).fit(
        U[is_pheno] if is_pheno.sum() >= 3 else U)
    evr = pca.explained_variance_ratio_
    P = pca.transform(U)
    cum2 = float(evr[:2].sum())
    not_faithful = cum2 < pc12_min

    order = sorted(np.unique(labels[is_pheno]).astype(int).tolist(),
                   key=lambda c: (labels == c).sum())
    C = _centroids(U, labels, order)

    # orient PC1 so the LARGEST phenotype projects positive (blue pole);
    # the SAME flip is applied to controls so both share one axis convention.
    flip = 1.0
    if order and P[is_pheno & (labels == order[-1]), 0].mean() < 0:
        flip = -1.0
    P[:, 0] *= flip
    ang = np.arctan2(P[:, 1], P[:, 0])
    axis_coord = P[:, 0]                                   # position on the axis

    # decide diverging vs categorical
    one_axis = axis_json is not None
    diverging = (axis_mode == "diverging") or (axis_mode == "auto" and one_axis)

    # core-alignment (still reported; core stays neutral in diverging mode)
    p_core, obs_core, null_core = _core_alignment_test(U[st], C, seed, n_perm)

    # ---- controls (same PCA, same PC1 flip) ----
    ctrl = None
    cpath = config.out("control_deltas.npz")
    if os.path.exists(cpath):
        cz = np.load(cpath, allow_pickle=True)
        cdelta = cz["delta"].astype(float)
        cnorm = np.linalg.norm(cdelta, axis=1)
        cP = pca.transform(_unit(cdelta))
        cP1 = cP[:, 0] * flip
        cP3 = cP[:, 2] if cP.shape[1] > 2 else np.zeros(len(cnorm))
        cang = np.arctan2(cP[:, 1], cP1)
        ctrl = {"norm": cnorm, "ang": cang, "P1": cP1, "P3": cP3,
                "frac_reliable": float(np.mean(cnorm >= cut))}

    div_cmap = _diverging_cmap()
    vext = float(np.max(np.abs(axis_coord))) if len(axis_coord) else 1.0
    dnorm = TwoSlopeNorm(vmin=-vext, vcenter=0.0, vmax=vext)
    ref = max(cut * 1.5, float(np.percentile(norm, 90)))
    alpha = np.clip(a_min + (1 - a_min) * norm / ref, a_min, 1.0)

    # ============================ figure ============================
    fig = plt.figure(figsize=(19.5, 6.6))

    # (A) compass -------------------------------------------------------------
    axA = fig.add_subplot(1, 3, 1, projection="polar")
    axA.set_theta_zero_location("E"); axA.set_theta_direction(1)
    tf = np.linspace(0, 2 * np.pi, 240)
    for frac in np.linspace(1.0, 0.0, 6, endpoint=False):
        axA.fill_between(tf, 0, cut * frac, color="#E9ECEF", alpha=0.20, zorder=0, lw=0)
    axA.plot(tf, np.full_like(tf, cut), color="#9aa4ae", lw=1.1, ls=(0, (5, 4)), zorder=1)

    if diverging:
        rgba = div_cmap(dnorm(axis_coord)); rgba[:, 3] = alpha
        axA.scatter(ang, norm, s=np.where(is_pheno, 70, 30), facecolors=rgba,
                    edgecolor="white", lw=0.5, zorder=3)
        # draw validated axis as a diameter through the two poles
        if order:
            ca = _circular_mean(ang[is_pheno & (labels == order[-1])])
            axA.plot([ca, ca + np.pi], [norm.max() * 1.02, norm.max() * 1.02],
                     color="#444", lw=1.4, ls=(0, (6, 4)), zorder=2, alpha=0.7)
        sub = f"color = axis position (PC1); opacity = magnitude"
        if axis_json:
            sub += f"\nvalidated axis {axis_json['pair']}: {axis_json['angle_deg']:.0f}° antipodal, p={axis_json['p']:.4f}"
    else:
        for c in order:
            m = is_pheno & (labels == c)
            rr = np.array([to_rgba(_cat_color(c), alpha[i]) for i in np.where(m)[0]])
            axA.scatter(ang[m], norm[m], s=72, facecolors=rr, edgecolor="white",
                        lw=0.8, zorder=5, label=f"pheno {int(c)} (n={int(m.sum())})")
        cr = np.array([to_rgba(_STABLE, alpha[i]) for i in np.where(st)[0]])
        axA.scatter(ang[st], norm[st], s=28, facecolors=cr, edgecolor="white", lw=0.35, zorder=3)
        sub = "categorical phenotypes; opacity = magnitude"

    if ctrl is not None:
        axA.scatter(ctrl["ang"], ctrl["norm"], marker="x", s=66, color=_CONTROL,
                    lw=1.8, zorder=7, label=f"control (n={len(ctrl['norm'])})")
    axA.set_rlim(0, norm.max() * 1.08); axA.set_rlabel_position(112)
    axA.set_yticks([cut]); axA.set_yticklabels([f"cut={cut:.2f}"], fontsize=8, color="#6b7580")
    axA.tick_params(axis="x", labelsize=9); axA.grid(color="#dfe3e7", lw=0.7)
    axA.set_title(f"(A) change compass  -  radius = magnitude\n{sub}", fontsize=10, pad=14)
    if diverging:
        sm = ScalarMappable(norm=dnorm, cmap=div_cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=axA, fraction=0.045, pad=0.10, aspect=30)
        cb.set_label("← pole 1        axis position (PC1)        pole 0 →", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        axA.legend(loc="upper left", bbox_to_anchor=(-0.18, 1.14), fontsize=8, frameon=False)

    # (B) magnitude distribution (faithful) -----------------------------------
    axB = fig.add_subplot(1, 3, 2)
    m_stable = norm[st]; m_rel = norm[is_pheno]
    grid = np.linspace(0, norm.max() * 1.05, 200)
    series = [("treated stable", m_stable, OI["gray"]),
              ("treated reliable", m_rel, OI["blue"])]
    if ctrl is not None:
        series.append(("control (untreated)", ctrl["norm"], OI["orange"]))
    for name, x, col in series:
        if len(x) == 0:
            continue
        k = _kde_curve(x, grid)
        if k is not None:
            axB.fill_between(grid, k, color=col, alpha=0.18, zorder=1)
            axB.plot(grid, k, color=col, lw=2.0, zorder=2, label=f"{name} (n={len(x)})")
        axB.plot(x, np.full_like(x, -0.02 * (1 + series.index((name, x, col)))),
                 "|", color=col, ms=9, mew=1.4, zorder=3)   # rug
    axB.axvline(cut, color="#666", ls=(0, (5, 4)), lw=1.2, zorder=4)
    axB.text(cut, axB.get_ylim()[1] * 0.96, "  reliability cut", fontsize=8, color="#666", va="top")
    if ctrl is not None and len(m_rel) and len(ctrl["norm"]):
        try:
            U_, pmw = mannwhitneyu(m_rel, ctrl["norm"], alternative="greater")
            axB.text(0.97, 0.80, f"treated>control\nMann-Whitney p={pmw:.1e}",
                     transform=axB.transAxes, ha="right", fontsize=8.5,
                     bbox=dict(boxstyle="round", fc="#f4f7fb", ec="#cfd8e3"))
        except Exception:
            pass
    axB.set_xlabel("magnitude of change  |Δ|"); axB.set_ylabel("density")
    axB.set_title("(B) magnitude distribution (faithful - no projection)", fontsize=10, pad=8)
    axB.legend(fontsize=8, frameon=False, loc="upper right")
    for s in ["top", "right"]:
        axB.spines[s].set_visible(False)

    # (C) faithfulness: PC1 x PC3 --------------------------------------------
    axC = fig.add_subplot(1, 3, 3)
    P3 = P[:, 2] if P.shape[1] > 2 else np.zeros(len(P))
    if diverging:
        rgbaC = div_cmap(dnorm(axis_coord)); rgbaC[:, 3] = np.where(is_pheno, 0.95, 0.5)
        axC.scatter(axis_coord, P3, s=np.where(is_pheno, 64, 24), facecolors=rgbaC,
                    edgecolor="white", lw=0.4, zorder=2)
    else:
        axC.scatter(P[st, 0], P3[st], s=22, color=_STABLE, alpha=0.5, zorder=1)
        for c in order:
            m = is_pheno & (labels == c)
            axC.scatter(P[m, 0], P3[m], s=62, color=_cat_color(c), edgecolor="white",
                        lw=0.6, zorder=3, label=f"pheno {int(c)}")
    if ctrl is not None:
        axC.scatter(ctrl["P1"], ctrl["P3"], marker="x", s=60, color=_CONTROL, lw=1.6, zorder=4)
    axC.axhline(0, color="#ccd2d8", lw=0.8); axC.axvline(0, color="#ccd2d8", lw=0.8)
    axC.set_xlabel(f"PC1 = validated axis ({evr[0]*100:.0f}%)")
    axC.set_ylabel(f"PC3 ({evr[2]*100:.0f}%)" if len(evr) > 2 else "PC3")
    axC.set_title("(C) faithfulness check: PC1 is the axis; PC3 = residual", fontsize=10, pad=8)
    for s in ["top", "right"]:
        axC.spines[s].set_visible(False)

    faith = "faithful" if not not_faithful else "a PROJECTION - trust color(PC1)/radius, not fine angle"
    fig.suptitle(f"EEG change: continuum core + one bipolar axis   "
                 f"(reliable PC1+PC2={cum2*100:.0f}%; 2-D angle is {faith})", fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = config.out("phenotype_compass.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    # ---- JSON ----
    result = {
        "framing": "bipolar_axis" if diverging else "categorical",
        "label_source": which, "validated_axis": axis_json,
        "n_total": int(len(pid)), "n_stable": int(st.sum()), "n_phenotype": int(is_pheno.sum()),
        "magnitude_cut": cut,
        "reliable_pca_evr": [float(x) for x in evr],
        "pc12_captures": cum2, "angle_2d_faithful": bool(not not_faithful),
        "core_alignment": {"p_value": p_core, "observed": obs_core, "null_mean": null_core},
        "control": None if ctrl is None else {
            "n": int(len(ctrl["norm"])), "fraction_reliable": ctrl["frac_reliable"]},
        "palette": "okabe_ito",
    }
    io.write_json(result, config.out("phenotype_compass.json"))

    # single source of truth for the validated axis coordinate (used by axis_qeeg)
    pd.DataFrame({"progression_id": pid, "axis_coord": axis_coord, "magnitude": norm,
                  "is_reliable": reliable,
                  "pole": np.where(axis_coord >= 0, "pole0", "pole1")}
                 ).to_csv(config.out("phenotype_axis.csv"), index=False)

    print(f"[compass] framing={'BIPOLAR-AXIS (diverging)' if diverging else 'categorical'}; "
          f"palette=Okabe-Ito")
    if axis_json:
        print(f"[compass] validated axis {axis_json['pair']}: {axis_json['angle_deg']:.0f} deg "
              f"antipodal, p={axis_json['p']:.4f}")
    print(f"[compass] reliable PC1+PC2={cum2*100:.0f}% -> angle {'faithful' if not not_faithful else 'PROJECTION'}; "
          f"PC1 alone={evr[0]*100:.0f}%")
    print(f"[compass] core-alignment p={p_core:.3f} (obs={obs_core:.3f} vs null={null_core:.3f})")
    if ctrl is not None:
        print(f"[compass] controls: {ctrl['frac_reliable']*100:.0f}% reach the treated cut.")
    print(f"[compass] wrote phenotype_compass.{{png,json}} to {config.output_dir}")
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phenotype compass (bipolar-axis)")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
