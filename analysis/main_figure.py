"""Main figure - the consolidated, honest result in four panels.

Assembles the definitive figure from the pipeline's real outputs (no re-fitting;
reads what the modules already wrote). The four panels are the four claims, in
narrative order, each annotated with its actual (clean-embedding) statistic:

  (A) There IS change, more than controls - magnitude distribution of |delta| for
      treated-stable / treated-reliable / untreated-control, with the reliability
      cut and the treated>control Mann-Whitney p (marginal).
  (B) Directions are RELIABLE but NON-PERSISTENT - within-session split-half
      direction cosine (~1.0: each change is precisely measured) vs consecutive
      between-visit step cosine (negative: changes mean-revert).
  (C) The changes lie on ONE bipolar axis, on a continuum - reliable directions in
      PC1xPC2, coloured by axis position (PC1); annotated with the angle-null p and
      the 'continuum, not discrete' caveat.
  (D) The axis is ANCHORED to baseline relative theta - baseline rel-theta vs axis
      position, with the within-panel FDR q. Baseline correlate (change is RTM).

Reads (all optional; a missing input greys its panel): deltas.npz, phenotype_axis.csv,
control_deltas.npz, qeeg_connectivity.*, directional_phenotype.json,
delta_reliability.json, phenotype_geometry.json, control_analysis.json,
axis_slowing.json, ordering_auc.json.

Outputs: main_figure.png
Run:  python -m analysis.main_figure --config analysis/config.yaml
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

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from .config import Config, load_config, add_arg
from . import io

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "gray": "#999999", "black": "#000000", "purple": "#CC79A7"}


def _unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True); n[n == 0] = 1.0
    return A / n


def _jload(config, name):
    p = config.out(name)
    try:
        return io.read_json(p) if os.path.exists(p) else None
    except Exception:
        return None


def _greyout(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="#888"); ax.set_xticks([]); ax.set_yticks([])


def _kde(x, grid):
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    try:
        return gaussian_kde(x)(grid)
    except Exception:
        return None


def main(config: Config) -> str:
    dpi = int(config["report"]["fig_dpi"])
    dcm = LinearSegmentedColormap.from_list("oi", [OI["vermillion"], "#F4F4F4", OI["blue"]], N=256)

    dz = np.load(config.out("deltas.npz"), allow_pickle=True) if os.path.exists(config.out("deltas.npz")) else None
    axis_df = (pd.read_csv(config.out("phenotype_axis.csv"))
               if os.path.exists(config.out("phenotype_axis.csv")) else None)
    dir_j, rel_j = _jload(config, "directional_phenotype.json"), _jload(config, "delta_reliability.json")
    geo_j, ctl_j = _jload(config, "phenotype_geometry.json"), _jload(config, "control_analysis.json")
    slow_j, auc_j = _jload(config, "axis_slowing.json"), _jload(config, "ordering_auc.json")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    (axA, axC), (axB, axD) = axes

    # ---------- (A) magnitude & treatment ----------
    if dz is not None and axis_df is not None:
        pid = np.array([str(x) for x in dz["progression_id"]])
        norm = np.linalg.norm(dz["delta"].astype(float), axis=1)
        relmap = dict(zip(axis_df["progression_id"].astype(str), axis_df["is_reliable"].astype(bool)))
        rel = np.array([bool(relmap.get(p, False)) for p in pid])
        cut = float(np.min(norm[rel])) if rel.any() else float(np.median(norm))
        grid = np.linspace(0, norm.max() * 1.05, 200)
        series = [("treated stable", norm[~rel], OI["gray"]), ("treated reliable", norm[rel], OI["blue"])]
        cpath = config.out("control_deltas.npz")
        if os.path.exists(cpath):
            cz = np.load(cpath, allow_pickle=True)
            series.append(("control (untreated)", np.linalg.norm(cz["delta"].astype(float), axis=1), OI["orange"]))
        for name, x, col in series:
            k = _kde(x, grid)
            if k is not None:
                axA.fill_between(grid, k, color=col, alpha=0.18); axA.plot(grid, k, color=col, lw=2, label=f"{name} (n={len(x)})")
            axA.plot(x, np.full_like(x, -0.02), "|", color=col, ms=8, mew=1.3)
        axA.axvline(cut, color="#666", ls=(0, (5, 4)), lw=1.2)
        p_mw = (ctl_j or {}).get("mannwhitney_treated_gt_control_p")
        if p_mw is not None:
            axA.text(0.97, 0.75, f"treated > control\nMann-Whitney p={p_mw:.3f}", transform=axA.transAxes,
                     ha="right", fontsize=9, bbox=dict(boxstyle="round", fc="#f4f7fb", ec="#cfd8e3"))
        axA.set_xlabel("magnitude of between-visit change  |Δ|"); axA.set_ylabel("density")
        axA.legend(fontsize=8, frameon=False, loc="upper right")
    else:
        _greyout(axA, "deltas.npz / phenotype_axis.csv missing")
    axA.set_title("(A) There is change — more than untreated controls", fontsize=11, loc="left")

    # ---------- (B) reliable but non-persistent ----------
    if dir_j is not None:
        pairs = dir_j.get("trajectory_consistency", {}).get("pairs", [])
        cos = np.array([p["cosine"] for p in pairs], float)
        meanc = dir_j.get("trajectory_consistency", {}).get("mean_cosine", np.nan)
        if len(cos):
            axB.hist(cos, bins=np.linspace(-1, 1, 13), color=OI["green"], alpha=0.8)
            axB.axvline(meanc, color=OI["vermillion"], ls="--", lw=1.6, label=f"mean = {meanc:.2f}")
        axB.axvline(0, color="#bbb", lw=1)
        shm = (rel_j or {}).get("median_split_half_cos_reliable")
        note = "within-session split-half direction cos ≈ %.2f\n(each change precisely measured)" % shm if shm else ""
        axB.text(0.03, 0.97, note + "\n\nconsecutive between-visit steps →\ndirections REVERSE (mean-revert)",
                 transform=axB.transAxes, va="top", fontsize=8.5,
                 bbox=dict(boxstyle="round", fc="#eef7f1", ec="#cbe6d6"))
        axB.set_xlabel("cosine between consecutive change steps"); axB.set_ylabel("patients")
        axB.legend(fontsize=8, frameon=False, loc="upper right")
    else:
        _greyout(axB, "directional_phenotype.json missing")
    axB.set_title("(B) Directions are reliable but non-persistent", fontsize=11, loc="left")

    # ---------- (C) the bipolar axis, as a change compass (controls included) ----------
    # polar: radius = magnitude, colour = axis position (PC1), controls = black x in
    # the low-magnitude core. Read colour/radius, NOT fine angle (2-D projection).
    axC.remove()
    axC = fig.add_subplot(2, 2, 2, projection="polar")
    if dz is not None and axis_df is not None:
        U = _unit(dz["delta"].astype(float)); norm = np.linalg.norm(dz["delta"].astype(float), axis=1)
        relmap = dict(zip(axis_df["progression_id"].astype(str), axis_df["is_reliable"].astype(bool)))
        rel = np.array([bool(relmap.get(p, False)) for p in pid])
        pca = PCA(n_components=2, random_state=int(config["seed"])).fit(U[rel]) if rel.sum() >= 3 else PCA(2).fit(U)
        P = pca.transform(U); evr = pca.explained_variance_ratio_
        amap = dict(zip(axis_df["progression_id"].astype(str), axis_df["axis_coord"].astype(float)))
        acoord = np.array([amap.get(p, 0.0) for p in pid])
        flip = -1.0 if (rel.any() and np.mean(acoord[rel] * P[rel, 0]) < 0) else 1.0
        P[:, 0] *= flip
        ang = np.arctan2(P[:, 1], P[:, 0]); cut = float(np.min(norm[rel])) if rel.any() else float(np.median(norm))
        vext = float(np.max(np.abs(P[rel, 0]))) if rel.any() else 1.0
        dn = TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)
        tf = np.linspace(0, 2 * np.pi, 240)
        for frac in np.linspace(1.0, 0.0, 6, endpoint=False):
            axC.fill_between(tf, 0, cut * frac, color="#E9ECEF", alpha=0.2, zorder=0, lw=0)
        axC.plot(tf, np.full_like(tf, cut), color="#9aa4ae", lw=1.1, ls=(0, (5, 4)), zorder=1)
        axC.scatter(ang[~rel], norm[~rel], s=24, color=OI["gray"], alpha=0.7, zorder=3, label="stable / continuum core")
        axC.scatter(ang[rel], norm[rel], c=dcm(dn(P[rel, 0])), s=58, edgecolor="white", lw=0.5,
                    zorder=4, label="reliable (colour = axis pos.)")
        cpath = config.out("control_deltas.npz")
        if os.path.exists(cpath):
            cz = np.load(cpath, allow_pickle=True)
            cU = _unit(cz["delta"].astype(float)); cnorm = np.linalg.norm(cz["delta"].astype(float), axis=1)
            cP = pca.transform(cU); cang = np.arctan2(cP[:, 1], cP[:, 0] * flip)
            axC.scatter(cang, cnorm, marker="x", s=64, color=OI["black"], lw=1.8, zorder=6,
                        label=f"control untreated (n={len(cnorm)})")
        axC.set_rlim(0, float(norm.max()) * 1.08); axC.set_rlabel_position(112)
        axC.set_yticks([cut]); axC.set_yticklabels([f"cut={cut:.2f}"], fontsize=7, color="#6b7580")
        axC.tick_params(axis="x", labelsize=8); axC.grid(color="#dfe3e7", lw=0.7)
        pa = None
        if geo_j is not None:
            sig = [(k, v) for k, v in geo_j.get("pairwise_angles", {}).items() if v.get("p_two_sided", 1) < 0.05]
            pa = max(sig, key=lambda kv: kv[1]["angle_deg"])[1] if sig else None
        sub = (f"axis = PC1 ({evr[0]*100:.0f}%); angle-null p={pa['p_two_sided']:.3f} (borderline)\n"
               "continuum, not discrete — read colour/radius, not fine angle"
               if pa else f"axis = PC1 ({evr[0]*100:.0f}%); continuum, not discrete")
        axC.legend(loc="upper left", bbox_to_anchor=(-0.16, 1.12), fontsize=7.5, frameon=False)
    else:
        _greyout(axC, "deltas.npz / phenotype_axis.csv missing"); sub = ""
    axC.set_title("(C) Change compass: one bipolar axis; controls sit in the core\n" + sub,
                  fontsize=10.5, loc="left", pad=12)

    # ---------- (D) baseline relative-theta anchor ----------
    fc_path = config.out("qeeg_connectivity")
    have_fc = os.path.exists(fc_path + ".parquet") or os.path.exists(fc_path + ".csv")
    if have_fc and axis_df is not None:
        fc = io.read_table(fc_path); fc["progression_id"] = fc["progression_id"].astype(str)
        m = fc.merge(axis_df[["progression_id", "axis_coord", "is_reliable"]].assign(
            progression_id=lambda d: d["progression_id"].astype(str)), on="progression_id", how="inner")
        m = m[m["is_reliable"].astype(bool)]
        col = "rel_theta_global_baseline"
        if col in m.columns and len(m) > 3:
            x = m["axis_coord"].to_numpy(float); y = m[col].to_numpy(float)
            good = np.isfinite(x) & np.isfinite(y); x, y = x[good], y[good]
            vext = float(np.max(np.abs(x))) if len(x) else 1.0
            axD.scatter(x, y, c=dcm(TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)(x)), s=52, edgecolor="white", lw=0.5, zorder=3)
            if len(x) >= 2:
                b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
                axD.plot(xs, np.polyval(b, xs), color="#333", lw=1.6, zorder=2)
            q = None
            if slow_j is not None:
                for r in slow_j.get("baseline_anchor", {}).get("per_feature", []):
                    if r.get("feature") == col:
                        q = r.get("q_value")
            axD.text(0.03, 0.05, (f"baseline rel-theta ~ axis\nwithin-panel FDR q={q:.3f}" if q else "baseline rel-theta ~ axis"),
                     transform=axD.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="#eef7f1", ec="#cbe6d6"))
            axD.set_xlabel("axis position (PC1)"); axD.set_ylabel("baseline relative theta power")
        else:
            _greyout(axD, "rel_theta_global_baseline not in table")
    else:
        _greyout(axD, "qeeg_connectivity missing (run module 6)")
    axD.set_title("(D) The axis is anchored to baseline relative theta", fontsize=11, loc="left")

    for ax in (axA, axB, axD):                        # axC is polar - no top/right spines
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    auc_bits = ""
    if auc_j is not None:
        auc_bits = (f"   |   ordering OOF AUC {auc_j.get('ordering_auc', float('nan')):.2f} "
                    f"[{auc_j.get('ordering_auc_ci95',[np.nan,np.nan])[0]:.2f},"
                    f"{auc_j.get('ordering_auc_ci95',[np.nan,np.nan])[1]:.2f}]")
    fig.suptitle("Leqembi EEG: a reliable, non-persistent bipolar axis of between-visit change, "
                 "anchored to baseline relative theta" + auc_bits, fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = config.out("main_figure.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"[main_figure] wrote {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidated main figure")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
