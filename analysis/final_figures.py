"""Final, publication-oriented figures - four SEPARATE files, minimal on-figure text.

Each figure is a single claim, drawn clean (axis labels + a short legend only); all
prose lives in the companion captions file so the panels stay uncluttered.

  final_figure_1  magnitude of between-visit change (treated stable / reliable /
                  untreated control), with the reliability cut.
  final_figure_2  change compass WITH controls and the bipolar axis drawn in
                  (radius = |Δ|, colour = axis position; axis poles marked).
  final_figure_3  the baseline-theta anchor: (left) baseline relative theta vs axis
                  position + fit; (right) mean baseline theta across axis-position
                  quartiles (the correlation, as bars).
  final_figure_4  annotated compass: the validated theta axis as a double arrow,
                  plus the strongest embedding-mode lead (c3, α-connectivity).

Also writes final_figure_captions.txt (Caption 1..4 + a Summary paragraph), filled
with the real statistics read from the pipeline's JSON/CSV outputs.

Reads (all optional; missing input degrades gracefully): deltas.npz,
phenotype_axis.csv, control_deltas.npz, directional_phenotype_labels.csv,
direction_modes.json, mode_audit.json, phenotype_geometry.json, qeeg_connectivity.*,
axis_slowing.json, axis_qeeg.json, control_analysis.json, delta_reliability.json,
directional_phenotype.json, ordering_auc.json.

Run:  python -m analysis.final_figures --config analysis/config.yaml
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
DCM = LinearSegmentedColormap.from_list("oi", [OI["vermillion"], "#F4F4F4", OI["blue"]], N=256)


def _unit(A):
    n = np.linalg.norm(A, axis=1, keepdims=True); n[n == 0] = 1.0
    return A / n


def _jload(config, name):
    p = config.out(name)
    try:
        return io.read_json(p) if os.path.exists(p) else None
    except Exception:
        return None


def _kde(x, grid):
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    try:
        return gaussian_kde(x)(grid)
    except Exception:
        return None


def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# shared loading / geometry
# --------------------------------------------------------------------------- #
def _load(config):
    d = {}
    dp = config.out("deltas.npz")
    d["dz"] = np.load(dp, allow_pickle=True) if os.path.exists(dp) else None
    ap = config.out("phenotype_axis.csv")
    d["axis_df"] = pd.read_csv(ap) if os.path.exists(ap) else None
    lp = config.out("directional_phenotype_labels.csv")
    d["lab"] = pd.read_csv(lp) if os.path.exists(lp) else None
    d["modes"] = _jload(config, "direction_modes.json")
    d["audit"] = _jload(config, "mode_audit.json")
    d["geo"] = _jload(config, "phenotype_geometry.json")
    d["slow"] = _jload(config, "axis_slowing.json")
    d["axq"] = _jload(config, "axis_qeeg.json")
    d["ctl"] = _jload(config, "control_analysis.json")
    d["rel"] = _jload(config, "delta_reliability.json")
    d["dir"] = _jload(config, "directional_phenotype.json")
    d["auc"] = _jload(config, "ordering_auc.json")
    return d


def _compass_geometry(config, d):
    """PCA(2) of reliable unit directions, oriented so axis_coord ~ +PC1. Returns a
    dict with per-progression angle/magnitude/reliability + per-cluster centroid
    angles, matching the compass used across figures 2 and 4."""
    dz, axis_df, lab = d["dz"], d["axis_df"], d["lab"]
    if dz is None or axis_df is None:
        return None
    delta = dz["delta"].astype(float)
    pid = np.array([str(x) for x in dz["progression_id"]])
    U = _unit(delta); norm = np.linalg.norm(delta, axis=1)
    relmap = dict(zip(axis_df["progression_id"].astype(str), axis_df["is_reliable"].astype(bool)))
    rel = np.array([bool(relmap.get(p, False)) for p in pid])
    seed = int(config["seed"])
    pca = PCA(n_components=2, random_state=seed).fit(U[rel] if rel.sum() >= 3 else U)
    P = pca.transform(U); evr = pca.explained_variance_ratio_
    amap = dict(zip(axis_df["progression_id"].astype(str), axis_df["axis_coord"].astype(float)))
    acoord = np.array([amap.get(p, 0.0) for p in pid])
    flip = -1.0 if (rel.any() and np.mean(acoord[rel] * P[rel, 0]) < 0) else 1.0
    P[:, 0] *= flip
    ang = np.arctan2(P[:, 1], P[:, 0])
    cut = float(np.min(norm[rel])) if rel.any() else float(np.median(norm))
    # per-cluster centroid angles (spherical_label on the reliable set)
    cent = {}
    if lab is not None and "spherical_label" in lab.columns:
        lmap = dict(zip(lab["progression_id"].astype(str), lab["spherical_label"]))
        labels = np.array([lmap.get(p, -1) for p in pid])
        for c in sorted(set(labels[rel].tolist())):
            m = (labels == c) & rel
            if m.any():
                cent[int(c)] = float(np.arctan2(np.sin(ang[m]).sum(), np.cos(ang[m]).sum()))
        d["_labels"] = labels
    return {"pid": pid, "P": P, "ang": ang, "norm": norm, "rel": rel, "cut": cut,
            "evr": evr, "pca": pca, "flip": flip, "cent": cent}


def _axis_poles(d, g):
    """(pole_neg, pole_pos) cluster ids from the validated angle-null pair, oriented
    by centroid PC1 sign; falls back to the two most-antipodal centroids."""
    pair = None
    if d.get("modes") and d["modes"].get("validated_axis_pairs"):
        pair = d["modes"]["validated_axis_pairs"][0][0]
    if pair is None and d.get("geo"):
        sig = [(k, v) for k, v in d["geo"].get("pairwise_angles", {}).items()
               if v.get("p_two_sided", 1) < 0.05]
        if sig:
            pair = max(sig, key=lambda kv: kv[1]["angle_deg"])[0]
    if pair is None or "-" not in pair:
        return None
    a, b = (int(x) for x in pair.split("-"))
    if a not in g["cent"] or b not in g["cent"]:
        return None
    # positive pole = the one nearer angle 0 (i.e. +PC1)
    ca, cb = g["cent"][a], g["cent"][b]
    return (a, b) if abs(np.cos(ca)) and np.cos(ca) < np.cos(cb) else (b, a)


def _lead_cluster(d, g):
    """The strongest embedding-mode lead: reliable, geometric, NOT on the axis, most
    in-group patients. (On the real data this is c3, the α-connectivity bundle.)"""
    if not d.get("audit"):
        return None
    cand = [m for m in d["audit"].get("per_mode", [])
            if m.get("is_geometric_mode") and not m.get("in_validated_axis")]
    if not cand:
        return None
    best = max(cand, key=lambda m: (m.get("n_patients", 0), m.get("n", 0)))
    return int(best["cluster"]) if best["cluster"] in g["cent"] else None


# --------------------------------------------------------------------------- #
# figure 1 - magnitude
# --------------------------------------------------------------------------- #
def fig1_magnitude(config, d):
    dpi = int(config["report"]["fig_dpi"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    dz, axis_df = d["dz"], d["axis_df"]
    if dz is not None and axis_df is not None:
        pid = np.array([str(x) for x in dz["progression_id"]])
        norm = np.linalg.norm(dz["delta"].astype(float), axis=1)
        relmap = dict(zip(axis_df["progression_id"].astype(str), axis_df["is_reliable"].astype(bool)))
        rel = np.array([bool(relmap.get(p, False)) for p in pid])
        cut = float(np.min(norm[rel])) if rel.any() else float(np.median(norm))
        grid = np.linspace(0, norm.max() * 1.05, 200)
        series = [("treated · stable", norm[~rel], OI["gray"]),
                  ("treated · reliable", norm[rel], OI["blue"])]
        cpath = config.out("control_deltas.npz")
        if os.path.exists(cpath):
            cz = np.load(cpath, allow_pickle=True)
            series.append(("untreated control", np.linalg.norm(cz["delta"].astype(float), axis=1), OI["orange"]))
        for name, x, col in series:
            k = _kde(x, grid)
            if k is not None:
                ax.fill_between(grid, k, color=col, alpha=0.18)
                ax.plot(grid, k, color=col, lw=2, label=f"{name} (n={len(x)})")
            ax.plot(x, np.full_like(x, -0.02 * (1 + 0.0)), "|", color=col, ms=9, mew=1.3)
        ax.axvline(cut, color="#666", ls=(0, (5, 4)), lw=1.1)
        ax.set_xlabel("magnitude of between-visit change  |Δ|"); ax.set_ylabel("density")
        ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    else:
        ax.text(0.5, 0.5, "deltas.npz / phenotype_axis.csv missing", ha="center", transform=ax.transAxes)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = config.out("final_figure_1.png"); fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# figure 2 - compass with controls + axis
# --------------------------------------------------------------------------- #
def fig2_compass(config, d, g):
    dpi = int(config["report"]["fig_dpi"])
    fig = plt.figure(figsize=(6.8, 6.4)); ax = fig.add_subplot(111, projection="polar")
    if g is None:
        ax.text(0.5, 0.5, "inputs missing", ha="center", transform=ax.transAxes)
        out = config.out("final_figure_2.png"); fig.savefig(out, dpi=dpi); plt.close(fig); return out
    P, ang, norm, rel, cut = g["P"], g["ang"], g["norm"], g["rel"], g["cut"]
    vext = float(np.max(np.abs(P[rel, 0]))) if rel.any() else 1.0
    dn = TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)
    tf = np.linspace(0, 2 * np.pi, 240)
    ax.plot(tf, np.full_like(tf, cut), color="#9aa4ae", lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.scatter(ang[~rel], norm[~rel], s=22, color=OI["gray"], alpha=0.65, zorder=3, label="stable / core")
    ax.scatter(ang[rel], norm[rel], c=DCM(dn(P[rel, 0])), s=58, edgecolor="white", lw=0.5,
               zorder=4, label="reliable (colour = axis pos.)")
    cpath = config.out("control_deltas.npz")
    if os.path.exists(cpath):
        cz = np.load(cpath, allow_pickle=True)
        cU = _unit(cz["delta"].astype(float)); cnorm = np.linalg.norm(cz["delta"].astype(float), axis=1)
        cP = g["pca"].transform(cU); cang = np.arctan2(cP[:, 1], cP[:, 0] * g["flip"])
        ax.scatter(cang, cnorm, marker="x", s=64, color=OI["black"], lw=1.8, zorder=6,
                   label=f"control (n={len(cnorm)})")
    # draw the bipolar axis through the two poles
    poles = _axis_poles(d, g)
    rmax = float(norm.max()) * 1.02
    if poles is not None:
        neg, pos = poles
        a_pos, a_neg = g["cent"][pos], g["cent"][neg]
        ax.plot([a_neg, a_pos], [rmax, rmax], color=OI["black"], lw=0, zorder=5)
        ax.annotate("", xy=(a_pos, rmax), xytext=(a_neg, rmax),
                    arrowprops=dict(arrowstyle="<->", color="#222", lw=2.0), zorder=7)
        ax.text(a_pos, rmax * 1.02, "axis +", fontsize=9, ha="center", color=OI["blue"], fontweight="bold")
        ax.text(a_neg, rmax * 1.02, "axis −", fontsize=9, ha="center", color=OI["vermillion"], fontweight="bold")
    ax.set_rlim(0, rmax * 1.12); ax.set_yticks([cut]); ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=8); ax.grid(color="#dfe3e7", lw=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.14, 1.10), fontsize=7.5, frameon=False)
    fig.tight_layout()
    out = config.out("final_figure_2.png"); fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# figure 3 - baseline theta anchor + quartile bars
# --------------------------------------------------------------------------- #
def fig3_theta(config, d):
    dpi = int(config["report"]["fig_dpi"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    axis_df = d["axis_df"]
    fc_path = config.out("qeeg_connectivity")
    have = (os.path.exists(fc_path + ".parquet") or os.path.exists(fc_path + ".csv")) and axis_df is not None
    col = "rel_theta_global_baseline"
    if have:
        fc = io.read_table(fc_path); fc["progression_id"] = fc["progression_id"].astype(str)
        m = fc.merge(axis_df.assign(progression_id=lambda x: x["progression_id"].astype(str)),
                     on="progression_id", how="inner")
        m = m[m["is_reliable"].astype(bool)]
    if have and col in m.columns and len(m) > 3:
        x = m["axis_coord"].to_numpy(float); y = m[col].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y); x, y = x[good], y[good]
        vext = float(np.max(np.abs(x))) if len(x) else 1.0
        # left: scatter + fit
        axL.scatter(x, y, c=DCM(TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)(x)), s=52,
                    edgecolor="white", lw=0.5, zorder=3)
        if len(x) >= 2:
            b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
            axL.plot(xs, np.polyval(b, xs), color="#333", lw=1.6, zorder=2)
        axL.set_xlabel("axis position (PC1)"); axL.set_ylabel("baseline relative theta power")
        # right: mean baseline theta across axis-position quartiles (the correlation as bars)
        qs = np.quantile(x, [0, .25, .5, .75, 1.0])
        binid = np.clip(np.digitize(x, qs[1:-1]), 0, 3)
        means = [np.nanmean(y[binid == q]) if np.any(binid == q) else np.nan for q in range(4)]
        sems = [np.nanstd(y[binid == q]) / max(1, np.sqrt(np.sum(binid == q))) if np.any(binid == q) else 0
                for q in range(4)]
        cols = [DCM(TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)(v))
                for v in [np.mean(x[binid == q]) if np.any(binid == q) else 0 for q in range(4)]]
        axR.bar(range(4), means, yerr=sems, color=cols, edgecolor="white", capsize=3, zorder=3)
        axR.set_xticks(range(4)); axR.set_xticklabels(["Q1\n(axis −)", "Q2", "Q3", "Q4\n(axis +)"], fontsize=8.5)
        axR.set_ylabel("mean baseline relative theta")
    else:
        for ax in (axL, axR):
            ax.text(0.5, 0.5, "qeeg_connectivity / theta missing", ha="center", transform=ax.transAxes)
    for ax in (axL, axR):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = config.out("final_figure_3.png"); fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# figure 4 - annotated compass: axis arrow + c3 lead
# --------------------------------------------------------------------------- #
def fig4_annotated(config, d, g):
    dpi = int(config["report"]["fig_dpi"])
    fig = plt.figure(figsize=(6.8, 6.4)); ax = fig.add_subplot(111, projection="polar")
    if g is None or not g["cent"]:
        ax.text(0.5, 0.5, "inputs missing", ha="center", transform=ax.transAxes)
        out = config.out("final_figure_4.png"); fig.savefig(out, dpi=dpi); plt.close(fig); return out
    ang, norm, rel = g["ang"], g["norm"], g["rel"]
    ax.scatter(ang[rel], norm[rel], s=34, color=OI["gray"], alpha=0.45, zorder=2)
    rmax = float(norm.max()) * 1.02
    poles = _axis_poles(d, g); lead = _lead_cluster(d, g)
    labels = d.get("_labels")
    # faint spokes for all other centroids
    for c, a in g["cent"].items():
        if poles and c in poles:
            continue
        if lead is not None and c == lead:
            continue
        rr = np.nanmean(norm[(labels == c) & rel]) if labels is not None else rmax * 0.6
        ax.plot([a, a], [0, rr], color=OI["gray"], lw=1.2, alpha=0.5, zorder=3)
    # validated theta axis as a double arrow
    if poles is not None:
        neg, pos = poles
        ax.annotate("", xy=(g["cent"][pos], rmax), xytext=(g["cent"][neg], rmax),
                    arrowprops=dict(arrowstyle="<->", color=OI["blue"], lw=2.6), zorder=6)
        ax.text(g["cent"][pos], rmax * 1.04, "theta axis", fontsize=10, ha="center",
                color=OI["blue"], fontweight="bold")
    # the c3 lead as its own arrow
    if lead is not None:
        rr = np.nanmean(norm[(labels == lead) & rel]) if labels is not None else rmax * 0.8
        ax.annotate("", xy=(g["cent"][lead], rr), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=OI["green"], lw=2.4), zorder=6)
        ax.text(g["cent"][lead], rr * 1.06, f"c{lead}: α-conn. (lead)", fontsize=9, ha="center",
                color=OI["green"], fontweight="bold")
    ax.set_rlim(0, rmax * 1.16); ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=8); ax.grid(color="#dfe3e7", lw=0.7)
    fig.tight_layout()
    out = config.out("final_figure_4.png"); fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# captions + summary
# --------------------------------------------------------------------------- #
def _captions(d):
    ctl = d.get("ctl") or {}; geo = d.get("geo") or {}; slow = d.get("slow") or {}
    axq = d.get("axq") or {}; rel = d.get("rel") or {}; dirj = d.get("dir") or {}
    modes = d.get("modes") or {}; audit = d.get("audit") or {}; auc = d.get("auc") or {}
    # pull key numbers
    p_mw = ctl.get("mannwhitney_treated_gt_control_p")
    nctl = ctl.get("n_control")
    angp = None; angdeg = None
    if modes.get("validated_axis_pairs"):
        vp = modes["validated_axis_pairs"][0]; angdeg = vp[1]; angp = vp[2]
    theta_base_q = None
    for r in slow.get("baseline_anchor", {}).get("per_feature", []):
        if r.get("feature") == "rel_theta_global_baseline":
            theta_base_q = r.get("q_value")
    theta_delta_q = None
    for r in (axq.get("primary_results") or []):
        if r.get("feature") == "rel_theta_global_delta":
            theta_delta_q = r.get("q_value")
    sh = rel.get("median_split_half_cos_reliable")
    persist = dirj.get("trajectory_consistency", {}).get("mean_cosine")
    n_geo = modes.get("n_geometric_modes"); n_axis = modes.get("n_validated_axes")
    n_cand = audit.get("n_candidate_phenotype"); n_nuis = audit.get("n_nuisance_linked")
    aucv = auc.get("ordering_auc"); aucci = auc.get("ordering_auc_ci95")
    # lead cluster info
    lead = None
    for m in audit.get("per_mode", []):
        if m.get("is_geometric_mode") and not m.get("in_validated_axis"):
            if lead is None or m.get("n_patients", 0) > lead.get("n_patients", 0):
                lead = m

    C = []
    C.append(
        "Caption 1 — Magnitude of between-visit change.\n"
        "Kernel-density estimates of the change magnitude |Δ| for treated progressions above "
        "(reliable) and below (stable) the reliability cut (dashed), and for untreated controls. "
        f"Treated change exceeds control only marginally (Mann-Whitney p={_fmt(p_mw)}, "
        f"n_control={nctl if nctl is not None else 'n/a'}); the effect is qualitative, not powered. "
        "Ticks are individual progressions.")
    C.append(
        "Caption 2 — Change compass with controls and the bipolar axis. "
        "Each point is a between-visit change: radius = |Δ|, colour = position on the embedding axis "
        "(PC1, diverging red↔blue); grey points are the low-magnitude/stable core, black × are untreated "
        "controls (which sit in the core). The double arrow marks the one bipolar axis that clears the "
        f"angle-null permutation test (poles at {_fmt(angdeg,0)}°, p={_fmt(angp)}). The layout is a 2-D "
        "projection of a continuum, so read colour and radius, not fine angle.")
    C.append(
        "Caption 3 — The axis is anchored to baseline relative theta. "
        "Left: baseline relative theta power vs axis position over the reliable subset, with least-squares "
        f"fit; the association survives the pre-specified primary-family FDR (baseline q={_fmt(theta_base_q)}"
        f"; corroborated on the change coordinate, q={_fmt(theta_delta_q)}). Right: mean baseline relative "
        "theta across quartiles of axis position (Q1 = axis −, Q4 = axis +), showing the monotone trend; "
        "error bars are SEM. Theta is a canonical AD slowing marker, so the axis has a real neural correlate "
        "rather than a session nuisance.")
    lead_txt = ""
    if lead is not None:
        feats = ", ".join(h.get("feature", "") for h in lead.get("qeeg_hits", [])) or "alpha-band connectivity"
        lead_txt = (f" The one non-axis lead worth noting, c{lead.get('cluster')} "
                    f"(n={lead.get('n')}/{lead.get('n_patients')} patients), shows increased {feats} but does "
                    "not survive global FDR correction across modes (reported as a sub-threshold lead, not a phenotype).")
    C.append(
        "Caption 4 — Annotated compass: one validated axis, the rest embedding sub-structure. "
        "Centroids of the reproducible embedding direction-bundles are drawn as spokes. Only the theta axis "
        "(blue double arrow) clears the angle-null test AND carries a corrected QEEG signature; the remaining "
        "bundles are geometrically tight but have no distinct QEEG meaning after correction for the number of "
        f"modes ({n_cand if n_cand is not None else 0} candidate phenotypes; {n_nuis if n_nuis is not None else 0} "
        f"nuisance-linked).{lead_txt}")

    summary = (
        "Summary. "
        f"Spherical k-means and a von Mises–Fisher mixture over the reliable between-visit change directions "
        f"(within-session split-half direction cosine ≈ {_fmt(sh,2)}: each change is precisely measured) "
        "described a continuum rather than discrete clusters. An angle-null permutation test resolved a single "
        f"bipolar axis (poles {_fmt(angdeg,0)}° apart, p={_fmt(angp)}); a fair, equally-hard continuum null "
        "plus a patient-clustered bootstrap identified several additional geometrically concentrated, reproducible "
        f"embedding modes ({n_geo if n_geo is not None else 'several'} in total), but a per-mode QEEG audit — "
        "pre-specified primary family, patient-clustered cluster-robust OLS, global FDR across all mode×feature "
        "tests plus a patient-block permutation, and a small-cluster guard — found no distinct, corrected QEEG "
        "signature for any of them, and no nuisance (interval/magnitude/calendar) link. Only the bipolar axis is "
        f"anchored to a canonical marker, baseline relative theta power (FDR q={_fmt(theta_base_q)}). The change is "
        f"reliable but non-persistent (consecutive-step cosine {_fmt(persist,2)}, i.e. mean-reverting), and its "
        f"magnitude exceeds untreated controls only marginally (p={_fmt(p_mw)})"
        + (f"; the ordering model is weak (pooled OOF AUC {_fmt(aucv,2)}"
           + (f" [{_fmt(aucci[0],2)}, {_fmt(aucci[1],2)}]" if isinstance(aucci, (list, tuple)) and len(aucci) == 2 else "")
           + ")" if aucv is not None else "") +
        ". Net: a reliable, non-persistent, theta-anchored bipolar axis of between-visit EEG change; the other "
        "embedding modes are sub-structure without distinct electrophysiology. Clinical correlation is the decisive "
        "pending test.")
    return C, summary


def main(config: Config) -> str:
    d = _load(config)
    g = _compass_geometry(config, d)
    outs = [fig1_magnitude(config, d), fig2_compass(config, d, g),
            fig3_theta(config, d), fig4_annotated(config, d, g)]
    caps, summary = _captions(d)
    cap_path = config.out("final_figure_captions.txt")
    with open(cap_path, "w") as f:
        f.write("FINAL FIGURE CAPTIONS\n=====================\n\n")
        f.write("\n\n".join(caps))
        f.write("\n\n\n" + summary + "\n")
    for o in outs:
        print(f"[final_figures] wrote {os.path.basename(o)}")
    print(f"[final_figures] wrote {os.path.basename(cap_path)}")
    print("\n" + summary)
    return cap_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final publication figures + captions")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
