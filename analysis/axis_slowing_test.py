"""Confirmatory test of the SLOWING hypothesis for the bipolar axis.

The exploratory axis_qeeg sweep (102 change features, 0 survived FDR) nonetheless
showed a coherent, non-EMG thread: median frequency down, SEF95 down, theta-band
network clustering up - i.e. EEG SLOWING - plus the single strongest association
of all, at BASELINE: baseline median frequency predicts axis position (r=0.65,
p~5e-4). This script tests that hypothesis directly and honestly, three ways:

  1. BASELINE ANCHOR - does axis position track baseline spectral slowing? Tested
     on a small a-priori panel with within-panel BH-FDR, so median-freq etc. face
     a ~10-test burden, not 200. If it survives here, it "survives correction on
     its own".

  2. COHERENCE (beyond chance) - the a-priori SLOWING PANEL (median freq, SEF95,
     PAF, slowing ratio, theta/alpha, rel theta/delta/alpha, spectral entropy;
     each with a physiologically pre-declared sign) is tested for whether its
     members move COHERENTLY along the axis - a sign-concordance (binomial) test.
     A composite slowing score aggregates them for one powerful test.

  3. CHANGE vs REGRESSION-TO-THE-MEAN - baseline median-freq predicts the axis AND
     delta median-freq predicts it with opposite sign; since delta = after-before,
     that can be mechanical. So the delta / composite-delta association is re-tested
     controlling for the feature's OWN baseline. Survives -> real change; collapses
     -> it was RTM.

HONESTY: the spectral-slowing panel is TEXTBOOK a-priori dementia-EEG physiology,
declared independent of this dataset, so testing it as a panel is legitimate (not
double-dipping). The theta-CONNECTIVITY features were surfaced by the data, so they
are reported separately as "concordant, exploratory" - NOT part of the confirmatory
panel. Same-sample throughout (n small); true confirmation is out-of-sample / the
clinical cohort. Inference is patient-clustered.

Reads qeeg_connectivity.* + phenotype_axis.csv (+ deltas.npz for dt). Outputs:
  axis_slowing.json, axis_slowing.png, axis_slowing.csv

Run:  python -m analysis.axis_slowing_test --config analysis/config.yaml
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

from scipy.stats import spearmanr, binomtest

from .config import Config, load_config, add_arg
from . import io
from .phenotype_stats import benjamini_hochberg

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "gray": "#999999", "black": "#000000"}

# a-priori spectral-slowing panel: (base_name, slow_sign, label)
#   slow_sign = +1  -> feature INCREASES with slowing;  -1 -> DECREASES with slowing
PANEL = [
    ("median_freq_global", -1, "median freq"),
    ("sef95_global",       -1, "SEF95"),
    ("paf_global",         -1, "peak alpha freq"),
    ("rel_alpha_global",   -1, "rel. alpha"),
    ("spectral_entropy_global", -1, "spectral entropy"),
    ("slowing_ratio_global", +1, "slowing ratio"),
    ("theta_alpha_global",   +1, "theta/alpha"),
    ("rel_theta_global",     +1, "rel. theta"),
    ("rel_delta_global",     +1, "rel. delta"),
]
# data-surfaced, reported as concordant-but-exploratory (NOT confirmatory)
EXPLORATORY = [
    ("graph_imcoh_theta_weighted_clustering", +1, "θ imcoh clustering"),
    ("graph_wpli_theta_weighted_clustering",  +1, "θ wPLI clustering"),
]


def _z(a):
    a = np.asarray(a, float); s = np.nanstd(a)
    return (a - np.nanmean(a)) / s if s > 0 else np.zeros_like(a)


_NAN_FIT = {"beta": np.nan, "se": np.nan, "p": np.nan, "lo": np.nan, "hi": np.nan,
            "method": "insufficient_finite_n"}


def _fit(y, x, covars, groups):
    """Standardized regression of y on x controlling covars; patient-clustered SE
    via statsmodels when available (else classical OLS fallback). Rows with any
    non-finite value (NaN/inf) in y/x/covars are dropped first (some spectral
    features - e.g. unresolved PAF - are legitimately NaN). Returns dict(beta, se,
    p, lo, hi, method)."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    cov = [np.asarray(c, float) for c in covars] if covars else []
    groups = np.asarray(groups)
    mask = np.isfinite(y) & np.isfinite(x)
    for c in cov:
        mask &= np.isfinite(c)
    if int(mask.sum()) < max(5, len(cov) + 3):
        return dict(_NAN_FIT)
    y, x = y[mask], x[mask]; cov = [c[mask] for c in cov]; groups = groups[mask]

    Z = np.column_stack([_z(x)] + [_z(c) for c in cov]) if cov else _z(x).reshape(-1, 1)
    try:
        import statsmodels.api as sm
        X = sm.add_constant(Z)
        m = sm.OLS(_z(y), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        ci = m.conf_int()
        return {"beta": float(m.params[1]), "se": float(m.bse[1]), "p": float(m.pvalues[1]),
                "lo": float(ci[1][0]), "hi": float(ci[1][1]), "method": "ols_cluster_robust"}
    except Exception:
        X = np.column_stack([np.ones(len(y)), Z]); yz = _z(y)
        beta, *_ = np.linalg.lstsq(X, yz, rcond=None)
        resid = yz - X @ beta; dof = max(len(y) - X.shape[1], 1)
        s2 = float(resid @ resid) / dof
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
        from scipy.stats import t
        p = float(2 * t.sf(abs(beta[1] / se), dof)) if se > 0 else np.nan
        return {"beta": float(beta[1]), "se": se, "p": p,
                "lo": beta[1] - 1.96 * se, "hi": beta[1] + 1.96 * se, "method": "ols_classic_fallback"}


def _diverging_cmap():
    return LinearSegmentedColormap.from_list("oi", [OI["vermillion"], "#F4F4F4", OI["blue"]], N=256)


def main(config: Config):
    ps = config["phenotype_stats"]
    seed = int(config["seed"])
    alpha = float(ps.get("fdr_alpha", 0.05))

    fc = io.read_table(config.out("qeeg_connectivity"))
    fc["progression_id"] = fc["progression_id"].astype(str)
    axis = pd.read_csv(config.out("phenotype_axis.csv"))
    axis["progression_id"] = axis["progression_id"].astype(str)
    df = fc.merge(axis[["progression_id", "axis_coord", "magnitude", "is_reliable"]],
                  on="progression_id", how="inner")
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    dtmap = dict(zip([str(x) for x in dz["progression_id"]], dz["dt"]))
    df["dt"] = df["progression_id"].astype(str).map(dtmap)
    if "patient_id" not in df.columns:
        df["patient_id"] = df["progression_id"].str.split("__").str[0]

    dft = df[df["is_reliable"].astype(bool)].dropna(subset=["axis_coord", "magnitude", "dt"]).copy()
    covars = [dft["magnitude"].to_numpy(float), dft["dt"].to_numpy(float)]
    groups = dft["patient_id"].to_numpy()
    axis0 = dft["axis_coord"].to_numpy(float)

    have = [(b, s, lab) for (b, s, lab) in PANEL if (b + "_baseline") in dft.columns]

    # ---- composite baseline slowing score (sign-aligned z; NaN-robust) ----
    def _sign_aligned(kind):  # kind = "baseline" | "delta"
        cols, used = [], []
        for b, s, lab in have:
            c = f"{b}_{kind}"
            if c in dft.columns and dft[c].notna().sum() >= 8 and dft[c].std() > 0:
                cols.append(s * _z(dft[c].to_numpy(float))); used.append((b, s, lab))
        M = np.vstack(cols).T if cols else np.zeros((len(dft), 0))
        # per-row nanmean over available features -> a NaN in one feature (e.g.
        # unresolved PAF) doesn't nuke the whole progression's composite.
        if M.shape[1]:
            allnan = ~np.isfinite(M).any(axis=1)
            comp = np.where(allnan, np.nan,
                            np.nanmean(np.where(np.isfinite(M), M, np.nan), axis=1))
        else:
            comp = np.zeros(len(dft))
        return comp, used

    comp_base, used_b = _sign_aligned("baseline")
    comp_delta, used_d = _sign_aligned("delta")

    # orient axis so that + = MORE baseline slowing (interpretability only)
    anchor_fit0 = _fit(comp_base, axis0, covars, groups)
    orient = -1.0 if np.isfinite(anchor_fit0["beta"]) and anchor_fit0["beta"] < 0 else 1.0
    axis = axis0 * orient

    # ---- (1) BASELINE ANCHOR: each panel feature ~ axis, within-panel FDR ----
    rows_b = []
    for b, s, lab in have:
        col = f"{b}_baseline"
        y = dft[col].to_numpy(float)
        f = _fit(y, axis, covars, groups)
        r_sp = float(spearmanr(y, axis)[0])
        rows_b.append({"feature": col, "label": lab, "slow_sign": s, "spearman_r": r_sp,
                       "beta": f["beta"], "beta_aligned": s * f["beta"],
                       "lo": f["lo"], "hi": f["hi"], "p": f["p"]})
    pv = np.array([r["p"] for r in rows_b], float); ok = ~np.isnan(pv)
    q = np.full(len(pv), np.nan); rej = np.zeros(len(pv), bool)
    if ok.any():
        rj, qq = benjamini_hochberg(pv[ok], alpha=alpha); rej[ok] = rj; q[ok] = qq
    for i, r in enumerate(rows_b):
        r["q_value"] = None if np.isnan(q[i]) else float(q[i]); r["fdr_sig"] = bool(rej[i])

    comp_base_fit = _fit(comp_base, axis, covars, groups)

    # ---- (2) COHERENCE: sign-concordance of the panel (aligned betas) ----
    aligned = np.array([r["beta_aligned"] for r in rows_b])
    ref = 1.0  # after orientation, coherent slowing => aligned betas POSITIVE
    n_conc = int(np.sum(aligned > 0)); n_tot = len(aligned)
    sign_p = binomtest(n_conc, n_tot, 0.5).pvalue if n_tot else np.nan

    # ---- (3) CHANGE vs RTM ----
    def _change_block(comp_or_col, is_col):
        y = dft[comp_or_col].to_numpy(float) if is_col else comp_or_col
        raw = _fit(y, axis, covars, groups)
        # add own baseline as covariate
        if is_col:
            bcol = comp_or_col.replace("_delta", "_baseline")
            extra = dft[bcol].to_numpy(float) if bcol in dft.columns else None
        else:
            extra = comp_base
        adj = _fit(y, axis, covars + [extra], groups) if extra is not None else None
        return raw, adj

    comp_delta_raw, comp_delta_adj = _change_block(comp_delta, is_col=False)
    mf_raw = mf_adj = None
    if "median_freq_global_delta" in dft.columns:
        mf_raw, mf_adj = _change_block("median_freq_global_delta", is_col=True)

    # ---- exploratory concordant (theta connectivity), NOT confirmatory ----
    rows_expl = []
    for b, s, lab in EXPLORATORY:
        col = f"{b}_delta"
        if col in dft.columns and dft[col].std() > 0:
            f = _fit(dft[col].to_numpy(float), axis, covars, groups)
            rows_expl.append({"feature": col, "label": lab, "beta_aligned": s * f["beta"], "p": f["p"]})

    # ============================ figure ============================
    dcm = _diverging_cmap()
    vext = float(np.max(np.abs(axis))) if len(axis) else 1.0
    dnorm = TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)
    fig = plt.figure(figsize=(18, 5.6))

    # (A) baseline anchor - median freq (concrete, Hz)
    axA = fig.add_subplot(1, 3, 1)
    if "median_freq_global_baseline" in dft.columns:
        y = dft["median_freq_global_baseline"].to_numpy(float)
        axA.scatter(axis, y, c=dcm(dnorm(axis)), s=46, edgecolor="white", lw=0.5, zorder=3)
        b = np.polyfit(axis, y, 1); xs = np.linspace(axis.min(), axis.max(), 50)
        axA.plot(xs, np.polyval(b, xs), color="#333", lw=1.6, zorder=2)
        mfr = next((r for r in rows_b if r["feature"] == "median_freq_global_baseline"), None)
        txt = f"Spearman r={mfr['spearman_r']:.2f}\np={mfr['p']:.4f}"
        if mfr and mfr["q_value"] is not None:
            txt += f"  q={mfr['q_value']:.3f}" + ("  *FDR" if mfr["fdr_sig"] else "")
        axA.text(0.04, 0.05, txt, transform=axA.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round", fc="#f4f7fb", ec="#cfd8e3"))
    axA.set_xlabel("axis position  (+ = more baseline slowing)")
    axA.set_ylabel("baseline median frequency (Hz)")
    axA.set_title("(A) Baseline anchor: axis tracks baseline EEG frequency", fontsize=10.5)
    for s in ["top", "right"]:
        axA.spines[s].set_visible(False)

    # (B) coherence forest - aligned betas per panel feature
    axB = fig.add_subplot(1, 3, 2)
    order = np.argsort(aligned)
    ys = np.arange(len(order))
    for yi, idx in zip(ys, order):
        r = rows_b[idx]
        col = OI["blue"] if r["beta_aligned"] > 0 else OI["gray"]
        axB.plot([r["slow_sign"] * r["lo"], r["slow_sign"] * r["hi"]], [yi, yi], color=col, lw=2, zorder=2)
        axB.scatter([r["beta_aligned"]], [yi], color=col, s=42, zorder=3, edgecolor="white", lw=0.6)
    axB.axvline(0, color="#888", lw=1)
    axB.axvspan(0, axB.get_xlim()[1] if axB.get_xlim()[1] > 0 else 1, color=OI["blue"], alpha=0.05)
    axB.set_yticks(ys); axB.set_yticklabels([rows_b[i]["label"] for i in order], fontsize=8)
    axB.set_xlabel("aligned β  (+ = consistent with slowing along axis)")
    axB.set_title(f"(B) Coherent a-priori slowing panel\n{n_conc}/{n_tot} concordant, "
                  f"sign-test p={sign_p:.3f}", fontsize=10.5)
    for s in ["top", "right"]:
        axB.spines[s].set_visible(False)

    # (C) composite slowing vs axis + RTM annotation
    axC = fig.add_subplot(1, 3, 3)
    axC.scatter(axis, comp_base, c=dcm(dnorm(axis)), s=40, edgecolor="white", lw=0.4,
                zorder=3, label="baseline slowing")
    b = np.polyfit(axis, comp_base, 1); xs = np.linspace(axis.min(), axis.max(), 50)
    axC.plot(xs, np.polyval(b, xs), color="#333", lw=1.6, zorder=2)
    lines = [f"composite BASELINE slowing ~ axis: β={comp_base_fit['beta']:.2f}, p={comp_base_fit['p']:.4f}"]
    lines.append(f"composite ΔCHANGE ~ axis: β={comp_delta_raw['beta']:.2f}, p={comp_delta_raw['p']:.3f}")
    if comp_delta_adj is not None:
        lines.append(f"   Δ after baseline-control (RTM): β={comp_delta_adj['beta']:.2f}, p={comp_delta_adj['p']:.3f}")
    if mf_raw is not None and mf_adj is not None:
        lines.append(f"median-freq Δ ~ axis: p={mf_raw['p']:.3f} → RTM-adj p={mf_adj['p']:.3f}")
    axC.text(0.03, 0.97, "\n".join(lines), transform=axC.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", fc="#fbf7ee", ec="#e5d9bf"))
    axC.set_xlabel("axis position  (+ = more baseline slowing)")
    axC.set_ylabel("composite slowing score (baseline, z)")
    axC.set_title("(C) Composite slowing + change/RTM check", fontsize=10.5)
    for s in ["top", "right"]:
        axC.spines[s].set_visible(False)

    fig.suptitle("Is the bipolar axis a spectral-SLOWING axis?  (n=%d reliable / %d patients; "
                 "same-sample, patient-clustered)" % (len(dft), dft["patient_id"].nunique()),
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = config.out("axis_slowing.png")
    fig.savefig(out_png, dpi=int(config["report"]["fig_dpi"]), bbox_inches="tight"); plt.close(fig)

    pd.DataFrame(rows_b).to_csv(config.out("axis_slowing.csv"), index=False)

    # ---- verdict logic ----
    anchor_survives = any(r["fdr_sig"] for r in rows_b)
    coherent = bool(np.isfinite(sign_p) and sign_p < 0.05 and n_conc >= max(2, n_tot - 2))
    change_real = (comp_delta_adj is not None and comp_delta_adj["p"] < 0.05)
    verdict = ("SUPPORTED" if (anchor_survives and coherent) else
               "PARTIAL" if (anchor_survives or coherent) else "NOT SUPPORTED")

    io.write_json({
        "hypothesis": "the bipolar axis is a spectral-slowing axis: baseline-frequency "
                      "anchor (survives within-panel FDR) + coherent a-priori slowing panel; "
                      "change signal checked against regression-to-the-mean.",
        "verdict": verdict,
        "n_reliable": int(len(dft)), "n_patients": int(dft["patient_id"].nunique()),
        "axis_oriented_so_positive_is": "more baseline slowing",
        "baseline_anchor": {
            "composite": comp_base_fit,
            "per_feature": rows_b,
            "any_survives_panel_fdr": anchor_survives,
            "strongest": min(rows_b, key=lambda r: (r["p"] if r["p"] == r["p"] else 1.0))["feature"],
        },
        "coherence": {"n_concordant": n_conc, "n_panel": n_tot, "sign_test_p": float(sign_p),
                      "coherent": coherent},
        "change_vs_rtm": {
            "composite_delta_raw": comp_delta_raw, "composite_delta_baseline_adjusted": comp_delta_adj,
            "median_freq_delta_raw": mf_raw, "median_freq_delta_baseline_adjusted": mf_adj,
            "change_survives_rtm": bool(change_real)},
        "exploratory_concordant_connectivity": rows_expl,
        "caveat": "Same-sample; spectral panel is a-priori physiology (legitimate), theta-"
                  "connectivity is data-surfaced (exploratory). n small -> confirmation needs "
                  "out-of-sample / clinical cohort.",
    }, config.out("axis_slowing.json"))

    print(f"[slowing] n={len(dft)} reliable / {dft['patient_id'].nunique()} patients; "
          f"axis oriented so + = more baseline slowing.")
    print(f"[slowing] (1) ANCHOR: composite baseline slowing ~ axis  β={comp_base_fit['beta']:.2f} "
          f"p={comp_base_fit['p']:.4f}; panel-FDR survivors: "
          f"{[r['feature'] for r in rows_b if r['fdr_sig']] or 'none'}")
    print(f"[slowing] (2) COHERENCE: {n_conc}/{n_tot} panel features concordant, sign-test p={sign_p:.3f}")
    if comp_delta_adj is not None:
        print(f"[slowing] (3) CHANGE: Δcomposite ~ axis raw p={comp_delta_raw['p']:.3f} -> "
              f"baseline-adjusted (RTM) p={comp_delta_adj['p']:.3f} "
              f"({'survives -> real change' if change_real else 'collapses -> regression to the mean'})")
    print(f"[slowing] VERDICT: {verdict}")
    return config.out("axis_slowing.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confirmatory slowing-axis test")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
