"""Does the axis track the APERIODIC (1/f) spectral tilt - and does that EXPLAIN
the baseline median-frequency anchor, or just restate it?

Follow-up to axis_slowing_test, which found the axis anchored to baseline median
frequency (q=0.004) but NOT to band-power slowing ratios, and the *change* signal
= regression-to-the-mean. The dissociation (spectral-edge moves, band ratios don't)
is what you'd see if the effect lives in the APERIODIC component (1/f tilt, an E/I
proxy) rather than oscillatory band redistribution. spectral.py now emits
``aperiodic_exponent_*`` (re-run qeeg first). This tests it, honestly:

  1. ANCHOR - aperiodic_exponent (baseline) ~ axis, reliable subset, magnitude/dt
     controlled, patient-clustered.
  2. CHANGE vs RTM - aperiodic_exponent (delta) ~ axis, re-tested controlling its
     own baseline.
  3. WHO EXPLAINS WHOM (the point) -
       a. median_freq ~ axis, then + aperiodic covariate: does the median-freq
          anchor SHRINK when aperiodic is controlled? (aperiodic mediates it)
       b. aperiodic ~ axis, then + median_freq covariate: does aperiodic SURVIVE
          controlling median freq? (aperiodic is independent / the driver)
     Plus the aperiodic<->median_freq correlation (are they the same thing?).
  4. Band-power ratios re-checked (should stay flat -> supports a tilt, not
     oscillatory, story).

HONESTY: baseline-only interpretation (change is RTM); aperiodic is correlated with
median freq so at best this SHARPENS one anchor, it is not a new independent finding;
same-sample, n small; NOT a clinical result (needs the clinical table). The estimator
is a lightweight peak-masked log-log slope, not full specparam.

Reads qeeg_connectivity.* (with aperiodic_exponent_*) + phenotype_axis.csv
(+ deltas.npz). Outputs: axis_aperiodic.json, axis_aperiodic.png

Run:  python -m analysis.axis_aperiodic_test --config analysis/config.yaml
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

from scipy.stats import spearmanr

from .config import Config, load_config, add_arg
from . import io
from .axis_slowing_test import _fit, _z, OI

APX = "aperiodic_exponent_global"
MFQ = "median_freq_global"
RATIOS = ["slowing_ratio_global", "theta_alpha_global", "rel_theta_global", "rel_delta_global"]


def _cmap():
    return LinearSegmentedColormap.from_list("oi", [OI["vermillion"], "#F4F4F4", OI["blue"]], N=256)


def main(config: Config):
    fc = io.read_table(config.out("qeeg_connectivity"))
    fc["progression_id"] = fc["progression_id"].astype(str)
    if f"{APX}_baseline" not in fc.columns:
        raise SystemExit("[aperiodic] aperiodic_exponent_* not in qeeg_connectivity - "
                         "re-run `python -m analysis.qeeg` after updating spectral.py.")
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

    cov = [dft["magnitude"].to_numpy(float), dft["dt"].to_numpy(float)]
    grp = dft["patient_id"].to_numpy()
    axc = dft["axis_coord"].to_numpy(float)

    def col(name):
        return dft[name].to_numpy(float) if name in dft.columns else None

    apx_b, apx_d = col(f"{APX}_baseline"), col(f"{APX}_delta")
    mfq_b, mfq_d = col(f"{MFQ}_baseline"), col(f"{MFQ}_delta")

    # orient axis so + = LOWER baseline median freq (i.e. the "slower" pole), to
    # match axis_slowing_test's convention (interpretability only).
    if mfq_b is not None:
        f0 = _fit(mfq_b, axc, cov, grp)
        axc = axc * (1.0 if f0["beta"] <= 0 else -1.0)

    # (1) anchor
    ap_anchor = _fit(apx_b, axc, cov, grp)
    ap_anchor["spearman_r"] = float(spearmanr(apx_b, axc)[0])

    # (2) change vs RTM
    ap_chg_raw = _fit(apx_d, axc, cov, grp) if apx_d is not None else None
    ap_chg_adj = _fit(apx_d, axc, cov + [apx_b], grp) if apx_d is not None else None

    # (3) who explains whom
    mfq_raw = _fit(mfq_b, axc, cov, grp) if mfq_b is not None else None
    mfq_ctrl_ap = _fit(mfq_b, axc, cov + [apx_b], grp) if mfq_b is not None else None
    ap_ctrl_mfq = _fit(apx_b, axc, cov + [mfq_b], grp) if mfq_b is not None else None
    r_ap_mfq = float(spearmanr(apx_b, mfq_b)[0]) if mfq_b is not None else np.nan

    # (4) ratios still flat?
    ratio_res = {}
    for rname in RATIOS:
        c = col(f"{rname}_baseline")
        if c is not None:
            ratio_res[rname] = _fit(c, axc, cov, grp)["p"]

    # ---- verdict ----
    def sig(fit):
        return fit is not None and fit["p"] < 0.05
    anchors = sig(ap_anchor)
    independent = sig(ap_ctrl_mfq)                              # aperiodic beats median-freq control
    mediates = (mfq_raw is not None and mfq_ctrl_ap is not None and sig(mfq_raw)
                and (not sig(mfq_ctrl_ap) or abs(mfq_ctrl_ap["beta"]) < 0.5 * abs(mfq_raw["beta"])))
    change_real = sig(ap_chg_adj)

    if anchors and independent and mediates:
        interp = ("Axis tracks the APERIODIC 1/f tilt, and it accounts for the median-"
                  "frequency anchor (median-freq assoc shrinks when aperiodic is controlled; "
                  "aperiodic survives controlling median-freq). Mechanistic upgrade: the "
                  "anchor is spectral tilt (E/I proxy), not oscillatory slowing.")
    elif anchors and not independent:
        interp = ("Aperiodic tracks the axis but is collinear with median frequency and does "
                  "NOT survive controlling for it -> aperiodic and median-freq are the same "
                  "signal; median frequency stays the cleaner summary. No mechanistic upgrade.")
    elif not anchors:
        interp = ("Aperiodic 1/f tilt does NOT track the axis -> the anchor is the oscillatory/"
                  "edge median-frequency measure, not the aperiodic component.")
    else:
        interp = ("Mixed: aperiodic anchors and is partly independent but does not clearly "
                  "mediate the median-frequency anchor. Underpowered; treat as suggestive.")

    # ============================ figure ============================
    dcm = _cmap()
    vext = float(np.max(np.abs(axc))) if len(axc) else 1.0
    dn = TwoSlopeNorm(vmin=-vext, vcenter=0, vmax=vext)
    fig = plt.figure(figsize=(18, 5.6))

    # (A) aperiodic baseline vs axis
    axA = fig.add_subplot(1, 3, 1)
    axA.scatter(axc, apx_b, c=dcm(dn(axc)), s=46, edgecolor="white", lw=0.5, zorder=3)
    b = np.polyfit(axc, apx_b, 1); xs = np.linspace(axc.min(), axc.max(), 50)
    axA.plot(xs, np.polyval(b, xs), color="#333", lw=1.6, zorder=2)
    axA.text(0.04, 0.05, f"β={ap_anchor['beta']:.2f}  p={ap_anchor['p']:.4f}\n"
                         f"Spearman r={ap_anchor['spearman_r']:.2f}",
             transform=axA.transAxes, fontsize=9,
             bbox=dict(boxstyle="round", fc="#f4f7fb", ec="#cfd8e3"))
    axA.set_xlabel("axis position  (+ = slower-freq pole)")
    axA.set_ylabel("baseline aperiodic exponent (1/f χ)")
    axA.set_title("(A) Aperiodic anchor: does 1/f tilt track the axis?", fontsize=10.5)
    for s in ["top", "right"]:
        axA.spines[s].set_visible(False)

    # (B) who explains whom
    axB = fig.add_subplot(1, 3, 2)
    labels, betas, ps, cols = [], [], [], []
    def add(lbl, fit, c):
        if fit is not None:
            labels.append(lbl); betas.append(abs(fit["beta"])); ps.append(fit["p"]); cols.append(c)
    add("median-freq ~ axis", mfq_raw, OI["orange"])
    add("…controlling aperiodic", mfq_ctrl_ap, "#f0c890")
    add("aperiodic ~ axis", ap_anchor, OI["blue"])
    add("…controlling median-freq", ap_ctrl_mfq, "#9ec6e6")
    yy = np.arange(len(labels))[::-1]
    axB.barh(yy, betas, color=cols, edgecolor="white")
    for y, bb, pp in zip(yy, betas, ps):
        axB.text(bb + 0.01, y, f"p={pp:.3f}", va="center", fontsize=8)
    axB.set_yticks(yy); axB.set_yticklabels(labels, fontsize=8.5)
    axB.set_xlabel("|standardized β| vs axis")
    axB.set_title("(B) Who explains whom? (bar shrinks = association absorbed)", fontsize=10.5)
    for s in ["top", "right"]:
        axB.spines[s].set_visible(False)

    # (C) aperiodic vs median freq + RTM
    axC = fig.add_subplot(1, 3, 3)
    if mfq_b is not None:
        axC.scatter(mfq_b, apx_b, c=dcm(dn(axc)), s=44, edgecolor="white", lw=0.4, zorder=3)
        axC.set_xlabel("baseline median frequency (Hz)")
    axC.set_ylabel("baseline aperiodic exponent (χ)")
    lines = [f"aperiodic ↔ median-freq: r={r_ap_mfq:.2f}"]
    if ap_chg_raw is not None and ap_chg_adj is not None:
        lines.append(f"aperiodic Δ ~ axis: p={ap_chg_raw['p']:.3f} → RTM-adj p={ap_chg_adj['p']:.3f}")
    axC.text(0.03, 0.97, "\n".join(lines), transform=axC.transAxes, fontsize=8.5, va="top",
             bbox=dict(boxstyle="round", fc="#fbf7ee", ec="#e5d9bf"))
    axC.set_title("(C) Same thing as median-freq? + change/RTM", fontsize=10.5)
    for s in ["top", "right"]:
        axC.spines[s].set_visible(False)

    fig.suptitle("Is the axis an APERIODIC (1/f tilt) anchor, or just median frequency?  "
                 "(n=%d reliable / %d patients; baseline; patient-clustered)"
                 % (len(dft), dft["patient_id"].nunique()), fontsize=11.5, y=1.02)
    fig.tight_layout()
    out = config.out("axis_aperiodic.png")
    fig.savefig(out, dpi=int(config["report"]["fig_dpi"]), bbox_inches="tight"); plt.close(fig)

    io.write_json({
        "hypothesis": "the axis's baseline anchor is the aperiodic 1/f tilt (spectral-edge "
                      "measures move, band-power ratios don't).",
        "n_reliable": int(len(dft)), "n_patients": int(dft["patient_id"].nunique()),
        "aperiodic_anchor": ap_anchor,
        "aperiodic_change_raw": ap_chg_raw, "aperiodic_change_baseline_adjusted": ap_chg_adj,
        "median_freq_vs_axis_raw": mfq_raw,
        "median_freq_vs_axis_controlling_aperiodic": mfq_ctrl_ap,
        "aperiodic_vs_axis_controlling_median_freq": ap_ctrl_mfq,
        "aperiodic_median_freq_corr": r_ap_mfq,
        "band_power_ratio_pvalues": ratio_res,
        "flags": {"anchors": bool(anchors), "independent_of_median_freq": bool(independent),
                  "mediates_median_freq_anchor": bool(mediates), "change_survives_rtm": bool(change_real)},
        "interpretation": interp,
        "caveat": "Baseline-only (change is RTM); aperiodic correlated with median freq -> "
                  "at best sharpens one anchor; same-sample, n small; NOT clinical; lightweight "
                  "peak-masked slope, not full specparam.",
    }, config.out("axis_aperiodic.json"))

    print(f"[aperiodic] n={len(dft)} reliable / {dft['patient_id'].nunique()} patients")
    print(f"[aperiodic] (1) anchor: aperiodic ~ axis  β={ap_anchor['beta']:.2f} p={ap_anchor['p']:.4f} "
          f"(r={ap_anchor['spearman_r']:.2f})")
    if mfq_raw is not None:
        print(f"[aperiodic] (3) median-freq ~ axis p={mfq_raw['p']:.3f} -> controlling aperiodic p="
              f"{mfq_ctrl_ap['p']:.3f} (β {mfq_raw['beta']:.2f}->{mfq_ctrl_ap['beta']:.2f})")
        print(f"[aperiodic]     aperiodic ~ axis controlling median-freq p={ap_ctrl_mfq['p']:.3f}; "
              f"aperiodic<->median-freq r={r_ap_mfq:.2f}")
    if ap_chg_adj is not None:
        print(f"[aperiodic] (2) change RTM: p={ap_chg_raw['p']:.3f} -> adj p={ap_chg_adj['p']:.3f}")
    print(f"[aperiodic] VERDICT: {interp}")
    return config.out("axis_aperiodic.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axis vs aperiodic 1/f tilt")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
