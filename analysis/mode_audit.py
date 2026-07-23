"""Audit each geometric mode: is it a nuisance bundle, and does it have distinct QEEG?

direction_modes certifies that several change-direction bundles are GEOMETRIC modes
- genuinely near-parallel in the embedding and reproducible under resampling. That
is necessary but NOT sufficient to call them phenotypes. This module runs the two
tests that decide interpretation, per mode:

  1. NUISANCE ASSOCIATION - is the bundle explained by a measured confound rather
     than neurophysiology? For each mode vs the rest of the reliable set we test
     inter-visit interval (dt), delta magnitude, and calendar-time batching of the
     sessions (a scanner/protocol-drift proxy), plus patient concentration. A mode
     that is really just "the long-interval progressions" or "the high-magnitude
     progressions" or "the ones recorded in one week" is embedding sub-structure,
     not a phenotype. BH-FDR across modes within each factor.

  2. PER-MODE QEEG DISTINCTIVENESS - does the mode's QEEG CHANGE differ from the
     rest, beyond magnitude/dt? Same pre-specified primary family and the same
     patient-clustered cluster-robust OLS used by axis_qeeg, membership (0/1) as
     the regressor. Effect sizes + CIs are reported, not just p, with an explicit
     small-n power note. BH-FDR within the primary family, per mode.

Verdict per mode:
  - validated axis (angle-null)          -> phenotype-eligible (interpret via axis_qeeg)
  - nuisance q<0.05 (dt/magnitude/time)  -> likely nuisance-linked (embedding artifact)
  - >=1 primary QEEG FDR-significant AND not nuisance -> CANDIDATE phenotype (verify)
  - otherwise                            -> embedding mode only (no distinct QEEG; note n)

Reads deltas.npz + directional_phenotype_labels.csv + direction_modes.json
(+ progressions.* for dates, + qeeg_connectivity.* for the QEEG block; both optional).
Outputs: mode_audit.{json,csv,png}
Run:  python -m analysis.mode_audit --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

from .config import Config, load_config, add_arg
from . import io
from .phenotype_stats import benjamini_hochberg
from .axis_qeeg import _cluster_ols, _perm_pvalue, _EMG

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "gray": "#999999", "black": "#000000"}

# same pre-specified AD-qEEG primary family as axis_qeeg (config-overridable)
PRIMARY_DEFAULT = ["median_freq_global", "rel_alpha_posterior", "rel_theta_global",
                   "slowing_ratio_posterior", "alpha_cog_global",
                   "aperiodic_exponent_global", "wpli_alpha_global",
                   "wpli_alpha_posterior", "graph_wpli_alpha_global_efficiency"]


def _mwu(mode_vals, rest_vals):
    """Mann-Whitney mode-vs-rest with rank-biserial effect size (in [-1,1];
    positive = mode ranks higher). NaN-robust; returns (effect, p, med_mode, med_rest)."""
    a = np.asarray(mode_vals, float); b = np.asarray(rest_vals, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, (float(np.median(a)) if len(a) else np.nan), (float(np.median(b)) if len(b) else np.nan)
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    rbc = 2.0 * U / (len(a) * len(b)) - 1.0
    return float(rbc), float(p), float(np.median(a)), float(np.median(b))


def _bh_over_modes(pvals, alpha=0.05):
    """BH-FDR across modes for one nuisance factor. Returns q per input position."""
    pv = np.array(pvals, float); q = np.full(len(pv), np.nan); ok = np.isfinite(pv)
    if ok.any():
        _, qq = benjamini_hochberg(pv[ok], alpha=alpha)
        q[ok] = qq
    return q


def main(config: Config):
    seed = int(config["seed"])
    dp = config.get("directional_phenotype", {})
    ps = config["phenotype_stats"]
    which = dp.get("geometry_label", "spherical_label")
    alpha = float(ps.get("fdr_alpha", 0.05))
    n_perm = int(dp.get("axis_qeeg_perm", 2000))
    dpi = int(config["report"]["fig_dpi"])

    # ---- inputs ----
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    pid = np.array([str(x) for x in dz["progression_id"]])
    patient = np.array([str(x) for x in dz["patient_id"]])
    norm = np.linalg.norm(dz["delta"].astype(float), axis=1)
    dt = np.asarray(dz["dt"], float) if "dt" in dz else np.full(len(pid), np.nan)

    lab = pd.read_csv(config.out("directional_phenotype_labels.csv"))
    lab["progression_id"] = lab["progression_id"].astype(str)
    lut = dict(zip(lab["progression_id"], lab[which]))
    rlut = dict(zip(lab["progression_id"], lab["is_reliable"].astype(bool)))
    labels = np.array([lut.get(p, -1) for p in pid], dtype=float)
    rel = np.array([bool(rlut.get(p, False)) for p in pid]) & (labels >= 0)

    # which clusters are geometric modes / validated axes (from direction_modes)
    modes_json = {}
    mj = config.out("direction_modes.json")
    if os.path.exists(mj):
        try:
            modes_json = io.read_json(mj)
        except Exception:
            modes_json = {}
    mode_flag = {int(d["cluster"]): d for d in modes_json.get("per_direction", [])}

    base = pd.DataFrame({"progression_id": pid, "patient_id": patient,
                         "label": labels, "magnitude": norm, "dt": dt, "reliable": rel})
    base = base[base["reliable"]].copy()
    base["label"] = base["label"].astype(int)

    # ---- calendar-time (session batching) from the progressions table, if present ----
    have_dates = False
    prog_path = config.out("progressions")
    if os.path.exists(prog_path + ".parquet") or os.path.exists(prog_path + ".csv"):
        prog = io.read_table(prog_path)
        prog["progression_id"] = prog["progression_id"].astype(str)
        if "before_date" in prog.columns and "after_date" in prog.columns:
            bd = pd.to_datetime(prog["before_date"]); ad = pd.to_datetime(prog["after_date"])
            mid = bd + (ad - bd) / 2
            origin = mid.min()
            dmap = dict(zip(prog["progression_id"], (mid - origin).dt.days.astype(float)))
            base["cal_day"] = base["progression_id"].map(dmap)
            have_dates = base["cal_day"].notna().any()

    # ---- QEEG table (optional) ----
    fc = None
    fc_path = config.out("qeeg_connectivity")
    if os.path.exists(fc_path + ".parquet") or os.path.exists(fc_path + ".csv"):
        fc = io.read_table(fc_path)
        fc["progression_id"] = fc["progression_id"].astype(str)
    prim_base = ps.get("axis_primary_features", PRIMARY_DEFAULT)
    primary_cols = [f"{b}_delta" for b in prim_base]

    clusters = sorted(base["label"].unique().tolist())
    if not clusters:
        raise SystemExit("[mode_audit] no reliable labeled progressions - run directional_phenotype.")

    # ============================ per-mode nuisance ============================
    nuis_factors = ["dt", "magnitude"] + (["cal_day"] if have_dates else [])
    raw = {c: {} for c in clusters}
    for c in clusters:
        m = base["label"] == c
        for f in nuis_factors:
            rbc, p, med_m, med_r = _mwu(base.loc[m, f], base.loc[~m, f])
            raw[c][f] = {"effect_rbc": rbc, "p": p, "median_mode": med_m, "median_rest": med_r}
        # temporal compactness: mode's date span / overall span (small = bunched)
        if have_dates:
            allspan = np.nanmax(base["cal_day"]) - np.nanmin(base["cal_day"])
            mspan = np.nanmax(base.loc[m, "cal_day"]) - np.nanmin(base.loc[m, "cal_day"]) if m.sum() > 1 else np.nan
            raw[c]["cal_compactness"] = float(mspan / allspan) if allspan and np.isfinite(mspan) else np.nan

    # BH across modes within each factor
    for f in nuis_factors:
        q = _bh_over_modes([raw[c][f]["p"] for c in clusters], alpha)
        for i, c in enumerate(clusters):
            raw[c][f]["q"] = float(q[i]) if np.isfinite(q[i]) else None

    # ============================ per-mode QEEG =============================
    # Three hardenings over a naive per-mode FDR:
    #   (a) GLOBAL BH across ALL mode x feature tests (not within-mode only) - the
    #       honest multiplicity is #modes x #features, so a within-mode q under-
    #       corrects by the number of modes.
    #   (b) PATIENT-BLOCK PERMUTATION p per test - cluster-robust OLS SEs are
    #       anti-conservative when the in-group has few patients; the permutation
    #       is the trustworthy small-n companion.
    #   (c) SMALL-CLUSTER GUARD - a mode with < mode_min_patients in-group patients
    #       cannot be a 'candidate'; its OLS inference is unreliable by construction.
    min_pat = int(dp.get("mode_min_patients", 5))
    q_perm = int(dp.get("mode_qeeg_perm", dp.get("axis_qeeg_perm", 2000)))
    qeeg_by_mode = {c: {"tested": 0, "n_global_sig_non_emg": 0, "hits": [], "skipped": True}
                    for c in clusters}
    if fc is not None:
        present_primary = [c for c in primary_cols if c in fc.columns]
        dfm = base.merge(fc[["progression_id"] + present_primary], on="progression_id", how="left")
        covM = [dfm["magnitude"].to_numpy(float), dfm["dt"].to_numpy(float)]
        pats = dfm["patient_id"].to_numpy()
        flat = []                                        # every (mode, feature) test
        for c in clusters:
            y_mem = (dfm["label"].to_numpy() == c).astype(float)
            for feat in present_primary:
                yv = dfm[feat].to_numpy(float)
                mask = np.isfinite(yv) & np.isfinite(covM[0]) & np.isfinite(covM[1])
                rec = {"cluster": c, "feature": feat, "beta": np.nan, "p_cluster": np.nan,
                       "p_perm": np.nan, "emg_prone": bool(_EMG.search(feat))}
                if mask.sum() >= 8 and np.nanstd(yv[mask]) > 0 and y_mem[mask].sum() >= 2:
                    beta, pcl, method = _cluster_ols(yv[mask], y_mem[mask],
                                                     [covM[0][mask], covM[1][mask]], pats[mask])
                    rec["beta"], rec["p_cluster"], rec["method"] = float(beta), float(pcl), method
                    rec["p_perm"] = float(_perm_pvalue(yv[mask], y_mem[mask],
                                                       [covM[0][mask], covM[1][mask]],
                                                       pats[mask], q_perm, seed))
                flat.append(rec)
        # (a) GLOBAL BH across every finite p_cluster
        pv = np.array([r["p_cluster"] for r in flat], float); ok = np.isfinite(pv)
        gq = np.full(len(pv), np.nan); grej = np.zeros(len(pv), bool)
        if ok.any():
            rj, qq = benjamini_hochberg(pv[ok], alpha=alpha); grej[ok] = rj; gq[ok] = qq
        for i, r in enumerate(flat):
            r["global_q"] = float(gq[i]) if np.isfinite(gq[i]) else None
            r["global_fdr_significant"] = bool(grej[i])
        for c in clusters:
            rows = [r for r in flat if r["cluster"] == c]
            # a hit must clear GLOBAL FDR, be non-EMG, AND pass the permutation
            hits = [r for r in rows if r["global_fdr_significant"] and not r["emg_prone"]
                    and np.isfinite(r["p_perm"]) and r["p_perm"] < alpha]
            npat_c = int(base.loc[base["label"] == c, "patient_id"].nunique())
            qeeg_by_mode[c] = {
                "tested": int(sum(np.isfinite(r["p_cluster"]) for r in rows)),
                "n_global_sig_non_emg": len(hits),
                "inference_reliable": bool(npat_c >= min_pat),
                "hits": [{"feature": r["feature"], "beta": r["beta"],
                          "global_q": r["global_q"], "p_perm": r["p_perm"]} for r in hits],
                "results": rows, "skipped": False}

    # ============================ per-mode verdict =========================
    rows_out = []
    for c in clusters:
        info = mode_flag.get(c, {})
        n = int((base["label"] == c).sum())
        npat = int(base.loc[base["label"] == c, "patient_id"].nunique())
        in_axis = bool(info.get("in_validated_axis", False))
        is_geom = bool(info.get("is_geometric_mode", False))
        nuis_hit = [f for f in nuis_factors if (raw[c][f].get("q") is not None and raw[c][f]["q"] < alpha)]
        q = qeeg_by_mode[c]
        reliable = q.get("inference_reliable", npat >= min_pat)
        qeeg_distinct = (not q["skipped"]) and q.get("n_global_sig_non_emg", 0) >= 1
        if in_axis:
            verdict = "phenotype-eligible (validated angle-null axis) - interpret via axis_qeeg"
        elif nuis_hit:
            verdict = f"likely NUISANCE-linked ({'/'.join(nuis_hit)}) - embedding sub-structure, not a phenotype"
        elif qeeg_distinct and reliable:
            verdict = "CANDIDATE phenotype (distinct QEEG survives GLOBAL FDR + permutation) - verify with more n"
        elif qeeg_distinct and not reliable:
            verdict = (f"weak lead - QEEG hit but inference UNRELIABLE (only {npat} in-group patients "
                       f"< {min_pat}; cluster-robust SEs anti-conservative)")
        elif is_geom:
            verdict = (f"embedding mode only - tight & reproducible but no distinct QEEG after GLOBAL "
                       f"FDR{' (UNDERPOWERED: n=%d/%d patients)' % (n, npat) if npat < min_pat else ''}")
        else:
            verdict = "continuum / projection (not a geometric mode)"
        rows_out.append({
            "cluster": c, "n": n, "n_patients": npat,
            "is_geometric_mode": is_geom, "in_validated_axis": in_axis,
            "nuisance_flags": nuis_hit,
            "dt_effect": raw[c]["dt"]["effect_rbc"], "dt_q": raw[c]["dt"].get("q"),
            "magnitude_effect": raw[c]["magnitude"]["effect_rbc"], "magnitude_q": raw[c]["magnitude"].get("q"),
            "cal_effect": raw[c].get("cal_day", {}).get("effect_rbc") if have_dates else None,
            "cal_q": raw[c].get("cal_day", {}).get("q") if have_dates else None,
            "cal_compactness": raw[c].get("cal_compactness") if have_dates else None,
            "qeeg_primary_tested": q["tested"],
            "qeeg_n_global_sig_non_emg": q.get("n_global_sig_non_emg", 0),
            "qeeg_inference_reliable": q.get("inference_reliable", npat >= min_pat),
            "qeeg_hits": q.get("hits", []),
            "verdict": verdict})

    n_candidate = sum(1 for r in rows_out if r["verdict"].startswith("CANDIDATE"))
    n_nuisance = sum(1 for r in rows_out if "NUISANCE" in r["verdict"])
    n_axis = sum(1 for r in rows_out if r["in_validated_axis"])

    # ============================ figure ============================
    order = sorted(clusters, key=lambda c: (not mode_flag.get(c, {}).get("in_validated_axis", False),
                                            -int((base["label"] == c).sum())))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.4))

    # (A) nuisance effect sizes per mode (signed rank-biserial); * marks q<0.05
    facs = nuis_factors
    M = np.array([[raw[c][f]["effect_rbc"] for f in facs] for c in order], float)
    im = axA.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axA.set_xticks(range(len(facs)))
    axA.set_xticklabels([{"dt": "inter-visit dt", "magnitude": "delta magnitude",
                          "cal_day": "calendar time"}.get(f, f) for f in facs], fontsize=9)
    axA.set_yticks(range(len(order))); axA.set_yticklabels([f"c{c}" for c in order], fontsize=9)
    for i, c in enumerate(order):
        for j, f in enumerate(facs):
            q = raw[c][f].get("q"); e = raw[c][f]["effect_rbc"]
            star = "*" if (q is not None and q < alpha) else ""
            axA.text(j, i, f"{e:+.2f}{star}" if np.isfinite(e) else "-", ha="center", va="center",
                     fontsize=8, color="black" if abs(e) < 0.6 else "white")
    axA.set_title("(A) Nuisance association per mode (rank-biserial; * = FDR q<0.05)\n"
                  "strong signed cell = the mode IS that confound", fontsize=10)
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04, label="mode vs rest effect")

    # (B) per-mode primary QEEG distinctiveness + verdict colour
    xs = np.arange(len(order))
    nsig = [qeeg_by_mode[c].get("n_global_sig_non_emg", 0) for c in order]
    def _vcol(c):
        v = next(r["verdict"] for r in rows_out if r["cluster"] == c)
        if v.startswith("phenotype-eligible"): return OI["blue"]
        if "NUISANCE" in v: return OI["orange"]
        if v.startswith("CANDIDATE"): return OI["green"]
        if v.startswith("weak lead"): return OI["vermillion"]
        return OI["gray"]
    axB.bar(xs, nsig, color=[_vcol(c) for c in order], edgecolor="white", zorder=3)
    for i, c in enumerate(order):
        r = next(r for r in rows_out if r["cluster"] == c)
        axB.text(i, nsig[i] + 0.05, f"c{c}\nn={r['n']}/{r['n_patients']}p", ha="center", fontsize=8)
    axB.set_xticks(xs); axB.set_xticklabels([f"c{c}" for c in order], fontsize=9)
    axB.set_ylabel("# primary QEEG features distinct (non-EMG, GLOBAL FDR + perm)")
    axB.set_ylim(0, max(1.0, (max(nsig) if nsig else 0) + 1))
    axB.set_title("(B) Distinct QEEG per mode (global FDR across all mode×feature + permutation)\n"
                  "blue=axis, green=candidate, red=weak lead (unreliable n), orange=nuisance, grey=none",
                  fontsize=8.8)
    for s in ["top", "right"]:
        axB.spines[s].set_visible(False)
    if fc is None:
        axB.text(0.5, 0.5, "qeeg_connectivity not found\n(run module 6)", transform=axB.transAxes,
                 ha="center", va="center", color=OI["gray"])

    n_weak = sum(1 for r in rows_out if r["verdict"].startswith("weak lead"))
    fig.suptitle(f"Mode audit: {n_axis} validated axis, {n_candidate} candidate phenotype(s) "
                 f"(global FDR + perm), {n_weak} weak lead(s), {n_nuisance} nuisance-linked, "
                 f"rest embedding-only", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = config.out("mode_audit.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close(fig)

    pd.DataFrame(rows_out).to_csv(config.out("mode_audit.csv"), index=False)
    io.write_json({
        "label_source": which, "n_modes_audited": len(clusters),
        "n_validated_axis": n_axis, "n_candidate_phenotype": n_candidate,
        "n_nuisance_linked": n_nuisance, "have_dates": bool(have_dates),
        "qeeg_available": fc is not None, "fdr_alpha": alpha,
        "primary_family": [c for c in primary_cols if fc is None or c in (fc.columns if fc is not None else [])],
        "min_patients_for_reliable_inference": int(dp.get("mode_min_patients", 5)),
        "qeeg_perms": int(dp.get("mode_qeeg_perm", dp.get("axis_qeeg_perm", 2000))),
        "note": "Two gates per geometric mode. (1) NUISANCE: Mann-Whitney mode-vs-rest on dt, "
                "magnitude and calendar time (rank-biserial effect; BH-FDR across modes). A "
                "significant hit means the bundle IS that confound -> embedding artifact, not a "
                "phenotype. (2) QEEG: pre-specified primary family, patient-clustered cluster-robust "
                "OLS (membership as regressor, magnitude+dt controlled). Multiplicity is corrected by "
                "a GLOBAL BH across ALL mode x feature tests (NOT within-mode - that under-corrects by "
                "the number of modes), and every hit must additionally pass a PATIENT-BLOCK "
                "PERMUTATION (cluster-robust SEs are anti-conservative at small n). A mode with fewer "
                "than min_patients_for_reliable_inference in-group patients cannot be a candidate - "
                "its inference is unreliable by construction and a QEEG hit there is a 'weak lead' at "
                "most. A phenotype claim needs distinct non-EMG QEEG surviving global FDR + permutation "
                "AND no nuisance link AND adequate n. Absence at small n is not proof of absence.",
        "per_mode": rows_out,
    }, config.out("mode_audit.json"))

    n_weak = sum(1 for r in rows_out if r["verdict"].startswith("weak lead"))
    print(f"[mode_audit] {len(clusters)} modes: {n_axis} axis, {n_candidate} candidate, "
          f"{n_weak} weak-lead, {n_nuisance} nuisance-linked, dates={have_dates}, "
          f"qeeg={'yes' if fc is not None else 'no'}.")
    for r in rows_out:
        print(f"[mode_audit]   c{r['cluster']} n={r['n']}/{r['n_patients']}p "
              f"dt={r['dt_effect']:+.2f}(q={r['dt_q']}) mag={r['magnitude_effect']:+.2f}(q={r['magnitude_q']}) "
              f"qeeg_glob_sig={r['qeeg_n_global_sig_non_emg']} reliable={r['qeeg_inference_reliable']} "
              f"-> {r['verdict']}")
    print(f"[mode_audit] wrote mode_audit.{{json,csv,png}} to {config.output_dir}")
    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit geometric modes: nuisance + per-mode QEEG")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
