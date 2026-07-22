"""Is the validated bipolar AXIS QEEG-related?

The categorical phenotype x QEEG screen (phenotype_qeeg) found no artifact-safe
signature - but it discretized an over-split set of directions and lost power. Now
that phenotype_geometry has validated a single bipolar axis (pheno0 <-> pheno1,
127 deg, p=0.0015) and phenotype_compass writes its CONTINUOUS coordinate
(phenotype_axis.csv), we can ask the more powerful, more principled question:

    does a progression's QEEG CHANGE track WHERE it sits on the axis?

This is a continuous association (keeps the full gradient), not a group test, and
it is NOT circular: QEEG comes from raw EEG, independent of the LaBraM embeddings
the axis is built from. If a feature survives, it is an interpretable EEG correlate
of the axis; if not, the negative is reinforced.

Honesty controls (same spirit as phenotype_qeeg):
  * RELIABLE subset only - the axis direction of a low-magnitude delta is noise.
  * control MAGNITUDE + dt (a feature must beat the confounds, not ride them).
  * PATIENT-CLUSTERED inference - cluster-robust OLS SE by patient (statsmodels);
    a patient-block permutation p is reported alongside as a small-cluster check.
  * BH-FDR across the QEEG-change family; EMG-prone bands (gamma / beta2) flagged
    because they are the classic muscle-artifact tell.

Primary family = QEEG *change* (``*_delta``) features (a direction of change should
partner QEEG change). Baseline (``*_baseline``) features are exploratory.

Reads qeeg_connectivity.* + phenotype_axis.csv (+ deltas.npz for dt). Outputs:
  axis_qeeg.json, axis_qeeg.png, axis_qeeg.csv

Run:  python -m analysis.axis_qeeg --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from .config import Config, load_config, add_arg
from . import io
from .phenotype_stats import benjamini_hochberg

_EMG = re.compile(r"(gamma|beta2|beta_2|high_?beta)", re.I)
OI = {"blue": "#0072B2", "vermillion": "#D55E00", "gray": "#999999"}


def _residualize(y, C):
    X = np.column_stack([np.ones(len(y)), C])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _z(a):
    s = a.std()
    return (a - a.mean()) / s if s > 0 else a * 0.0


def _cluster_ols(y, axis, covars, groups):
    """Standardized cluster-robust OLS: y ~ axis + covars, SE clustered by patient.
    Returns (beta_axis, p_cluster, method). Falls back to residual-Spearman if
    statsmodels is unavailable."""
    try:
        import statsmodels.api as sm
        X = np.column_stack([_z(axis)] + [_z(c) for c in covars])
        X = sm.add_constant(X)
        m = sm.OLS(_z(y), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        return float(m.params[1]), float(m.pvalues[1]), "ols_cluster_robust"
    except Exception:
        ry = _residualize(y, np.column_stack(covars))
        rx = _residualize(axis, np.column_stack(covars))
        r, p = spearmanr(rx, ry)
        return float(r), float(p), "spearman_resid_fallback"


def _perm_pvalue(y, axis, covars, groups, n_perm, seed):
    """Patient-block permutation: break the axis<->QEEG association while keeping
    within-patient structure, via residual-Spearman. Conservative small-n check."""
    C = np.column_stack(covars)
    ry = _residualize(y, C)
    rx = _residualize(axis, C)
    obs = abs(spearmanr(rx, ry)[0])
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    # map each patient to its block of indices
    blocks = {g: np.where(groups == g)[0] for g in uniq}
    cnt = 0
    for _ in range(n_perm):
        perm_order = rng.permutation(uniq)
        rx_perm = np.empty_like(rx)
        for src, dst in zip(uniq, perm_order):
            # place src patient's axis-residuals onto dst patient's positions
            s_idx, d_idx = blocks[src], blocks[dst]
            take = rx[s_idx]
            if len(take) >= len(d_idx):
                rx_perm[d_idx] = take[:len(d_idx)]
            else:
                rx_perm[d_idx[:len(take)]] = take
                rx_perm[d_idx[len(take):]] = rx[d_idx[len(take):]]
        if abs(spearmanr(rx_perm, ry)[0]) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def _test_feature(df, feat, axis_col, covars, patient_col, n_perm, seed):
    need = [feat, axis_col, patient_col] + covars
    sub = df.dropna(subset=need)
    if len(sub) < 8 or sub[feat].std() == 0:
        return {"feature": feat, "n": len(sub), "beta": np.nan, "p_cluster": np.nan,
                "spearman_r": np.nan, "p_perm": np.nan, "emg_prone": bool(_EMG.search(feat))}
    y = sub[feat].to_numpy(float)
    axis = sub[axis_col].to_numpy(float)
    covs = [sub[c].to_numpy(float) for c in covars]
    groups = sub[patient_col].to_numpy()
    beta, pcl, method = _cluster_ols(y, axis, covs, groups)
    rx = _residualize(axis, np.column_stack(covs)); ry = _residualize(y, np.column_stack(covs))
    r_sp = float(spearmanr(rx, ry)[0])
    p_perm = _perm_pvalue(y, axis, covs, groups, n_perm, seed)
    return {"feature": feat, "n": int(len(sub)), "beta": beta, "p_cluster": pcl,
            "method": method, "spearman_r": r_sp, "p_perm": float(p_perm),
            "emg_prone": bool(_EMG.search(feat))}


def main(config: Config):
    ps = config["phenotype_stats"]
    dp = config.get("directional_phenotype", {})
    n_perm = int(dp.get("axis_qeeg_perm", 2000))
    seed = int(config["seed"])

    fc_path = config.out("qeeg_connectivity")
    if not (os.path.exists(fc_path + ".parquet") or os.path.exists(fc_path + ".csv")):
        raise SystemExit("[axis_qeeg] qeeg_connectivity not found - run module 6 (qeeg) first.")
    ax_path = config.out("phenotype_axis.csv")
    if not os.path.exists(ax_path):
        raise SystemExit("[axis_qeeg] phenotype_axis.csv not found - run "
                         "`python -m analysis.phenotype_compass` first (it writes the axis).")

    fc = io.read_table(fc_path)
    fc["progression_id"] = fc["progression_id"].astype(str)
    axis = pd.read_csv(ax_path); axis["progression_id"] = axis["progression_id"].astype(str)
    df = fc.merge(axis[["progression_id", "axis_coord", "magnitude", "is_reliable"]],
                  on="progression_id", how="inner")
    # dt from deltas.npz
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    dtmap = dict(zip([str(x) for x in dz["progression_id"]], dz["dt"]))
    df["dt"] = df["progression_id"].astype(str).map(dtmap)
    if "patient_id" not in df.columns:
        df["patient_id"] = df["progression_id"].str.split("__").str[0]

    # RELIABLE subset - axis direction is meaningful only above the magnitude cut
    dft = df[df["is_reliable"].astype(bool)].copy()
    covars = ["magnitude", "dt"]

    fc_cols = [c for c in fc.columns if c not in ("progression_id", "patient_id")]
    delta_feats = [c for c in fc_cols if c.endswith("_delta")]
    base_feats = [c for c in fc_cols if c.endswith("_baseline")]

    # PRE-SPECIFIED PRIMARY FAMILY (Phase 3). FDR is applied WITHIN this small set of
    # established AD qEEG markers only; everything else is reported secondary and
    # explicitly uncorrected. Target claim: "the axis is not explained by pre-specified
    # markers" - not "nothing survived out of ~100 tests". Config-overridable.
    PRIMARY_DEFAULT = ["median_freq_global", "rel_alpha_posterior", "rel_theta_global",
                       "slowing_ratio_posterior", "alpha_cog_global",
                       "aperiodic_exponent_global", "wpli_alpha_global",
                       "wpli_alpha_posterior", "graph_wpli_alpha_global_efficiency"]
    prim_base = ps.get("axis_primary_features", PRIMARY_DEFAULT)
    primary_cols = {f"{b}_delta" for b in prim_base}
    present_primary = [c for c in primary_cols if c in delta_feats]
    missing_primary = sorted(primary_cols - set(delta_feats))

    res_delta = [_test_feature(dft, f, "axis_coord", covars, "patient_id", n_perm, seed)
                 for f in delta_feats]
    for r in res_delta:
        r["family"] = "primary" if r["feature"] in primary_cols else "secondary"

    # BH-FDR WITHIN the primary family only
    prim = [r for r in res_delta if r["family"] == "primary"]
    pv = np.array([r["p_cluster"] for r in prim], float)
    ok = ~np.isnan(pv); rej = np.zeros(len(pv), bool); q = np.full(len(pv), np.nan)
    if ok.any():
        rj, qq = benjamini_hochberg(pv[ok], alpha=float(ps.get("fdr_alpha", 0.05)))
        rej[ok] = rj; q[ok] = qq
    for i, r in enumerate(prim):
        r["q_value"] = float(q[i]) if not np.isnan(q[i]) else None
        r["fdr_significant"] = bool(rej[i])
    for r in res_delta:                                 # secondary: uncorrected, no q
        if r["family"] == "secondary":
            r["q_value"] = None; r["fdr_significant"] = False
    res_base = [{**_test_feature(dft, f, "axis_coord", covars, "patient_id", n_perm, seed),
                 "family": "baseline"} for f in base_feats]

    def _pc(r):
        v = r.get("p_cluster"); return v if v is not None and not np.isnan(v) else 1.0

    n_sig = int(sum(r.get("fdr_significant") for r in res_delta))
    # survivors that are NOT EMG-prone and also pass the permutation check
    clean = sorted([r for r in res_delta if r.get("fdr_significant")
                    and not r["emg_prone"] and r["p_perm"] < 0.05], key=_pc)
    # uncorrected hits (context)
    uncorr = sorted([r for r in res_delta if _pc(r) < 0.05], key=_pc)

    out = pd.DataFrame(res_delta + res_base).sort_values("p_cluster", na_position="last")
    out.to_csv(config.out("axis_qeeg.csv"), index=False)

    # figure: axis vs the 4 most-associated change features
    top = [r["feature"] for r in sorted(res_delta, key=_pc)][:4]
    fig, axes = plt.subplots(1, len(top), figsize=(4.2 * len(top), 4.4), squeeze=False)
    for ax, feat in zip(axes[0], top):
        sub = dft.dropna(subset=[feat, "axis_coord"])
        x = sub["axis_coord"].to_numpy(float); y = sub[feat].to_numpy(float)
        col = np.where(x >= 0, OI["blue"], OI["vermillion"])
        ax.scatter(x, y, c=col, s=34, edgecolor="white", lw=0.5, zorder=3)
        if len(x) >= 2 and np.ptp(x) > 0:
            b = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(b, xs), color="#444", lw=1.4, zorder=2)
        rr = next(r for r in res_delta if r["feature"] == feat)
        tag = " [EMG?]" if rr["emg_prone"] else ""
        ax.axvline(0, color="#ccd2d8", lw=0.8, zorder=1)
        ax.set_title(f"{feat}{tag}\nbeta={rr['beta']:.2f} p_clu={_pc(rr):.3f} "
                     f"q={'' if rr['q_value'] is None else format(rr['q_value'],'.2f')}", fontsize=8)
        ax.set_xlabel("axis position (PC1)"); ax.tick_params(labelsize=7)
    fig.suptitle("QEEG change vs bipolar-axis position (reliable subset, magnitude/dt-adjusted, "
                 "patient-clustered)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.out("axis_qeeg.png"), dpi=int(config["report"]["fig_dpi"]))
    plt.close(fig)

    io.write_json({
        "note": "Continuous axis (PC1) vs QEEG. RELIABLE subset only; magnitude+dt "
                "controlled; patient-clustered (cluster-robust OLS SE) + patient-block "
                "permutation companion. BH-FDR is applied WITHIN a PRE-SPECIFIED PRIMARY "
                "FAMILY of established AD qEEG markers; all other features are SECONDARY "
                "and uncorrected. EMG-prone (gamma/beta2) flagged. NOT circular (QEEG "
                "independent of embeddings). Confirmatory clinical test still pending.",
        "n_reliable_tested": int(len(dft)),
        "n_patients": int(dft["patient_id"].nunique()),
        "covariates": covars, "fdr_alpha": float(ps.get("fdr_alpha", 0.05)),
        "primary_family": sorted(present_primary),
        "primary_family_missing_from_table": missing_primary,
        "n_primary_tested": len(prim), "n_secondary_uncorrected": len(delta_feats) - len(prim),
        "n_primary_fdr_significant": n_sig,
        "n_clean_survivors_non_emg_perm_ok": len(clean),
        "clean_survivors": [{"feature": r["feature"], "beta": r["beta"],
                             "p_cluster": r["p_cluster"], "q_value": r["q_value"],
                             "p_perm": r["p_perm"], "spearman_r": r["spearman_r"]}
                            for r in clean],
        "primary_results": [r for r in res_delta if r["family"] == "primary"],
        "secondary_uncorrected_hits": [{"feature": r["feature"], "beta": r["beta"],
                              "p_cluster": _pc(r), "emg_prone": r["emg_prone"],
                              "p_perm": r["p_perm"]} for r in uncorr if r["family"] == "secondary"][:15],
        "change_features": res_delta, "baseline_features": res_base,
    }, config.out("axis_qeeg.json"))

    if missing_primary:
        print(f"[axis_qeeg] NOTE: {len(missing_primary)} primary features not in table "
              f"(re-run qeeg after spectral.py update?): {missing_primary}")
    print(f"[axis_qeeg] {len(dft)} reliable progressions / {dft['patient_id'].nunique()} patients; "
          f"{len(prim)} PRIMARY (FDR) + {len(delta_feats)-len(prim)} secondary (uncorrected).")
    print(f"[axis_qeeg] {n_sig} primary FDR-significant; {len(clean)} clean (non-EMG & permutation-ok).")
    for r in uncorr[:8]:
        star = " *FDR" if r.get("fdr_significant") else ""
        emg = " [EMG?]" if r["emg_prone"] else ""
        print(f"[axis_qeeg]   {r['feature']}{emg}: beta={r['beta']:.2f} "
              f"p_clu={_pc(r):.3f} p_perm={r['p_perm']:.3f}{star}")
    if not clean:
        print("[axis_qeeg] -> no artifact-safe QEEG correlate of the axis survives correction.")
    else:
        print("[axis_qeeg] -> candidate EEG correlate(s) of the axis: "
              + ", ".join(r["feature"] for r in clean))
    return config.out("axis_qeeg.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axis vs QEEG association")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
