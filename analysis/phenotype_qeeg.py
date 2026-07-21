"""Phenotype x QEEG - do the directional phenotypes differ in connectivity?

EXPLORATORY / UNADJUSTED. This compares the functional-connectivity features
(module 6) across the directional phenotype groups WITHOUT clinical covariate
adjustment, because the clinical table is not in yet. It is a descriptive
screen, not the confirmatory test - the covariate-adjusted mixedlm
(phenotype_stats / module 7) is what you run once clinical data arrives.

Nonparametric Kruskal-Wallis across groups (robust at small n) + epsilon-squared
effect size; BH-FDR across the primary FC features; the rest flagged exploratory.
Tiny groups (< min_group_size) are dropped from testing and reported separately.
Duplicate patients are NOT modeled here (flagged) - another reason it is a screen.

Reads qeeg_connectivity.* + directional_phenotype_labels.csv. Outputs:
  phenotype_qeeg.json, phenotype_qeeg.png, phenotype_qeeg.csv

Run:  python -m analysis.phenotype_qeeg --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import kruskal

from .config import Config, load_config, add_arg
from . import io
from .phenotype_stats import benjamini_hochberg

_PAL = ["#BBBBBB", "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]


def _epsilon_squared(H, n, k):
    return float((H - k + 1) / (n - k)) if n > k else np.nan


def _kw(values, groups):
    """Kruskal-Wallis on `values` split by `groups` (array of labels)."""
    parts = [values[groups == g] for g in np.unique(groups)]
    parts = [p for p in parts if len(p) >= 1]
    n = int(sum(len(p) for p in parts))
    if len(parts) < 2 or n <= len(parts) or min(len(p) for p in parts) < 1:
        return np.nan, np.nan, np.nan, n
    try:
        H, p = kruskal(*parts)
        return float(H), float(p), _epsilon_squared(H, n, len(parts)), n
    except Exception:
        return np.nan, np.nan, np.nan, n


def _residualize(y, C):
    """Residuals of y after OLS on covariates C (magnitude, dt), with intercept."""
    X = np.column_stack([np.ones(len(y)), C])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _test_feature(df, feature, covars):
    """Return unadjusted + magnitude/dt-adjusted KW, all-groups + phenotype-only.

    p / p_adj             : all groups (incl stable), raw / covariate-residualized
    p_pheno / p_adj_pheno : dropping stable -> do the phenotypes differ from EACH
                            OTHER (the confound-controlled, decisive test)
    """
    need = [feature, "group"] + covars
    sub = df.dropna(subset=need)
    if len(sub) < 6:
        return {"feature": feature, "p": np.nan, "epsilon_sq": np.nan,
                "p_pheno": np.nan, "p_adj": np.nan, "p_adj_pheno": np.nan, "n": len(sub)}
    y = sub[feature].to_numpy(float)
    g = sub["group"].to_numpy()
    C = sub[covars].to_numpy(float)
    resid = _residualize(y, C)
    pheno = g != "stable"

    H, p, eps, n = _kw(y, g)
    _, p_pheno, eps_pheno, _ = _kw(y[pheno], g[pheno])
    _, p_adj, _, _ = _kw(resid, g)
    _, p_adj_pheno, eps_adj_pheno, _ = _kw(resid[pheno], g[pheno])
    return {"feature": feature, "H": H, "p": p, "epsilon_sq": eps, "n": n,
            "p_pheno": p_pheno, "epsilon_sq_pheno": eps_pheno,
            "p_adj": p_adj, "p_adj_pheno": p_adj_pheno,
            "epsilon_sq_adj_pheno": eps_adj_pheno}


def main(config: Config):
    dp = config.get("directional_phenotype", {})
    ps = config["phenotype_stats"]
    fc_path = config.out("qeeg_connectivity")
    if not (os.path.exists(fc_path + ".parquet") or os.path.exists(fc_path + ".csv")):
        raise SystemExit("[pheno_qeeg] qeeg_connectivity not found - run module 6 (qeeg) "
                         "first (downstream: [clustering] runs it, or "
                         "`python -m analysis.qeeg --config ...`).")
    fc = io.read_table(fc_path)
    lab = pd.read_csv(config.out("directional_phenotype_labels.csv"))
    which = dp.get("geometry_label", "spherical_label")
    df = fc.merge(lab[["progression_id", which, "magnitude"]], on="progression_id", how="inner")
    df = df.rename(columns={which: "phenotype"})
    # dt covariate from deltas.npz
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    dtmap = dict(zip([str(x) for x in dz["progression_id"]], dz["dt"]))
    df["dt"] = df["progression_id"].astype(str).map(dtmap)

    # keep groups with enough members; label -1 as "stable"
    min_g = int(dp.get("qeeg_min_group_size", 4))
    counts = df["phenotype"].value_counts()
    keep = counts[counts >= min_g].index.tolist()
    dropped = {int(k): int(v) for k, v in counts.items() if k not in keep}
    dft = df[df["phenotype"].isin(keep)].copy()
    dft["group"] = dft["phenotype"].map(lambda c: "stable" if c == -1 else f"pheno{c}")

    covars = ["magnitude", "dt"]
    fc_cols = [c for c in fc.columns if c not in ("progression_id", "patient_id")]
    primary = [c for c in ps.get("fc_confirmatory_features", []) if c in fc_cols]
    exploratory = [c for c in fc_cols if c not in primary]

    # FDR is applied on the DECISIVE test: magnitude/dt-adjusted, phenotype-only.
    res_primary = [_test_feature(dft, f, covars) for f in primary]
    if res_primary:
        pv = np.array([r["p_adj_pheno"] for r in res_primary], dtype=float)
        ok = ~np.isnan(pv)
        rej = np.zeros(len(pv), bool); q = np.full(len(pv), np.nan)
        if ok.any():
            rj, qq = benjamini_hochberg(pv[ok], alpha=float(ps.get("fdr_alpha", 0.05)))
            rej[ok] = rj; q[ok] = qq
        for i, r in enumerate(res_primary):
            r["q_value_adj_pheno"] = float(q[i]) if not np.isnan(q[i]) else None
            r["fdr_significant"] = bool(rej[i]); r["family"] = "primary"
    res_expl = [{**_test_feature(dft, f, covars), "family": "exploratory"}
                for f in exploratory]

    all_res = res_primary + res_expl
    out = pd.DataFrame(all_res).sort_values(["family", "p_adj_pheno"], na_position="last")
    out.to_csv(config.out("phenotype_qeeg.csv"), index=False)

    def _pa(r):
        v = r.get("p_adj_pheno"); return v if v is not None and not np.isnan(v) else 1.0

    # which exploratory features survive the DECISIVE (adjusted, phenotype-only) test
    expl_hits = sorted([r for r in res_expl if _pa(r) < 0.05], key=_pa)

    # figure: the features most separating phenotypes AFTER adjustment
    plot_feats = [r["feature"] for r in sorted(all_res, key=_pa)][:4]
    groups = sorted(dft["group"].unique())
    fig, axes = plt.subplots(1, len(plot_feats), figsize=(4 * len(plot_feats), 4.5), squeeze=False)
    for ax, feat in zip(axes[0], plot_feats):
        data = [dft[dft["group"] == g][feat].dropna().values for g in groups]
        ax.boxplot(data, showfliers=False)
        for i, (g, d) in enumerate(zip(groups, data), start=1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(d)), d,
                       s=18, color=_PAL[i % len(_PAL)], alpha=0.8, zorder=3)
        ax.set_xticklabels(groups, rotation=30, fontsize=8)
        rr = next(r for r in all_res if r["feature"] == feat)
        ax.set_title(f"{feat}\nraw p={rr['p']:.3f} | adj+phenoOnly p={_pa(rr):.3f}", fontsize=8)
    fig.suptitle("QEEG by phenotype - raw vs magnitude/dt-adjusted, stable excluded (EXPLORATORY)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(config.out("phenotype_qeeg.png"), dpi=int(config["report"]["fig_dpi"]))
    plt.close(fig)

    n_sig = int(sum(1 for r in res_primary if r.get("fdr_significant")))
    io.write_json({"note": "EXPLORATORY, no clinical covariates. Columns: p/p_adj = "
                           "all groups incl stable, raw / magnitude+dt-residualized; "
                           "p_pheno/p_adj_pheno = stable EXCLUDED (do phenotypes differ "
                           "from each other?). p_adj_pheno is the confound-controlled "
                           "decisive test. Duplicate patients still not modeled; "
                           "module 7 with clinical covariates is the confirmatory test.",
                   "covariates_adjusted": covars,
                   "label_source": which, "groups_tested": sorted(dft["group"].unique()),
                   "dropped_small_groups": dropped, "primary": res_primary,
                   "exploratory": res_expl, "n_primary_fdr_significant": n_sig,
                   "exploratory_surviving_adjusted_phenotype_only":
                       [{"feature": r["feature"], "p_adj_pheno": _pa(r),
                         "epsilon_sq_adj_pheno": r.get("epsilon_sq_adj_pheno")}
                        for r in expl_hits]},
                  config.out("phenotype_qeeg.json"))

    print(f"[pheno_qeeg] {len(dft)} progressions, groups {sorted(dft['group'].unique())} "
          f"(dropped {dropped}); covariates adjusted: {covars}")
    print("[pheno_qeeg] PRIMARY (p=all groups raw | p_pheno=stable excluded | "
          "p_adj_pheno=adjusted+stable excluded):")
    for r in res_primary:
        star = " *FDR" if r.get("fdr_significant") else ""
        print(f"[pheno_qeeg]   {r['feature']}: p={r['p']:.3f} | p_pheno={r['p_pheno']:.3f} "
              f"| p_adj_pheno={_pa(r):.3f}{star}")
    print(f"[pheno_qeeg] {n_sig}/{len(res_primary)} primary survive adjusted+phenotype-only FDR.")
    print(f"[pheno_qeeg] {len(expl_hits)} exploratory features survive adjusted+phenotype-only "
          f"(p<0.05, uncorrected): " + ", ".join(f"{r['feature']}({_pa(r):.3f})" for r in expl_hits[:8]))
    print(f"[pheno_qeeg] {n_sig}/{len(res_primary)} primary FC features FDR-significant "
          f"(exploratory). Re-run with covariates (module 7) when clinical is in.")
    return config.out("phenotype_qeeg.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phenotype x QEEG (exploratory)")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
