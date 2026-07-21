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


def _test_feature(df, feature, group_col):
    groups = [g[feature].dropna().values for _, g in df.groupby(group_col)]
    groups = [g for g in groups if len(g) >= 1]
    if len(groups) < 2 or sum(len(g) for g in groups) <= len(groups):
        return {"feature": feature, "H": np.nan, "p": np.nan, "epsilon_sq": np.nan,
                "n": int(sum(len(g) for g in groups))}
    try:
        H, p = kruskal(*groups)
        n = int(sum(len(g) for g in groups))
        return {"feature": feature, "H": float(H), "p": float(p),
                "epsilon_sq": _epsilon_squared(H, n, len(groups)), "n": n}
    except Exception:
        return {"feature": feature, "H": np.nan, "p": np.nan, "epsilon_sq": np.nan, "n": 0}


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
    df = fc.merge(lab[["progression_id", which]], on="progression_id", how="inner")
    df = df.rename(columns={which: "phenotype"})

    # keep groups with enough members; label -1 as "stable"
    min_g = int(dp.get("qeeg_min_group_size", 4))
    counts = df["phenotype"].value_counts()
    keep = counts[counts >= min_g].index.tolist()
    dropped = {int(k): int(v) for k, v in counts.items() if k not in keep}
    dft = df[df["phenotype"].isin(keep)].copy()
    dft["group"] = dft["phenotype"].map(lambda c: "stable" if c == -1 else f"pheno{c}")

    fc_cols = [c for c in fc.columns if c not in ("progression_id", "patient_id")]
    primary = [c for c in ps.get("fc_confirmatory_features", []) if c in fc_cols]
    exploratory = [c for c in fc_cols if c not in primary]

    res_primary = [_test_feature(dft, f, "group") for f in primary]
    if res_primary:
        pv = np.array([r["p"] for r in res_primary], dtype=float)
        ok = ~np.isnan(pv)
        rej = np.zeros(len(pv), bool); q = np.full(len(pv), np.nan)
        if ok.any():
            rj, qq = benjamini_hochberg(pv[ok], alpha=float(ps.get("fdr_alpha", 0.05)))
            rej[ok] = rj; q[ok] = qq
        for i, r in enumerate(res_primary):
            r["q_value"] = float(q[i]) if not np.isnan(q[i]) else None
            r["fdr_significant"] = bool(rej[i]); r["family"] = "primary"
    res_expl = [{**_test_feature(dft, f, "group"), "family": "exploratory"}
                for f in exploratory]

    out = pd.DataFrame(res_primary + res_expl).sort_values(
        ["family", "p"], na_position="last")
    out.to_csv(config.out("phenotype_qeeg.csv"), index=False)

    # figure: primary features by group
    plot_feats = [r["feature"] for r in sorted(res_primary, key=lambda r: (r["p"] if not np.isnan(r["p"]) else 1))][:4]
    if plot_feats:
        groups = sorted(dft["group"].unique())
        fig, axes = plt.subplots(1, len(plot_feats), figsize=(4 * len(plot_feats), 4.5), squeeze=False)
        for ax, feat in zip(axes[0], plot_feats):
            data = [dft[dft["group"] == g][feat].dropna().values for g in groups]
            ax.boxplot(data, showfliers=False)
            for i, (g, d) in enumerate(zip(groups, data), start=1):
                ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(d)), d,
                           s=18, color=_PAL[i % len(_PAL)], alpha=0.8, zorder=3)
            ax.set_xticklabels(groups, rotation=30, fontsize=8)
            rr = next(r for r in res_primary if r["feature"] == feat)
            ax.set_title(f"{feat}\nKW p={rr['p']:.3f}, eps2={rr['epsilon_sq']:.2f}", fontsize=8)
        fig.suptitle("QEEG connectivity by directional phenotype (EXPLORATORY, unadjusted)", fontsize=10)
        fig.tight_layout()
        fig.savefig(config.out("phenotype_qeeg.png"), dpi=int(config["report"]["fig_dpi"]))
        plt.close(fig)

    n_sig = int(sum(1 for r in res_primary if r.get("fdr_significant")))
    io.write_json({"note": "EXPLORATORY, unadjusted (no clinical covariates, "
                           "duplicate patients not modeled). Confirmatory test is "
                           "module 7 with covariates once clinical data is available.",
                   "label_source": which, "groups_tested": sorted(dft["group"].unique()),
                   "dropped_small_groups": dropped, "primary": res_primary,
                   "exploratory": res_expl, "n_primary_fdr_significant": n_sig},
                  config.out("phenotype_qeeg.json"))
    print(f"[pheno_qeeg] EXPLORATORY: {len(dft)} progressions in groups "
          f"{sorted(dft['group'].unique())}; dropped small groups {dropped}")
    for r in res_primary:
        star = " *FDR" if r.get("fdr_significant") else ""
        print(f"[pheno_qeeg]   {r['feature']}: KW p={r['p']:.3f}, eps2={r['epsilon_sq']:.2f}{star}")
    print(f"[pheno_qeeg] {n_sig}/{len(res_primary)} primary FC features FDR-significant "
          f"(exploratory). Re-run with covariates (module 7) when clinical is in.")
    return config.out("phenotype_qeeg.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phenotype x QEEG (exploratory)")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
