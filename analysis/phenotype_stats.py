"""Module 7 - phenotype_stats: covariate-adjusted comparison across clusters.

Per feature, tests the cluster effect adjusting for dt, baseline_severity,
APOE4, ARIA, age, with a PATIENT random effect (statsmodels mixedlm) or
cluster-robust GEE for duplicate patients (invariant 4, enforced eagerly via
``covariate_spec``). Benjamini-Hochberg FDR is applied across the confirmatory
feature family (the pre-specified primary QEEG + FC features); everything else
is reported uncorrected and flagged exploratory.

This module now compares the **functional-connectivity** features produced by
module 6: confirmatory global/posterior alpha-band wPLI change, plus the
graph-metric summaries as exploratory. MMSE (baseline + delta) is a secondary
coarse reference.

statsmodels is imported lazily so this module (and the invariant-4 validator +
BH-FDR helper) import without it.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from .config import Config, load_config, add_arg
from . import io
from . import invariants


# ---------------------------------------------------------------------------
# Model spec (invariant 4)
# ---------------------------------------------------------------------------
def covariate_spec(config: Config) -> dict:
    ps = config["phenotype_stats"]
    covariates = list(ps["covariates"])
    has_patient_effect = bool(ps["patient_random_effect"]) or ps["model"] == "gee"
    invariants.validate_phenotype_model_spec(covariates, has_patient_effect)
    return {
        "covariates": covariates,
        "groups": "patient_id",
        "model": ps["model"],
        "fdr_alpha": float(ps["fdr_alpha"]),
    }


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR (pure numpy)
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvals, alpha: float = 0.05):
    """Return (rejected_mask, qvalues) for BH step-up FDR control."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity
    qvals = np.empty(n)
    qvals[order] = np.clip(q, 0, 1)
    rejected = qvals <= alpha
    return rejected, qvals


# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------
def select_features(feature_cols, config) -> pd.DataFrame:
    """Tag each feature column confirmatory vs exploratory."""
    ps = config["phenotype_stats"]
    confirmatory = set(ps["fc_confirmatory_features"])
    expl_pat = re.compile(ps["fc_exploratory_pattern"])
    rows = []
    for c in feature_cols:
        rows.append({"feature": c,
                     "family": "confirmatory" if c in confirmatory
                     else ("exploratory_graph" if expl_pat.search(c) else "exploratory")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-feature fit (statsmodels, lazy)
# ---------------------------------------------------------------------------
def fit_cluster_effect(df: pd.DataFrame, feature: str, spec: dict) -> dict:
    """Fit feature ~ C(cluster) + covariates with a patient random effect;
    return the omnibus cluster-effect p-value."""
    import statsmodels.formula.api as smf

    covs = " + ".join(spec["covariates"])
    formula = f"Q('{feature}') ~ C(cluster) + {covs}"
    sub = df.dropna(subset=[feature, "cluster", *spec["covariates"], spec["groups"]])
    if sub["cluster"].nunique() < 2 or len(sub) < (len(spec["covariates"]) + 4):
        return {"feature": feature, "p_cluster": np.nan, "n": int(len(sub)),
                "note": "insufficient data / <2 clusters"}
    try:
        if spec["model"] == "gee":
            import statsmodels.api as sm
            model = smf.gee(formula, groups=sub[spec["groups"]], data=sub,
                            cov_struct=sm.cov_struct.Exchangeable())
            res = model.fit()
        else:
            model = smf.mixedlm(formula, sub, groups=sub[spec["groups"]])
            res = model.fit(reml=False)
        names = list(res.model.exog_names)
        cl_idx = [i for i, nm in enumerate(names) if nm.startswith("C(cluster)")]
        R = np.zeros((len(cl_idx), len(res.params)))
        for i, j in enumerate(cl_idx):
            R[i, j] = 1.0
        wt = res.wald_test(R, scalar=True)
        p = float(np.ravel(wt.pvalue)[0])
        return {"feature": feature, "p_cluster": p, "n": int(len(sub)), "note": ""}
    except Exception as e:  # noqa: BLE001
        return {"feature": feature, "p_cluster": np.nan, "n": int(len(sub)),
                "note": f"fit failed: {type(e).__name__}"}


def compare_across_clusters(features_df, meta_df, labels_df, config) -> pd.DataFrame:
    """Merge features + covariates + cluster labels, fit each feature, apply FDR."""
    spec = covariate_spec(config)                       # invariant 4
    df = (features_df.merge(meta_df, on=["progression_id", "patient_id"], how="inner")
          .merge(labels_df, on="progression_id", how="inner"))

    feature_cols = [c for c in features_df.columns
                    if c not in ("progression_id", "patient_id")]
    fam = select_features(feature_cols, config).set_index("feature")["family"].to_dict()

    results = [fit_cluster_effect(df, f, spec) for f in feature_cols]
    res = pd.DataFrame(results)
    res["family"] = res["feature"].map(fam)

    # BH-FDR within the confirmatory family only.
    conf = res["family"].eq("confirmatory") & res["p_cluster"].notna()
    res["q_value"] = np.nan
    res["fdr_significant"] = False
    if conf.any():
        rej, q = benjamini_hochberg(res.loc[conf, "p_cluster"].to_numpy(),
                                    alpha=spec["fdr_alpha"])
        res.loc[conf, "q_value"] = q
        res.loc[conf, "fdr_significant"] = rej
    return res.sort_values(["family", "p_cluster"], na_position="last").reset_index(drop=True)


def main(config: Config) -> str:
    spec = covariate_spec(config)
    print(f"[phenotype_stats] model: feature ~ C(cluster) + "
          f"{' + '.join(spec['covariates'])} | groups={spec['groups']} "
          f"| FDR alpha={spec['fdr_alpha']}")

    meta_df = io.read_table(config.out("progressions"))[
        ["progression_id", "patient_id", "dt", "baseline_severity",
         "APOE4", "ARIA", "age", "MMSE_delta"]]
    # The covariate-adjusted model needs real covariates. If the clinical table
    # was never provided, these are all NaN -> nothing to adjust for.
    if meta_df[list(spec["covariates"])].isna().all().any():
        raise SystemExit("[phenotype_stats] required covariates are all-NaN "
                         "(no clinical table). Provide paths.clinical_csv and "
                         "re-run assemble; skipping covariate-adjusted stats.")
    features_df = io.read_table(config.out("qeeg_connectivity"))

    labels_path = config.out("labels.npz")
    import os
    if not os.path.exists(labels_path):
        raise SystemExit("[phenotype_stats] cluster labels (labels.npz) not found - "
                         "run module 4 (cluster) first, or the gate routed to "
                         "graded_score (no clusters to compare).")
    z = np.load(labels_path, allow_pickle=True)
    labels_df = pd.DataFrame({"progression_id": z["progression_id"],
                              "cluster": z["cluster"]})

    res = compare_across_clusters(features_df, meta_df, labels_df, config)
    out_path = io.write_table(res, config.out("phenotype_stats_fc"))
    n_sig = int(res["fdr_significant"].sum())
    print(f"[phenotype_stats] {len(res)} features tested; "
          f"{n_sig} confirmatory FC feature(s) FDR-significant -> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 7 - covariate-adjusted cluster stats")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
