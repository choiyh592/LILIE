"""Module 8 - report: letter outputs.

Assembles whatever the run produced, degrading gracefully when a piece is
absent (no clinical table -> clinical columns are dropped; no QEEG -> the FC
panel/columns are skipped).

Figure panels: (a) PCA scree (gate evidence); (b) 2-D PCA scatter by cluster;
(c) stability summary (clusterwise Jaccard); (d) QEEG gradient across clusters
(only if module 6 ran). Table: per-cluster n, dt, and any available clinical +
primary QEEG summaries.

Output (paths.output_dir):
  report_figure.png, report_table.csv, report.md

Run:  python -m analysis.report --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, load_config, add_arg
from . import io

# colorblind-safe qualitative palette
_PAL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]


def _load(config):
    L = np.load(config.out("labels.npz"), allow_pickle=True)
    labels = pd.DataFrame({"progression_id": L["progression_id"],
                           "patient_id": L["patient_id"], "cluster": L["cluster"].astype(int)})
    k = int(L["k"])
    X = np.load(config.out("X_pca.npz"), allow_pickle=True)
    prog = io.read_table(config.out("progressions"))
    stability = io.read_json(config.out("stability.json")) if os.path.exists(config.out("stability.json")) else None
    qeeg = None
    for ext in (".parquet", ".csv"):
        if os.path.exists(config.out("qeeg_connectivity") + ext):
            qeeg = io.read_table(config.out("qeeg_connectivity"))
            break
    return labels, k, X, prog, stability, qeeg


def _cluster_colors(k):
    return [_PAL[i % len(_PAL)] for i in range(k)]


def main(config: Config) -> str:
    labels, k, X, prog, stability, qeeg = _load(config)
    colors = _cluster_colors(k)
    evr = X["explained_variance_ratio"]
    scores = X["X_pca"]
    order = [pid for pid in X["progression_id"]]
    lab_map = dict(zip(labels["progression_id"], labels["cluster"]))
    clab = np.array([lab_map[p] for p in order])

    df = prog.merge(labels[["progression_id", "cluster"]], on="progression_id", how="inner")

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) scree
    ax = axes[0, 0]
    ax.plot(np.arange(1, len(evr) + 1), evr, "o-", color="#3b6ea5")
    ax.set_title("(a) PCA scree (gate evidence)")
    ax.set_xlabel("component"); ax.set_ylabel("explained var. ratio")

    # (b) PC1 vs PC2 by cluster
    ax = axes[0, 1]
    for c in range(k):
        m = clab == c
        pc2 = scores[m, 1] if scores.shape[1] > 1 else np.zeros(m.sum())
        ax.scatter(scores[m, 0], pc2, s=28, color=colors[c], label=f"cluster {c} (n={int(m.sum())})",
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("(b) PCA scatter by cluster")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False, fontsize=8)

    # (c) stability
    ax = axes[1, 0]
    if stability:
        jac = stability["jaccard_per_cluster"]
        ks = sorted(int(i) for i in jac)
        vals = [jac[str(i)] if str(i) in jac else jac.get(i, np.nan) for i in ks]
        bars = ax.bar([str(i) for i in ks], vals, color=[colors[i] for i in ks])
        ax.axhline(stability["jaccard_flag_below"], ls="--", color="#c0603a",
                   label=f"flag < {stability['jaccard_flag_below']}")
        ax.set_ylim(0, 1)
        ax.set_title(f"(c) stability: clusterwise Jaccard\nARI={stability['ari_mean']:.2f}")
        ax.set_xlabel("cluster"); ax.set_ylabel("mean Jaccard"); ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "stability not run", ha="center", va="center"); ax.axis("off")

    # (d) QEEG gradient (primary FC feature) across clusters
    ax = axes[1, 1]
    fc_col = None
    if qeeg is not None:
        for cand in config["phenotype_stats"]["fc_confirmatory_features"]:
            if cand in qeeg.columns:
                fc_col = cand; break
    if fc_col is not None:
        qd = qeeg[["progression_id", fc_col]].merge(labels[["progression_id", "cluster"]], on="progression_id")
        means = qd.groupby("cluster")[fc_col].mean()
        sems = qd.groupby("cluster")[fc_col].sem()
        ax.bar([str(c) for c in means.index], means.values,
               yerr=sems.values, color=[colors[int(c)] for c in means.index], capsize=4)
        ax.set_title(f"(d) QEEG gradient: {fc_col}")
        ax.set_xlabel("cluster"); ax.set_ylabel(fc_col)
    else:
        ax.text(0.5, 0.5, "QEEG/FC not available", ha="center", va="center"); ax.axis("off")

    fig.tight_layout()
    fig.savefig(config.out("report_figure.png"), dpi=int(config["report"]["fig_dpi"]))
    plt.close(fig)

    # --- per-cluster table (drop all-NaN clinical columns) ---
    agg = {"n": ("progression_id", "count"), "dt_mean": ("dt", "mean"), "dt_sd": ("dt", "std")}
    for col in ["baseline_severity", "age", "MMSE_delta"]:
        if col in df and df[col].notna().any():
            agg[f"{col}_mean"] = (col, "mean")
    for col in ["APOE4", "ARIA"]:
        if col in df and df[col].notna().any():
            agg[f"{col}_pct"] = (col, lambda s: 100.0 * np.nanmean((s.values > 0).astype(float)))
    table = df.groupby("cluster").agg(**agg)
    if qeeg is not None and fc_col is not None:
        qd = qeeg[["progression_id", fc_col]].merge(labels[["progression_id", "cluster"]], on="progression_id")
        table[f"{fc_col}_mean"] = qd.groupby("cluster")[fc_col].mean()
    table.to_csv(config.out("report_table.csv"))

    # --- markdown summary ---
    gate = io.read_json(config.out("gate.json")) if os.path.exists(config.out("gate.json")) else {}
    lines = ["# EEG Trajectory Phenotype - run summary", "",
             f"- Gate: **{gate.get('route','?').upper()}** - {gate.get('rationale','')}",
             f"- k = {k}; cluster sizes: {df.groupby('cluster').size().to_dict()}"]
    if stability:
        lines.append(f"- Stability: ARI={stability['ari_mean']:.2f} {stability['ari_ci95']}, "
                     f"flagged clusters (Jaccard<{stability['jaccard_flag_below']}): "
                     f"{stability['flagged_clusters'] or 'none'}")
    if not df.get("baseline_severity", pd.Series([np.nan])).notna().any():
        lines.append("- Clinical covariates absent -> module 7 (covariate-adjusted "
                     "phenotype stats) skipped; add the clinical table to enable it.")
    if qeeg is None:
        lines.append("- QEEG/FC not computed (no raw EEG) -> FC panels/columns omitted.")
    lines += ["", "Files: report_figure.png, report_table.csv, scree.png, gate.json, "
              "labels.npz, stability.json"]
    with open(config.out("report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"[report] wrote report_figure.png, report_table.csv, report.md to {config.output_dir}")
    return config.out("report_figure.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 8 - letter outputs")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
