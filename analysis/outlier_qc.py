"""Outlier QC diagnostic (non-destructive).

Ranks progressions by how extreme their delta is (robust Mahalanobis distance on
the retained PCs) and reports the descriptors that tell an *artifact* from a
*real* large-change case:
  - spread_std / spread_iqr : segment-pair disagreement within the progression
    (high spread -> the median delta is unreliable),
  - n_before_segs / n_after_segs : how many segments backed the delta,
  - dt : inter-session interval (long gaps -> legitimately larger change),
  - delta_norm, robust Mahalanobis distance, PC1/PC2.

Writes outlier_qc.csv (sorted, most extreme first) and prints the top rows. Makes
no changes to the pipeline -- it's there so you can decide keep/remove/robustify.

Run:  python -m analysis.outlier_qc --config analysis/config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import Config, load_config, add_arg
from . import io, outliers


def _segment_counts(metadata_csv: str):
    meta = pd.read_csv(metadata_csv)
    parts = meta["group_name"].str.split("_", expand=True)
    meta["patient_id"] = parts[0].astype(int)
    meta["date"] = pd.to_datetime(dict(
        year=parts[1].astype(int), month=parts[2].astype(int), day=parts[3].astype(int)))
    return meta.groupby(["patient_id", "date"]).size().to_dict()


def main(config: Config) -> str:
    seed = int(config["seed"])
    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    delta = dz["delta"].astype(float)
    df = pd.DataFrame({
        "progression_id": dz["progression_id"], "patient_id": dz["patient_id"],
        "dt": dz["dt"], "spread_std": dz["spread_std"], "spread_iqr": dz["spread_iqr"],
        "delta_norm": np.linalg.norm(delta, axis=1),
    })

    xz = np.load(config.out("X_pca.npz"), allow_pickle=True)
    X = xz["X_pca"].astype(float)
    xdf = pd.DataFrame({"progression_id": xz["progression_id"],
                        "PC1": X[:, 0], "PC2": X[:, 1] if X.shape[1] > 1 else 0.0})
    cc = config["cluster"]
    method = str(cc.get("outlier_method", "lof"))
    # Detect on the SAME space clustering uses: directional (unit-normalized PC
    # scores) when metric=cosine, else the raw PC scores. Otherwise this
    # diagnostic would flag magnitude fliers the directional clustering ignores.
    Xdet = X.copy()
    if str(cc.get("metric", "euclidean")) == "cosine":
        nrm = np.linalg.norm(Xdet, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
        Xdet = Xdet / nrm
    mask, score, cutoff = outliers.outlier_mask(
        Xdet, method=method, quantile=float(cc.get("outlier_quantile", 0.975)),
        n_neighbors=int(cc.get("outlier_n_neighbors", 20)), seed=seed)
    xdf["outlier_score"] = score
    xdf["is_outlier"] = mask
    df = df.merge(xdf, on="progression_id", how="left")

    # segment counts per session
    counts = _segment_counts(config.path("paths", "metadata_csv"))
    prog = io.read_table(config.out("progressions"))
    prog["before_date"] = pd.to_datetime(prog["before_date"]).dt.normalize()
    prog["after_date"] = pd.to_datetime(prog["after_date"]).dt.normalize()
    nb, na = {}, {}
    for _, r in prog.iterrows():
        nb[r["progression_id"]] = counts.get((int(r["patient_id"]), r["before_date"]), 0)
        na[r["progression_id"]] = counts.get((int(r["patient_id"]), r["after_date"]), 0)
    df["n_before_segs"] = df["progression_id"].map(nb)
    df["n_after_segs"] = df["progression_id"].map(na)

    df = df.sort_values("outlier_score", ascending=False).reset_index(drop=True)
    out_path = config.out("outlier_qc.csv")
    df.to_csv(out_path, index=False)

    n_out = int(df["is_outlier"].sum())
    print(f"[outlier_qc] {n_out}/{len(df)} progressions flagged as outliers "
          f"(method={method}). Top extremes:")
    cols = ["progression_id", "patient_id", "outlier_score", "delta_norm", "spread_std",
            "n_before_segs", "n_after_segs", "dt", "is_outlier"]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df[cols].head(min(10, len(df))).to_string(index=False))
    print(f"[outlier_qc] full table -> {out_path}")
    print("[outlier_qc] read: high spread_std or few segments => likely artifact; "
          "clean + long dt => possibly real large change.")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outlier QC diagnostic")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
