"""Module 1 - assemble: progressions + metadata.

Reuses ``create_data_splits`` (repo) to build consecutive-session progressions
AND the subject-wise fold assignment in one pass -- that function already sorts
each patient's sessions by date and pairs t_i -> t_{i+1} via shift(-1), keeping
BOTH progressions for a 3-session patient and dropping only the final unpaired
session. We then attach clinical metadata, derive ``dt`` and
``baseline_severity``, and emit a patient -> group map so a patient's multiple
progressions stay in one CV fold everywhere (invariant 1).

Outputs (in paths.output_dir):
  progressions.parquet|csv  - one row per progression with covariates + fold
  patient_group_map.csv     - patient_id -> fold (disjoint; invariant 1)

Run:  python -m analysis.assemble --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .config import Config, load_config, add_arg
from . import io
from . import invariants


def _ensure_repo_on_path(config: Config) -> None:
    repo_root = config.path("repo_root")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _fold_date(row, which: str) -> pd.Timestamp:
    y, m, d = (int(row[f"Year_{which}"]), int(row[f"Month_{which}"]), int(row[f"Day_{which}"]))
    return pd.Timestamp(year=y, month=m, day=d)


def _load_fold_progressions(splits_dir: str, num_folds: int) -> pd.DataFrame:
    """Read longitudinal_pairs_fold_{i}.csv (1-indexed) and tag each with its fold."""
    frames = []
    for i in range(1, num_folds + 1):
        p = os.path.join(splits_dir, f"longitudinal_pairs_fold_{i}.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected fold file missing: {p}")
        df = pd.read_csv(p)
        df["fold"] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_clinical(config: Config) -> pd.DataFrame | None:
    c = config["clinical"]
    path = config.path("paths", "clinical_csv")
    if not os.path.exists(path):
        print(f"[assemble] WARNING: clinical_csv not found ({path}); "
              f"covariates will be NaN. Downstream stats (module 7) require it.")
        return None
    df = pd.read_csv(path)
    rename = {
        c["patient_id_col"]: "patient_id",
        c["session_date_col"]: "session_date",
        c["mmse_col"]: "MMSE",
        c["age_col"]: "age",
        c["apoe4_col"]: "APOE4",
        c["aria_col"]: "ARIA",
    }
    df = df.rename(columns=rename)
    df["patient_id"] = df["patient_id"].astype(int)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.normalize()
    return df[["patient_id", "session_date", "MMSE", "age", "APOE4", "ARIA"]]


def _session_lookup(clinical: pd.DataFrame, cols):
    return clinical.set_index(["patient_id", "session_date"])[cols].to_dict("index")


def main(config: Config) -> str:
    _ensure_repo_on_path(config)
    from dataset.dataset_creation.create_data_splits import create_data_splits

    splits_dir = config.path("paths", "splits_dir")
    metadata_csv = config.path("paths", "metadata_csv")
    num_folds = int(config["assemble"]["num_folds"])
    seed = int(config["seed"])
    os.makedirs(splits_dir, exist_ok=True)

    # 1. Reuse create_data_splits: builds progressions + subject-wise folds.
    print(f"[assemble] create_data_splits(metadata={metadata_csv}, "
          f"num_folds={num_folds}, seed={seed})")
    create_data_splits(metadata_csv, splits_dir, num_splits=num_folds, seed=seed)

    prog = _load_fold_progressions(splits_dir, num_folds)
    prog["patient_id"] = prog["ID"].astype(int)
    prog["before_date"] = prog.apply(lambda r: _fold_date(r, "Before"), axis=1)
    prog["after_date"] = prog.apply(lambda r: _fold_date(r, "After"), axis=1)
    prog["progression_id"] = (
        prog["patient_id"].astype(str)
        + "__" + prog["before_date"].dt.strftime("%Y%m%d")
        + "__" + prog["after_date"].dt.strftime("%Y%m%d")
    )
    if prog["progression_id"].duplicated().any():
        dups = prog.loc[prog["progression_id"].duplicated(), "progression_id"].tolist()
        raise invariants.InvariantError(f"Duplicate progression_id(s): {dups[:5]}")

    # dt (inter-session interval, days) + orientation sanity (invariant 2).
    prog["dt"] = (prog["after_date"] - prog["before_date"]).dt.days
    invariants.assert_earlier_to_later(prog["before_date"], prog["after_date"])

    # 2. Clinical covariates.
    clinical = _load_clinical(config)
    for col in ["MMSE_before", "MMSE_after", "MMSE_delta", "age",
                "APOE4", "ARIA", "baseline_severity"]:
        prog[col] = np.nan
    if clinical is not None:
        mmse = _session_lookup(clinical, ["MMSE"])
        allc = _session_lookup(clinical, ["MMSE", "age", "APOE4", "ARIA"])
        # patient-level fallback for constant covariates
        pat_level = (clinical.sort_values("session_date")
                     .groupby("patient_id")[["APOE4", "ARIA"]].first().to_dict("index"))
        first_session = (clinical.sort_values("session_date")
                         .groupby("patient_id")["session_date"].first().to_dict())
        mmse_by_session = clinical.set_index(["patient_id", "session_date"])["MMSE"].to_dict()

        baseline_from = config["clinical"]["baseline_severity_from"]
        for i, r in prog.iterrows():
            pid, bd, ad = r["patient_id"], r["before_date"], r["after_date"]
            mb = allc.get((pid, bd), {})
            prog.at[i, "MMSE_before"] = mmse.get((pid, bd), {}).get("MMSE", np.nan)
            prog.at[i, "MMSE_after"] = mmse.get((pid, ad), {}).get("MMSE", np.nan)
            prog.at[i, "age"] = mb.get("age", np.nan)
            prog.at[i, "APOE4"] = mb.get("APOE4", pat_level.get(pid, {}).get("APOE4", np.nan))
            prog.at[i, "ARIA"] = mb.get("ARIA", pat_level.get(pid, {}).get("ARIA", np.nan))
            if baseline_from == "patient_baseline":
                prog.at[i, "baseline_severity"] = mmse_by_session.get(
                    (pid, first_session.get(pid)), np.nan)
            else:  # "before"
                prog.at[i, "baseline_severity"] = prog.at[i, "MMSE_before"]
        prog["MMSE_delta"] = prog["MMSE_after"] - prog["MMSE_before"]

    # 3. patient -> group map (invariant 1: disjoint folds).
    invariants.assert_disjoint_groups(prog, "patient_id", "fold")
    group_map = (prog[["patient_id", "fold"]].drop_duplicates()
                 .sort_values("patient_id").reset_index(drop=True))
    map_path = config.out("patient_group_map.csv")
    group_map.to_csv(map_path, index=False)

    # Log multi-progression patients.
    per_patient = prog.groupby("patient_id").size()
    n_multi = int((per_patient > 1).sum())
    print(f"[assemble] {len(prog)} progressions from {prog['patient_id'].nunique()} "
          f"patients across {num_folds} folds.")
    print(f"[assemble] {n_multi} patient(s) contribute >1 progression "
          f"(kept; confined to one fold each).")

    cols = ["progression_id", "patient_id", "fold", "before_date", "after_date",
            "dt", "MMSE_before", "MMSE_after", "MMSE_delta", "age", "APOE4",
            "ARIA", "baseline_severity"]
    out = prog[cols].copy()
    out_path = io.write_table(out, config.out("progressions"))
    print(f"[assemble] wrote {out_path}")
    print(f"[assemble] wrote {map_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1 - assemble progressions + metadata")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
