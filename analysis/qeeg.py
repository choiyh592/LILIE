"""Module 6 - qeeg: QEEG features from preprocessed raw EEG.

IMPLEMENTED: functional connectivity (VC-robust wPLI / imaginary coherence) plus
graph-theoretic summaries, computed per session with an identical pipeline, then
expressed as BOTH a baseline value and a within-progression delta, keyed by
progression_id (invariant 3).

SCAFFOLD (marked TODO): the spectral-power / PAF / slowing-ratio / entropy
features from the plan's primary+exploratory list. This module currently
delivers the connectivity family; the power features are added in the same
per-session loop.

Feature naming (per method in {wpli, imcoh}, band in connectivity_bands):
  <method>_<band>_global      - mean connectivity over all channel pairs
  <method>_<band>_posterior   - mean connectivity within the posterior subset
  graph_<method>_<band>_<m>   - graph metric m (exploratory)
Each appears as ``<name>_baseline`` and ``<name>_delta`` per progression.

Output (paths.output_dir): qeeg_connectivity.parquet|csv keyed by progression_id.

Run:  python -m analysis.qeeg --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .config import Config, load_config, add_arg
from . import io
from . import invariants
from . import connectivity as fc


# ---------------------------------------------------------------------------
# Per-session feature computation (pure: numpy/scipy only, no IO)
# ---------------------------------------------------------------------------
def session_features(eeg: np.ndarray, fs: float, config: Config) -> dict:
    """Connectivity + graph features for one session's ``[C, T]`` EEG."""
    q = config["qeeg"]
    bands = q["bands"]
    channels = list(q["channels"])
    if eeg.shape[0] != len(channels):
        raise ValueError(f"EEG has {eeg.shape[0]} channels but config lists "
                         f"{len(channels)}")
    post_idx = [channels.index(ch) for ch in q["posterior_channels"] if ch in channels]

    epochs = fc.make_epochs(eeg, fs, float(q["epoch_len_s"]), float(q["epoch_overlap"]))
    feats: dict = {}
    for method in q["connectivity_methods"]:
        for band_name in q["connectivity_bands"]:
            W = fc.spectral_connectivity(epochs, fs, bands[band_name], method=method)
            pref = f"{method}_{band_name}"
            feats[f"{pref}_global"] = fc.global_mean(W)
            feats[f"{pref}_posterior"] = fc.submatrix_mean(W, post_idx)
            for mname, val in fc.graph_metrics(W, q["graph_metrics"]).items():
                feats[f"graph_{pref}_{mname}"] = val
    # TODO(power): relative alpha/theta power, slowing ratio, PAF, SEF95, entropy
    # per the plan; slot into this dict with the same baseline/delta treatment.
    return feats


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _group_name(patient_id: int, date: pd.Timestamp) -> str:
    return f"{int(patient_id)}_{date.year}_{date.month:02d}_{date.day:02d}"


def _load_session_eeg(config: Config, group_name: str):
    """Load one session's raw EEG as ``[C, T]``. Supports an HDF5 store keyed by
    group_name (repo convention) or a per-session ``.npy`` in raw_eeg_dir."""
    raw = config.path("paths", "raw_eeg_dir")
    key = config["qeeg"]["hdf5_eeg_key"]
    if raw.endswith((".h5", ".hdf5")) or os.path.isfile(raw):
        import h5py
        with h5py.File(raw, "r") as f:
            g = f[group_name]
            data = g[key][:] if key in g else g["eeg"][:]
        return np.asarray(data, dtype=np.float32)
    npy = os.path.join(raw, f"{group_name}.npy")
    if os.path.exists(npy):
        return np.load(npy).astype(np.float32)
    h5 = os.path.join(raw, f"{group_name}.h5")
    if os.path.exists(h5):
        import h5py
        with h5py.File(h5, "r") as f:
            return np.asarray(f[key][:], dtype=np.float32)
    raise FileNotFoundError(f"No raw EEG for session {group_name} under {raw}")


def main(config: Config) -> str:
    q = config["qeeg"]
    fs = q["fs"]
    if fs is None:
        raise ValueError("qeeg.fs (sampling rate, Hz) must be set in config.")
    fs = float(fs)

    prog = io.read_table(config.out("progressions"))
    prog["before_date"] = pd.to_datetime(prog["before_date"])
    prog["after_date"] = pd.to_datetime(prog["after_date"])

    # Compute per-session features once (sessions are shared across progressions).
    session_cache: dict = {}
    def _feats(pid, date):
        gname = _group_name(pid, date)
        if gname not in session_cache:
            eeg = _load_session_eeg(config, gname)
            session_cache[gname] = session_features(eeg, fs, config)
        return session_cache[gname]

    rows = []
    skipped = []
    for _, r in prog.iterrows():
        try:
            fb = _feats(r["patient_id"], r["before_date"])
            fa = _feats(r["patient_id"], r["after_date"])
        except FileNotFoundError as e:
            skipped.append((r["progression_id"], str(e)))
            continue
        row = {"progression_id": r["progression_id"], "patient_id": int(r["patient_id"])}
        for name in fb:
            row[f"{name}_baseline"] = fb[name]
            row[f"{name}_delta"] = fa[name] - fb[name]     # within-progression change
        rows.append(row)

    if skipped:
        print(f"[qeeg] WARNING: {len(skipped)} progression(s) skipped (missing raw EEG); "
              f"e.g. {skipped[0]}")

    out_df = pd.DataFrame(rows)
    invariants.assert_progression_unit(len(out_df), out_df["progression_id"].tolist())
    out_path = io.write_table(out_df, config.out("qeeg_connectivity"))
    print(f"[qeeg] {len(out_df)} progressions x {out_df.shape[1] - 2} FC features "
          f"-> {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 6 - QEEG connectivity features")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
