"""Module 6 - qeeg: QEEG features from preprocessed raw EEG.

IMPLEMENTED: functional connectivity (VC-robust wPLI / imaginary coherence) plus
graph-theoretic summaries, per session, expressed as BOTH a baseline value and a
within-progression delta, keyed by progression_id (invariant 3).

SCAFFOLD (TODO): spectral-power / PAF / slowing-ratio / entropy features.

Raw EEG loading supports MNE ``.fif`` (sampling rate + channel names read from
the file), a per-session ``.npy``, or an HDF5 store. For ``.fif`` the session is
matched by prefix ``{patient_id}_{YYYY}_{MM}_{DD}_*.fif`` (the recording's
timestamp suffix is ignored).

Output (paths.output_dir): qeeg_connectivity.parquet|csv keyed by progression_id.

Run:  python -m analysis.qeeg --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import glob
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
def _match_channels(ch_names, wanted):
    """Case-insensitive index lookup of `wanted` channel names in `ch_names`."""
    lut = {c.strip().lower(): i for i, c in enumerate(ch_names)}
    return [lut[w.strip().lower()] for w in wanted if w.strip().lower() in lut]


def session_features(eeg: np.ndarray, fs: float, ch_names, config: Config) -> dict:
    """Connectivity + graph features for one session's ``[C, T]`` EEG."""
    q = config["qeeg"]
    bands = q["bands"]
    post_idx = _match_channels(ch_names, q["posterior_channels"])

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
    # TODO(power): relative alpha/theta power, slowing ratio, PAF, SEF95, entropy.
    return feats


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _group_name(patient_id: int, date: pd.Timestamp) -> str:
    return f"{int(patient_id)}_{date.year}_{date.month:02d}_{date.day:02d}"


def _load_session_eeg(config: Config, group_name: str):
    """Load one session's EEG -> (data[C,T], fs_or_None, ch_names_or_None).

    Supports MNE .fif (prefix match, timestamp suffix ignored), .npy, or HDF5.
    """
    raw = config.path("paths", "raw_eeg_dir")
    key = config["qeeg"]["hdf5_eeg_key"]

    # 1. MNE .fif directory (their format: {gname}_{HH}_{MM}_{SS}_processed_raw.fif)
    if os.path.isdir(raw):
        matches = sorted(glob.glob(os.path.join(raw, f"{group_name}_*.fif"))) \
            or sorted(glob.glob(os.path.join(raw, f"{group_name}*.fif")))
        if matches:
            import mne
            if len(matches) > 1:
                print(f"[qeeg] {group_name}: {len(matches)} .fif files; using {os.path.basename(matches[0])}")
            r = mne.io.read_raw_fif(matches[0], preload=True, verbose="ERROR")
            try:
                r.pick("eeg")
            except Exception:
                r.pick_types(eeg=True)
            return r.get_data().astype(np.float32), float(r.info["sfreq"]), list(r.ch_names)

    # 2. single HDF5 store keyed by group_name
    if raw.endswith((".h5", ".hdf5")) or (os.path.isfile(raw)):
        import h5py
        with h5py.File(raw, "r") as f:
            g = f[group_name]
            data = g[key][:] if key in g else g["eeg"][:]
        return np.asarray(data, dtype=np.float32), None, None

    # 3. per-session .npy
    npy = os.path.join(raw, f"{group_name}.npy")
    if os.path.exists(npy):
        return np.load(npy).astype(np.float32), None, None
    raise FileNotFoundError(f"No raw EEG for session {group_name} under {raw}")


def main(config: Config):
    q = config["qeeg"]
    fs_cfg = q["fs"]

    prog = io.read_table(config.out("progressions"))
    prog["before_date"] = pd.to_datetime(prog["before_date"])
    prog["after_date"] = pd.to_datetime(prog["after_date"])

    session_cache: dict = {}

    def _feats(pid, date):
        gname = _group_name(pid, date)
        if gname not in session_cache:
            eeg, fs_file, ch_names = _load_session_eeg(config, gname)
            fs = fs_file if fs_file is not None else (float(fs_cfg) if fs_cfg else None)
            if fs is None:
                raise ValueError("qeeg.fs must be set (or use .fif so it is read "
                                 "from the recording).")
            if ch_names is None:
                ch_names = list(q["channels"])
            session_cache[gname] = session_features(eeg, fs, ch_names, config)
        return session_cache[gname]

    rows, skipped = [], []
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
            row[f"{name}_delta"] = fa[name] - fb[name]
        rows.append(row)

    if skipped:
        print(f"[qeeg] WARNING: {len(skipped)} progression(s) skipped (missing raw EEG); "
              f"e.g. {skipped[0]}")
    if not rows:
        print("[qeeg] no progressions had raw EEG -> nothing written. Check "
              "paths.raw_eeg_dir and the session file naming.")
        return None

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
