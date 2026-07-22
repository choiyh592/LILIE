"""Line-noise / notch diagnostic: did the 50 Hz notch match the recording's mains?

LaBraM-standard preprocessing applies a 50 Hz notch. If the recordings were made
on 60 Hz mains, that notch removed a clean frequency and left the real 60 Hz line
peak - which contaminates the (pre-fix) full-range spectral features and is baked
into the embeddings. This checks it directly: average the raw-EEG PSD across a
sample of sessions and measure the peak-to-neighbourhood ratio at 50 and 60 Hz.

  ratio(line) = max PSD in [line±0.5] / median PSD in [line-4,line-1]∪[line+1,line+4]
    ratio >> 1  -> a line peak IS present at that frequency
    ratio  < 1  -> a notch DIP is present there

Read as: a dip at 50 + a peak at 60  => notch missed your 60 Hz mains (re-notch at 60).
         a dip at 50 + flat at 60     => notch matched your 50 Hz mains (fine).

Reads raw .fif under paths.raw_eeg_dir. Outputs: psd_diagnostic.png, psd_diagnostic.json
Run:  python -m analysis.psd_diagnostic --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import welch

from .config import Config, load_config, add_arg
from . import io


def _line_ratio(f, psd, line, half=0.5, lo=4.0, hi=1.0):
    """Peak-to-neighbourhood ratio at `line` Hz (pure: testable without EEG)."""
    peak_m = (f >= line - half) & (f <= line + half)
    nb_m = ((f >= line - lo) & (f <= line - hi)) | ((f >= line + hi) & (f <= line + lo))
    if not peak_m.any() or not nb_m.any():
        return np.nan
    return float(np.max(psd[peak_m]) / (np.median(psd[nb_m]) + 1e-30))


def main(config: Config):
    q = config["qeeg"]
    raw = config.path("paths", "raw_eeg_dir")
    fs_cfg = q.get("fs", None)
    sample = int(q.get("psd_diagnostic_sample", 40))
    lines = q.get("mains_candidates", [50, 60])

    files = sorted(glob.glob(os.path.join(raw, "*.fif"))) if os.path.isdir(raw) else []
    if not files:
        raise SystemExit(f"[psd] no .fif under {raw} (this diagnostic needs raw EEG; "
                         "run it where the recordings live).")
    files = files[:sample]

    import mne
    acc, grid, fs_used, n_ok = None, None, None, 0
    for fp in files:
        try:
            r = mne.io.read_raw_fif(fp, preload=True, verbose="ERROR")
            try:
                r.pick("eeg")
            except Exception:
                r.pick_types(eeg=True)
            data = r.get_data().astype(np.float64)
            fs = float(r.info["sfreq"]) if r.info.get("sfreq") else float(fs_cfg)
        except Exception as e:
            print(f"[psd] skip {os.path.basename(fp)}: {e}"); continue
        nper = int(min(data.shape[1], max(256, round(4 * fs))))
        f, P = welch(data, fs=fs, nperseg=nper, axis=-1)     # [C,F]
        pm = P.mean(0)                                        # channel-mean PSD
        if grid is None:
            grid, acc, fs_used = f, np.zeros_like(pm), fs
            acc += pm; n_ok = 1
        elif f.shape == grid.shape:
            acc += pm; n_ok += 1
    if n_ok == 0:
        raise SystemExit("[psd] no readable .fif files.")
    psd = acc / n_ok

    ratios = {int(L): _line_ratio(grid, psd, float(L)) for L in lines}
    # verdict
    r50, r60 = ratios.get(50, np.nan), ratios.get(60, np.nan)
    def _peak(x): return np.isfinite(x) and x > 1.5
    def _dip(x):  return np.isfinite(x) and x < 0.8
    if _dip(r50) and _peak(r60):
        verdict = ("NOTCH MISMATCH: 50 Hz notch applied but a 60 Hz line peak remains "
                   "-> your mains is 60 Hz. Re-notch at 60 (and, strictly, regenerate "
                   "embeddings) if the 60 Hz peak is large.")
    elif _dip(r50) and not _peak(r60):
        verdict = "OK: 50 Hz notch matched (dip at 50, no 60 Hz peak) -> 50 Hz mains."
    elif _peak(r60) and not _dip(r50):
        verdict = "60 Hz line peak present and 50 Hz not clearly notched -> check preprocessing."
    else:
        verdict = "Inconclusive: no clear line peak or dip at 50/60 Hz (well-cleaned or low line noise)."

    # figure
    fig, ax = plt.subplots(figsize=(9, 5))
    band = (grid >= 1) & (grid <= min(100, grid.max()))
    ax.semilogy(grid[band], psd[band], color="#0072B2", lw=1.4)
    for L, col in [(50, "#009E73"), (60, "#D55E00")]:
        if grid.max() >= L:
            ax.axvline(L, color=col, ls="--", lw=1.3)
            ax.text(L, psd[band].max(), f" {L}Hz\n ratio={ratios.get(L, float('nan')):.2f}",
                    color=col, fontsize=9, va="top")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("mean PSD (log)")
    ax.set_title(f"Line-noise check - mean PSD over {n_ok} sessions (fs={fs_used:.0f} Hz)", fontsize=11)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.out("psd_diagnostic.png"), dpi=int(config["report"]["fig_dpi"]))
    plt.close(fig)

    io.write_json({"n_sessions": n_ok, "fs": fs_used, "nyquist": fs_used / 2,
                   "line_ratios": {str(k): v for k, v in ratios.items()},
                   "verdict": verdict,
                   "note": "ratio>1.5 = line peak; ratio<0.8 = notch dip. Analysis-band "
                           "[1,45] spectral features already exclude 50/60 Hz; this matters "
                           "mainly for the embeddings and any full-band feature."},
                  config.out("psd_diagnostic.json"))

    print(f"[psd] {n_ok} sessions, fs={fs_used:.0f} (Nyquist {fs_used/2:.0f}); "
          f"line ratios: " + ", ".join(f"{k}Hz={v:.2f}" for k, v in ratios.items()))
    print(f"[psd] VERDICT: {verdict}")
    return config.out("psd_diagnostic.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Line-noise / notch PSD diagnostic")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
