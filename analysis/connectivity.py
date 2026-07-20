"""Volume-conduction-robust functional connectivity + graph-theoretic summaries.

Deliberately built on ``scipy`` + ``numpy`` only (no hard ``mne-connectivity``
dependency), so the connectivity math is runnable and testable on a minimal
stack. Two spectral metrics, both insensitive to zero-lag (volume-conduction)
leakage:

* **wPLI** (weighted phase-lag index) -
  ``|E[Im(S_ij)]| / E[|Im(S_ij)|]`` over epochs, per frequency, band-averaged.
* **imcoh** (imaginary part of coherency) -
  ``|Im( E[S_ij] / sqrt(E[S_ii] E[S_jj]) )|``, band-averaged.

Both return a symmetric ``[C, C]`` matrix in ``[0, 1]`` with a zero diagonal.
Graph metrics operate on such a weighted, non-negative, symmetric matrix.

These are undirected, non-negative weights; raw coherence / PLV are intentionally
NOT provided because they are corrupted by volume conduction on scalp EEG.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Epoching
# ---------------------------------------------------------------------------
def make_epochs(x: np.ndarray, fs: float, epoch_len_s: float,
                overlap: float = 0.5) -> np.ndarray:
    """Slice continuous ``[C, T]`` EEG into ``[E, C, L]`` epochs (Hann-tapered later)."""
    C, T = x.shape
    L = int(round(epoch_len_s * fs))
    if L <= 0 or L > T:
        raise ValueError(f"epoch length {L} incompatible with signal length {T}")
    step = max(1, int(round(L * (1.0 - overlap))))
    starts = list(range(0, T - L + 1, step))
    if not starts:
        starts = [0]
    return np.stack([x[:, s:s + L] for s in starts], axis=0)


# ---------------------------------------------------------------------------
# Spectral connectivity
# ---------------------------------------------------------------------------
def _epoch_ffts(epochs: np.ndarray, fs: float):
    """Return (freqs, Xf) where Xf is [E, C, F] complex, Hann-tapered, detrended."""
    E, C, L = epochs.shape
    win = np.hanning(L)
    xw = (epochs - epochs.mean(axis=2, keepdims=True)) * win[None, None, :]
    Xf = np.fft.rfft(xw, axis=2)
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    return freqs, Xf


def spectral_connectivity(epochs: np.ndarray, fs: float, band: Sequence[float],
                          method: str = "wpli") -> np.ndarray:
    """Connectivity matrix over epochs for one frequency band.

    Parameters
    ----------
    epochs : ``[E, C, L]`` array of segmented EEG.
    fs     : sampling rate (Hz).
    band   : ``(f_lo, f_hi)`` in Hz.
    method : ``"wpli"`` or ``"imcoh"``.
    """
    method = method.lower()
    if method not in ("wpli", "imcoh"):
        raise ValueError(f"unknown method {method!r} (use 'wpli' or 'imcoh')")

    freqs, Xf = _epoch_ffts(epochs, fs)
    E, C, F = Xf.shape
    fmask = (freqs >= band[0]) & (freqs <= band[1])
    if not fmask.any():
        raise ValueError(f"no FFT bins in band {band} at fs={fs}")
    eps = 1e-20

    # Accumulate cross-spectral quantities over epochs.
    acc_Sxy = np.zeros((C, C, F), dtype=complex)      # E[S_ij]
    acc_abs_imS = np.zeros((C, C, F))                 # E[|Im(S_ij)|]
    acc_Sxx = np.zeros((C, F))                        # E[|X_i|^2]
    for e in range(E):
        Xe = Xf[e]                                    # [C, F]
        Sxy = Xe[:, None, :] * np.conj(Xe[None, :, :])  # [C, C, F]
        acc_Sxy += Sxy
        acc_abs_imS += np.abs(Sxy.imag)
        acc_Sxx += (Xe.real ** 2 + Xe.imag ** 2)
    mean_Sxy = acc_Sxy / E
    mean_abs_imS = acc_abs_imS / E
    mean_Sxx = acc_Sxx / E

    if method == "wpli":
        # Magnitude-pooled across the band: sum|E[Im]| / sum E[|Im|]. Pooling
        # (rather than averaging per-bin wPLI) weights bins by their imaginary
        # cross-power, so noise-only bins don't dilute a narrowband coupling.
        num = np.abs(mean_Sxy.imag)[:, :, fmask].sum(axis=2)
        den = mean_abs_imS[:, :, fmask].sum(axis=2)
        W = num / (den + eps)
    else:  # imcoh
        denom = np.sqrt(mean_Sxx[:, None, :] * mean_Sxx[None, :, :]) + eps
        coh = mean_Sxy / denom
        W = np.abs(coh.imag)[:, :, fmask].mean(axis=2)    # band-average
    W = 0.5 * (W + W.T)                               # enforce symmetry
    np.fill_diagonal(W, 0.0)
    return np.clip(W, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Derived scalar summaries
# ---------------------------------------------------------------------------
def global_mean(W: np.ndarray) -> float:
    """Mean of the off-diagonal upper triangle (global connectivity)."""
    iu = np.triu_indices_from(W, k=1)
    return float(W[iu].mean())


def submatrix_mean(W: np.ndarray, idx: Sequence[int]) -> float:
    """Mean connectivity within a subset of nodes (e.g. posterior)."""
    idx = list(idx)
    sub = W[np.ix_(idx, idx)]
    iu = np.triu_indices_from(sub, k=1)
    return float(sub[iu].mean()) if len(iu[0]) else float("nan")


# ---------------------------------------------------------------------------
# Graph-theoretic metrics (weighted, undirected, non-negative)
# ---------------------------------------------------------------------------
def _shortest_path_lengths(W: np.ndarray) -> np.ndarray:
    """Floyd-Warshall on distance = 1/weight (inf where no edge)."""
    C = W.shape[0]
    with np.errstate(divide="ignore"):
        D = 1.0 / W
    D[W <= 0] = np.inf
    np.fill_diagonal(D, 0.0)
    for k in range(C):
        D = np.minimum(D, D[:, k][:, None] + D[k, :][None, :])
    return D


def mean_strength(W: np.ndarray) -> float:
    """Mean nodal strength (sum of a node's weights, averaged over nodes)."""
    return float(W.sum(axis=1).mean())


def global_efficiency(W: np.ndarray) -> float:
    """Average inverse shortest-path length over all node pairs."""
    D = _shortest_path_lengths(W)
    C = W.shape[0]
    iu = np.triu_indices(C, k=1)
    inv = np.where(np.isfinite(D[iu]) & (D[iu] > 0), 1.0 / D[iu], 0.0)
    return float(inv.mean()) if len(iu[0]) else float("nan")


def char_path_length(W: np.ndarray) -> float:
    """Mean of finite shortest-path lengths (characteristic path length)."""
    D = _shortest_path_lengths(W)
    iu = np.triu_indices(W.shape[0], k=1)
    d = D[iu]
    d = d[np.isfinite(d) & (d > 0)]
    return float(d.mean()) if d.size else float("inf")


def weighted_clustering(W: np.ndarray) -> float:
    """Onnela weighted clustering coefficient, averaged over nodes."""
    C = W.shape[0]
    Wn = W / (W.max() + 1e-20)          # normalize weights to [0, 1]
    A = (W > 0).astype(float)
    k = A.sum(axis=1)
    cuberoot = np.cbrt(Wn)
    # number of weighted triangles around each node
    tri = np.diag(cuberoot @ cuberoot @ cuberoot)
    denom = k * (k - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(denom > 0, tri / denom, 0.0)
    return float(ci.mean())


def modularity(W: np.ndarray) -> float:
    """Newman leading-eigenvector modularity Q for a single 2-way split.

    Lightweight, deterministic community summary: split nodes by the sign of the
    leading eigenvector of the modularity matrix B = W - (k k^T)/(2m), then score
    Q for that partition. Returns 0 if the split does not improve modularity.
    """
    k = W.sum(axis=1)
    m2 = k.sum()                        # = 2m
    if m2 <= 0:
        return 0.0
    B = W - np.outer(k, k) / m2
    vals, vecs = np.linalg.eigh(B)
    lead = vecs[:, np.argmax(vals)]
    s = np.where(lead >= 0, 1.0, -1.0)
    if np.all(s == s[0]):               # no split
        return 0.0
    Q = float(s @ B @ s) / (2.0 * m2)
    return max(Q, 0.0)


GRAPH_METRIC_FNS = {
    "mean_strength": mean_strength,
    "global_efficiency": global_efficiency,
    "char_path_length": char_path_length,
    "weighted_clustering": weighted_clustering,
    "modularity": modularity,
}


def graph_metrics(W: np.ndarray, which: Sequence[str]) -> dict:
    return {name: GRAPH_METRIC_FNS[name](W) for name in which}
