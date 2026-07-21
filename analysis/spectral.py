"""Spectral / power QEEG features (the plan's non-connectivity primaries).

Per session, from Welch PSD:
  - absolute + relative band power (all bands), global and posterior
  - relative alpha power (posterior) and relative theta - plan primaries
  - (delta+theta)/(alpha+beta) slowing ratio (global + posterior)
  - peak alpha frequency (PAF), global + posterior
  - theta/alpha ratio, median frequency, SEF95, spectral entropy

Feature names are shared with the connectivity block (qeeg.session_features
merges both); baseline + within-progression delta are added downstream.
scipy only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def _trapz(y, x, axis=-1):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x, axis=axis)


def _psd(eeg, fs):
    nper = int(min(eeg.shape[1], max(64, round(4 * fs))))
    f, P = welch(eeg, fs=fs, nperseg=nper, axis=-1)     # [C,F]
    return f, P


def _bandpow(f, P, band):
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return np.zeros(P.shape[0])
    return _trapz(P[:, m], f[m], axis=1)                # [C]


def _aperiodic(f, P, fmin=2.0, fmax=40.0, n_iter=3):
    """Aperiodic (1/f) exponent per channel: chi where PSD ~ f^(-chi), chi>0.

    Lightweight specparam-style fit: OLS of log10(PSD) on log10(f) over [fmin,fmax]
    with iterative masking of points that sit ABOVE the fit (oscillatory peaks), so
    alpha/beta bumps don't bias the aperiodic slope. Returns (exponent[C], offset[C]).
    """
    m = (f >= fmin) & (f <= fmax) & (f > 0)
    if m.sum() < 6:
        return np.full(P.shape[0], np.nan), np.full(P.shape[0], np.nan)
    lf = np.log10(f[m])
    exps, offs = [], []
    for c in range(P.shape[0]):
        lp = np.log10(P[c, m] + 1e-30)
        keep = np.ones(len(lf), bool)
        b = np.polyfit(lf, lp, 1)
        for _ in range(n_iter):
            resid = lp - np.polyval(b, lf)
            thr = resid[keep].std() if keep.sum() > 3 else resid.std()
            keep = resid <= 1.0 * thr                    # drop peaks (above the fit)
            if keep.sum() < 5:
                keep = np.ones(len(lf), bool); break
            b = np.polyfit(lf[keep], lp[keep], 1)
        exps.append(-float(b[0]))                        # exponent = -slope (positive)
        offs.append(float(b[1]))
    return np.array(exps), np.array(offs)


def spectral_features(eeg: np.ndarray, fs: float, bands: dict, post_idx) -> dict:
    f, P = _psd(eeg, fs)
    total = _trapz(P, f, axis=1) + 1e-20                # [C]
    feats: dict = {}
    absp = {}
    for bn, br in bands.items():
        bp = _bandpow(f, P, br)
        absp[bn] = bp
        rel = bp / total
        feats[f"abs_{bn}_global"] = float(bp.mean())
        feats[f"rel_{bn}_global"] = float(rel.mean())
        if post_idx:
            feats[f"rel_{bn}_posterior"] = float(rel[post_idx].mean())

    # slowing ratio (delta+theta)/(alpha+beta)
    num = absp.get("delta", 0) + absp.get("theta", 0)
    den = absp.get("alpha", 0) + absp.get("beta1", 0) + absp.get("beta2", 0) + 1e-20
    sr = num / den
    feats["slowing_ratio_global"] = float(np.mean(sr))
    if post_idx:
        feats["slowing_ratio_posterior"] = float(np.mean(sr[post_idx]))

    # theta/alpha
    ta = absp.get("theta", 0) / (absp.get("alpha", 1e-20) + 1e-20)
    feats["theta_alpha_global"] = float(np.mean(ta))

    # peak alpha frequency
    am = (f >= bands["alpha"][0]) & (f <= bands["alpha"][1])
    if am.any():
        fa = f[am]
        paf = fa[np.argmax(P[:, am], axis=1)]
        feats["paf_global"] = float(paf.mean())
        if post_idx:
            feats["paf_posterior"] = float(paf[post_idx].mean())

    # median freq, SEF95, spectral entropy (channel-averaged)
    csum = np.cumsum(P, axis=1) / (P.sum(axis=1, keepdims=True) + 1e-20)
    feats["median_freq_global"] = float(f[np.argmax(csum >= 0.5, axis=1)].mean())
    feats["sef95_global"] = float(f[np.argmax(csum >= 0.95, axis=1)].mean())
    Pn = P / (P.sum(axis=1, keepdims=True) + 1e-20)
    ent = -np.sum(Pn * np.log(Pn + 1e-20), axis=1) / np.log(P.shape[1])
    feats["spectral_entropy_global"] = float(ent.mean())

    # aperiodic (1/f) exponent + offset - the spectral-tilt / E-I-proxy component
    exp_, off_ = _aperiodic(f, P)
    if np.isfinite(exp_).any():
        feats["aperiodic_exponent_global"] = float(np.nanmean(exp_))
        feats["aperiodic_offset_global"] = float(np.nanmean(off_))
        if post_idx:
            feats["aperiodic_exponent_posterior"] = float(np.nanmean(exp_[post_idx]))
    return feats
