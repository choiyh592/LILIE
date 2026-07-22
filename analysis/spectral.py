"""Spectral / power QEEG features (the plan's non-connectivity primaries).

Per session, from Welch PSD, all computed within an ANALYSIS BAND (default
[1, 45] Hz) so session-level nuisance outside the bands of interest (sub-1 Hz
drift, 45-75 Hz EMG range, filter roll-off, residual line noise) does not
contaminate the relative-power denominators, SEF95, or spectral entropy:

  - absolute (log10) + relative band power (all bands), global and posterior
  - (delta+theta)/(alpha+beta) slowing ratio (global + posterior)
  - alpha centre-of-gravity (primary) and aperiodic-flattened peak alpha freq
    (secondary; NaN when no resolvable peak) - both global + posterior
  - theta/alpha ratio, median frequency, SEF95, spectral entropy (in-band)
  - aperiodic (1/f) exponent + offset

Feature names are shared with the connectivity block (qeeg.session_features
merges both); baseline + within-progression delta are added downstream. scipy only.
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
    alpha/beta bumps don't bias the slope. Returns (exponent[C], offset[C]) where the
    per-channel log-log fit is  log10(P) ~ offset - exponent*log10(f)  (offset is the
    intercept, so the fitted line is reusable for PAF flattening).
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
        offs.append(float(b[1]))                         # intercept
    return np.array(exps), np.array(offs)


def _resolve_beta(bands):
    return [k for k in bands if k.startswith("beta")]


def spectral_features(eeg: np.ndarray, fs: float, bands: dict, post_idx,
                      analysis_band=(1.0, 45.0)) -> dict:
    # --- band-name guard (1.4): raise, do not silently degrade ---
    required = ("delta", "theta", "alpha")
    missing = [b for b in required if b not in bands]
    if missing:
        raise ValueError(f"spectral_features: config bands missing {missing}; "
                         f"have {list(bands)}")
    beta_keys = _resolve_beta(bands)
    if not beta_keys:
        raise ValueError("spectral_features: no 'beta*' band in config for the "
                         "slowing-ratio denominator.")

    f, P = _psd(eeg, fs)
    ab0, ab1 = float(analysis_band[0]), float(analysis_band[1])
    mb = (f >= ab0) & (f <= ab1)
    if mb.sum() < 4:
        raise ValueError(f"analysis_band {analysis_band} yields <4 PSD bins.")
    fb, Pb = f[mb], P[:, mb]
    total = _trapz(Pb, fb, axis=1) + 1e-20              # [C], within analysis band

    feats: dict = {}
    absp = {}                                           # RAW band power (for ratios)
    for bn, br in bands.items():
        bp = _bandpow(f, P, br)
        absp[bn] = bp
        rel = bp / total
        feats[f"abs_{bn}_global"] = float(np.log10(bp + 1e-20).mean())   # (1.3) log power
        feats[f"rel_{bn}_global"] = float(rel.mean())
        if post_idx:
            feats[f"rel_{bn}_posterior"] = float(rel[post_idx].mean())

    # slowing ratio (delta+theta)/(alpha + all beta*) -- raw powers
    num = absp["delta"] + absp["theta"]
    den = absp["alpha"] + sum(absp[k] for k in beta_keys) + 1e-20
    sr = num / den
    feats["slowing_ratio_global"] = float(np.mean(sr))
    if post_idx:
        feats["slowing_ratio_posterior"] = float(np.mean(sr[post_idx]))

    # theta/alpha (raw)
    ta = absp["theta"] / (absp["alpha"] + 1e-20)
    feats["theta_alpha_global"] = float(np.mean(ta))

    # aperiodic fit FIRST (reused by PAF flattening)
    exp_, off_ = _aperiodic(f, P)

    # alpha centre-of-gravity (primary) + aperiodic-flattened PAF (secondary)
    am = (f >= bands["alpha"][0]) & (f <= bands["alpha"][1])
    if am.any():
        fa = f[am]
        Pa = P[:, am]
        cog = (fa[None, :] * Pa).sum(1) / (Pa.sum(1) + 1e-30)          # [C]
        feats["alpha_cog_global"] = float(cog.mean())
        if post_idx:
            feats["alpha_cog_posterior"] = float(cog[post_idx].mean())
        # flattened peak: subtract aperiodic fit, peak-pick residual. Accept only a
        # PROMINENT interior peak (height > peak_sd * in-band residual SD), else NaN,
        # so channels without a resolvable alpha peak do not return a spurious bin.
        peak_sd = 2.5
        lf_all = np.log10(np.where(f > 0, f, np.nan))
        paf = np.full(P.shape[0], np.nan)
        for c in range(P.shape[0]):
            if not np.isfinite(exp_[c]):
                continue
            fit_c = off_[c] - exp_[c] * lf_all                        # log10 aperiodic line
            resid = np.log10(P[c] + 1e-30) - fit_c
            band_sd = np.nanstd(resid[mb]) + 1e-12
            ra = resid[am]
            j = int(np.nanargmax(ra))
            if j == 0 or j == len(ra) - 1:                            # edge -> no peak
                continue
            if ra[j] < peak_sd * band_sd:                            # not prominent -> unresolved
                continue
            paf[c] = fa[j]
        feats["paf_nan_frac_global"] = float(np.mean(~np.isfinite(paf)))
        if np.isfinite(paf).any():
            feats["paf_global"] = float(np.nanmean(paf))
            if post_idx and np.isfinite(paf[post_idx]).any():
                feats["paf_posterior"] = float(np.nanmean(paf[post_idx]))

    # median freq, SEF95, spectral entropy -- WITHIN analysis band
    csum = np.cumsum(Pb, axis=1) / (Pb.sum(axis=1, keepdims=True) + 1e-20)
    feats["median_freq_global"] = float(fb[np.argmax(csum >= 0.5, axis=1)].mean())
    feats["sef95_global"] = float(fb[np.argmax(csum >= 0.95, axis=1)].mean())
    Pn = Pb / (Pb.sum(axis=1, keepdims=True) + 1e-20)
    ent = -np.sum(Pn * np.log(Pn + 1e-20), axis=1) / np.log(Pb.shape[1])
    feats["spectral_entropy_global"] = float(ent.mean())

    # aperiodic (1/f) exponent + offset - the spectral-tilt / E-I-proxy component
    if np.isfinite(exp_).any():
        feats["aperiodic_exponent_global"] = float(np.nanmean(exp_))
        feats["aperiodic_offset_global"] = float(np.nanmean(off_))
        if post_idx:
            feats["aperiodic_exponent_posterior"] = float(np.nanmean(exp_[post_idx]))
    return feats
