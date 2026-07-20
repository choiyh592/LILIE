"""Robust multivariate outlier detection for the delta / PC space.

A handful of extreme-delta progressions can dominate PCA and make k-means peel
each tail into its own tiny cluster. This module flags such points so clustering
can run on the core population and mark the outliers separately (cluster = -1).

Default method is **Local Outlier Factor** (density-based): unlike a single
robust-covariance (Mahalanobis/MCD) fit, LOF does NOT assume one elliptical
core, so a genuine minority cluster is not mistaken for outliers -- it only
flags points sitting in locally sparse regions. Mahalanobis (MCD) is available
as an option for the clean "one dominant blob + tails" case.
"""
from __future__ import annotations

import numpy as np


def _mahalanobis_mask(X, quantile, seed):
    from scipy.stats import chi2
    n, p = X.shape
    try:
        from sklearn.covariance import MinCovDet
        if n < 2 * (p + 1):
            raise ValueError("too few points for MCD")
        d2 = np.asarray(MinCovDet(random_state=seed).fit(X).mahalanobis(X), float)
    except Exception:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-9
        d2 = (((X - med) / mad) ** 2).sum(axis=1)
    cutoff = float(chi2.ppf(quantile, df=p))
    return d2 > cutoff, d2, cutoff


def _lof_mask(X, n_neighbors, contamination):
    from sklearn.neighbors import LocalOutlierFactor
    nn = int(min(n_neighbors, max(2, X.shape[0] - 1)))
    lof = LocalOutlierFactor(n_neighbors=nn, contamination=contamination)
    pred = lof.fit_predict(X)
    score = -lof.negative_outlier_factor_          # >1 => more outlier-ish
    return pred == -1, score, None


def outlier_mask(X, method="lof", quantile=0.975, n_neighbors=20,
                 contamination="auto", seed=0):
    """Return (mask, score, cutoff). mask[i]=True marks progression i an outlier.

    method="lof"          -> Local Outlier Factor (default; multimodal-safe).
    method="mahalanobis"  -> robust MCD distance vs chi-square cutoff.
    """
    X = np.asarray(X, dtype=float)
    if method == "mahalanobis":
        return _mahalanobis_mask(X, quantile, seed)
    return _lof_mask(X, n_neighbors, contamination)
