"""Small numerical helpers shared by the whole package (not meant for direct use)."""
from __future__ import annotations

import socket

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


class SingularMeasureError(ValueError):
    """Raised when two laws are mutually singular, so no Radon-Nikodym derivative exists."""


class NotComparableError(ValueError):
    """Raised when two processes live on different spaces and cannot be compared."""


def ess(w) -> float:
    """Effective sample size (sum w)^2 / sum w^2 of a weight vector."""
    w = np.asarray(w, float)
    s = w.sum()
    return float(s * s / np.sum(w * w)) if s > 0 else 0.0


def _edges(g):
    d = g[1] - g[0]
    return np.concatenate([g - d / 2, [g[-1] + d / 2]]), d


def hist_density(S, w, xg, yg, smooth=1.3):
    """Weighted, lightly smoothed 2-D histogram density on the grid xg x yg -> shape (ny, nx)."""
    xe, dx = _edges(xg)
    ye, dy = _edges(yg)
    H, _, _ = np.histogram2d(S[:, 0], S[:, 1], bins=[xe, ye], weights=w)
    H = gaussian_filter(H, smooth)
    tot = float(np.sum(w)) if w is not None else float(len(S))
    return (H / (tot * dx * dy)).T


def hist_density_1d(S, w, xg, smooth=1.3):
    xe, dx = _edges(xg)
    H, _ = np.histogram(S, bins=xe, weights=w)
    H = gaussian_filter1d(H.astype(float), smooth)
    tot = float(np.sum(w)) if w is not None else float(len(S))
    return H / (tot * dx)


def quantile_range(S, q=(0.5, 99.5), pad=0.06, include=None):
    """Per-axis robust range of samples S (N, d) -> (d, 2); optional extra points to include."""
    S = np.asarray(S, float)
    S = S[np.all(np.isfinite(S), axis=1)]
    lo = np.percentile(S, q[0], axis=0)
    hi = np.percentile(S, q[1], axis=0)
    if include is not None:
        inc = np.asarray(include, float)
        lo, hi = np.minimum(lo, inc), np.maximum(hi, inc)
    span = np.maximum(hi - lo, 1e-9 + 1e-3 * np.maximum(np.abs(hi), np.abs(lo)))
    lo_p = lo - pad * span
    nonneg = S.min(axis=0) >= 0          # positive supports (prices, counts, exponential...) never go below 0
    lo_p = np.where(nonneg, np.maximum(lo_p, 0.0), lo_p)
    return np.stack([lo_p, hi + pad * span], axis=1)


def hdr_levels(P, cell, masses=(0.5, 0.9)):
    """Density thresholds whose super-level sets hold the given probability masses."""
    flat = np.sort(P.ravel())[::-1]
    c = np.cumsum(flat) * cell
    c = c / c[-1]
    return [flat[min(np.searchsorted(c, m), len(flat) - 1)] for m in masses]


def robust_zmax(peaks, margin=1.08):
    """Height limit that ignores Dirac-like spikes (e.g. bridge at t = T)."""
    peaks = np.asarray(peaks, float)
    peaks = peaks[np.isfinite(peaks)]
    cap = 5.0 * np.median(peaks)
    ok = peaks[peaks <= cap]
    return margin * (ok.max() if ok.size else np.median(peaks))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
