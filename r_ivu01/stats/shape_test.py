"""Step 3 — shape gate, v1.3/F-02 rewrite.

The original G1 ("unimodal fit SSE better than monotone fit, in >=X% of
resamples") is VOID: a peak-free unimodal isotonic fit nests a monotone fit
(peak at either boundary IS monotone), so SSE_unimodal <= SSE_monotone holds
in every resample by construction — the gate could never fail (fail-open).

Replacement (locked, do not modify without a new spec version): a resample
"supports" H1 only if BOTH hold:
  (a) peak internality  — the unimodal fit's peak lands in decile 2-9
                            (1-indexed), not at either boundary bin.
  (b) dual-wing floor    — fitted(peak) - fitted(bin 1)  >= DELTA_ATR, AND
                            fitted(peak) - fitted(bin 10) >= DELTA_ATR.
G1 passes if the resample support rate >= G1_SUPPORT_RATE_THRESHOLD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from r_ivu01.config import DELTA_ATR, PEAK_INTERNAL_DECILE_RANGE


@dataclass
class UnimodalFit:
    fitted: np.ndarray
    peak_index: int  # 0-indexed position of the peak within `fitted`
    sse: float


def fit_unimodal_isotonic(y: np.ndarray, weights: np.ndarray | None = None) -> UnimodalFit:
    """Best-SSE unimodal (single interior peak, free peak position) fit via
    exhaustive search over candidate peak positions, each evaluated with a
    left increasing-isotonic PAVA fit and a right decreasing-isotonic PAVA
    fit. O(k^2) — trivial at k=10 (deciles)."""

    k = len(y)
    best_sse = np.inf
    best_fitted = None
    best_peak = 0

    for p in range(k):
        left_y, left_w = y[: p + 1], (weights[: p + 1] if weights is not None else None)
        right_y, right_w = y[p:], (weights[p:] if weights is not None else None)

        left_fitted = IsotonicRegression(increasing=True).fit_transform(
            np.arange(len(left_y)), left_y, sample_weight=left_w
        )
        right_fitted = IsotonicRegression(increasing=False).fit_transform(
            np.arange(len(right_y)), right_y, sample_weight=right_w
        )

        fitted = np.concatenate([left_fitted[:-1], right_fitted]) if p < k - 1 else left_fitted

        sse = float(np.sum((y - fitted) ** 2 * (weights if weights is not None else 1.0)))
        if sse < best_sse:
            best_sse, best_fitted, best_peak = sse, fitted, p

    return UnimodalFit(fitted=best_fitted, peak_index=best_peak, sse=best_sse)


def peak_is_internal(peak_index_0based: int, n_bins: int, internal_range: tuple[int, int] = PEAK_INTERNAL_DECILE_RANGE) -> bool:
    peak_decile_1based = peak_index_0based + 1
    lo, hi = internal_range
    return bool(lo <= peak_decile_1based <= hi)


def dual_wing_floor(fit: UnimodalFit, delta: float = DELTA_ATR) -> bool:
    peak_val = fit.fitted[fit.peak_index]
    return bool((peak_val - fit.fitted[0] >= delta) and (peak_val - fit.fitted[-1] >= delta))


def resample_supports_h1(y: np.ndarray, weights: np.ndarray | None = None) -> bool:
    fit = fit_unimodal_isotonic(y, weights)
    return bool(peak_is_internal(fit.peak_index, len(y)) and dual_wing_floor(fit))
