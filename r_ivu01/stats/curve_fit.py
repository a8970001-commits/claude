"""Step 4 — parametric family fit, only run after Step 3 (G1) passes.

Candidate A (symmetric): log-Gaussian kernel.
Candidate B (asymmetric): Gamma kernel.
Both are 2-parameter families; fit is weighted least squares against per-bin
values, weight = 1/SE^2 (SE from the same cluster bootstrap as Step 3, F-11).
AIC = n*ln(RSS/n) + 2k; tie (|delta AIC| < 2) -> pick simpler (A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from r_ivu01.config import AIC_TIE_THRESHOLD


def log_gaussian_kernel(v: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = np.log(np.maximum(v, 1e-6)) - np.log(mu)
    return np.exp(-(z ** 2) / (2 * sigma ** 2))


def gamma_kernel(v: np.ndarray, mu: float, k: float) -> np.ndarray:
    r = np.maximum(v, 1e-6) / mu
    return np.power(r, k) * np.exp(k * (1 - r))


@dataclass
class KernelFitResult:
    name: str
    params: tuple[float, float]
    fitted: np.ndarray
    rss: float
    aic: float


def _weighted_aic(y: np.ndarray, fitted: np.ndarray, weights: np.ndarray, n_params: int) -> float:
    n = len(y)
    rss = float(np.sum(weights * (y - fitted) ** 2))
    return n * np.log(rss / n) + 2 * n_params, rss


def fit_kernel(
    rvol: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    kernel: str,
    p0: tuple[float, float],
) -> KernelFitResult:
    fn = log_gaussian_kernel if kernel == "log_gaussian" else gamma_kernel
    sigma = 1.0 / np.sqrt(np.maximum(weights, 1e-12))  # curve_fit's `sigma` = per-point stddev
    params, _ = curve_fit(fn, rvol, target, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=10000)
    fitted = fn(rvol, *params)
    aic, rss = _weighted_aic(target, fitted, weights, n_params=2)
    return KernelFitResult(name=kernel, params=tuple(params), fitted=fitted, rss=rss, aic=aic)


def select_kernel(
    rvol: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    p0_log_gaussian: tuple[float, float] = (1.5, 0.5),
    p0_gamma: tuple[float, float] = (1.5, 2.0),
) -> KernelFitResult:
    """Fits both candidates, returns the winner. Ties (< AIC_TIE_THRESHOLD)
    resolve to candidate A (log-Gaussian) per spec Step 4."""

    fit_a = fit_kernel(rvol, target, weights, "log_gaussian", p0_log_gaussian)
    fit_b = fit_kernel(rvol, target, weights, "gamma", p0_gamma)

    if fit_a.aic - fit_b.aic <= AIC_TIE_THRESHOLD:
        return fit_a
    return fit_b
