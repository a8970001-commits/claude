"""Step 1 — decile binning + anchor bins (F-04)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from r_ivu01.config import ANCHOR_BIN_HIGH, ANCHOR_BIN_LOW, MIN_BIN_SAMPLES, N_DECILES, TRIM_PCT


@dataclass
class DecileBinning:
    edges: np.ndarray       # frozen decile boundaries (from the in-sample fit)
    labels: pd.Series       # 1..N_DECILES per event, NaN if in an anchor bin
    anchor_low: pd.Series   # bool: RVOL < ANCHOR_BIN_LOW
    anchor_high: pd.Series  # bool: RVOL > ANCHOR_BIN_HIGH


def fit_decile_edges(rvol: pd.Series, n_deciles: int = N_DECILES) -> np.ndarray:
    """Compute decile boundaries on the (in-sample) event population. These
    edges must be FROZEN and reused verbatim for OOS/bootstrap resamples
    (v1.3/E-05) — never refit per resample."""
    quantiles = np.linspace(0, 1, n_deciles + 1)
    return rvol.quantile(quantiles).to_numpy()


def assign_deciles(rvol: pd.Series, edges: np.ndarray) -> DecileBinning:
    anchor_low = rvol < ANCHOR_BIN_LOW
    anchor_high = rvol > ANCHOR_BIN_HIGH

    labels = pd.Series(
        np.digitize(rvol.to_numpy(), edges[1:-1], right=True) + 1,
        index=rvol.index,
    ).astype(float)
    # F-04: anchor-bin members overlap decile membership (RVOL>4.0 subset of
    # decile 10, etc.) and must NOT double-count into Step 3's isotonic fit.
    labels[anchor_low | anchor_high] = np.nan

    return DecileBinning(edges=edges, labels=labels, anchor_low=anchor_low, anchor_high=anchor_high)


def trimmed_mean(values: pd.Series, trim_pct: float = TRIM_PCT) -> float:
    arr = np.sort(values.dropna().to_numpy())
    n = len(arr)
    if n == 0:
        return np.nan
    k = int(np.floor(n * trim_pct))
    trimmed = arr[k : n - k] if n - 2 * k > 0 else arr
    return float(np.mean(trimmed))


def beta_binomial_shrunk_win_rate(wins: pd.Series, prior_rate: float, n0: float = 100.0) -> float:
    """Win-rate shrinkage: prior = overall sample win rate, equivalent prior
    sample size n0=100. Only applies to win rate, NOT to R_fwd (spec Step 2)."""
    n = wins.count()
    k = wins.sum()
    return (k + prior_rate * n0) / (n + n0)


def bin_summary(
    df: pd.DataFrame,
    bin_labels: pd.Series,
    value_col: str,
    win_col: str,
    overall_win_rate: float,
    min_bin_samples: int = MIN_BIN_SAMPLES,
) -> pd.DataFrame:
    """Per-bin: n, trimmed mean, median, shrunk win rate. Bins under
    min_bin_samples are flagged for merge with an adjacent bin (caller's
    responsibility to actually merge, since adjacency depends on bin ordering
    context that varies between deciles and anchor bins)."""

    rows = []
    for label, group_idx in bin_labels.dropna().groupby(bin_labels.dropna()).groups.items():
        group = df.loc[group_idx]
        n = len(group)
        rows.append(
            {
                "bin": label,
                "n": n,
                "trimmed_mean": trimmed_mean(group[value_col]),
                "median": group[value_col].median(),
                "win_rate_shrunk": beta_binomial_shrunk_win_rate(group[win_col], overall_win_rate),
                "below_min_samples": n < min_bin_samples,
            }
        )
    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
