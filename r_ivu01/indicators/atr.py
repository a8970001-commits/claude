"""Wilder RMA-smoothed ATR (aligned with Pine's ta.atr). SMA-based ATR is banned (3.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA: seeded by an SMA of the first `period` values, then
    recursively rma[t] = rma[t-1] + (x[t] - rma[t-1]) / period.
    """
    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    if len(values) < period:
        return pd.Series(out, index=series.index)

    seed_slice = values[:period]
    if np.isnan(seed_slice).any():
        return pd.Series(out, index=series.index)

    out[period - 1] = seed_slice.mean()
    for t in range(period, len(values)):
        out[t] = out[t - 1] + (values[t] - out[t - 1]) / period
    return pd.Series(out, index=series.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = true_range(high, low, close)
    return wilder_rma(tr, period)
