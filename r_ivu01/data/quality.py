"""Data quality flagging — 2.2 军规2 (yfinance health check)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ANOMALY_DAILY_MOVE_THRESHOLD = 0.50  # |return| > 50% -> flag for manual review


def flag_extreme_moves(close: pd.DataFrame) -> pd.DataFrame:
    """Returns a long-format DataFrame of (date, ticker, pct_change) rows where
    |daily return| exceeds the threshold, destined for data_quality_flags.csv.

    Uses np.where on the boolean mask rather than `df[mask].stack()`: boolean
    DataFrame indexing preserves the original shape (non-matching cells become
    NaN, not dropped), and pandas' `.stack()` default `dropna` behavior has
    changed across versions -- np.where sidesteps both traps.
    """

    pct_change = close.pct_change()
    mask = (pct_change.abs() > ANOMALY_DAILY_MOVE_THRESHOLD).to_numpy()
    rows, cols = np.where(mask)
    values = pct_change.to_numpy()[rows, cols]
    return pd.DataFrame(
        {
            "date": pct_change.index[rows],
            "ticker": pct_change.columns[cols],
            "pct_change": values,
        }
    )


def write_quality_flags_csv(close: pd.DataFrame, path: str) -> pd.DataFrame:
    flagged = flag_extreme_moves(close)
    flagged.to_csv(path, index=False)
    return flagged
