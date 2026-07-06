"""Data quality flagging — 2.2 军规2 (yfinance health check)."""

from __future__ import annotations

import pandas as pd

ANOMALY_DAILY_MOVE_THRESHOLD = 0.50  # |return| > 50% -> flag for manual review


def flag_extreme_moves(close: pd.DataFrame) -> pd.DataFrame:
    """Returns a long-format DataFrame of (date, ticker, pct_change) rows where
    |daily return| exceeds the threshold, destined for data_quality_flags.csv."""

    pct_change = close.pct_change()
    mask = pct_change.abs() > ANOMALY_DAILY_MOVE_THRESHOLD
    flagged = pct_change[mask].stack()
    flagged.name = "pct_change"
    flagged.index.names = ["date", "ticker"]
    return flagged.reset_index()


def write_quality_flags_csv(close: pd.DataFrame, path: str) -> pd.DataFrame:
    flagged = flag_extreme_moves(close)
    flagged.to_csv(path, index=False)
    return flagged
