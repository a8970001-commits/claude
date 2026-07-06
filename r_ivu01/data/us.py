"""US data loader — 裁決B: yfinance. Enforces the three 軍規 from spec 2.2."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from r_ivu01.config import US_SPLIT_EXCLUDE_DAYS


@dataclass
class USUniverseData:
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    split_events: dict[str, pd.DatetimeIndex]  # ticker -> split dates
    survivorship_biased: bool = True  # yfinance has no historical index constituents


def download_us_universe(tickers: list[str], start: str, end: str) -> USUniverseData:
    """军规1: auto_adjust=True is written explicitly on every call (never rely
    on the yfinance default, to guard against version drift). As of yfinance
    0.2.28+, this means OHLC is split/dividend adjusted, there is no
    'Adj Close' column, and Volume is inverse-adjusted for splits.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # LOCKED, do not omit or set False (spec 2.2 军规1)
        group_by="ticker",
        progress=False,
    )

    close = pd.DataFrame({t: raw[t]["Close"] for t in tickers})
    high = pd.DataFrame({t: raw[t]["High"] for t in tickers})
    low = pd.DataFrame({t: raw[t]["Low"] for t in tickers})
    volume = pd.DataFrame({t: raw[t]["Volume"] for t in tickers})

    split_events = {t: _fetch_split_dates(t) for t in tickers}

    # 军规3: survivorship bias caveat is mandatory and non-negotiable — US
    # results are directional corroboration only, never a standalone basis
    # for adoption (see spec 2.2 and G5).
    return USUniverseData(
        close=close,
        high=high,
        low=low,
        volume=volume,
        split_events=split_events,
        survivorship_biased=True,
    )


def _fetch_split_dates(ticker: str) -> pd.DatetimeIndex:
    actions = yf.Ticker(ticker).splits
    return pd.DatetimeIndex(actions.index) if actions is not None else pd.DatetimeIndex([])


def split_exclusion_window(split_dates: pd.DatetimeIndex, date_index: pd.DatetimeIndex) -> pd.Series:
    excluded = pd.Series(False, index=date_index)
    for split_date in split_dates:
        pos = date_index.searchsorted(split_date)
        start = max(0, pos - US_SPLIT_EXCLUDE_DAYS)
        end = min(len(date_index), pos + US_SPLIT_EXCLUDE_DAYS + 1)
        excluded.iloc[start:end] = True
    return excluded
