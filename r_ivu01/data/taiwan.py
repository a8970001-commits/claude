"""TW data loader interface — 裁決A: finlab (指揮官既有管線).

finlab access/credentials are commander infrastructure, not something this
scaffold can assume. This module defines the exact contract the rest of the
pipeline depends on; wire it to the real finlab client (finlab.data.get(...))
in the execution environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from r_ivu01.config import TW_LIQUIDITY_MIN_TWD, TW_MIN_PRICE


@dataclass
class TaiwanUniverseData:
    """Long-format panel, one row per (date, ticker)."""

    close: pd.DataFrame       # index=date, columns=ticker
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    trading_value: pd.DataFrame  # close * volume, for liquidity screen
    is_delisted_history_included: bool
    disposition_flag: pd.DataFrame | None = None  # finlab 處置標記, if available
    volume_ex_adjusted: bool = False  # has finlab already restored volume across 股數變動 events?


def load_taiwan_universe(start: str, end: str) -> TaiwanUniverseData:
    """TODO(execution env): implement with finlab.data.get(...).

    Required finlab fields (per spec 2.1):
      - 收盤價 (close), 最高價 (high), 最低價 (low), 成交股數/成交量 (volume)
      - 個股市場別 (to filter to 上市普通股, excl. ETF/TDR/DR/特別股)
      - 處置股票 flag if available
      - full delisted-ticker history if available (else set
        is_delisted_history_included=False and the report must carry the
        survivorship-bias caveat verbatim, per 2.1's explicit instruction)
    """
    raise NotImplementedError(
        "Wire this to the finlab client in the execution environment. "
        "See docstring for required fields (spec 2.1)."
    )


def apply_liquidity_and_price_filters(
    data: TaiwanUniverseData,
    min_trading_value: float = TW_LIQUIDITY_MIN_TWD,
    min_price: float = TW_MIN_PRICE,
) -> pd.DataFrame:
    """Point-in-time 20-day avg trading value >= threshold, and close >= min
    price. Returns a boolean eligibility mask (date x ticker), computed from
    data available strictly at or before each date (no look-ahead)."""

    avg_trading_value_20d = data.trading_value.rolling(20).mean()
    eligible = (avg_trading_value_20d >= min_trading_value) & (data.close >= min_price)
    return eligible
