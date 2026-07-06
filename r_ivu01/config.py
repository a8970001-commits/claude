"""Locked constants for R-IVU01 v1.3-SPEC.

Every value here is fixed by the spec (奈米本金控 v5.25 SSoT-Freeze 適用範圍外,
獨立研究因子). Do not change these to "improve" fit — per HARD CONSTRAINTS #5/#8,
modifying definitions requires a new spec version, not a code edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---- 2.1 / 2.2 sample window -------------------------------------------------
SAMPLE_START = "2015-01-01"
SAMPLE_END = "2026-06-30"
IS_END = "2022-12-31"          # in-sample: 2015-2022
OOS_START = "2023-01-01"       # out-of-sample: 2023-2026

TW_LIQUIDITY_MIN_TWD = 50_000_000   # 20d avg trading value, point-in-time
TW_MIN_PRICE = 10.0

US_UNIVERSE = ("SP500", "NASDAQ100")  # current constituents; survivorship-biased

# ---- 2.3 / F-08 exclusion windows -------------------------------------------
EX_DIVIDEND_EXCLUDE_DAYS_BEFORE = 5      # cash dividend only, no share-count change
SHARE_CHANGE_EXCLUDE_DAYS_AFTER = 20     # rights issue / split / capital reduction
SHARE_CHANGE_EXCLUDE_DAYS_AFTER_RESTORED = 2  # if finlab volume already ex-adjusted
US_SPLIT_EXCLUDE_DAYS = 2                # +/- around split date

DISPOSITION_VOL_SHRINK_PROXY = 0.70  # 20d/60d avg volume shrink >70% if no flag

# F-03/E-03: global warm-up, shared by BOTH layers (H4 sample-comformity)
GLOBAL_WARMUP_BARS = 250

LIMIT_UP_MAX_INTRADAY_RANGE = 0.01  # limit-lock day: close==limit AND range<1%

# ---- 3.1 breakout event ------------------------------------------------------
BREAKOUT_LOOKBACK_HIGH = 20     # close[t] > max(high[t-20:t-1])
SMA_MID = 50
SMA_LONG = 200
DEDUP_MIN_GAP_DAYS = 10

# ---- 3.2 RVOL ----------------------------------------------------------------
RVOL_WINDOW = 20   # RVOL[t] = volume[t] / SMA(volume,20)[t-1]

# ---- 3.3 layer 1: shape identification --------------------------------------
ATR_PERIOD = 14           # Wilder RMA, denominator is ATR(14)[t-1]  (F-01)
FORWARD_WINDOWS = (1, 3, 5, 10, 20)

# ---- 3.3 layer 2: ACE white line (v5.25 SSoT-Freeze frozen text, verbatim) --
# Source: v5.25 file lines 98-104, 361-395, 3333-3413, 3513. DO NOT reconstruct
# or "improve" this from first principles (HARD CONSTRAINT #7) — these values
# and the formulas in indicators/ace.py are transcribed, not derived.
@dataclass(frozen=True)
class AceParams:
    atr_period: int = 22
    volatility_lookback: int = 100
    min_lookback: int = 50
    max_lookback: int = 150
    min_multiplier: float = 1.5
    max_multiplier: float = 3.5
    regime_up_ratio: float = 1.15    # atrValue > base*1.15 -> instant raise
    regime_down_ratio: float = 0.75  # atrValue < base*0.75 for 5 consecutive bars -> lower
    regime_down_confirm_bars: int = 5
    max_hold_days: int = 60


ACE_PARAMS = AceParams()

# ---- Step 1: binning ---------------------------------------------------------
N_DECILES = 10
ANCHOR_BIN_LOW = 0.8    # RVOL < 0.8
ANCHOR_BIN_HIGH = 4.0   # RVOL > 4.0
MIN_BIN_SAMPLES = 200
TRIM_PCT = 0.05                    # trimmed mean
BETA_BINOMIAL_N0 = 100             # win-rate shrinkage prior sample size

# ---- E-06/F-06: single primary test combination (multi-testing firewall) ---
PRIMARY_MARKET = "TW"
PRIMARY_H = 5
PRIMARY_METRIC = "trimmed_mean_R_fwd"

# ---- Step 3 / F-02: shape gate -----------------------------------------------
DELTA_ATR = 0.10          # dual-wing effect floor, ATR units, h=5
BOOTSTRAP_N_RESAMPLES = 1000
G1_SUPPORT_RATE_THRESHOLD = 0.80
PEAK_INTERNAL_DECILE_RANGE = (2, 9)   # inclusive, 1-indexed deciles

# ---- Step 4: parametric family ------------------------------------------------
AIC_TIE_THRESHOLD = 2.0

# ---- Gate thresholds ----------------------------------------------------------
G2_HALF_RATIO_RANGE = (0.7, 1.4)

# ---- G4 / E-07: real friction structure (2026-07-06 web-verified rates) ------
TW_BROKER_FEE_RATE = 0.001425      # 0.1425% per side, cap rate
TW_BROKER_FEE_DISCOUNT = 0.6       # discount applies ONLY to broker fee, not tax
TW_TRANSACTION_TAX_RATE = 0.003    # 0.3%, sell-side only, government tax, NOT discountable
TW_SLIPPAGE_ROUNDTRIP = 0.0015     # 0.15%, from 67-trade audit gap/slippage evidence
US_BROKER_FEE_ROUNDTRIP = 0.0005   # 0.05%
US_SLIPPAGE_ROUNDTRIP = 0.0015     # 0.15%, same evidence base as TW


def tw_roundtrip_cost() -> float:
    """0.1425%*0.6*2 + 0.3% + 0.15% = 0.621% (v1.3/F-07, replaces v1.2's 0.585%)."""
    return TW_BROKER_FEE_RATE * TW_BROKER_FEE_DISCOUNT * 2 + TW_TRANSACTION_TAX_RATE + TW_SLIPPAGE_ROUNDTRIP


def us_roundtrip_cost() -> float:
    return US_BROKER_FEE_ROUNDTRIP + US_SLIPPAGE_ROUNDTRIP


# ---- F-14: replication precision check ---------------------------------------
REPLICATION_SAMPLE_TICKERS = 10
REPLICATION_SAMPLE_EVENTS_PER_TICKER = 3
REPLICATION_TOLERANCE = 0.02  # +/- 2%
