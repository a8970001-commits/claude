"""ACE white line (trail_Struct_L) — layer 2 production-realism engine.

TRANSCRIBED, NOT DERIVED. Source: 奈米本金控 v5.25 SSoT-Freeze, lines
98-104, 361-395, 3333-3413, 3513, embedded verbatim in spec section 3.3.
HARD CONSTRAINT #7 forbids reconstructing this formula from first principles
— every operator below traces to a specific frozen line. If a discrepancy is
found against the TradingView reference during the F-14 replication check,
STOP and report; do not "fix" the formula here.

Known reconstruction traps (recorded in spec v1.1->v1.2 changelog, #2):
  1. stopVal_Struct_L anchors on highest(CLOSE, ...), not highest(HIGH, ...).
  2. The white line is NOT affected by aceMode (TIGHT/MELTDOWN) or CSM Exit
     Escalation — those tighten only the tactical line trail_Main_L. This
     engine deliberately does not implement CSM/FAL/aceMode (scope guard,
     HARD CONSTRAINT #5).
  3. base_atr_regime is an asymmetric sticky state machine: it snaps up
     instantly but requires 5 consecutive confirming bars to snap down.

F-10 vectorization ban: dynamicLookback is a per-bar varying window length.
ta.highest/ta.lowest reconstructions MUST take the window at each bar's own
current length — no fixed-window rolling() substitution is permitted, since
that would silently change every bar's window to today's snapshot length.
This reference implementation is therefore a bar-by-bar loop (O(n * w)).
Performance optimization (e.g. a variable-window monotonic deque) may be
substituted only if proven bar-for-bar identical to this loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from r_ivu01.config import AceParams, ACE_PARAMS
from r_ivu01.indicators.atr import atr as compute_atr


@dataclass
class AceWhiteLineResult:
    atr_value: pd.Series
    base_atr_regime: pd.Series
    dynamic_multiplier: pd.Series
    dynamic_lookback: pd.Series
    stop_val_struct_l: pd.Series
    lowest_low_c: pd.Series
    trail_struct_l: pd.Series
    ace_dir: pd.Series

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "atrValue_ACE": self.atr_value,
                "base_atr_regime": self.base_atr_regime,
                "dynamicMultiplier": self.dynamic_multiplier,
                "dynamicLookback": self.dynamic_lookback,
                "stopVal_Struct_L": self.stop_val_struct_l,
                "lowestLow_C": self.lowest_low_c,
                "trail_Struct_L": self.trail_struct_l,
                "aceDir": self.ace_dir,
            }
        )


def _rolling_window_extreme(values: np.ndarray, lookbacks: np.ndarray, t: int, mode: str) -> float:
    w = int(lookbacks[t])
    start = max(0, t - w + 1)
    window = values[start : t + 1]
    if window.size == 0 or np.isnan(window).all():
        return np.nan
    return np.nanmax(window) if mode == "max" else np.nanmin(window)


def run_ace_white_line(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    params: AceParams = ACE_PARAMS,
) -> AceWhiteLineResult:
    """Bar-by-bar state machine over a single ticker's full history (Pine `var`
    persistent semantics — NOT reset per simulated trade)."""

    n = len(close)
    idx = close.index
    close_v = close.to_numpy(dtype=float)

    atr_value = compute_atr(high, low, close, params.atr_period)
    atr_v = atr_value.to_numpy(dtype=float)

    base_regime = np.full(n, np.nan)
    down_confirm_count = np.zeros(n, dtype=int)

    normalized_atr = np.full(n, np.nan)
    highest_atr = np.full(n, np.nan)
    lowest_atr = np.full(n, np.nan)
    vol_score = np.zeros(n)
    dyn_mult = np.full(n, np.nan)
    dyn_lookback = np.full(n, np.nan)
    stop_val = np.full(n, np.nan)
    lowest_low_c = np.full(n, np.nan)
    trail = np.full(n, np.nan)
    ace_dir = np.full(n, np.nan)

    base = np.nan
    consec_calm = 0

    for t in range(n):
        a = atr_v[t]
        if np.isnan(a):
            continue

        if np.isnan(base):
            base = a  # base_atr_regime init: first non-na atrValue_ACE
            consec_calm = 0
        elif a > base * params.regime_up_ratio:
            base = a  # instant raise
            consec_calm = 0
        elif a < base * params.regime_down_ratio:
            consec_calm += 1
            if consec_calm >= params.regime_down_confirm_bars:
                base = a  # calm_confirmed: lower base
                consec_calm = 0
        else:
            consec_calm = 0

        base_regime[t] = base
        down_confirm_count[t] = consec_calm

        c = close_v[t]
        if c == 0 or np.isnan(c):
            continue
        normalized_atr[t] = base / c

        window_start = max(0, t - params.volatility_lookback + 1)
        na_window = normalized_atr[window_start : t + 1]
        highest_atr[t] = np.nanmax(na_window)
        lowest_atr[t] = np.nanmin(na_window)

        rng = highest_atr[t] - lowest_atr[t]
        vol_score[t] = (
            (normalized_atr[t] - lowest_atr[t]) / rng if rng > 0 else 0.0
        )

        dyn_mult[t] = params.min_multiplier + (
            params.max_multiplier - params.min_multiplier
        ) * vol_score[t]
        dyn_lookback[t] = round(
            max(1.0, params.max_lookback - (params.max_lookback - params.min_lookback) * vol_score[t])
        )

        stop_val[t] = (
            _rolling_window_extreme(close_v, dyn_lookback, t, "max") - a * dyn_mult[t]
        )
        lowest_low_c[t] = _rolling_window_extreme(close_v, dyn_lookback, t, "min")

        if t == 0 or np.isnan(trail[t - 1]):
            trail[t] = stop_val[t]
            ace_dir[t] = 1
            continue

        prev_trail = trail[t - 1]
        prev_dir = ace_dir[t - 1]

        if prev_dir == 1:
            cur_trail = max(prev_trail, stop_val[t])
            if c < cur_trail:
                trail[t] = cur_trail
                ace_dir[t] = -1
            else:
                trail[t] = cur_trail
                ace_dir[t] = 1
        else:
            rearm_level = lowest_low_c[t] + a * dyn_mult[t]
            if c > rearm_level:
                trail[t] = stop_val[t]
                ace_dir[t] = 1
            else:
                trail[t] = prev_trail
                ace_dir[t] = -1

    return AceWhiteLineResult(
        atr_value=pd.Series(atr_v, index=idx),
        base_atr_regime=pd.Series(base_regime, index=idx),
        dynamic_multiplier=pd.Series(dyn_mult, index=idx),
        dynamic_lookback=pd.Series(dyn_lookback, index=idx),
        stop_val_struct_l=pd.Series(stop_val, index=idx),
        lowest_low_c=pd.Series(lowest_low_c, index=idx),
        trail_struct_l=pd.Series(trail, index=idx),
        ace_dir=pd.Series(ace_dir, index=idx),
    )


@dataclass
class ExitSimResult:
    exit_index: int | None
    exit_reason: str  # "trail_flip" | "max_hold" | "excluded_ace_dir"
    exit_price: float | None


def simulate_exit(
    ace_result: AceWhiteLineResult,
    close: pd.Series,
    entry_pos: int,
    max_hold_days: int = ACE_PARAMS.max_hold_days,
) -> ExitSimResult:
    """P2 Option A (strict version, 2026-07-06 commander ruling): exit event
    = first close < trail_Struct_L after entry, or max_hold_days forced close,
    whichever comes first. Event day aceDir != 1 -> excluded (F-09: end-of-bar
    state after the bar's state machine has fully executed)."""

    ace_dir = ace_result.ace_dir.to_numpy()
    trail = ace_result.trail_struct_l.to_numpy()
    close_v = close.to_numpy(dtype=float)
    n = len(close_v)

    if np.isnan(ace_dir[entry_pos]) or ace_dir[entry_pos] != 1:
        return ExitSimResult(None, "excluded_ace_dir", None)

    last_pos = min(entry_pos + max_hold_days, n - 1)
    for t in range(entry_pos + 1, last_pos + 1):
        if not np.isnan(trail[t]) and close_v[t] < trail[t]:
            return ExitSimResult(t, "trail_flip", close_v[t])

    return ExitSimResult(last_pos, "max_hold", close_v[last_pos])
