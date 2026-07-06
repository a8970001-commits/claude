"""Section 5 — falsification gates G1-G5.

Any single gate failure -> report the failure mode as-is and archive
(spec: "禁止回頭調整第3節定義重跑"). Do not loosen thresholds to pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from r_ivu01.config import G1_SUPPORT_RATE_THRESHOLD, G2_HALF_RATIO_RANGE
from r_ivu01.stats.bootstrap import cluster_bootstrap_resample
from r_ivu01.stats.shape_test import resample_supports_h1


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict


def g1_shape_gate(
    events: pd.DataFrame,
    cluster_col: str,
    decile_col: str,
    value_col: str,
    n_bins: int,
    n_resamples: int,
    random_state: int | None = None,
) -> GateResult:
    """v1.3/F-02: cluster-bootstrap support rate for peak-internality +
    dual-wing floor, evaluated on the primary combination only (TW x h=5 x
    trimmed-mean R_fwd, E-06)."""

    def statistic(sample: pd.DataFrame) -> bool:
        bin_means = (
            sample.dropna(subset=[decile_col])
            .groupby(decile_col)[value_col]
            .mean()
            .reindex(range(1, n_bins + 1))
        )
        if bin_means.isna().any():
            return False  # a resample that empties a bin cannot support H1
        return resample_supports_h1(bin_means.to_numpy())

    supports = cluster_bootstrap_resample(events, cluster_col, n_resamples, statistic, random_state)
    support_rate = float(np.mean(supports))
    passed = support_rate >= G1_SUPPORT_RATE_THRESHOLD
    return GateResult("G1_shape_existence", passed, {"support_rate": support_rate})


def g2_time_stability(mu_first_half: float, mu_second_half: float) -> GateResult:
    ratio = mu_second_half / mu_first_half if mu_first_half else np.nan
    lo, hi = G2_HALF_RATIO_RANGE
    passed = bool(lo <= ratio <= hi)
    return GateResult("G2_time_stability", passed, {"ratio": ratio, "mu_first_half": mu_first_half, "mu_second_half": mu_second_half})


def g3_oos_validity(spread_bootstrap_samples: np.ndarray) -> GateResult:
    point_estimate = float(np.mean(spread_bootstrap_samples))
    ci_low, ci_high = np.percentile(spread_bootstrap_samples, [2.5, 97.5])
    passed = bool(point_estimate > 0 and ci_low > 0)
    return GateResult(
        "G3_oos_validity",
        passed,
        {"point_estimate": point_estimate, "ci_low": float(ci_low), "ci_high": float(ci_high)},
    )


def g4_economic_significance(pct_returns_high_weight_group: np.ndarray, roundtrip_cost: float) -> GateResult:
    """Judged on LAYER 2 (ACE realized exits), not layer 1 (spec Section 5).
    `pct_returns_high_weight_group` = (exit_price - entry_price)/entry_price
    for the high-weight (w >= 0.7) group; NOT the ATR-normalized realized R —
    fees are a % of notional, not a multiple of ATR."""

    net_ev = float(np.mean(pct_returns_high_weight_group)) - roundtrip_cost
    passed = net_ev > 0
    return GateResult("G4_economic_significance", passed, {"net_expected_value": net_ev, "roundtrip_cost": roundtrip_cost})


def g5_cross_market_consistency(tw_peak_direction_positive: bool, us_peak_direction_positive: bool) -> GateResult:
    """Non-binding / descriptive only (spec: 非必要條件，僅記錄)."""
    same_direction = tw_peak_direction_positive == us_peak_direction_positive
    return GateResult("G5_cross_market_consistency", same_direction, {"descriptive_only": True})
