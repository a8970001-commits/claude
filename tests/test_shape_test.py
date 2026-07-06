import numpy as np

from r_ivu01.stats.shape_test import (
    dual_wing_floor,
    fit_unimodal_isotonic,
    peak_is_internal,
    resample_supports_h1,
)


def test_true_inverted_u_is_supported():
    # clean inverted-U peaking at decile 5 (0-indexed 4), well above the
    # 0.10 ATR dual-wing floor
    y = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.0, 0.7, 0.4, 0.1, -0.1])
    assert resample_supports_h1(y) is True


def test_monotone_increasing_is_not_supported():
    # F-02's core regression test: the OLD gate would have nested-fit this
    # to "success" every time. The new gate must reject it (peak at boundary).
    y = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    fit = fit_unimodal_isotonic(y)
    assert not peak_is_internal(fit.peak_index, len(y))
    assert resample_supports_h1(y) is False


def test_flat_noise_is_not_supported():
    rng = np.random.default_rng(42)
    y = rng.normal(0, 0.01, 10)
    assert resample_supports_h1(y) is False


def test_shallow_dip_fails_dual_wing_floor():
    # peak is internal but the wings barely dip - below delta=0.10
    y = np.array([0.5, 0.52, 0.54, 0.56, 0.58, 0.60, 0.58, 0.56, 0.54, 0.52])
    fit = fit_unimodal_isotonic(y)
    assert peak_is_internal(fit.peak_index, len(y))
    assert not dual_wing_floor(fit)
