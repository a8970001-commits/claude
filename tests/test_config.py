import pytest

from r_ivu01.config import tw_roundtrip_cost, us_roundtrip_cost


def test_tw_roundtrip_cost_matches_spec_e07():
    # 0.1425% * 0.6 * 2 + 0.3% + 0.15% = 0.621%  (v1.3/F-07; tax is NOT discountable)
    assert tw_roundtrip_cost() == pytest.approx(0.00621, abs=1e-9)


def test_us_roundtrip_cost_matches_spec_e07():
    assert us_roundtrip_cost() == pytest.approx(0.0020, abs=1e-9)
