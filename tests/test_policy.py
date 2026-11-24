import numpy as np
import pytest
from scipy.stats import poisson

from spares_opt.data import build_network
from spares_opt.policy import abc_xyz, expected_backorders, fill_rate, stock_for_fill_rate


@pytest.mark.parametrize("mean", [0.05, 1.5, 12.0])
@pytest.mark.parametrize("stock", [0, 1, 3, 20])
def test_expected_backorders_matches_brute_force(mean, stock):
    x = np.arange(0, 400)
    brute = ((x - stock).clip(0) * poisson.pmf(x, mean)).sum()
    assert expected_backorders(stock, mean) == pytest.approx(brute, abs=1e-9)


def test_fill_rate_is_zero_without_stock_and_increases_with_stock():
    rates = fill_rate(np.arange(10), 2.0)
    assert rates[0] == pytest.approx(0.0)
    assert np.all(np.diff(rates) > 0)
    assert rates[-1] < 1.0


def test_stock_for_fill_rate_is_minimal():
    s = stock_for_fill_rate(0.95, 6.0)
    assert fill_rate(s, 6.0) >= 0.95
    assert fill_rate(s - 1, 6.0) < 0.95


def test_abc_xyz_covers_every_sku_and_ranks_by_spend():
    seg = abc_xyz(build_network().demand)
    assert len(seg) == 40
    assert set(seg.abc) <= {"A", "B", "C"}
    assert seg.annual_spend.is_monotonic_decreasing
    # A items carry most of the spend
    a_share = seg.loc[seg.abc == "A", "annual_spend"].sum() / seg.annual_spend.sum()
    assert a_share >= 0.80
