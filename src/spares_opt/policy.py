"""Inventory policy math for periodic-review, order-up-to (base-stock) control with Poisson demand.

Each week a service center orders exactly what it used, so its inventory position returns to S.
An order placed at the end of week t arrives at the start of week t + L + 1, so the stock on hand
must cover demand over the protection period L + R (lead time plus review period).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .data import REPLENISHMENT_LEAD_TIME_WEEKS, REVIEW_PERIOD_WEEKS


def expected_backorders(stock: int | np.ndarray, mean: float) -> float | np.ndarray:
    """E[(D - S)^+] for D ~ Poisson(mean), computed in closed form.

    E[(D-S)^+] = mean * P(D >= S) - S * P(D >= S + 1)
    """
    s = np.asarray(stock, dtype=float)
    return mean * poisson.sf(s - 1, mean) - s * poisson.sf(s, mean)


def expected_short_per_week(stock: int | np.ndarray, weekly_rate: float) -> float | np.ndarray:
    """Expected units demanded but not filled from shelf in one review period.

    For order-up-to S with lead time L and review period R, units short per period equal
    EBO(S; demand over L + R) - EBO(S; demand over L).
    """
    lt = REPLENISHMENT_LEAD_TIME_WEEKS
    r = REVIEW_PERIOD_WEEKS
    return expected_backorders(stock, weekly_rate * (lt + r)) - expected_backorders(stock, weekly_rate * lt)


def fill_rate(stock: int | np.ndarray, weekly_rate: float) -> float | np.ndarray:
    """Share of demand filled immediately from shelf stock (Type 2 service level)."""
    if weekly_rate <= 0:
        return np.ones_like(np.asarray(stock, dtype=float))
    return 1.0 - expected_short_per_week(stock, weekly_rate) / (weekly_rate * REVIEW_PERIOD_WEEKS)


def stock_for_fill_rate(target: float, weekly_rate: float, max_stock: int = 500) -> int:
    """Smallest base-stock level that meets a fill-rate target."""
    levels = np.arange(max_stock + 1)
    ok = np.nonzero(fill_rate(levels, weekly_rate) >= target)[0]
    if len(ok) == 0:
        raise ValueError(f"Target {target} not reachable within {max_stock} units")
    return int(ok[0])


def stock_grid(weekly_rate: float, coverage: float = 0.9999) -> np.ndarray:
    """Candidate base-stock levels from 0 up to a high quantile of protection-period demand."""
    mean = weekly_rate * (REPLENISHMENT_LEAD_TIME_WEEKS + REVIEW_PERIOD_WEEKS)
    upper = int(poisson.ppf(coverage, mean)) + 1
    return np.arange(max(upper, 2) + 1)


def abc_xyz(demand: pd.DataFrame) -> pd.DataFrame:
    """Classic ABC (annual spend) x XYZ (demand intermittency) segmentation at SKU level.

    ABC: A = top 80% of annual spend, B = next 15%, C = last 5%.
    XYZ: coefficient of variation of weekly Poisson demand (1 / sqrt(rate));
         X < 0.5 (steady), Y < 1.0, Z >= 1.0 (intermittent).
    """
    sku = (
        demand.assign(annual_spend=lambda d: d.weekly_rate * 52 * d.unit_cost)
        .groupby(["sku", "category", "unit_cost", "critical"], as_index=False)
        .agg(weekly_rate=("weekly_rate", "sum"), annual_spend=("annual_spend", "sum"))
        .sort_values("annual_spend", ascending=False)
        .reset_index(drop=True)
    )
    cum_share = sku.annual_spend.cumsum() / sku.annual_spend.sum()
    # Classify by the cumulative share *before* each SKU so the SKU that crosses 80% is still A.
    prior_share = cum_share.shift(fill_value=0.0)
    sku["abc"] = np.select([prior_share < 0.80, prior_share < 0.95], ["A", "B"], default="C")
    cv = 1.0 / np.sqrt(sku.weekly_rate.clip(lower=1e-9))
    sku["xyz"] = np.select([cv < 0.5, cv < 1.0], ["X", "Y"], default="Z")
    sku["segment"] = sku.abc + sku.xyz
    return sku
