"""Budget-constrained multi-location spare-parts allocation as a mixed-integer program (PuLP + CBC).

Decision:   z[i, k] = 1 if item i (a SKU at a service center) is stocked at base-stock level k.
Objective:  minimize criticality-weighted expected units short per week across all items.
            Vehicle-down-critical parts carry a higher weight, since a missing part keeps a car off the road.
Constraints:
    - exactly one stock level per item
    - total inventory investment (unit cost x base-stock) <= budget
    - optional minimum fill rate for every critical item

Expected shortfall is convex and decreasing in stock, so the optimal solution spends each extra dollar
where it removes the most weighted shortfall: cheap, fast-moving, critical parts first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from .data import weekly_holding_cost
from .policy import expected_short_per_week, fill_rate, stock_for_fill_rate, stock_grid

CRITICAL_WEIGHT = 3.0


@dataclass
class Allocation:
    stock: pd.DataFrame  # demand rows plus chosen base_stock and per-item metrics
    status: str

    @property
    def investment(self) -> float:
        return float((self.stock.unit_cost * self.stock.base_stock).sum())

    def summary(self) -> dict[str, float]:
        return summarize(self.stock)


def summarize(stock: pd.DataFrame) -> dict[str, float]:
    """Demand-weighted service metrics for a set of base-stock levels."""
    s = stock.copy()
    s["short"] = [expected_short_per_week(b, r) for b, r in zip(s.base_stock, s.weekly_rate)]
    crit = s[s.critical]
    total_rate = s.weekly_rate.sum()
    return {
        "investment": float((s.unit_cost * s.base_stock).sum()),
        "annual_holding_cost": float((weekly_holding_cost(s.unit_cost) * s.base_stock).sum() * 52),
        "fill_rate": float(1 - s.short.sum() / total_rate),
        "critical_fill_rate": float(1 - crit.short.sum() / crit.weekly_rate.sum()),
        "weighted_short_per_week": float((s.short * np.where(s.critical, CRITICAL_WEIGHT, 1.0)).sum()),
        "critical_units_short_per_year": float(crit.short.sum() * 52),
    }


def optimize_allocation(
    demand: pd.DataFrame,
    budget: float,
    min_critical_fill: float | None = None,
    time_limit: int = 60,
) -> Allocation:
    """Solve the budget-constrained allocation MILP."""
    prob = pulp.LpProblem("spares_allocation", pulp.LpMinimize)

    z: dict[tuple[int, int], pulp.LpVariable] = {}
    objective_terms = []
    budget_terms = []

    for i, row in enumerate(demand.itertuples(index=False)):
        levels = stock_grid(row.weekly_rate)
        shorts = expected_short_per_week(levels, row.weekly_rate)
        fills = fill_rate(levels, row.weekly_rate)
        weight = CRITICAL_WEIGHT if row.critical else 1.0

        allowed = levels
        if min_critical_fill is not None and row.critical:
            allowed = levels[fills >= min_critical_fill]
            if len(allowed) == 0:
                allowed = levels[-1:]

        item_vars = []
        for k in allowed:
            var = pulp.LpVariable(f"z_{i}_{k}", cat="Binary")
            z[(i, int(k))] = var
            item_vars.append(var)
            objective_terms.append(weight * float(shorts[k]) * var)
            budget_terms.append(float(row.unit_cost) * int(k) * var)
        prob += pulp.lpSum(item_vars) == 1, f"one_level_{i}"

    prob += pulp.lpSum(objective_terms)
    prob += pulp.lpSum(budget_terms) <= budget, "budget"

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[prob.status]
    if prob.status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Allocation MILP ended with status {status}")

    chosen = np.zeros(len(demand), dtype=int)
    for (i, k), var in z.items():
        if var.value() is not None and var.value() > 0.5:
            chosen[i] = k

    stock = demand.copy()
    stock["base_stock"] = chosen
    stock["fill_rate"] = [fill_rate(b, r) for b, r in zip(stock.base_stock, stock.weekly_rate)]
    return Allocation(stock=stock, status=status)


def uniform_target_policy(demand: pd.DataFrame, target: float) -> pd.DataFrame:
    """Common baseline: give every SKU at every center the same fill-rate target."""
    stock = demand.copy()
    stock["base_stock"] = [stock_for_fill_rate(target, r) for r in stock.weekly_rate]
    stock["fill_rate"] = [fill_rate(b, r) for b, r in zip(stock.base_stock, stock.weekly_rate)]
    return stock


def efficient_frontier(demand: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    """Solve the MILP across a range of budgets to trace investment vs. service level."""
    rows = []
    for budget in budgets:
        alloc = optimize_allocation(demand, budget)
        rows.append({"budget": budget, **alloc.summary()})
    return pd.DataFrame(rows)


def uniform_frontier(demand: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    rows = []
    for t in targets:
        rows.append({"target": t, **summarize(uniform_target_policy(demand, t))})
    return pd.DataFrame(rows)


def min_budget_for_service(
    demand: pd.DataFrame,
    target_fill: float,
    target_critical_fill: float,
    lo: float,
    hi: float,
    tol: float = 500.0,
) -> Allocation:
    """Bisection on budget: cheapest optimized allocation that matches BOTH the overall and the
    critical-part fill rate of a reference policy, so savings never come from starving critical parts."""

    def meets(alloc: Allocation) -> bool:
        s = alloc.summary()
        return s["fill_rate"] >= target_fill and s["critical_fill_rate"] >= target_critical_fill

    best = optimize_allocation(demand, hi)
    if not meets(best):
        raise ValueError("Service targets not reachable at the upper budget")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        alloc = optimize_allocation(demand, mid)
        if meets(alloc):
            hi, best = mid, alloc
        else:
            lo = mid
    return best
