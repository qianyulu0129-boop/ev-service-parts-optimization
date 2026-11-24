"""Monte Carlo simulation of weekly base-stock replenishment under supply and demand disruptions.

Each week at every service center:
  1. replenishment that was due this week arrives and first clears any backorders
  2. Poisson demand occurs; it is filled from shelf stock if possible, otherwise backordered
  3. the center orders exactly what was demanded (order-up-to S), which arrives L + 1 weeks later

Scenarios change demand multipliers or lead times for chosen parts and weeks, so each policy
can be stress-tested under the same random demand streams (common random numbers).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import REPLENISHMENT_LEAD_TIME_WEEKS

WARMUP_WEEKS = 8
HORIZON_WEEKS = 52


@dataclass
class Scenario:
    name: str
    description: str
    # (item mask, first week, last week exclusive, multiplier)
    demand_shocks: list[tuple[np.ndarray, int, int, float]] = field(default_factory=list)
    # (item mask, first week, last week exclusive, extra lead-time weeks for orders placed then)
    lead_time_shocks: list[tuple[np.ndarray, int, int, int]] = field(default_factory=list)
    # (first week, last week exclusive): DC closed, orders placed then ship when it reopens
    dc_outage: tuple[int, int] | None = None


def build_scenarios(demand: pd.DataFrame) -> list[Scenario]:
    long_lead_critical = demand.category.isin(["HV", "Thermal", "Powertrain"]).to_numpy()
    control_arm = demand.sku.str.startswith("Front control arm").to_numpy()
    w = WARMUP_WEEKS
    return [
        Scenario("Baseline", "Normal demand and 1-week DC replenishment"),
        Scenario(
            "Supplier delay",
            "HV, thermal and powertrain parts take +3 weeks to replenish for 12 weeks",
            lead_time_shocks=[(long_lead_critical, w + 10, w + 22, 3)],
        ),
        Scenario(
            "Recall surge",
            "Front control arm demand triples for 8 weeks after a service campaign",
            demand_shocks=[(control_arm, w + 15, w + 23, 3.0)],
        ),
        Scenario(
            "DC outage",
            "Central DC closed for 2 weeks; no replenishment ships",
            dc_outage=(w + 30, w + 32),
        ),
        Scenario(
            "Combined",
            "Supplier delay, recall surge and DC outage in the same year",
            demand_shocks=[(control_arm, w + 15, w + 23, 3.0)],
            lead_time_shocks=[(long_lead_critical, w + 10, w + 22, 3)],
            dc_outage=(w + 30, w + 32),
        ),
    ]


def simulate(
    demand: pd.DataFrame,
    base_stock: np.ndarray,
    scenario: Scenario,
    n_runs: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Run the simulation and return service metrics averaged over runs and the post-warmup horizon."""
    rng = np.random.default_rng(seed)
    n_items = len(demand)
    total_weeks = WARMUP_WEEKS + HORIZON_WEEKS
    rate = demand.weekly_rate.to_numpy()
    critical = demand.critical.to_numpy()
    s = np.asarray(base_stock, dtype=np.int64)

    # Week-by-item demand multipliers and lead times
    mult = np.ones((total_weeks, n_items))
    for mask, start, end, factor in scenario.demand_shocks:
        mult[start:end, mask] *= factor
    lead = np.full((total_weeks, n_items), REPLENISHMENT_LEAD_TIME_WEEKS, dtype=np.int64)
    for mask, start, end, extra in scenario.lead_time_shocks:
        lead[start:end, mask] += extra
    if scenario.dc_outage is not None:
        start, end = scenario.dc_outage
        for t in range(start, end):
            lead[t, :] += end - t  # wait until the DC reopens

    max_lead = int(lead.max())
    arrivals = np.zeros((n_runs, n_items, total_weeks + max_lead + 2), dtype=np.int64)
    on_hand = np.tile(s, (n_runs, 1))
    backlog = np.zeros((n_runs, n_items), dtype=np.int64)

    demanded = np.zeros(n_items)
    filled_from_shelf = np.zeros(n_items)
    critical_backlog_weeks = 0.0

    for t in range(total_weeks):
        on_hand += arrivals[:, :, t]
        cleared = np.minimum(on_hand, backlog)
        on_hand -= cleared
        backlog -= cleared

        d = rng.poisson(rate * mult[t], size=(n_runs, n_items))
        filled = np.minimum(on_hand, d)
        on_hand -= filled
        backlog += d - filled

        # Order-up-to: reorder what was demanded; arrives after lead time + 1 week
        due = t + 1 + lead[t]
        np.add.at(arrivals, (slice(None), np.arange(n_items), due), d)

        if t >= WARMUP_WEEKS:
            demanded += d.sum(axis=0)
            filled_from_shelf += filled.sum(axis=0)
            critical_backlog_weeks += backlog[:, critical].sum()

    crit_demand = demanded[critical].sum()
    return {
        "scenario": scenario.name,
        "fill_rate": float(filled_from_shelf.sum() / demanded.sum()),
        "critical_fill_rate": float(filled_from_shelf[critical].sum() / crit_demand),
        # Average number of critical parts on backorder in a given week across the network:
        # a proxy for vehicles waiting on parts.
        "avg_critical_backorders": float(critical_backlog_weeks / (n_runs * HORIZON_WEEKS)),
    }


def stress_test(
    demand: pd.DataFrame, policies: dict[str, np.ndarray], n_runs: int = 500, seed: int = 42
) -> pd.DataFrame:
    rows = []
    for scenario in build_scenarios(demand):
        for policy_name, stock in policies.items():
            result = simulate(demand, stock, scenario, n_runs=n_runs, seed=seed)
            rows.append({"policy": policy_name, **result})
    return pd.DataFrame(rows)
