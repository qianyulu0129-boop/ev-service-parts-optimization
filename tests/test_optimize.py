import pytest

from spares_opt.data import build_network
from spares_opt.optimize import (
    min_budget_for_service,
    optimize_allocation,
    summarize,
    uniform_target_policy,
)


@pytest.fixture(scope="module")
def demand():
    return build_network().demand


@pytest.fixture(scope="module")
def uniform(demand):
    return summarize(uniform_target_policy(demand, 0.95))


def test_budget_constraint_is_respected(demand):
    alloc = optimize_allocation(demand, budget=100_000)
    assert alloc.status == "Optimal"
    assert alloc.investment <= 100_000 + 1e-6


def test_more_budget_never_hurts(demand):
    small = optimize_allocation(demand, budget=80_000).summary()
    large = optimize_allocation(demand, budget=160_000).summary()
    assert large["weighted_short_per_week"] <= small["weighted_short_per_week"]
    assert large["fill_rate"] >= small["fill_rate"]


def test_optimized_beats_uniform_at_same_budget(demand, uniform):
    opt = optimize_allocation(demand, budget=uniform["investment"]).summary()
    assert opt["investment"] <= uniform["investment"] + 1e-6
    assert opt["fill_rate"] > uniform["fill_rate"]
    assert opt["critical_fill_rate"] > uniform["critical_fill_rate"]


def test_lean_allocation_matches_service_with_less_money(demand, uniform):
    lean = min_budget_for_service(
        demand,
        uniform["fill_rate"],
        uniform["critical_fill_rate"],
        lo=0.2 * uniform["investment"],
        hi=uniform["investment"],
        tol=5_000,
    ).summary()
    assert lean["fill_rate"] >= uniform["fill_rate"]
    assert lean["critical_fill_rate"] >= uniform["critical_fill_rate"]
    assert lean["investment"] < uniform["investment"]


def test_min_critical_fill_floor_is_enforced(demand):
    alloc = optimize_allocation(demand, budget=400_000, min_critical_fill=0.90)
    crit = alloc.stock[alloc.stock.critical]
    assert (crit.fill_rate >= 0.90).all()
