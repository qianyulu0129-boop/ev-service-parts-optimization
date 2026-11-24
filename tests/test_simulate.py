import pytest

from spares_opt.data import build_network
from spares_opt.optimize import summarize, uniform_target_policy
from spares_opt.simulate import build_scenarios, simulate


@pytest.fixture(scope="module")
def setup():
    demand = build_network().demand
    policy = uniform_target_policy(demand, 0.95)
    scenarios = {s.name: s for s in build_scenarios(demand)}
    return demand, policy, scenarios


def test_baseline_simulation_matches_analytic_fill_rate(setup):
    demand, policy, scenarios = setup
    analytic = summarize(policy)
    sim = simulate(demand, policy.base_stock.to_numpy(), scenarios["Baseline"], n_runs=300)
    assert sim["fill_rate"] == pytest.approx(analytic["fill_rate"], abs=0.003)
    assert sim["critical_fill_rate"] == pytest.approx(analytic["critical_fill_rate"], abs=0.005)


def test_disruptions_reduce_service(setup):
    demand, policy, scenarios = setup
    stock = policy.base_stock.to_numpy()
    base = simulate(demand, stock, scenarios["Baseline"], n_runs=200)
    combined = simulate(demand, stock, scenarios["Combined"], n_runs=200)
    assert combined["critical_fill_rate"] < base["critical_fill_rate"]
    assert combined["avg_critical_backorders"] > base["avg_critical_backorders"]


def test_simulation_is_reproducible(setup):
    demand, policy, scenarios = setup
    stock = policy.base_stock.to_numpy()
    a = simulate(demand, stock, scenarios["DC outage"], n_runs=50, seed=1)
    b = simulate(demand, stock, scenarios["DC outage"], n_runs=50, seed=1)
    assert a == b
