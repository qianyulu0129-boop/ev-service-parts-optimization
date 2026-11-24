"""Run the full analysis: segmentation, optimization, efficient frontier, and disruption stress test.

Usage:
    python run.py            # full run (about 1 minute)
    python run.py --quick    # fewer simulation runs and frontier points, for CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spares_opt.data import build_network  # noqa: E402
from spares_opt.optimize import (  # noqa: E402
    efficient_frontier,
    min_budget_for_service,
    optimize_allocation,
    summarize,
    uniform_frontier,
    uniform_target_policy,
)
from spares_opt.policy import abc_xyz  # noqa: E402
from spares_opt.simulate import build_scenarios, stress_test  # noqa: E402

RESULTS = ROOT / "results"

# Chart styling (validated categorical palette, recessive grid and axes)
SURFACE = "#fcfcfb"
TEXT = "#1f1f1e"
MUTED = "#898781"
GRID = "#e1e0d9"
COLORS = {
    "Uniform 95% target": "#2a78d6",
    "Optimized, same budget": "#eb6834",
    "Optimized, lean": "#1baf7a",
}


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_frontier(opt: pd.DataFrame, uni: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    style_axes(ax)
    ax.plot(uni.investment / 1e3, uni.fill_rate * 100, color=COLORS["Uniform 95% target"], lw=2, marker="o", ms=6,
            label="Uniform fill-rate target (same target for every SKU)")
    ax.plot(opt.investment / 1e3, opt.fill_rate * 100, color=COLORS["Optimized, same budget"], lw=2, marker="o", ms=6,
            label="Optimized allocation (MILP)")
    ax.text(uni.investment.iloc[-1] / 1e3, uni.fill_rate.iloc[-1] * 100 - 1.2, "Uniform target", color=TEXT,
            fontsize=9, ha="right")
    ax.text(opt.investment.iloc[len(opt) // 2] / 1e3, opt.fill_rate.iloc[len(opt) // 2] * 100 + 0.8, "Optimized",
            color=TEXT, fontsize=9, ha="right")
    ax.set_xlabel("Inventory investment across 6 service centers ($K)", color=MUTED, fontsize=10)
    ax.set_ylabel("Fill rate from shelf (%)", color=MUTED, fontsize=10)
    ax.set_title("Inventory investment vs. service level", color=TEXT, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_stress(stress: pd.DataFrame, path: Path) -> None:
    scenarios = list(dict.fromkeys(stress.scenario))
    policies = list(COLORS)
    y = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    for j, policy in enumerate(policies):
        vals = stress[stress.policy == policy].set_index("scenario").loc[scenarios, "critical_fill_rate"] * 100
        pos = y + (j - 1) * 0.24
        ax.scatter(vals, pos, s=70, color=COLORS[policy], edgecolor=SURFACE, linewidth=2, zorder=3, label=policy)
        for p, v in zip(pos, vals):
            ax.text(v + 0.25, p, f"{v:.1f}%", va="center", fontsize=8, color=TEXT)
    for i in y:
        ax.axhline(i, color=GRID, linewidth=0.8, zorder=1)
    ax.set_yticks(y, scenarios)
    ax.invert_yaxis()
    lo = np.floor(stress.critical_fill_rate.min() * 100) - 1
    ax.set_xlim(lo, 100.8)
    ax.set_xlabel("Critical-part fill rate (%), mean of simulated years", color=MUTED, fontsize=10)
    ax.set_title("Disruption stress test: vehicle-down parts available from shelf", color=TEXT, fontsize=12,
                 loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.22))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(quick: bool = False) -> dict:
    RESULTS.mkdir(exist_ok=True)
    net = build_network()
    demand = net.demand

    # 1. Segmentation
    seg = abc_xyz(demand)
    seg.to_csv(RESULTS / "abc_xyz_segmentation.csv", index=False)

    # 2. Baseline policy: same 95% fill-rate target for every SKU at every center
    uniform = uniform_target_policy(demand, 0.95)
    u = summarize(uniform)

    # 3. Optimized allocation at the same budget
    same_budget = optimize_allocation(demand, budget=u["investment"])
    same_budget.stock.to_csv(RESULTS / "allocation_optimized_same_budget.csv", index=False)

    # 4. Cheapest optimized allocation that matches the baseline's overall AND critical fill rates
    lean = min_budget_for_service(
        demand, u["fill_rate"], u["critical_fill_rate"], lo=0.2 * u["investment"], hi=u["investment"]
    )
    lean.stock.to_csv(RESULTS / "allocation_optimized_lean.csv", index=False)

    # 5. Efficient frontier
    n_points = 5 if quick else 12
    budgets = list(np.linspace(0.15, 1.3, n_points) * u["investment"])
    frontier = efficient_frontier(demand, budgets)
    frontier.to_csv(RESULTS / "frontier_optimized.csv", index=False)
    uni_frontier = uniform_frontier(demand, [0.80, 0.85, 0.90, 0.95, 0.97, 0.99])
    uni_frontier.to_csv(RESULTS / "frontier_uniform.csv", index=False)
    plot_frontier(frontier, uni_frontier, RESULTS / "frontier.png")

    # 6. Disruption stress test (Monte Carlo)
    policies = {
        "Uniform 95% target": uniform.base_stock.to_numpy(),
        "Optimized, same budget": same_budget.stock.base_stock.to_numpy(),
        "Optimized, lean": lean.stock.base_stock.to_numpy(),
    }
    stress = stress_test(demand, policies, n_runs=100 if quick else 500)
    stress.to_csv(RESULTS / "stress_test.csv", index=False)
    plot_stress(stress, RESULTS / "stress_test.png")

    # 7. Where the optimizer moved the money
    by_cat = (
        pd.concat(
            [
                uniform.assign(policy="Uniform 95% target"),
                same_budget.stock.assign(policy="Optimized, same budget"),
                lean.stock.assign(policy="Optimized, lean"),
            ]
        )
        .assign(investment=lambda d: d.unit_cost * d.base_stock)
        .pivot_table(index="category", columns="policy", values="investment", aggfunc="sum")
        .round(0)
    )
    by_cat.to_csv(RESULTS / "investment_by_category.csv")

    s = same_budget.summary()
    lean_s = lean.summary()
    summary = {
        "n_skus": int(net.parts.shape[0]),
        "n_service_centers": int(demand.center.nunique()),
        "n_stocking_decisions": int(net.n_items),
        "uniform_95": u,
        "optimized_same_budget": s,
        "optimized_lean": lean_s,
        "lean_investment_saving_pct": 1 - lean_s["investment"] / u["investment"],
        "lean_annual_holding_saving": u["annual_holding_cost"] - lean_s["annual_holding_cost"],
        "segments": seg.segment.value_counts().to_dict(),
        "scenarios": {sc.name: sc.description for sc in build_scenarios(demand)},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    out = main(quick=args.quick)
    u, s, lean = out["uniform_95"], out["optimized_same_budget"], out["optimized_lean"]
    print(f"Stocking decisions: {out['n_stocking_decisions']} ({out['n_skus']} SKUs x {out['n_service_centers']} centers)")
    print(f"Uniform 95% target : ${u['investment']:,.0f}  fill {u['fill_rate']:.1%}  critical {u['critical_fill_rate']:.1%}")
    print(f"Optimized, same $  : ${s['investment']:,.0f}  fill {s['fill_rate']:.1%}  critical {s['critical_fill_rate']:.1%}")
    print(f"Optimized, lean    : ${lean['investment']:,.0f}  fill {lean['fill_rate']:.1%}  critical "
          f"{lean['critical_fill_rate']:.1%}  (saves {out['lean_investment_saving_pct']:.0%})")
    print(f"Results written to {RESULTS}")
