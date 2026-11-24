"""Synthetic EV service-parts network.

One central distribution center (DC) replenishes six regional service centers every week.
Parts are generated from 20 base part types x 2 vehicle platforms = 40 SKUs, spanning
cheap fast-moving consumables to expensive, slow-moving, vehicle-down-critical HV components.

All numbers are simulated for a portfolio project; they are plausible in scale but not real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Relative demand scale per service center (bigger metro areas see more repairs).
CENTERS: dict[str, float] = {
    "Los Angeles": 1.4,
    "SF Bay Area": 1.3,
    "Seattle": 0.8,
    "Phoenix": 0.7,
    "Dallas": 1.0,
    "Chicago": 0.9,
}

# name, category, unit cost (USD), baseline weekly demand per center, vehicle-down critical?
BASE_PARTS: list[tuple[str, str, float, float, bool]] = [
    ("Brake pad set", "Wear", 120, 6.0, False),
    ("Wiper blade kit", "Wear", 35, 8.0, False),
    ("Cabin air filter", "Wear", 40, 7.0, False),
    ("12V battery", "Electrical", 180, 2.5, True),
    ("Tire pressure sensor", "Electrical", 65, 3.0, False),
    ("Front control arm", "Suspension", 260, 1.2, True),
    ("Rear toe link", "Suspension", 140, 0.8, False),
    ("Wheel bearing hub", "Suspension", 310, 0.6, True),
    ("Door handle assembly", "Body", 150, 1.5, False),
    ("Side mirror assembly", "Body", 320, 0.7, False),
    ("Headlamp assembly", "Body", 900, 0.4, False),
    ("Windshield", "Body", 1100, 0.5, False),
    ("Charge port assembly", "HV", 600, 0.30, True),
    ("HV contactor", "HV", 450, 0.20, True),
    ("Coolant pump", "Thermal", 380, 0.35, True),
    ("Heat pump valve", "Thermal", 520, 0.15, True),
    ("Media control unit", "Electronics", 1500, 0.10, False),
    ("Body control module", "Electronics", 800, 0.12, True),
    ("HV battery module", "HV", 3000, 0.04, True),
    ("Drive unit", "Powertrain", 4500, 0.03, True),
]

PLATFORMS: dict[str, float] = {"Platform A": 0.6, "Platform B": 0.4}

ANNUAL_HOLDING_RATE = 0.25  # cost of capital, storage, obsolescence
REPLENISHMENT_LEAD_TIME_WEEKS = 1  # DC -> service center truck
REVIEW_PERIOD_WEEKS = 1  # centers reorder weekly


@dataclass(frozen=True)
class Network:
    parts: pd.DataFrame  # one row per SKU
    demand: pd.DataFrame  # one row per (SKU, center) with weekly demand rate

    @property
    def n_items(self) -> int:
        return len(self.demand)


def build_network(seed: int = 7) -> Network:
    """Create the SKU master and per-center weekly Poisson demand rates."""
    rng = np.random.default_rng(seed)

    part_rows = []
    for base_name, category, cost, base_rate, critical in BASE_PARTS:
        for platform, share in PLATFORMS.items():
            # Platform B parts cost a bit more; small random cost noise for realism.
            unit_cost = cost * (1.08 if platform == "Platform B" else 1.0) * rng.uniform(0.95, 1.05)
            part_rows.append(
                {
                    "sku": f"{base_name} ({platform})",
                    "base_part": base_name,
                    "platform": platform,
                    "category": category,
                    "unit_cost": round(unit_cost, 2),
                    "critical": critical,
                    "base_rate": base_rate * share,
                }
            )
    parts = pd.DataFrame(part_rows)

    demand_rows = []
    for part in parts.itertuples():
        for center, scale in CENTERS.items():
            # Center-specific mix noise so the optimizer has something to exploit.
            rate = part.base_rate * scale * rng.lognormal(mean=0.0, sigma=0.25)
            demand_rows.append(
                {
                    "sku": part.sku,
                    "center": center,
                    "category": part.category,
                    "unit_cost": part.unit_cost,
                    "critical": part.critical,
                    "weekly_rate": round(rate, 4),
                }
            )
    demand = pd.DataFrame(demand_rows)
    return Network(parts=parts.drop(columns="base_rate"), demand=demand)


def weekly_holding_cost(unit_cost: float | np.ndarray) -> float | np.ndarray:
    return unit_cost * ANNUAL_HOLDING_RATE / 52
