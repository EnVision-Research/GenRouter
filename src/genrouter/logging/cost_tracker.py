from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class CostEvent:
    primitive: str
    cost: float = 0.0
    latency: float = 0.0


class CostTracker:
    def __init__(self) -> None:
        self.events: list[CostEvent] = []

    def record(self, primitive: str, cost: float = 0.0, latency: float = 0.0) -> None:
        self.events.append(CostEvent(primitive=primitive, cost=float(cost), latency=float(latency)))

    @property
    def total_cost(self) -> float:
        return sum(item.cost for item in self.events)

    @property
    def total_latency(self) -> float:
        return sum(item.latency for item in self.events)

    def by_primitive(self) -> dict[str, dict[str, float]]:
        grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"cost": 0.0, "latency": 0.0})
        for event in self.events:
            grouped[event.primitive]["cost"] += event.cost
            grouped[event.primitive]["latency"] += event.latency
        return dict(grouped)
