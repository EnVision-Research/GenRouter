from __future__ import annotations

from genrouter.schemas import PlanEstimate


def dominates(a: PlanEstimate, b: PlanEstimate) -> bool:
    better_or_equal = (
        a.estimated_score >= b.estimated_score
        and a.estimated_cost <= b.estimated_cost
        and a.estimated_latency <= b.estimated_latency
    )
    strictly_better = (
        a.estimated_score > b.estimated_score
        or a.estimated_cost < b.estimated_cost
        or a.estimated_latency < b.estimated_latency
    )
    return better_or_equal and strictly_better


def pareto_filter(plans: list[PlanEstimate]) -> list[PlanEstimate]:
    kept: list[PlanEstimate] = []
    for plan in plans:
        if any(dominates(other, plan) for other in plans if other is not plan):
            continue
        kept.append(
            PlanEstimate(
                workflow=plan.workflow,
                generator=plan.generator,
                execution_config=plan.execution_config,
                estimated_score=plan.estimated_score,
                estimated_cost=plan.estimated_cost,
                estimated_latency=plan.estimated_latency,
                estimated_utility=plan.estimated_utility,
                pareto_status="non_dominated",
                evidence=plan.evidence,
            )
        )
    return kept
