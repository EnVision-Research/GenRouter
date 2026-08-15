from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from genrouter.schemas import PlanEstimate, TaskSignature
from genrouter.workflows.signature import HYBRID_NEED_FIELDS, requires_hybrid


def assign_bucket(signature: TaskSignature | dict[str, Any]) -> str:
    value = signature if isinstance(signature, TaskSignature) else TaskSignature.from_dict(signature)
    data = value.to_dict()

    def high(key: str) -> bool:
        return data[key] >= 3

    if requires_hybrid(value):
        return "hybrid"
    for field in HYBRID_NEED_FIELDS:
        if high(field):
            return field
    if high("rewrite"):
        return "rewrite"
    if max(data.values()) < 2:
        return "simple"
    return "general"


class RouteMemoryBank:
    """Bucket-level plan statistics distilled from trajectory records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def memories(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def write(self, memories: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for item in memories:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    def estimate(self, signature: TaskSignature, workflow: str, generator: str) -> dict[str, Any] | None:
        bucket = assign_bucket(signature)
        plan_key = f"{workflow}+{generator}"
        for memory in self.memories():
            if memory.get("bucket") != bucket:
                continue
            stats = memory.get("plan_stats", {}).get(plan_key)
            if not isinstance(stats, dict):
                continue
            score = float(stats.get("mean_score", 0.0) or 0.0)
            cost = float(stats.get("mean_cost", 0.0) or 0.0)
            latency = float(stats.get("mean_latency", 0.0) or 0.0)
            utility = float(stats.get("mean_utility", score) or 0.0)
            return {
                "score": score,
                "cost": cost,
                "latency": latency,
                "utility": utility,
                "source": "route_memory",
                "bucket": bucket,
                "plan_key": plan_key,
                "num_records": int(stats.get("n", 0) or 0),
                "std_score": float(stats.get("std_score", 0.0) or 0.0),
                "is_pareto": plan_key in memory.get("pareto_plans", []),
                "default_best_plan": memory.get("default_best_plan", ""),
                "fast_plan": memory.get("fast_plan", ""),
                "balanced_plan": memory.get("balanced_plan", ""),
            }
        return None

    @classmethod
    def distill(
        cls,
        records: list[dict[str, Any]],
        lambda_c: float = 0.0,
        lambda_l: float = 0.0,
    ) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, list[dict[str, float]]]] = {}
        for record in records:
            signature = TaskSignature.from_dict(record["input"]["task_signature"])
            plan = record["plan"]
            workflow = str(plan["workflow"])
            generator = str(plan["generator"])
            if not workflow or not generator:
                continue
            metrics = record["metrics"]
            execution = record["execution"]
            score = float(metrics["score"] or 0.0)
            cost = float(execution["cost"] or 0.0)
            latency = float(execution["latency_seconds"] or 0.0)
            utility = float(metrics.get("utility", score - float(lambda_c) * cost - float(lambda_l) * latency) or 0.0)
            bucket = assign_bucket(signature)
            plan_key = f"{workflow}+{generator}"
            groups.setdefault(bucket, {}).setdefault(plan_key, []).append(
                {
                    "score": score,
                    "cost": cost,
                    "latency": latency,
                    "utility": utility,
                }
            )

        route_memory: list[dict[str, Any]] = []
        for bucket in sorted(groups):
            plan_stats = {
                plan_key: _summarize_results(results)
                for plan_key, results in sorted(groups[bucket].items())
            }
            pareto_plans = _pareto_plan_keys(plan_stats)
            best_plan = max(plan_stats, key=lambda key: plan_stats[key]["mean_utility"])
            fast_plan = min(plan_stats, key=lambda key: plan_stats[key]["mean_latency"])
            balanced_plan = max(pareto_plans or plan_stats.keys(), key=lambda key: plan_stats[key]["mean_utility"])
            route_memory.append(
                {
                    "bucket": bucket,
                    "trigger": _bucket_trigger(bucket),
                    "plan_stats": plan_stats,
                    "pareto_plans": pareto_plans,
                    "default_best_plan": best_plan,
                    "fast_plan": fast_plan,
                    "balanced_plan": balanced_plan,
                }
            )
        return route_memory


def _summarize_results(results: list[dict[str, float]]) -> dict[str, float | int]:
    scores = [item["score"] for item in results]
    return {
        "n": len(results),
        "mean_score": _mean(scores),
        "mean_cost": _mean([item["cost"] for item in results]),
        "mean_latency": _mean([item["latency"] for item in results]),
        "mean_utility": _mean([item["utility"] for item in results]),
        "std_score": _std(scores),
    }


def _pareto_plan_keys(plan_stats: dict[str, dict[str, Any]]) -> list[str]:
    estimates = [
        PlanEstimate(
            workflow=plan_key.split("+", 1)[0],
            generator=plan_key.split("+", 1)[1] if "+" in plan_key else "",
            execution_config={},
            estimated_score=float(stats["mean_score"]),
            estimated_cost=float(stats["mean_cost"]),
            estimated_latency=float(stats["mean_latency"]),
            estimated_utility=float(stats["mean_utility"]),
        )
        for plan_key, stats in plan_stats.items()
    ]
    kept: list[str] = []
    for current_key, current in zip(plan_stats.keys(), estimates):
        dominated = False
        for other in estimates:
            if other is current:
                continue
            better_or_equal = (
                other.estimated_score >= current.estimated_score
                and other.estimated_cost <= current.estimated_cost
                and other.estimated_latency <= current.estimated_latency
            )
            strictly_better = (
                other.estimated_score > current.estimated_score
                or other.estimated_cost < current.estimated_cost
                or other.estimated_latency < current.estimated_latency
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(current_key)
    return kept


def _bucket_trigger(bucket: str) -> str:
    triggers = {
        "hybrid": "at least two primitive needs >= 3",
        "search_text": "search_text >= 3",
        "search_image": "search_image >= 3",
        "reason": "reason >= 3",
        "skill": "skill >= 3",
        "verify_refine": "verify_refine >= 3",
        "code_sketch": "code_sketch >= 3",
        "rewrite": "rewrite >= 3",
        "simple": "max(signature) <= 1",
        "general": "no primary high-need bucket",
    }
    return triggers.get(bucket, "")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
