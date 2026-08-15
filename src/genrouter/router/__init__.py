"""Routing namespace."""

from genrouter.router.plans import CandidatePlan, compatible_plan, construct_candidate_plans
from genrouter.router.genrouter import GenRouter, RouteDecision
from genrouter.router.pareto import pareto_filter
from genrouter.router.route_memory import RouteMemoryBank, assign_bucket
from genrouter.router.task_signature import TaskSignatureExtractor

__all__ = [
    "CandidatePlan",
    "GenRouter",
    "RouteDecision",
    "RouteMemoryBank",
    "TaskSignatureExtractor",
    "assign_bucket",
    "compatible_plan",
    "construct_candidate_plans",
    "pareto_filter",
]
