from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from genrouter.backends.embedding import build_embedding_backend
from genrouter.memory.experience_bank import ExperienceBank
from genrouter.registries import Registry, WorkflowSpec
from genrouter.router.pareto import pareto_filter
from genrouter.router.plans import construct_candidate_plans
from genrouter.router.route_memory import RouteMemoryBank
from genrouter.router.task_signature import TaskSignatureExtractor
from genrouter.schemas import GeneratorSpec, PlanEstimate, TaskSignature
from genrouter.workflows.signature import requires_hybrid


@dataclass(frozen=True)
class RouteDecision:
    task_signature: TaskSignature
    selected_plan: PlanEstimate
    candidate_plans: list[PlanEstimate]
    pareto_plans: list[PlanEstimate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_signature": self.task_signature.to_dict(),
            "selected_plan": self.selected_plan.to_dict(),
            "candidate_plans": [item.to_dict() for item in self.candidate_plans],
            "pareto_plans": [item.to_dict() for item in self.pareto_plans],
        }


class GenRouter:
    def __init__(
        self,
        workflows: Registry[WorkflowSpec],
        generators: Registry[GeneratorSpec],
        experience_bank: ExperienceBank,
        route_memory: RouteMemoryBank,
        lambda_c: float = 0.0,
        lambda_l: float = 0.0,
        trajectory_alpha0: float = 0.7,
        trajectory_top_k: int = 20,
        generator_options: list[str] | None = None,
        signature_extractor: TaskSignatureExtractor | None = None,
    ) -> None:
        self.workflows = workflows
        self.generators = generators
        self.experience_bank = experience_bank
        self.route_memory = route_memory
        self.lambda_c = float(lambda_c)
        self.lambda_l = float(lambda_l)
        self.trajectory_alpha0 = float(trajectory_alpha0)
        self.trajectory_top_k = int(trajectory_top_k)
        self.generator_options = list(generator_options or [])
        self.signature_extractor = signature_extractor or TaskSignatureExtractor()

    @classmethod
    def from_config(
        cls,

        workflows: Registry[WorkflowSpec],
        generators: Registry[GeneratorSpec],
        default_config: dict[str, Any],
        llm: Any | None = None,
        embedding_backend: Any | None = None,
    ) -> "GenRouter":
        paths = dict(default_config.get("paths", {}))
        routing = dict(default_config.get("routing", {}))
        generator_config = dict(default_config.get("generator", {}))
        embedding_config = dict(default_config.get("embedding", {}))
        if embedding_backend is None and embedding_config:
            embedding_backend = build_embedding_backend(embedding_config)
        return cls(
            workflows=workflows,
            generators=generators,
            experience_bank=ExperienceBank(
                paths.get("experience_bank", "data/experience_bank.jsonl"),
                embedding_backend=embedding_backend,
                signature_weight=float(routing.get("signature_similarity_weight", 0.7)),
            ),
            route_memory=RouteMemoryBank(paths.get("route_memory", "data/route_memory.jsonl")),
            lambda_c=float(routing.get("lambda_c", 0.0)),
            lambda_l=float(routing.get("lambda_l", 0.0)),
            trajectory_alpha0=float(routing.get("trajectory_alpha0", 0.7)),
            trajectory_top_k=int(routing.get("trajectory_top_k", 20)),
            generator_options=_string_list(generator_config.get("options")),
            signature_extractor=TaskSignatureExtractor(llm),
        )

    def select(self, prompt: str, max_generator_cost: float | None = None) -> RouteDecision:
        signature = self.signature_extractor.extract(prompt)
        candidates = construct_candidate_plans(
            self.workflows,
            self.generators,
            task_signature=signature,
            max_generator_cost=max_generator_cost,
            generator_options=self.generator_options,
        )
        prior_workflow = _threshold_workflow(
            signature,
            allowed_workflows={candidate.workflow for candidate in candidates},
        )
        estimates = [
            self._estimate_plan(
                prompt,
                signature,
                candidate.workflow,
                candidate.generator,
                prior_workflow=prior_workflow,
            )
            for candidate in candidates
        ]
        pareto = pareto_filter(estimates)
        ranked = sorted(pareto, key=lambda item: item.estimated_utility, reverse=True)
        if not ranked:
            raise RuntimeError("No compatible candidate plans are available")
        return RouteDecision(
            task_signature=signature,
            selected_plan=ranked[0],
            candidate_plans=estimates,
            pareto_plans=pareto,
        )

    def _estimate_plan(
        self,
        prompt: str,
        signature: TaskSignature,
        workflow_name: str,
        generator_name: str,
        prior_workflow: str | None = None,
    ) -> PlanEstimate:
        workflow = self.workflows.get(workflow_name)
        generator = self.generators.get(generator_name)
        trajectory = self.experience_bank.estimate(
            signature,
            workflow_name,
            generator_name,
            prompt=prompt,
            top_k=self.trajectory_top_k,
        )
        route = self.route_memory.estimate(signature, workflow_name, generator_name)
        trajectory_available = trajectory is not None
        route_available = route is not None
        trajectory_confidence = 0.0
        if trajectory_available and self.trajectory_top_k > 0:
            trajectory_confidence = min(1.0, float(trajectory.get("num_prompts", 0) or 0) / float(self.trajectory_top_k))
        trajectory_alpha = max(0.0, min(1.0, self.trajectory_alpha0 * trajectory_confidence))

        trajectory_utility = None
        route_utility = None
        if trajectory_available and route_available:
            trajectory_score = max(0.0, min(1.0, float(trajectory["score"])))
            trajectory_cost = float(trajectory["cost"])
            trajectory_latency = float(trajectory["latency"])
            trajectory_utility = float(trajectory.get("utility", trajectory_score - self.lambda_c * trajectory_cost - self.lambda_l * trajectory_latency))
            route_score = max(0.0, min(1.0, float(route["score"])))
            route_cost = float(route["cost"])
            route_latency = float(route["latency"])
            route_utility = float(route["utility"])
            score = trajectory_alpha * trajectory_score + (1.0 - trajectory_alpha) * route_score
            cost = trajectory_alpha * trajectory_cost + (1.0 - trajectory_alpha) * route_cost
            latency = trajectory_alpha * trajectory_latency + (1.0 - trajectory_alpha) * route_latency
            utility = score - self.lambda_c * cost - self.lambda_l * latency
            estimate_source = "trajectory_route"
        elif trajectory_available:
            score = max(0.0, min(1.0, float(trajectory["score"])))
            cost = float(trajectory["cost"])
            latency = float(trajectory["latency"])
            trajectory_utility = float(trajectory.get("utility", score - self.lambda_c * cost - self.lambda_l * latency))
            utility = trajectory_utility
            estimate_source = "trajectory"
        elif route_available:
            score = max(0.0, min(1.0, float(route["score"])))
            cost = float(route["cost"])
            latency = float(route["latency"])
            route_utility = float(route["utility"])
            utility = route_utility
            estimate_source = "route_memory"
        else:
            prior = self._prior_estimate(signature, workflow, generator, prior_workflow=prior_workflow)
            score = float(prior["score"])
            cost = float(prior["cost"])
            latency = float(prior["latency"])
            utility = float(prior["utility"])
            estimate_source = "prior"

        evidence = {
            "estimate_source": estimate_source,
            "trajectory_source": trajectory["source"] if trajectory else None,
            "trajectory_num_records": trajectory.get("num_records", 0) if trajectory else 0,
            "trajectory_num_prompts": trajectory.get("num_prompts", 0) if trajectory else 0,
            "trajectory_confidence": trajectory_confidence,
            "trajectory_alpha0": self.trajectory_alpha0,
            "trajectory_alpha": trajectory_alpha,
            "trajectory_top_k": self.trajectory_top_k,
            "trajectory_utility": trajectory_utility,
            "route_bucket": route["bucket"] if route else None,
            "route_source": route["source"] if route else None,
            "route_num_records": route["num_records"] if route else 0,
            "route_utility": route_utility,
            "route_is_pareto": route["is_pareto"] if route else False,
        }
        return PlanEstimate(
            workflow=workflow_name,
            generator=generator_name,
            execution_config=workflow.config.to_dict(),
            estimated_score=score,
            estimated_cost=cost,
            estimated_latency=latency,
            estimated_utility=utility,
            pareto_status="candidate",
            evidence=evidence,
        )

    def _prior_estimate(
        self,
        signature: TaskSignature,
        workflow: WorkflowSpec,
        generator: GeneratorSpec,
        prior_workflow: str | None = None,
    ) -> dict[str, Any]:
        preferred = prior_workflow or _threshold_workflow(signature)
        workflow_score = 0.8 if workflow.name == preferred else 0.45
        cost = float(generator.cost_per_call)
        latency = 0.0
        return {
            "score": min(0.95, workflow_score),
            "cost": cost,
            "latency": latency,
            "utility": min(0.95, workflow_score) - self.lambda_c * cost - self.lambda_l * latency,
            "source": "prior",
            "num_records": 0,
            "preferred_workflow": preferred,
        }


def _threshold_workflow(signature: TaskSignature, allowed_workflows: set[str] | None = None) -> str:
    allowed = allowed_workflows or set()

    def choose(name: str) -> str | None:
        if not allowed or name in allowed:
            return name
        return None

    high = 3
    low = 2
    if requires_hybrid(signature, threshold=high):
        selected = choose("HybridGen")
        if selected:
            return selected
    if signature.code_sketch >= high:
        selected = choose("CodeSketchGen")
        if selected:
            return selected
    if signature.search_image >= high:
        selected = choose("RefGen")
        if selected:
            return selected
    if signature.search_text >= high:
        selected = choose("SearchGen")
        if selected:
            return selected
    if signature.reason >= high:
        selected = choose("ReasonGen")
        if selected:
            return selected
    if signature.skill >= high:
        selected = choose("SkillGen")
        if selected:
            return selected
    if signature.verify_refine >= high:
        selected = choose("VerifyRefine")
        if selected:
            return selected
    if signature.rewrite >= high:
        selected = choose("RewriteGen")
        if selected:
            return selected
    if max(signature.to_dict().values()) < low:
        selected = choose("DirectGen")
        if selected:
            return selected
    selected = choose("RewriteGen")
    if selected:
        return selected
    if "DirectGen" in allowed:
        return "DirectGen"
    return sorted(allowed)[0] if allowed else "RewriteGen"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []
