from __future__ import annotations

from dataclasses import dataclass

from genrouter.registries import Registry, WorkflowSpec, compatible
from genrouter.schemas import GeneratorSpec, TaskSignature


@dataclass(frozen=True)
class CandidatePlan:
    workflow: str
    generator: str

    def to_dict(self) -> dict[str, str]:
        return {
            "workflow": self.workflow,
            "generator": self.generator,
        }


def construct_candidate_plans(
    workflows: Registry[WorkflowSpec],
    generators: Registry[GeneratorSpec],
    *,
    task_signature: TaskSignature,
    max_generator_cost: float | None = None,
    generator_options: list[str] | None = None,
) -> list[CandidatePlan]:
    plans: list[CandidatePlan] = []
    allowed_generators = set(generator_options or [])
    for workflow in workflows.values():
        for generator in generators.values():
            if allowed_generators and generator.name not in allowed_generators:
                continue
            if max_generator_cost is not None and generator.cost_per_call > max_generator_cost:
                continue
            if compatible_plan(workflow, generator, task_signature):
                plans.append(CandidatePlan(workflow=workflow.name, generator=generator.name))
    return plans


def compatible_plan(
    workflow: WorkflowSpec,
    generator: GeneratorSpec,
    task_signature: TaskSignature,
) -> bool:
    return compatible(workflow, generator, task_signature)
