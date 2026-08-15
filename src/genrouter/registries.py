from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

from genrouter.schemas import (
    GENERATION_MODES,
    ExecutionConfig,
    GeneratorSpec,
    TaskSignature,
)
from genrouter.workflows.signature import hybrid_branches, requires_hybrid


T = TypeVar("T")
class Registry(Generic[T]):
    def __init__(self, items: Iterable[T], name_attr: str = "name") -> None:
        self._name_attr = name_attr
        self._items: dict[str, T] = {}
        for item in items:
            name = str(getattr(item, name_attr))
            self._items[name] = item

    def get(self, name: str) -> T:
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items.keys())

    def values(self) -> list[T]:
        return [self._items[name] for name in self.names()]

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    config: ExecutionConfig
    enabled: bool = True
    required_generation_mode: str | None = None


def build_workflow_registry(config: dict[str, dict]) -> Registry[WorkflowSpec]:
    workflows: list[WorkflowSpec] = []
    for name, values in config.items():
        if not bool(values.get("enabled", True)):
            continue
        mode = values.get("required_generation_mode")
        if name == "HybridGen":
            required_mode = None
        else:
            required_mode = str(mode or "")
            if required_mode not in GENERATION_MODES:
                raise ValueError(f"unknown generation mode: {required_mode}")
        workflows.append(
            WorkflowSpec(
                name=name,
                config=ExecutionConfig.from_config(values),
                enabled=True,
                required_generation_mode=required_mode,
            )
        )
    return Registry(workflows)


def build_generator_registry(config: dict[str, dict]) -> Registry[GeneratorSpec]:
    specs: list[GeneratorSpec] = []
    for name, values in config.items():
        if not bool(values.get("enabled", True)):
            continue
        spec = GeneratorSpec.from_config(name, values)
        specs.append(spec)
    return Registry(specs)


def required_generation_mode(
    workflow: WorkflowSpec,
    task_signature: TaskSignature,
) -> str | None:
    if workflow.name != "HybridGen":
        return workflow.required_generation_mode
    if not requires_hybrid(task_signature):
        return None
    if hybrid_branches(task_signature)["requires_reference"]:
        return "image2image"
    return "text2image"


def compatible(
    workflow: WorkflowSpec,
    generator: GeneratorSpec,
    task_signature: TaskSignature,
) -> bool:
    if getattr(generator, "provider", None) in {"api", "placeholder"}:
        return False
    mode = required_generation_mode(workflow, task_signature)
    return mode is not None and mode in generator.generation_modes
