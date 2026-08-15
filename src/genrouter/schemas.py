from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


GENERATION_MODES = frozenset({"text2image", "image2image"})


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def parse_generation_modes(value: Any) -> list[str]:
    modes = _string_list(value)
    unknown = sorted(set(modes) - GENERATION_MODES)
    if unknown:
        raise ValueError(f"unknown generation mode: {', '.join(unknown)}")
    if not modes:
        raise ValueError("generator requires at least one generation mode")
    return modes


def summarize_token_usage(trace: list["PrimitiveTrace"]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for item in trace:
        details = item.details or {}
        raw = details.get("token_usage", {})
        if not isinstance(raw, dict):
            continue
        for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens"):
            value = _int(raw.get(key))
            if value:
                usage[key] = usage.get(key, 0) + value
    explicit_total = usage.get("total_tokens", 0)
    derived_total = (
        usage.get("prompt_tokens", 0)
        + usage.get("completion_tokens", 0)
        + usage.get("input_tokens", 0)
        + usage.get("output_tokens", 0)
    )
    if derived_total:
        usage["total_tokens"] = derived_total
    elif explicit_total:
        usage["total_tokens"] = explicit_total
    return usage


@dataclass(frozen=True)
class TaskSignature:
    rewrite: int = 0
    search_text: int = 0
    search_image: int = 0
    reason: int = 0
    skill: int = 0
    verify_refine: int = 0
    code_sketch: int = 0

    def __post_init__(self) -> None:
        for value in self.to_dict().values():
            _signature_score(value)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSignature":
        return cls(
            rewrite=_signature_score(data["rewrite"]),
            search_text=_signature_score(data["search_text"]),
            search_image=_signature_score(data["search_image"]),
            reason=_signature_score(data["reason"]),
            skill=_signature_score(data["skill"]),
            verify_refine=_signature_score(data["verify_refine"]),
            code_sketch=_signature_score(data["code_sketch"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _signature_score(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
        raise ValueError("Task signature scores must be integers from 0 to 5")
    return value


@dataclass(frozen=True)
class GeneratorSpec:
    name: str
    endpoint: str
    provider: str = "http"
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    generation_modes: list[str] = field(default_factory=list)
    cost_per_call: float = 0.0
    default_params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def supports_reference(self) -> bool:
        return "image2image" in self.generation_modes

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any]) -> "GeneratorSpec":
        return cls(
            name=name,
            endpoint=str(data.get("endpoint", "")),
            provider=str(data.get("provider", "http")),
            base_url=str(data.get("base_url", "")),
            model=str(data.get("model", "")),
            api_key_env=str(data.get("api_key_env", "")),
            generation_modes=parse_generation_modes(data.get("generation_modes")),
            cost_per_call=_float(data.get("cost_per_call")),
            default_params=dict(data.get("default_params", {}) or {}),
            tags=[str(item) for item in data.get("tags", [])],
            enabled=bool(data.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionConfig:
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "ExecutionConfig":
        return cls(values=dict(data))

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class PrimitiveTrace:
    primitive: str
    backend: str
    input_summary: str = ""
    output_summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    latency: float = 0.0
    status: str = "completed"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        raw = self.details.get("token_usage", {}) if isinstance(self.details, dict) else {}
        if isinstance(raw, dict) and raw:
            payload["token_usage"] = {key: _int(value) for key, value in raw.items() if _int(value)}
        return payload


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    question: str
    type: str = "general"
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> "ChecklistItem":
        return cls(
            id=str(data.get("id") or f"c{index + 1}"),
            question=str(data.get("question") or data.get("text") or ""),
            type=str(data.get("type", "general")),
            weight=_float(data.get("weight"), 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpecEntity:
    id: str
    name: str
    priority: str = "supporting"

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> "SpecEntity":
        return cls(
            id=str(data.get("id") or f"e{index + 1}"),
            name=str(data.get("name") or data.get("label") or ""),
            priority=str(data.get("priority") or "supporting"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpecConstraint:
    id: str
    text: str
    type: str = "general"
    priority: str = "major"
    depends_on: list[str] = field(default_factory=list)
    spec: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> "SpecConstraint":
        return cls(
            id=str(data.get("id") or f"k{index + 1}"),
            text=str(data.get("text") or data.get("question") or data.get("description") or ""),
            type=str(data.get("type") or "general"),
            priority=str(data.get("priority") or "major"),
            depends_on=_string_list(data.get("depends_on")),
            spec=dict(data.get("spec", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpecUnknown:
    id: str
    kind: str
    owner_id: str
    owner_kind: str
    question: str
    owner_name: str = ""
    status: str = "open"
    source: str = "decompose"

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> "SpecUnknown":
        return cls(
            id=str(data.get("id") or f"u{index + 1}"),
            kind=str(data.get("kind") or "semantic_reasoning"),
            owner_id=str(data.get("owner_id") or ""),
            owner_kind=str(data.get("owner_kind") or "prompt"),
            question=str(data.get("question") or ""),
            owner_name=str(data.get("owner_name") or ""),
            status=str(data.get("status") or "open"),
            source=str(data.get("source") or "decompose"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecomposeResult:
    checklist: list[ChecklistItem]
    trace: PrimitiveTrace
    entities: list[SpecEntity] = field(default_factory=list)
    constraints: list[SpecConstraint] = field(default_factory=list)
    unknowns: list[SpecUnknown] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checklist": [item.to_dict() for item in self.checklist],
            "entities": [item.to_dict() for item in self.entities],
            "constraints": [item.to_dict() for item in self.constraints],
            "unknowns": [item.to_dict() for item in self.unknowns],
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True)
class AnalyzeTargets:
    search_text: list[str] = field(default_factory=list)
    search_image: list[str] = field(default_factory=list)
    reason: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyzeResult:
    targets: AnalyzeTargets
    trace: PrimitiveTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "targets": self.targets.to_dict(),
            "trace": self.trace.to_dict(),
        }

    def as_evidence(self) -> "Evidence":
        targets = self.targets.to_dict()
        parts = []
        for key, values in targets.items():
            if values:
                parts.append(f"{key}: {', '.join(values)}")
        return Evidence(
            kind="analyze_targets",
            content="; ".join(parts) if parts else "No analyze targets.",
            source="Analyze",
            payload={"targets": targets},
        )


@dataclass(frozen=True)
class GenerateResult:
    image: bytes
    generator: str
    cost: float
    latency: float
    trace: PrimitiveTrace


@dataclass(frozen=True)
class VerificationItem:
    constraint_id: str
    passed: bool
    rationale: str = ""
    failure_family: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationItem":
        constraint_id = str(data.get("constraint_id") or data.get("id") or "")
        passed = bool(data.get("passed", False))
        return cls(
            constraint_id=constraint_id,
            passed=passed,
            rationale=str(data.get("rationale") or ""),
            failure_family=str(data.get("failure_family") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationFeedback:
    overall_score: float
    items: list[VerificationItem]
    failed_items: list[str]
    trace: PrimitiveTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "items": [item.to_dict() for item in self.items],
            "failed_items": list(self.failed_items),
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True)
class RefineResult:
    refined_prompt: str
    trace: PrimitiveTrace

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillQueryResult:
    selected_skills: list[str]
    instructions: list[str]
    trace: PrimitiveTrace

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    kind: str
    content: str
    source: str = ""
    reference_path: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SketchResult:
    sketch_type: str
    code: str
    image_path: str
    code_path: str
    records: list[dict[str, Any]]
    render_prompt: str
    reasoning: list[str]
    trace: PrimitiveTrace

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_evidence(self) -> "Evidence":
        return Evidence(
            kind="sketch",
            content=self.render_prompt,
            reference_path=self.image_path,
            payload={"sketch_type": self.sketch_type},
        )


@dataclass(frozen=True)
class WorkflowResult:
    prompt_id: str
    workflow: str
    generator: str
    final_prompt: str
    final_image: bytes = field(default=b"", repr=False, compare=False)
    final_image_path: str = ""
    trace: list[PrimitiveTrace] = field(default_factory=list)
    score: float = 0.0
    cost: float = 0.0
    latency: float = 0.0
    utility: float = 0.0

    def with_utility(self, lambda_c: float, lambda_l: float) -> "WorkflowResult":
        utility = self.score - lambda_c * self.cost - lambda_l * self.latency
        return WorkflowResult(
            prompt_id=self.prompt_id,
            workflow=self.workflow,
            generator=self.generator,
            final_prompt=self.final_prompt,
            final_image=self.final_image,
            final_image_path=self.final_image_path,
            trace=list(self.trace),
            score=self.score,
            cost=self.cost,
            latency=self.latency,
            utility=utility,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("final_image", None)
        payload["trace"] = [item.to_dict() for item in self.trace]
        token_usage = summarize_token_usage(self.trace)
        if token_usage:
            payload["token_usage"] = token_usage
        return payload


@dataclass(frozen=True)
class PlanEstimate:
    workflow: str
    generator: str
    execution_config: dict[str, Any]
    estimated_score: float
    estimated_cost: float
    estimated_latency: float
    estimated_utility: float
    pareto_status: str = "unknown"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
