from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from eval.common.benchmark import Benchmark, Case, GenerationRecord, Score
from eval.common.records import RunStore
from genrouter.artifacts import write_workflow_artifacts
from genrouter.backends.chat import (
    build_llm_backend,
    build_mllm_backend,
    build_signature_llm_backend,
)
from genrouter.backends.generator import build_generator_backend
from genrouter.backends.search import build_search_backend
from genrouter.config import ProjectConfig, load_project_config
from genrouter.knowledge.skill_bank import SkillBank
from genrouter.registries import build_generator_registry, build_workflow_registry
from genrouter.router import (
    CandidatePlan,
    GenRouter,
    TaskSignatureExtractor,
    construct_candidate_plans,
)
from genrouter.schemas import TaskSignature
from genrouter.workflows.factory import build_workflow


class EvaluationRuntime(Protocol):
    def extract_signature(self, case: Case) -> TaskSignature:
        ...

    def candidate_plans(
        self,
        signature: TaskSignature,
        generator_options: list[str],
    ) -> list[CandidatePlan]:
        ...

    def begin_routed_phase(
        self,
        phase: str,
        generator_options: list[str],
    ) -> None:
        ...

    def route(
        self,
        case: Case,
    ) -> tuple[str, str, TaskSignature, dict[str, Any]]:
        ...

    def generate(
        self,
        case: Case,
        *,
        workflow: str,
        generator: str,
        signature: TaskSignature,
        image_path: Path,
        phase: str,
        selected_by: str,
        route_decision: dict[str, Any] | None,
    ) -> GenerationRecord:
        ...


ApplyPhase = Callable[..., dict[str, int]]


class GenRouterEvaluationRuntime:
    def __init__(
        self,
        *,
        config_dir: Path,
        benchmark_config: Mapping[str, Any] | None = None,
        experience_path: Path,
        route_memory_path: Path,
    ) -> None:
        self.config_dir = config_dir
        self.experience_path = experience_path
        self.route_memory_path = route_memory_path
        project_config = load_project_config(config_dir)
        self.config = evaluation_project_config(
            project_config,
            benchmark_config or {},
        )
        self.workflows = build_workflow_registry(self.config.workflows)
        self.generators = build_generator_registry(self.config.generators)
        self.skills = SkillBank.from_config(self.config.skills)
        self.llm = build_llm_backend(self.config.default.get("llm", {}))
        self.mllm = build_mllm_backend(self.config.default.get("mllm", {}))
        self.search = build_search_backend(self.config.default.get("search", {}))
        signature_llm = (
            build_signature_llm_backend(self.config.default)
            if "signature_llm" in self.config.default
            else self.llm
        )
        self.signature_extractor = TaskSignatureExtractor(signature_llm)
        self.router = None

    def enabled_workflows(self) -> list[str]:
        return self.workflows.names()

    def enabled_generators(self) -> list[str]:
        return self.generators.names()

    def configured_generator_options(self) -> list[str]:
        generator_config = dict(self.config.default.get("generator") or {})
        options = [str(item) for item in generator_config.get("options") or []]
        if not options:
            raise ValueError("configs/default.yaml must select generator options")
        missing = [name for name in options if name not in self.generators]
        if missing:
            raise ValueError(
                f"Unknown configured generators: {', '.join(missing)}"
            )
        return options

    def concrete_generator_configs(
        self,
        options: list[str],
    ) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "provider": spec.provider,
                "base_url": spec.base_url,
                "endpoint": spec.endpoint,
                "model": spec.model,
                "generation_modes": list(spec.generation_modes),
            }
            for name in options
            for spec in [self.generators.get(name)]
        }

    def extract_signature(self, case: Case) -> TaskSignature:
        return self.signature_extractor.extract(case.prompt)

    def candidate_plans(
        self,
        signature: TaskSignature,
        generator_options: list[str],
    ) -> list[CandidatePlan]:
        return construct_candidate_plans(
            self.workflows,
            self.generators,
            task_signature=signature,
            generator_options=generator_options,
        )

    def begin_routed_phase(
        self,
        phase: str,
        generator_options: list[str],
    ) -> None:
        default = dict(self.config.default)
        default["paths"] = {
            **dict(default.get("paths", {})),
            "experience_bank": str(self.experience_path),
            "route_memory": str(self.route_memory_path),
        }
        default["generator"] = {"options": list(generator_options)}
        self.router = GenRouter.from_config(
            self.workflows,
            self.generators,
            default,
            llm=self.signature_extractor.llm,
        )

    def route(
        self,
        case: Case,
    ) -> tuple[str, str, TaskSignature, dict[str, Any]]:
        if self.router is None:
            raise RuntimeError("begin_routed_phase must run before route")
        decision = self.router.select(case.prompt)
        return (
            decision.selected_plan.workflow,
            decision.selected_plan.generator,
            decision.task_signature,
            decision.to_dict(),
        )

    def generate(
        self,
        case: Case,
        *,
        workflow: str,
        generator: str,
        signature: TaskSignature,
        image_path: Path,
        phase: str,
        selected_by: str,
        route_decision: dict[str, Any] | None,
    ) -> GenerationRecord:
        workflow_spec = self.workflows.get(workflow)
        generator_spec = self.generators.get(generator)
        workflow_impl = build_workflow(
            workflow,
            self.skills,
            llm=self.llm,
            mllm=self.mllm,
            search_backend=self.search,
        )
        workflow_config = workflow_spec.config.to_dict()
        workflow_config["task_signature"] = signature.to_dict()
        result = workflow_impl.run(
            prompt=case.prompt,
            generator=build_generator_backend(generator_spec),
            config=workflow_config,
            prompt_id=case.case_id,
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(result.final_image)
        artifact_root = image_path.parent / "runs"
        artifact_paths = write_workflow_artifacts(
            result,
            artifact_root,
            prompt=case.prompt,
            task_signature=signature,
        )
        result_path = Path(artifact_paths["result_path"])
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return GenerationRecord(
            case_id=case.case_id,
            prompt=case.prompt,
            workflow=workflow,
            generator=generator,
            image_path=image_path,
            result_path=result_path,
            task_signature=signature.to_dict(),
            trace=list(payload.get("trace") or []),
            metadata={
                "phase": phase,
                "selected_by": selected_by,
                "route_decision": route_decision,
                "case_metadata": dict(case.metadata),
            },
        )


def evaluation_project_config(
    project_config: ProjectConfig,
    benchmark_config: Mapping[str, Any],
) -> ProjectConfig:
    services = dict(benchmark_config.get("services") or {})
    models = dict(benchmark_config.get("models") or {})
    generators = {
        name: dict(values)
        for name, values in project_config.generators.items()
    }
    for name in generators.keys() & services.keys() & models.keys():
        generators[name].update(
            base_url=str(services[name]),
            model=str(models[name]),
            api_key_env="",
        )
        generators[name].pop("api_profile", None)

    default = dict(project_config.default)
    if "task_signature" in services and "task_signature" in models:
        signature_config = dict(default.get("signature_llm") or {})
        signature_config.update(
            default="task_signature",
            backend="openai",
            base_url=str(services["task_signature"]),
            model=str(models["task_signature"]),
            api_key_env="",
        )
        default["signature_llm"] = signature_config

    return ProjectConfig(
        default=default,
        workflows=project_config.workflows,
        generators=generators,
        skills=project_config.skills,
        config_dir=project_config.config_dir,
    )


class EvaluationEngine:
    def __init__(
        self,
        *,
        benchmark: Benchmark,
        runtime: EvaluationRuntime,
        generator_options: list[str],
        store: RunStore,
        apply_phase: ApplyPhase,
    ) -> None:
        self.benchmark = benchmark
        self.runtime = runtime
        self.generator_options = list(generator_options)
        self.store = store
        self.apply_phase = apply_phase

    def run(
        self,
        *,
        cold_start_size: int = 10,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        if cold_start_size <= 0 or batch_size <= 0:
            raise ValueError("cold_start_size and batch_size must be positive")
        cases = self.benchmark.load_cases()
        if len(cases) < cold_start_size:
            raise ValueError(
                f"Benchmark {self.benchmark.name} has {len(cases)} cases; "
                f"cold start requires {cold_start_size}"
            )
        cold_cases = cases[:cold_start_size]
        self._run_cold_start(cold_cases)
        routed_phases = [("batch:0000", cold_cases)]
        routed_phases.extend(
            (f"batch:{index:04d}", batch)
            for index, batch in enumerate(
                _chunks(cases[cold_start_size:], batch_size),
                1,
            )
        )
        for phase, phase_cases in routed_phases:
            self._run_routed_phase(phase, phase_cases)
        self.store.refresh_final_indexes()
        summary = {
            "benchmark": self.benchmark.name,
            "generator_options": list(self.generator_options),
            "cold_start_cases": len(cold_cases),
            "final_cases": len(cases),
            "completed_phases": self.store.manifest().completed_phases,
        }
        self.store.write_summary(summary)
        return summary

    def _run_cold_start(self, cases: list[Case]) -> None:
        phase = "cold_start"
        if self.store.is_complete(phase):
            return
        if self._commit_prepared_phase(phase, selected_by="cold_start"):
            return
        records: list[GenerationRecord] = []
        for case in cases:
            signature = self.runtime.extract_signature(case)
            for plan in self.runtime.candidate_plans(
                signature,
                self.generator_options,
            ):
                plan_dir = (
                    self.store.phase_dir(phase)
                    / plan.workflow
                    / plan.generator
                )
                records.append(
                    self.runtime.generate(
                        case,
                        workflow=plan.workflow,
                        generator=plan.generator,
                        signature=signature,
                        image_path=self.benchmark.image_path(case, plan_dir),
                        phase=phase,
                        selected_by="cold_start",
                        route_decision=None,
                    )
                )
        if not records:
            raise RuntimeError("Cold start produced no compatible workflow runs")
        scores = self._evaluate_cold_start(records)
        self.store.write_phase(phase, records, scores)
        self._commit_prepared_phase(phase, selected_by="cold_start")

    def _run_routed_phase(self, phase: str, cases: list[Case]) -> None:
        if self.store.is_complete(phase):
            return
        if self._commit_prepared_phase(phase, selected_by="route"):
            return
        self.runtime.begin_routed_phase(phase, self.generator_options)
        records: list[GenerationRecord] = []
        for case in cases:
            workflow, generator, signature, decision = self.runtime.route(case)
            allowed = {
                (plan.workflow, plan.generator)
                for plan in self.runtime.candidate_plans(
                    signature,
                    self.generator_options,
                )
            }
            if (workflow, generator) not in allowed:
                raise RuntimeError(
                    f"Router selected incompatible plan {workflow}+{generator}"
                )
            records.append(
                self.runtime.generate(
                    case,
                    workflow=workflow,
                    generator=generator,
                    signature=signature,
                    image_path=self.benchmark.image_path(case, self.store.images_dir),
                    phase=phase,
                    selected_by="route",
                    route_decision=decision,
                )
            )
        scores = self._evaluate_complete_phase(records, phase)
        self.store.write_phase(phase, records, scores)
        self._commit_prepared_phase(phase, selected_by="route")

    def _commit_prepared_phase(self, phase: str, *, selected_by: str) -> bool:
        prepared = self.store.read_prepared_phase(phase)
        if prepared is None:
            return False
        records, scores = prepared
        self._validate_scores(records, scores, phase)
        self.apply_phase(records, scores, selected_by=selected_by)
        self.store.complete_phase(phase)
        if phase.startswith("batch:"):
            self.store.refresh_final_indexes()
        return True

    def _evaluate_cold_start(
        self,
        records: list[GenerationRecord],
    ) -> list[Score]:
        grouped: dict[
            tuple[str, str],
            list[tuple[int, GenerationRecord]],
        ] = {}
        for index, record in enumerate(records):
            grouped.setdefault(
                (record.workflow, record.generator),
                [],
            ).append((index, record))

        merged: list[Score | None] = [None] * len(records)
        for (workflow, generator), indexed_records in grouped.items():
            group_records = [record for _, record in indexed_records]
            group_scores = self.benchmark.evaluate(
                group_records,
                self.store.phase_dir("cold_start")
                / "evaluation"
                / workflow
                / generator,
            )
            self._validate_scores(group_records, group_scores, "cold_start")
            for (index, _), score in zip(indexed_records, group_scores):
                merged[index] = score

        if any(score is None for score in merged):
            raise RuntimeError("Cold-start evaluation did not score every record")
        return [score for score in merged if score is not None]

    def _evaluate_complete_phase(
        self,
        records: list[GenerationRecord],
        phase: str,
    ) -> list[Score]:
        scores = self.benchmark.evaluate(
            records,
            self.store.phase_dir(phase) / "evaluation",
        )
        self._validate_scores(records, scores, phase)
        return scores

    @staticmethod
    def _validate_scores(
        records: list[GenerationRecord],
        scores: list[Score],
        phase: str,
    ) -> None:
        record_ids = [record.case_id for record in records]
        score_ids = [score.case_id for score in scores]
        if score_ids != record_ids:
            raise RuntimeError(
                f"Official evaluator returned {len(score_ids)} scores for "
                f"{len(record_ids)} generated records in {phase}"
            )


def _chunks(items: list[Case], size: int) -> list[list[Case]]:
    return [items[start : start + size] for start in range(0, len(items), size)]
