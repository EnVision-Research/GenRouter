from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from eval.common.benchmark import GenerationRecord, Score
from genrouter.memory.experience_bank import ExperienceBank
from genrouter.memory.experience_record import build_experience_record
from genrouter.router import RouteMemoryBank
from genrouter.schemas import (
    PrimitiveTrace,
    WorkflowResult,
    summarize_token_usage,
)


def experience_key(record: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    input_data = dict(record.get("input") or {})
    plan = dict(record.get("plan") or {})
    metrics = dict(record.get("metrics") or {})
    return (
        str(input_data.get("benchmark") or ""),
        str(input_data.get("prompt_id") or ""),
        str(plan.get("workflow") or ""),
        str(plan.get("generator") or ""),
        str(plan.get("selected_by") or ""),
        str(metrics.get("score_source") or ""),
    )


def trace_from_payload(items: list[Any]) -> list[PrimitiveTrace]:
    trace: list[PrimitiveTrace] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        details = dict(item.get("details") or {})
        if isinstance(item.get("token_usage"), dict) and "token_usage" not in details:
            details["token_usage"] = dict(item["token_usage"])
        trace.append(
            PrimitiveTrace(
                primitive=str(item.get("primitive") or ""),
                backend=str(item.get("backend") or ""),
                input_summary=str(item.get("input_summary") or ""),
                output_summary=str(item.get("output_summary") or ""),
                details=details,
                cost=float(item.get("cost", 0.0) or 0.0),
                latency=float(item.get("latency", 0.0) or 0.0),
                status=str(item.get("status") or "completed"),
                error=str(item.get("error") or ""),
            )
        )
    return trace


def token_usage_from_trace(trace: list[PrimitiveTrace]) -> dict[str, int]:
    return summarize_token_usage(trace)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(target)


def merge_experience_records(path: str | Path, new_records: list[dict[str, Any]]) -> int:
    """Replace matching benchmark/plan records and preserve unrelated experience."""
    target = Path(path)
    existing = ExperienceBank(target).records()
    new_keys = {experience_key(record) for record in new_records}
    kept = [record for record in existing if experience_key(record) not in new_keys]
    write_jsonl(target, [*kept, *new_records])
    return len(existing) - len(kept)


def apply_scores_and_refresh_memory(
    records: list[GenerationRecord],
    scores: list[Score],
    *,
    benchmark: str,
    selected_by: str,
    experience_path: Path,
    route_memory_path: Path,
    lambda_c: float,
    lambda_l: float,
) -> dict[str, int]:
    new_experience: list[dict[str, Any]] = []
    if [score.case_id for score in scores] != [record.case_id for record in records]:
        raise RuntimeError("Scores must preserve generation-record order")
    for record, score in zip(records, scores):
        payload = json.loads(record.result_path.read_text(encoding="utf-8"))
        trace = trace_from_payload(record.trace)
        cost = float(payload.get("cost", 0.0) or 0.0)
        latency = float(payload.get("latency", 0.0) or 0.0)
        utility = (
            float(score.value)
            - float(lambda_c) * cost
            - float(lambda_l) * latency
        )
        result = WorkflowResult(
            prompt_id=record.case_id,
            workflow=record.workflow,
            generator=record.generator,
            final_prompt=str(payload.get("final_prompt") or record.prompt),
            trace=trace,
            score=float(score.value),
            cost=cost,
            latency=latency,
            utility=utility,
        )
        payload["score"] = float(score.value)
        payload["utility"] = utility
        payload.setdefault("token_usage", token_usage_from_trace(trace))
        _write_json(record.result_path, payload)
        new_experience.append(
            build_experience_record(
                prompt_id=record.case_id,
                prompt=record.prompt,
                benchmark=benchmark,
                task_signature=record.task_signature,
                workflow_name=record.workflow,
                generator_name=record.generator,
                image_score=float(score.value),
                result=result,
                result_payload=payload,
                selected_by=selected_by,
                lambda_c=lambda_c,
                lambda_l=lambda_l,
                score_source=benchmark,
                workflow_data={benchmark: dict(score.metrics)},
            )
        )
    replaced = merge_experience_records(experience_path, new_experience)
    all_experience = ExperienceBank(experience_path).records()
    memories = RouteMemoryBank.distill(
        all_experience,
        lambda_c=lambda_c,
        lambda_l=lambda_l,
    )
    RouteMemoryBank(route_memory_path).write(memories)
    return {
        "written": len(new_experience),
        "replaced": replaced,
        "experience_records": len(all_experience),
        "route_memory_buckets": len(memories),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
