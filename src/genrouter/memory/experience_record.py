from __future__ import annotations

from typing import Any


def build_experience_record(
    *,
    prompt_id: str,
    prompt: str,
    benchmark: str,
    task_signature: Any,
    workflow_name: str,
    generator_name: str,
    image_score: float,
    result: Any,
    result_payload: dict[str, Any],
    selected_by: str,
    lambda_c: float,
    lambda_l: float,
    score_source: str = "scorer",
    workflow_data: dict[str, Any] | None = None,
    error: Any = None,
) -> dict[str, Any]:
    return {
        "input": {
            "benchmark": benchmark,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "task_signature": _signature_dict(task_signature),
        },
        "plan": {
            "workflow": workflow_name,
            "generator": generator_name,
            "selected_by": selected_by,
        },
        "execution": {
            "status": "completed",
            "primitive_sequence": [str(item.primitive) for item in result.trace],
            "latency_seconds": result.latency,
            "token_usage": normalize_experience_token_usage(result_payload.get("token_usage", {})),
            "cost": result.cost,
            "references_used": references_used(result),
        },
        "metrics": {
            "score": image_score,
            "score_source": score_source,
            "utility": result.utility,
            "utility_params": {
                "lambda_c": float(lambda_c),
                "lambda_l": float(lambda_l),
            },
        },
        "workflow_data": workflow_data or {},
        "error": error,
    }


def normalize_experience_token_usage(token_usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(token_usage.get("input_tokens", 0) or 0) + int(token_usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(token_usage.get("output_tokens", 0) or 0) + int(token_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(token_usage.get("total_tokens", 0) or 0) or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def references_used(result: Any) -> list[str]:
    if result.workflow not in {"RefGen", "HybridGen"}:
        return []
    references: list[str] = []
    seen: set[str] = set()
    for trace in result.trace:
        if trace.primitive != "Generate":
            continue
        details = trace.details or {}
        raw_references = details.get("references") or []
        if not isinstance(raw_references, list):
            continue
        for reference in raw_references:
            value = str(reference).strip()
            if value and value not in seen:
                seen.add(value)
                references.append(value)
    return references


def _signature_dict(task_signature: Any) -> dict[str, Any]:
    if hasattr(task_signature, "to_dict"):
        return task_signature.to_dict()
    return dict(task_signature)
