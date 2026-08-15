from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from genrouter.primitives.costing import backend_call_cost, backend_cost_details
from genrouter.schemas import PrimitiveTrace, VerificationFeedback


@dataclass(frozen=True)
class ExperienceSummaryResult:
    experience: str
    attempt: dict[str, Any]
    trace: PrimitiveTrace


class ExperienceSummarizer:
    name = "ExperienceSummary"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def __call__(
        self,
        original_prompt: str,
        current_prompt: str,
        feedback: VerificationFeedback,
        constraints: list[Any] | None = None,
        attempt_history: list[dict[str, Any]] | None = None,
        image: bytes | None = None,
        thought: str = "",
    ) -> ExperienceSummaryResult:
        start = time.perf_counter()
        failed_checklist = _failed_checklist(feedback, constraints or [])
        history = list(attempt_history or [])
        experience = self._from_llm(
            original_prompt=original_prompt,
            current_prompt=current_prompt,
            failed_checklist=failed_checklist,
            attempt_history=history,
            image=image,
            thought=thought,
        )
        if not experience:
            raise RuntimeError("ExperienceSummarizer requires a real LLM response with experience")

        attempt = {
            "prompt": current_prompt,
            "failed_checklist": failed_checklist,
            "experience": experience,
        }
        backend = str(getattr(self.llm, "name", "unknown_experience"))
        trace = PrimitiveTrace(
            primitive=self.name,
            backend=backend,
            input_summary=f"failed={','.join(feedback.failed_items)}",
            output_summary=experience[:160],
            details={
                "original_prompt": original_prompt,
                "current_prompt": current_prompt,
                "failed_checklist": failed_checklist,
                "experience": experience,
                "attempt_history": history,
                "thought": thought,
                **backend_cost_details(self.llm),
            },
            cost=backend_call_cost(self.llm),
            latency=time.perf_counter() - start,
        )
        return ExperienceSummaryResult(
            experience=experience,
            attempt=attempt,
            trace=trace,
        )

    def _from_llm(
        self,
        *,
        original_prompt: str,
        current_prompt: str,
        failed_checklist: list[dict[str, Any]],
        attempt_history: list[dict[str, Any]],
        image: bytes | None,
        thought: str,
    ) -> str:
        prompt = (
            "Task: Summarize the experience of the current image generation attempt.\n"
            "--- CURRENT FAILED ATTEMPT ---\n"
            f"Prompt used: {current_prompt}\n"
            f"Failed checklist: {json.dumps(failed_checklist, ensure_ascii=False)}\n"
            f"Reasoning/Thought before generation: {thought}\n"
            "Image: <image>\n"
            "--- PRIOR ATTEMPT HISTORY ---\n"
            f"{json.dumps(attempt_history, ensure_ascii=False) if attempt_history else 'None (First round)'}\n"
            "--- ANALYSIS ---\n"
            "Summarize why the attempt failed and the concrete strategy for the next attempt. "
            "Use the image, failed checklist, and prior attempts. Keep it under 100 words. "
            "Do not include introductory phrases. "
            "Return only JSON with schema: {\"experience\":\"under 100 words\"}.\n\n"
            f"Original prompt: {original_prompt}"
        )
        payload = self.llm.think_json(prompt, images=[image] if image else None)
        return str(payload.get("experience") or "").strip()


def _failed_checklist(
    feedback: VerificationFeedback,
    constraints: list[Any],
) -> list[dict[str, Any]]:
    constraint_by_id = {
        str(item.get("id") if isinstance(item, dict) else getattr(item, "id", "")): _as_dict(item)
        for item in constraints
    }
    failed: list[dict[str, Any]] = []
    for item in feedback.items:
        if item.passed:
            continue
        record: dict[str, Any] = {"constraint_id": item.constraint_id}
        constraint = constraint_by_id.get(item.constraint_id, {})
        if constraint.get("text"):
            record["text"] = constraint["text"]
        if constraint.get("type"):
            record["type"] = constraint["type"]
        record["rationale"] = item.rationale
        if item.failure_family:
            record["failure_family"] = item.failure_family
        failed.append(record)
    return failed


def _as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return dict(item)
    return {}
