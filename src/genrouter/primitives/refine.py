from __future__ import annotations

import time
import json
from typing import Any

from genrouter.primitives.costing import backend_call_cost, backend_cost_details
from genrouter.schemas import PrimitiveTrace, RefineResult, VerificationFeedback


class Refine:
    name = "Refine"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def __call__(
        self,
        original_prompt: str,
        current_prompt: str,
        feedback: VerificationFeedback,
        attempt_history: list[dict[str, Any]] | None = None,
        entities: list[Any] | None = None,
        constraints: list[Any] | None = None,
    ) -> RefineResult:
        start = time.perf_counter()
        repair_context = _repair_context(
            feedback=feedback,
            entities=entities or [],
            constraints=constraints or [],
            attempt_history=attempt_history or [],
        )
        refined_prompt = self._from_llm(original_prompt, current_prompt, repair_context)
        if not refined_prompt:
            raise RuntimeError("Refine requires a real LLM response with refined_prompt")
        latency = time.perf_counter() - start
        backend = str(getattr(self.llm, "name", "unknown_refine"))
        trace = PrimitiveTrace(
            primitive=self.name,
            backend=backend,
            input_summary=f"failed={','.join(feedback.failed_items)}",
            output_summary=refined_prompt[:160],
            details={
                "original_prompt": original_prompt,
                "current_prompt": current_prompt,
                "refined_prompt": refined_prompt,
                "repair_context": repair_context,
                **backend_cost_details(self.llm),
            },
            cost=backend_call_cost(self.llm),
            latency=latency,
        )
        return RefineResult(
            refined_prompt=refined_prompt,
            trace=trace,
        )

    def _from_llm(
        self,
        original_prompt: str,
        current_prompt: str,
        repair_context: dict[str, Any],
    ) -> str:
        if self.llm is None or not hasattr(self.llm, "think_json"):
            return ""
        prompt_text = (
            "Refine the image-generation prompt for the next attempt.\n\n"
            "Return only JSON:\n"
            "{\"refined_prompt\":\"...\"}\n\n"
            "Requirements:\n"
            "- Fix all failed targets in Repair Context.\n"
            "- Preserve passed requirements and the Original Intent.\n"
            "- Preserve exact visible text, names, numbers, counts, attributes, spatial relations, layout, and hard constraints.\n"
            "- Avoid mistakes from prior attempts.\n"
            "- Make the refined prompt self-contained, concise, and generator-ready.\n\n"
            "Repair hints:\n"
            "- Text: quote exact text and require readable spelling.\n"
            "- Count: state exact count and forbid extra/missing items.\n"
            "- Layout/relation: state positions, order, alignment, containment, or composition clearly.\n"
            "- Attribute/style: bind the attribute/style to the correct subject.\n\n"
            f"Original Intent:\n{original_prompt}\n\n"
            f"Current Prompt:\n{current_prompt}\n\n"
            f"Repair Context:\n{json.dumps(repair_context, ensure_ascii=False, separators=(',', ': '))}\n"
        )
        payload = self.llm.think_json(prompt_text)
        return str(payload.get("refined_prompt") or "").strip()


def _repair_context(
    *,
    feedback: VerificationFeedback,
    entities: list[Any],
    constraints: list[Any],
    attempt_history: list[dict[str, Any]],
) -> dict[str, Any]:
    constraint_by_id = {str(item.get("id") if isinstance(item, dict) else getattr(item, "id", "")): _as_dict(item) for item in constraints}
    entity_by_id = {str(item.get("id") if isinstance(item, dict) else getattr(item, "id", "")): _as_dict(item) for item in entities}
    failed_targets: list[dict[str, Any]] = []
    preserve_targets: list[dict[str, Any]] = []
    for item in feedback.items:
        target = {
            "constraint_id": item.constraint_id,
            "failure_family": item.failure_family,
            "rationale": item.rationale,
        }
        constraint = constraint_by_id.get(item.constraint_id, {})
        if constraint:
            target["constraint"] = constraint
            related_entities = [
                entity_by_id[entity_id]
                for entity_id in constraint.get("depends_on", [])
                if entity_id in entity_by_id
            ]
            if related_entities:
                target["entities"] = related_entities
        if item.passed:
            preserve_targets.append(target)
        else:
            failed_targets.append(target)
    return {
        "failed_targets": failed_targets,
        "preserve_targets": preserve_targets,
        "attempt_history": list(attempt_history),
    }


def _as_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return dict(item)
    return {}
