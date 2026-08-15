from __future__ import annotations

import time
import json
from typing import Any

from genrouter.primitives.costing import backend_call_cost, backend_token_pricing, backend_token_usage
from genrouter.schemas import ChecklistItem, PrimitiveTrace, SpecConstraint, SpecEntity, SpecUnknown, VerificationFeedback, VerificationItem


class Verify:
    name = "Verify"

    def __init__(self, mllm=None, max_retries: int = 2):
        self.mllm = mllm
        self.max_retries = max(0, int(max_retries))

    def __call__(
        self,
        image: bytes,
        checklist: list[ChecklistItem],
        prompt: str = "",
        entities: list[SpecEntity] | None = None,
        constraints: list[SpecConstraint] | None = None,
        unknowns: list[SpecUnknown] | None = None,
    ) -> VerificationFeedback:
        start = time.perf_counter()
        if not isinstance(image, bytes) or not image:
            raise ValueError("Verify requires non-empty image bytes")
        entities = entities or []
        constraints = constraints or []
        unknowns = unknowns or []
        items, attempts, raw_payloads, token_usage, call_cost = self._from_mllm(
            image=image,
            checklist=checklist,
            prompt=prompt,
            entities=entities,
            constraints=constraints,
            unknowns=unknowns,
        )
        if not items:
            raise RuntimeError(
                "Verify requires MLLM verification items with valid constraint ids; "
                f"responses={raw_payloads!r}"
            )
        items = self._complete_missing_items(items, checklist)
        failed_items = [item.constraint_id for item in items if not item.passed]
        weights = {item.id: item.weight for item in checklist}
        total_weight = sum(weights.get(item.constraint_id, 1.0) for item in items) or float(len(items) or 1)
        weighted_score = sum((1.0 if item.passed else 0.0) * weights.get(item.constraint_id, 1.0) for item in items)
        overall_score = max(0.0, min(1.0, weighted_score / total_weight))
        latency = time.perf_counter() - start
        backend = str(getattr(self.mllm, "name", "unknown_verify"))
        trace = PrimitiveTrace(
            primitive=self.name,
            backend=backend,
            input_summary=f"{len(checklist)} checklist items",
            output_summary=f"overall_score={overall_score:.3f}; failed={len(failed_items)}",
            details={
                "prompt": prompt,
                "entities": [_to_dict(item) for item in entities],
                "constraints": [_to_dict(item) for item in constraints],
                "unknowns": [_to_dict(item) for item in unknowns],
                "checklist": [item.to_dict() for item in checklist],
                "items": [item.to_dict() for item in items],
                "failed_items": failed_items,
                "overall_score": overall_score,
                "image_bytes": len(image),
                "verification_attempts": attempts,
                "token_usage": token_usage,
                "token_pricing": backend_token_pricing(self.mllm),
            },
            cost=call_cost,
            latency=latency,
        )
        return VerificationFeedback(
            overall_score=overall_score,
            items=items,
            failed_items=failed_items,
            trace=trace,
        )

    def _from_mllm(
        self,
        image: bytes,
        checklist: list[ChecklistItem],
        prompt: str,
        entities: list[SpecEntity],
        constraints: list[SpecConstraint],
        unknowns: list[SpecUnknown],
    ) -> tuple[list[VerificationItem], int, list[dict[str, Any]], dict[str, int], float]:
        if self.mllm is None or not hasattr(self.mllm, "think_json"):
            return [], 0, [], {}, 0.0
        checklist_payload = [item.to_dict() for item in checklist]

        constraint_payload = [
            {
                "id": item.id,
                "text": getattr(item, "text", ""),
                "type": getattr(item, "type", ""),
                "priority": getattr(item, "priority", ""),
            }
            for item in constraints
        ]

        checklist_json = json.dumps(
            checklist_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        constraints_json = json.dumps(
            constraint_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        checklist_ids = {item.id for item in checklist}
        constraint_ids = {item.id for item in constraints}
        allowed_ids = checklist_ids | constraint_ids

        prompt_payload = (
            "Verify whether the image satisfies the prompt and checklist.\n\n"
            "Return only JSON:\n"
            "{\"items\":[{\"constraint_id\":\"id\",\"passed\":true,\"rationale\":\"short visual reason\",\"failure_family\":\"\"}]}\n\n"
            "Rules:\n"
            "- Judge each checklist item using only visible image evidence.\n"
            "- Use only ids from Checklist or Constraints.\n"
            "- Pass only if the requirement is clearly satisfied; if unclear, mark failed.\n"
            "- Do not reward image quality when a requirement is wrong or missing.\n"
            "- For failures, set failure_family to one of: text, count, layout, relation, attribute, subject, style, other.\n"
            "- For passed items, use failure_family as an empty string.\n"
            "- Keep rationale short and visual.\n\n"
            "Failure guide:\n"
            "- text: missing, unreadable, misspelled, or extra visible text.\n"
            "- count: wrong number, missing repeated elements, or extra repeated elements.\n"
            "- layout/relation: wrong position, order, alignment, containment, foreground/background, left/right, above/below.\n"
            "- attribute/subject: wrong color, material, identity, expression, shape, state, or attribute binding.\n"
            "- style: wrong medium, genre, lighting, or rendering style.\n\n"
            f"Original Prompt:\n{prompt}\n\n"
            f"Constraints:\n{constraints_json}\n\n"
            f"Checklist:\n{checklist_json}"
        )

        raw_payloads: list[dict[str, Any]] = []
        token_usage: dict[str, int] = {}
        call_cost = 0.0

        for attempt in range(self.max_retries + 1):
            request_prompt = prompt_payload
            if attempt:
                request_prompt += (
                    "\n\nPrevious response was invalid. Return exactly one JSON object. "
                    f'Use a non-empty "items" list and only these constraint_id values: {sorted(allowed_ids)}.'
                )

            payload = self.mllm.think_json(request_prompt, images=[image])
            raw_payloads.append(payload)
            _merge_usage(token_usage, backend_token_usage(self.mllm))
            call_cost += backend_call_cost(self.mllm)

            parsed = _parse_items(payload, allowed_ids)
            if parsed:
                return parsed, attempt + 1, raw_payloads, token_usage, call_cost

        return [], self.max_retries + 1, raw_payloads, token_usage, call_cost

    def _complete_missing_items(
        self,
        items: list[VerificationItem],
        checklist: list[ChecklistItem],
    ) -> list[VerificationItem]:
        returned_ids = {item.constraint_id for item in items}
        missing = [
            VerificationItem(
                constraint_id=item.id,
                passed=False,
                rationale="Verifier did not return a judgment for this checklist item.",
                failure_family="prompt_repair",
            )
            for item in checklist
            if item.id not in returned_ids
        ]
        return [*items, *missing]


def _parse_items(payload: dict[str, Any], allowed_ids: set[str]) -> list[VerificationItem]:
    if not isinstance(payload, dict):
        return []
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return []
    parsed: list[VerificationItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = VerificationItem.from_dict(raw)
        if item.constraint_id in allowed_ids:
            parsed.append(item)
    return parsed


def _merge_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value


def _to_dict(item):
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return dict(item)
    return item
