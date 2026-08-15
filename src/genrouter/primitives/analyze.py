from __future__ import annotations

import time
from typing import Any

from genrouter.primitives.costing import backend_call_cost, backend_cost_details
from genrouter.schemas import AnalyzeResult, AnalyzeTargets, PrimitiveTrace


class Analyze:
    name = "Analyze"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def __call__(self, prompt: str) -> AnalyzeResult:
        start = time.perf_counter()
        payload = self._from_llm(prompt)
        if not payload:
            raise RuntimeError("Analyze requires a real LLM response")
        raw_targets = payload["targets"]
        if not isinstance(raw_targets, dict):
            raise TypeError("Analyze field targets must be an object")
        targets = AnalyzeTargets(
            search_text=_unique_texts(raw_targets.get("search_text", []), "search_text"),
            search_image=_unique_texts(raw_targets.get("search_image", []), "search_image"),
            reason=_unique_texts(raw_targets.get("reason", []), "reason"),
        )
        trace = PrimitiveTrace(
            primitive=self.name,
            backend=str(getattr(self.llm, "name", "unknown_analyze")),
            input_summary=prompt[:160],
            output_summary=(
                f"{len(targets.search_text)} text; "
                f"{len(targets.search_image)} image; "
                f"{len(targets.reason)} reason targets"
            ),
            details={
                "prompt": prompt,
                "targets": targets.to_dict(),
                **backend_cost_details(self.llm),
            },
            cost=backend_call_cost(self.llm),
            latency=time.perf_counter() - start,
        )
        return AnalyzeResult(targets=targets, trace=trace)

    def _from_llm(self, prompt: str) -> dict[str, Any]:
        if self.llm is None or not hasattr(self.llm, "think_json"):
            return {}
        payload = self.llm.think_json(
            "Analyze the image generation prompt and extract extract the minimal targets required for downstream Search Text, Search Image, and Reason.\n"
            "targets.search_text:\n"
            "External facts that must be retrieved before generation, such as names, dates, events, specifications, scientific knowledge, geographic knowledge, or official information.\n"
            "targets.search_image:\n"
            "Specific visual identities that require reference images, such as people, characters, products, brands, logos, landmarks, artworks, or famous places. Use canonical names whenever possible.\n"
            "targets.reason:\n"
            "Implicit logical, temporal, spatial, physical, causal, or mathematical sub-problems that must be solved before generation.\n"
            "Use [] when a category does not apply.\n"
            "Return only JSON with schema:\n"
            '{"targets":{"search_text":["..."],"search_image":["..."],"reason":["..."]}}.\n\n'
            f"Prompt: {prompt}"
        )
        return payload if isinstance(payload, dict) else {}


def _unique_texts(raw_items: object, field_name: str) -> list[str]:
    if not isinstance(raw_items, list):
        raise TypeError(f"Analyze field {field_name} must be a list")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = " ".join(str(item or "").split()).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized
