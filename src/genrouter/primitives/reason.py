from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Any

from genrouter.primitives.costing import backend_call_cost, backend_cost_details
from genrouter.schemas import Evidence, PrimitiveTrace


@dataclass(frozen=True)
class ReasonResult:
    evidence: list[Evidence]
    trace: PrimitiveTrace


class Reason:
    name = "Reason"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def __call__(
        self,
        prompt: str,
        reason_targets: list[str] | None = None,
        evidence: list[Evidence] | None = None,
    ) -> ReasonResult:
        start = time.perf_counter()
        targets = [str(item) for item in (reason_targets or []) if str(item).strip()]
        source_evidence = list(evidence or [])
        source_evidence_dicts = [item.to_dict() for item in source_evidence]
        notes = self._from_llm(prompt, targets=targets, evidence=source_evidence_dicts)
        reasoning_evidence = [Evidence(kind="reasoning", content=note) for note in notes]
        trace = PrimitiveTrace(
            primitive=self.name,
            backend=str(getattr(self.llm, "name", "unknown_reason")),
            input_summary=prompt[:160],
            output_summary=f"{len(reasoning_evidence)} reasoning notes",
            details={
                "prompt": prompt,
                "reason_targets": targets,
                "source_evidence": source_evidence_dicts,
                "evidence": [item.to_dict() for item in reasoning_evidence],
                **backend_cost_details(self.llm),
            },
            cost=backend_call_cost(self.llm),
            latency=time.perf_counter() - start,
        )
        return ReasonResult(evidence=reasoning_evidence, trace=trace)

    def _from_llm(
        self,
        prompt: str,
        targets: list[str],
        evidence: list[dict[str, Any]],
    ) -> list[str]:
        if self.llm is None or not hasattr(self.llm, "think_json"):
            return []
        if not targets:
            return []
        focus = json.dumps(targets, ensure_ascii=False)
        evidence_json = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
        payload = self.llm.think_json(
            "Infer the visual implications required for image generation from the reasoning targets and supporting evidence.\n"
            "Return only JSON:\n"
            '{"notes":["..."]}\n\n'
            "Rules:\n"
            "- Use the provided reasoning targets and evidence.\n"
            "- Each note should describe a final visual implication, not the reasoning process.\n"
            "- Preserve all explicit names, visible text, numbers, counts, temporal and spatial relations.\n"
            "- Do not invent unsupported or uncertain facts.\n"
            "- Keep each note concise and directly usable in an image prompt.\n"
            "- Avoid decorative details that are not implied by the prompt.\n"
            'If no useful implication exists, return {"notes":[]}.\n\n'
            f"Focus:\n{focus}\n\n"
            f"Evidence:\n{evidence_json}\n\n"
            f"Prompt:\n{prompt}"
        )
        raw = payload.get("notes") or []
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]
