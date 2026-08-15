from __future__ import annotations

import time
from typing import Any

from genrouter.primitives.costing import backend_call_cost, backend_cost_details
from genrouter.schemas import ChecklistItem, DecomposeResult, PrimitiveTrace, SpecConstraint, SpecEntity, SpecUnknown


class Decompose:
    name = "Decompose"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def __call__(self, prompt: str) -> DecomposeResult:
        start = time.perf_counter()
        payload = self._from_llm(prompt)
        entities, constraints, unknowns, checklist = self._parse_payload(payload)
        latency = time.perf_counter() - start
        backend = str(getattr(self.llm, "name", "unknown_decompose"))
        trace = PrimitiveTrace(
            primitive=self.name,
            backend=backend,
            input_summary=prompt[:160],
            output_summary=f"{len(entities)} entities; {len(constraints)} constraints; {len(checklist)} checklist items",
            details={
                "prompt": prompt,
                "entities": [item.to_dict() for item in entities],
                "constraints": [item.to_dict() for item in constraints],
                "unknowns": [item.to_dict() for item in unknowns],
                "checklist": [item.to_dict() for item in checklist],
                **backend_cost_details(self.llm),
            },
            cost=backend_call_cost(self.llm),
            latency=latency,
        )
        return DecomposeResult(
            checklist=checklist,
            trace=trace,
            entities=entities,
            constraints=constraints,
            unknowns=unknowns,
        )

    def _from_llm(self, prompt: str) -> dict[str, Any]:
        if self.llm is None or not hasattr(self.llm, "think_json"):
            raise RuntimeError("Decompose requires a real LLM response")
        payload = self.llm.think_json(
            "Decompose the image prompt into a visual spec. Return only strict JSON:\n"
            '{"entities":[{"id":"o1","name":"visible entity","priority":"primary|supporting|peripheral"}],'
            '"constraints":[{"id":"c1","text":"verifiable visual requirement","type":"attribute|count|relation|layout|style|text|general","priority":"critical|major|minor","spec":{}}],'
            '"unknowns":[{"id":"u1","kind":"external_reference|ambiguity|missing_detail|reasoning","owner_id":"o1","owner_kind":"object|constraint|prompt","question":"what is unresolved?"}]}\n'
            "Rules: extract only explicit visible entities and checkable visual constraints; preserve exact text, names, numbers, counts, attributes, positions, and relations; use count constraints for repeated objects; do not infer unstated details or reasoning conclusions; add unknowns only for unresolved information required by an explicit requirement, named reference, ambiguity, or hidden reasoning. "
            f"Prompt: {prompt}"
        )
        return payload if isinstance(payload, dict) else {}

    def _parse_payload(
        self,
        payload: dict[str, Any],
    ) -> tuple[list[SpecEntity], list[SpecConstraint], list[SpecUnknown], list[ChecklistItem]]:
        entities = self._parse_entities(payload.get("entities"))
        constraints = self._parse_constraints(payload.get("constraints"))
        unknowns = self._parse_unknowns(payload.get("unknowns"))
        checklist = [self._constraint_to_checklist(item) for item in constraints]
        return entities, constraints, unknowns, checklist

    def _parse_entities(self, raw_items: Any) -> list[SpecEntity]:
        if not isinstance(raw_items, list):
            return []
        parsed: list[SpecEntity] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            entity = SpecEntity.from_dict(item, index=index)
            if entity.name.strip():
                parsed.append(entity)
        return parsed

    def _parse_constraints(self, raw_items: Any) -> list[SpecConstraint]:
        if not isinstance(raw_items, list):
            return []
        parsed: list[SpecConstraint] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            constraint = SpecConstraint.from_dict(item, index=index)
            if constraint.text.strip():
                parsed.append(constraint)
        return parsed

    def _parse_unknowns(self, raw_items: Any) -> list[SpecUnknown]:
        if not isinstance(raw_items, list):
            return []
        parsed: list[SpecUnknown] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            unknown = SpecUnknown.from_dict(item, index=index)
            if unknown.question.strip():
                parsed.append(unknown)
        return parsed

    def _constraint_to_checklist(self, constraint: SpecConstraint) -> ChecklistItem:
        return ChecklistItem(
            id=constraint.id,
            question=f"Does the image satisfy this constraint: {constraint.text}?",
            type=constraint.type,
            weight=1.5 if constraint.priority == "critical" else 1.0,
        )
