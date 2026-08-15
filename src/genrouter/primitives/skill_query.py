from __future__ import annotations

import re
import time
from typing import Any

from genrouter.knowledge.skill_bank import SkillBank
from genrouter.primitives.costing import backend_call_cost, backend_cost_details
from genrouter.schemas import PrimitiveTrace, SkillQueryResult, TaskSignature


class SkillQuery:
    name = "SkillQuery"

    def __init__(self, skill_bank: SkillBank, llm: Any | None = None) -> None:
        self.skill_bank = skill_bank
        self.llm = llm

    def __call__(
        self,
        prompt: str,
        signature: TaskSignature | None = None,
        requested_skills: list[str] | None = None,
        max_skills: int | None = None,
    ) -> SkillQueryResult:
        start = time.perf_counter()
        selection_mode = "requested" if requested_skills else "signature"
        selected_names = requested_skills or []
        if not selected_names and self.llm is not None and hasattr(self.llm, "think_json"):
            llm_selected = self._llm_select(prompt)
            if llm_selected is not None:
                selected_names = llm_selected
                selection_mode = "llm" if selected_names else "llm_none"
        if not selected_names and selection_mode != "llm_none":
            selected_names = self._infer_skills(prompt)
            selection_mode = "signature" if selected_names else selection_mode
        effective_max_skills = self.skill_bank.max_skills if max_skills is None else max(0, int(max_skills))
        selected_names = selected_names[:effective_max_skills]
        selected = self.skill_bank.select(selected_names)
        latency = time.perf_counter() - start
        skill_details = [
            {
                "name": item.name,
                "source_path": item.source_path,
                "instruction_excerpt": item.instructions[:1000],
            }
            for item in selected
        ]
        trace = PrimitiveTrace(
            primitive=self.name,
            backend="SkillBank",
            input_summary=prompt[:160],
            output_summary=",".join(item.name for item in selected),
            details={
                "prompt": prompt,
                "selection_mode": selection_mode,
                "manifest": self.skill_bank.manifest(),
                "requested_skills": requested_skills or [],
                "inferred_skills": selected_names,
                "selected_skills": [item.name for item in selected],
                "max_skills": effective_max_skills,
                "skills": skill_details,
                **backend_cost_details(self.llm),
            },
            cost=backend_call_cost(self.llm) if selection_mode.startswith("llm") else 0.0,
            latency=latency,
        )
        return SkillQueryResult(
            selected_skills=[item.name for item in selected],
            instructions=[item.instructions for item in selected],
            trace=trace,
        )

    def _llm_select(self, prompt: str) -> list[str] | None:
        manifest = self.skill_bank.manifest()
        payload = self.llm.think_json(
            "You are a strategic Skill Router. Your goal is to determine if the user's request "
            "genuinely requires a specialized skill or if it can be handled by standard generation.\n\n"
            f"### Available Skills:\n{manifest}\n"
            f"### User Request:\n{prompt}\n\n"
            "### Evaluation Criteria:\n"
            "1. **Relevance**: Does the request explicitly match the DESCRIPTION of a skill?\n"
            "2. **Added Value**: Does using this skill provide significant benefits, such as specific artistic styles, "
            "complex logic, or reference handling, that standard generation lacks?\n"
            "3. **Default to NONE**: If the request is simple, generic, or does not strongly align with any skill, "
            "choose no skills.\n\n"
            "### Response Requirement:\n"
            "Return only JSON with schema: {\"skills\":[\"skill_id\"]}. "
            "If no skill is a strong match, return {\"skills\":[]}."
        )
        raw_items = payload["skills"]
        if not isinstance(raw_items, list):
            raise TypeError("SkillQuery LLM field skills must be a list")
        available = set(self.skill_bank.available())
        selected = []
        for item in raw_items:
            name = str(item).strip()
            if name in available:
                selected.append(name)
        return list(dict.fromkeys(selected))

    def _infer_skills(self, prompt: str) -> list[str]:
        lowered = prompt.lower()
        names: list[str] = []
        if '"' in prompt or "text" in lowered:
            names.append("text_rendering")
        if any(token in lowered for token in ["left", "right", "above", "below", "behind", "front"]):
            names.append("spatial_layout")
        if re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten|\d+)\b", lowered):
            names.append("quantity_counting")
        return list(dict.fromkeys(names))
