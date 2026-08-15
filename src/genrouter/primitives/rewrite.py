from __future__ import annotations

import time
import json
from typing import Any

from genrouter.primitives.costing import backend_call_cost, backend_cost_details
from genrouter.schemas import Evidence, PrimitiveTrace


class RewriteResult:
    def __init__(self, rewritten_prompt: str, trace: PrimitiveTrace) -> None:
        self.rewritten_prompt = rewritten_prompt
        self.trace = trace


class Rewrite:
    name = "Rewrite"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def __call__(
        self,
        prompt: str,
        evidence: list[Evidence] | None = None,
        skills: list[str] | None = None,
    ) -> RewriteResult:
        start = time.perf_counter()
        effective_evidence = list(evidence or [])
        evidence_dicts = [item.to_dict() for item in effective_evidence]
        rewritten = self._from_llm(prompt, evidence=evidence_dicts, skills=skills)
        if not rewritten:
            raise RuntimeError("Rewrite requires a real LLM response with rewritten_prompt")
        latency = time.perf_counter() - start
        backend = str(getattr(self.llm, "name", "unknown_rewrite"))
        trace = PrimitiveTrace(
            primitive=self.name,
            backend=backend,
            input_summary=prompt[:160],
            output_summary=rewritten[:160],
            details={
                "original_prompt": prompt,
                "rewritten_prompt": rewritten,
                "evidence": evidence_dicts,
                "skills": skills or [],
                **backend_cost_details(self.llm),
            },
            cost=backend_call_cost(self.llm),
            latency=latency,
        )
        return RewriteResult(rewritten_prompt=rewritten, trace=trace)

    def _from_llm(
        self,
        prompt: str,
        evidence: list[dict[str, Any]] | None,
        skills: list[str] | None,
    ) -> str:
        if self.llm is None or not hasattr(self.llm, "think_json"):
            return ""
        evidence_json = json.dumps(
            evidence or [],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        skill_text = self._skill_instructions(skills).strip() if skills else "None"

        prompt_text = (
            "Rewrite the user's image-generation prompt into a clear, self-contained prompt optimized for image generation.\n\n"
            "Return only JSON:\n"
            "{\"rewritten_prompt\":\"...\"}\n\n"
            "Requirements:\n"
            "- Preserve all explicit intent and constraints.\n"
            "- Clarify ambiguity with compatible visual details.\n"
            "- Use Evidence and Skill Instructions only to improve generation quality.\n"
            "- Never invent unsupported facts or alter the user's requested content.\n"
            "- Keep the prompt concise and coherent.\n"
            f"Original Prompt:\n{prompt}\n\n"
            f"Skill Instructions:\n{skill_text}\n\n"
            f"Evidence:\n{evidence_json}"
        )
        payload = self.llm.think_json(prompt_text)
        return str(payload.get("rewritten_prompt") or "").strip()

    def _skill_instructions(self, skills: list[str] | None) -> str:
        if not skills:
            return "None"
        return "\n".join(f"- {item}" for item in skills)
