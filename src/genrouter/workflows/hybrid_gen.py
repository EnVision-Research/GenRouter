from __future__ import annotations

import time
from typing import Any

from genrouter.primitives.analyze import Analyze
from genrouter.primitives.decompose import Decompose
from genrouter.primitives.experience import ExperienceSummarizer
from genrouter.primitives.generate import Generate
from genrouter.primitives.reason import Reason
from genrouter.primitives.refine import Refine
from genrouter.primitives.rewrite import Rewrite
from genrouter.primitives.search import Search
from genrouter.primitives.sketch import Sketch
from genrouter.primitives.skill_query import SkillQuery
from genrouter.primitives.verify import Verify
from genrouter.schemas import PrimitiveTrace, WorkflowResult
from genrouter.workflows.search_plan import planned_image_queries, planned_reason_targets, planned_text_queries
from genrouter.workflows.signature import hybrid_branches, signature_from_config


class HybridGenWorkflow:
    name = "HybridGen"

    def __init__(
        self,
        search: Search,
        reason: Reason,
        skill_query: SkillQuery,
        rewrite: Rewrite,
        decompose: Decompose,
        generate: Generate,
        verify: Verify,
        refine: Refine,
        analyze: Analyze,
        sketch: Sketch,
        experience: ExperienceSummarizer | None = None,
    ) -> None:
        self.analyze = analyze
        self.search = search
        self.reason = reason
        self.skill_query = skill_query
        self.rewrite = rewrite
        self.decompose = decompose
        self.generate = generate
        self.verify = verify
        self.refine = refine
        self.sketch = sketch
        self.experience = experience

    def run(self, prompt: str, generator: Any, config: dict[str, Any], prompt_id: str = "") -> WorkflowResult:
        start = time.perf_counter()
        trace: list[PrimitiveTrace] = []
        prompt_id = prompt_id or "manual"
        signature = signature_from_config(config)
        branches = hybrid_branches(signature)
        supports_reference = bool(getattr(generator, "supports_reference", False))
        if branches["requires_reference"] and not supports_reference:
            raise ValueError(f"Workflow {self.name} requires a reference-capable generator for this task signature")
        if not branches["requires_reference"] and not _supports_text_to_image(generator):
            raise ValueError(f"Workflow {self.name} requires a text-to-image generator for this task signature")

        analyze_result = self.analyze(prompt)
        trace.append(analyze_result.trace)

        text_evidence = []
        image_evidence = []
        references: list[str] = []
        reasoning_evidence = []
        sketch_evidence = []
        skill_instructions: list[str] = []

        if branches["text_search"]:
            text_search = self.search.text(
                planned_text_queries(analyze_result, prompt),
                top_k=int(config.get("search_top_k", 3)),
            )
            trace.append(text_search.trace)
            text_evidence = text_search.evidence
        if branches["image_search"]:
            image_search = self.search.image(
                planned_image_queries(analyze_result, prompt),
                top_k=int(config.get("image_top_k", 5)),
            )
            trace.append(image_search.trace)
            image_evidence = image_search.evidence
            references = image_search.references
        if branches["reasoning"]:
            reason_result = self.reason(
                prompt,
                reason_targets=planned_reason_targets(analyze_result, prompt),
                evidence=text_evidence,
            )
            trace.append(reason_result.trace)
            reasoning_evidence = reason_result.evidence
        if branches["skills"]:
            skill_result = self.skill_query(prompt, signature=signature)
            trace.append(skill_result.trace)
            skill_instructions = skill_result.instructions
        if branches["sketch"]:
            sketch_config = _sketch_config(config)
            sketch_result = self.sketch(
                prompt,
                signature=signature,
                config=sketch_config,
                prompt_id=prompt_id,
                targets=analyze_result.targets,
            )
            trace.append(sketch_result.trace)
            references.append(sketch_result.image_path)
            sketch_evidence = [sketch_result.as_evidence()]

        evidence = [
            analyze_result.as_evidence(),
            *text_evidence,
            *image_evidence,
            *reasoning_evidence,
            *sketch_evidence,
        ]
        rewrite_result = self.rewrite(prompt, evidence=evidence, skills=skill_instructions)
        trace.append(rewrite_result.trace)
        decompose_result = self.decompose(prompt)
        trace.append(decompose_result.trace)

        current_prompt = rewrite_result.rewritten_prompt
        final_prompt = current_prompt
        final_image = b""
        score = 0.0
        attempt_history: list[dict[str, Any]] = []
        max_iter = int(config.get("max_iter", 2))
        if branches["requires_reference"] and not references:
            raise RuntimeError("HybridGen reference branch produced no reference image")
        for iteration in range(max_iter + 1):
            generate_result = self.generate(current_prompt, generator, references=references)
            final_image = generate_result.image
            trace.append(generate_result.trace)
            feedback = self.verify(
                generate_result.image,
                decompose_result.checklist,
                prompt=prompt,
                entities=decompose_result.entities,
                constraints=decompose_result.constraints,
                unknowns=decompose_result.unknowns,
            )
            trace.append(feedback.trace)
            score = feedback.overall_score
            final_prompt = current_prompt
            if not feedback.failed_items:
                break
            if iteration < max_iter:
                if self.experience is not None:
                    experience_result = self.experience(
                        original_prompt=prompt,
                        current_prompt=current_prompt,
                        feedback=feedback,
                        constraints=decompose_result.constraints,
                        attempt_history=attempt_history,
                        image=generate_result.image,
                    )
                    trace.append(experience_result.trace)
                    attempt_history.append(experience_result.attempt)
                refine_result = self.refine(
                    original_prompt=prompt,
                    current_prompt=current_prompt,
                    feedback=feedback,
                    attempt_history=attempt_history,
                    entities=decompose_result.entities,
                    constraints=decompose_result.constraints,
                )
                trace.append(refine_result.trace)
                current_prompt = refine_result.refined_prompt

        return WorkflowResult(
            prompt_id=prompt_id,
            workflow=self.name,
            generator=str(getattr(generator, "name", "unknown_generator")),
            final_prompt=final_prompt,
            final_image=final_image,
            trace=trace,
            score=score,
            cost=sum(item.cost for item in trace),
            latency=time.perf_counter() - start,
        )


def _supports_text_to_image(generator: Any) -> bool:
    spec = getattr(generator, "spec", None)
    tags = getattr(spec, "tags", []) or []
    return "text_to_image" in tags or not bool(getattr(generator, "supports_reference", False))


def _sketch_config(config: dict[str, Any]) -> dict[str, Any]:
    values = dict(config)
    values.setdefault("sketch_type", "auto")
    values.setdefault("canvas_width", 1024)
    values.setdefault("canvas_height", 1024)
    values.setdefault("max_repair_attempts", 1)
    return values
