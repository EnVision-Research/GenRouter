from __future__ import annotations

import time
from typing import Any

from genrouter.primitives.analyze import Analyze
from genrouter.primitives.decompose import Decompose
from genrouter.primitives.experience import ExperienceSummarizer
from genrouter.primitives.generate import Generate
from genrouter.primitives.refine import Refine
from genrouter.primitives.rewrite import Rewrite
from genrouter.primitives.verify import Verify
from genrouter.schemas import PrimitiveTrace, WorkflowResult


class VerifyRefineWorkflow:
    name = "VerifyRefine"

    def __init__(
        self,
        analyze: Analyze,
        decompose: Decompose,
        generate: Generate,
        verify: Verify,
        refine: Refine,
        rewrite: Rewrite,
        experience: ExperienceSummarizer | None = None,
    ) -> None:
        self.analyze = analyze
        self.decompose = decompose
        self.generate = generate
        self.verify = verify
        self.refine = refine
        self.rewrite = rewrite
        self.experience = experience

    def run(self, prompt: str, generator: Any, config: dict[str, Any], prompt_id: str = "") -> WorkflowResult:
        start = time.perf_counter()
        prompt_id = prompt_id or "manual"
        max_iter = int(config.get("max_iter", 2))

        trace: list[PrimitiveTrace] = []
        analyze_result = self.analyze(prompt)
        trace.append(analyze_result.trace)
        decompose_result = self.decompose(prompt)
        checklist = decompose_result.checklist
        trace.append(decompose_result.trace)

        rewrite_result = self.rewrite(prompt, evidence=[analyze_result.as_evidence()])
        trace.append(rewrite_result.trace)
        current_prompt = rewrite_result.rewritten_prompt
        final_prompt = current_prompt
        final_image = b""
        score = 0.0
        attempt_history: list[dict[str, Any]] = []

        for iteration in range(max_iter + 1):
            generate_result = self.generate(current_prompt, generator)
            final_image = generate_result.image
            trace.append(generate_result.trace)

            feedback = self.verify(
                generate_result.image,
                checklist,
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
            final_image_path="",
            trace=trace,
            score=score,
            cost=sum(item.cost for item in trace),
            latency=time.perf_counter() - start,
        )
