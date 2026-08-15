from __future__ import annotations

import time
from typing import Any

from genrouter.primitives.analyze import Analyze
from genrouter.primitives.generate import Generate
from genrouter.primitives.sketch import Sketch
from genrouter.schemas import PrimitiveTrace, WorkflowResult
from genrouter.workflows.base import WorkflowExecutionError
from genrouter.workflows.signature import signature_from_config


class CodeSketchGenWorkflow:
    name = "CodeSketchGen"

    def __init__(self, analyze: Analyze, sketch: Sketch, generate: Generate) -> None:
        self.analyze = analyze
        self.sketch = sketch
        self.generate = generate

    def run(self, prompt: str, generator: Any, config: dict[str, Any], prompt_id: str = "") -> WorkflowResult:
        if not bool(getattr(generator, "supports_reference", False)):
            raise ValueError(f"Workflow {self.name} requires a reference-capable generator")
        start = time.perf_counter()
        analyze_result = self.analyze(prompt)
        signature = signature_from_config(config)
        sketch_result = self.sketch(
            prompt,
            signature=signature,
            config=config,
            prompt_id=prompt_id or "manual",
            targets=analyze_result.targets,
        )
        render_prompt = _render_prompt(prompt, sketch_result.render_prompt)
        try:
            generate_result = self.generate(
                render_prompt,
                generator,
                references=[sketch_result.image_path],
            )
        except Exception as exc:
            trace = [
                analyze_result.trace,
                sketch_result.trace,
                PrimitiveTrace(
                    primitive="Generate",
                    backend=str(getattr(generator, "name", "unknown_generator")),
                    input_summary=render_prompt[:160],
                    output_summary="failed",
                    details={
                        "prompt": render_prompt,
                        "references": [sketch_result.image_path],
                        "params": {},
                    },
                    cost=float(getattr(generator, "cost_per_call", 0.0)),
                    latency=0.0,
                    status="failed",
                    error=str(exc),
                ),
            ]
            partial = WorkflowResult(
                prompt_id=prompt_id or "manual",
                workflow=self.name,
                generator=str(getattr(generator, "name", "unknown_generator")),
                final_prompt=render_prompt,
                final_image=b"",
                trace=trace,
                cost=sum(item.cost for item in trace),
                latency=time.perf_counter() - start,
            )
            raise WorkflowExecutionError(str(exc), partial) from exc
        trace = [analyze_result.trace, sketch_result.trace, generate_result.trace]
        return WorkflowResult(
            prompt_id=prompt_id or "manual",
            workflow=self.name,
            generator=generate_result.generator,
            final_prompt=render_prompt,
            final_image=generate_result.image,
            trace=trace,
            cost=sum(item.cost for item in trace),
            latency=time.perf_counter() - start,
        )


def _render_prompt(prompt: str, sketch_instruction: str) -> str:
    return (
        f"{prompt}\n\n"
        "Refine the rendered code sketch provided as the reference image. "
        "Preserve its entity counts, spatial layout, text placement, geometry, and layer ordering. "
        "Add only visual detail, material, lighting, texture, and realism/style requested by the user.\n\n"
        f"Sketch instruction: {sketch_instruction}"
    )
