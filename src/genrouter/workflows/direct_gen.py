from __future__ import annotations

import time
from typing import Any

from genrouter.primitives.generate import Generate
from genrouter.schemas import WorkflowResult


class DirectGenWorkflow:
    name = "DirectGen"

    def __init__(self, generate: Generate) -> None:
        self.generate = generate

    def run(self, prompt: str, generator: Any, config: dict[str, Any], prompt_id: str = "") -> WorkflowResult:
        start = time.perf_counter()
        result = self.generate(prompt, generator)
        return WorkflowResult(
            prompt_id=prompt_id or "manual",
            workflow=self.name,
            generator=result.generator,
            final_prompt=prompt,
            final_image=result.image,
            trace=[result.trace],
            cost=result.cost,
            latency=time.perf_counter() - start,
        )
