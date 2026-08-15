from __future__ import annotations

import time
from typing import Any

from genrouter.schemas import GenerateResult, PrimitiveTrace


class Generate:
    name = "Generate"

    def __call__(
        self,
        prompt: str,
        generator: Any,
        references: list[str] | None = None,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> GenerateResult:
        start = time.perf_counter()
        image = generator.generate(prompt=prompt, references=references, seed=seed, params=params)
        latency = time.perf_counter() - start
        cost = float(getattr(generator, "cost_per_call", 0.0))
        generator_name = str(getattr(generator, "name", "unknown_generator"))
        trace = PrimitiveTrace(
            primitive=self.name,
            backend=generator_name,
            input_summary=prompt[:160],
            output_summary=f"{len(image)} bytes",
            details={
                "prompt": prompt,
                "references": references or [],
                "seed": seed,
                "params": params or {},
                "image_bytes": len(image),
            },
            cost=cost,
            latency=latency,
        )
        return GenerateResult(
            image=image,
            generator=generator_name,
            cost=cost,
            latency=latency,
            trace=trace,
        )
