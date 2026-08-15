from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from genrouter.primitives.costing import backend_call_cost, backend_cost_details
from genrouter.schemas import PrimitiveTrace


@dataclass(frozen=True)
class ScoreResult:
    score: float
    trace: PrimitiveTrace


class Score:
    name = "Score"

    def __init__(self, scorer: Any) -> None:
        self.scorer = scorer

    def __call__(self, prompt: str, image: bytes | str) -> ScoreResult:
        start = time.perf_counter()
        score = float(self.scorer.score(prompt, image))
        latency = time.perf_counter() - start
        backend = str(getattr(self.scorer, "name", "unknown_scorer"))
        return ScoreResult(
            score=max(0.0, min(1.0, score)),
            trace=PrimitiveTrace(
                primitive=self.name,
                backend=backend,
                input_summary=prompt[:160],
                output_summary=f"score={score:.3f}",
                details={
                    "prompt": prompt,
                    "score": max(0.0, min(1.0, score)),
                    "image_bytes": len(image) if isinstance(image, bytes) else 0,
                    "image_path": image if isinstance(image, str) else "",
                    **backend_cost_details(self.scorer),
                },
                cost=backend_call_cost(self.scorer),
                latency=latency,
            ),
        )
