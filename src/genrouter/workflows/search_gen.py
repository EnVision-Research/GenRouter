from __future__ import annotations

import time
from typing import Any

from genrouter.primitives.analyze import Analyze
from genrouter.primitives.generate import Generate
from genrouter.primitives.rewrite import Rewrite
from genrouter.primitives.search import Search
from genrouter.schemas import WorkflowResult
from genrouter.workflows.search_plan import planned_text_queries


class SearchGenWorkflow:
    name = "SearchGen"

    def __init__(self, search: Search, rewrite: Rewrite, generate: Generate, analyze: Analyze) -> None:
        self.analyze = analyze
        self.search = search
        self.rewrite = rewrite
        self.generate = generate

    def run(self, prompt: str, generator: Any, config: dict[str, Any], prompt_id: str = "") -> WorkflowResult:
        start = time.perf_counter()
        analyze_result = self.analyze(prompt)
        search_result = self.search.text(
            planned_text_queries(analyze_result, prompt),
            top_k=int(config.get("search_top_k", 3)),
        )
        rewrite_result = self.rewrite(prompt, evidence=[analyze_result.as_evidence(), *search_result.evidence])
        generate_result = self.generate(rewrite_result.rewritten_prompt, generator)
        trace = [analyze_result.trace, search_result.trace, rewrite_result.trace, generate_result.trace]
        return WorkflowResult(
            prompt_id=prompt_id or "manual",
            workflow=self.name,
            generator=generate_result.generator,
            final_prompt=rewrite_result.rewritten_prompt,
            final_image=generate_result.image,
            trace=trace,
            cost=sum(item.cost for item in trace),
            latency=time.perf_counter() - start,
        )
