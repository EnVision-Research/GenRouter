from __future__ import annotations

from genrouter.schemas import AnalyzeResult


def planned_text_queries(analysis: AnalyzeResult, prompt: str) -> list[str]:
    return list(analysis.targets.search_text or [prompt])


def planned_image_queries(analysis: AnalyzeResult, prompt: str) -> list[str]:
    return list(analysis.targets.search_image or [prompt])


def planned_reason_targets(analysis: AnalyzeResult, prompt: str) -> list[str]:
    return list(analysis.targets.reason or [prompt])
