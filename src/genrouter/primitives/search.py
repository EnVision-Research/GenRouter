from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from genrouter.primitives.costing import backend_query_cost
from genrouter.schemas import Evidence, PrimitiveTrace


@dataclass(frozen=True)
class SearchResult:
    evidence: list[Evidence]
    references: list[str]
    trace: PrimitiveTrace


class SearchBackend(Protocol):
    name: str

    def text(self, query: str, top_k: int = 3) -> list[dict[str, Any]]: ...

    def image(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class Search:
    name = "Search"

    def __init__(self, backend: SearchBackend | None = None) -> None:
        if backend is None:
            raise RuntimeError("Search requires a configured real search backend")
        self.backend = backend

    def text(self, query: str | list[str], top_k: int = 3) -> SearchResult:
        start = time.perf_counter()
        queries = _normalize_queries(query)
        limit = max(0, int(top_k))
        evidence: list[dict[str, Any]] = []
        if limit:
            for item in queries:
                evidence.extend(_backend_items(self.backend.text(item, top_k=limit), "text"))
        evidence = _dedupe_evidence(evidence)
        evidence_items = [_text_to_evidence(item) for item in evidence]
        trace = PrimitiveTrace(
            primitive="SearchText",
            backend=str(getattr(self.backend, "name", "unknown_search")),
            input_summary=_query_summary(queries),
            output_summary=f"{len(evidence_items)} text evidence items from {len(queries)} queries",
            details={
                "queries": queries,
                "top_k": limit,
                "evidence": [item.to_dict() for item in evidence_items],
            },
            cost=backend_query_cost(self.backend, len(queries) if limit else 0),
            latency=time.perf_counter() - start,
        )
        return SearchResult(evidence=evidence_items, references=[], trace=trace)

    def image(self, query: str | list[str], top_k: int = 5) -> SearchResult:
        start = time.perf_counter()
        queries = _normalize_queries(query)
        candidate_limit = max(0, int(top_k))
        evidence: list[dict[str, Any]] = []
        if candidate_limit:
            for item in queries:
                candidates = _backend_items(self.backend.image(item, top_k=candidate_limit), "image")
                if not candidates:
                    raise RuntimeError(f"Image search returned no reference for query: {item}")
                evidence.append(candidates[0])
        references = [_image_reference(index, item) for index, item in enumerate(evidence)]
        evidence_items = [
            _image_to_evidence(item, reference)
            for item, reference in zip(evidence, references, strict=True)
        ]
        trace = PrimitiveTrace(
            primitive="SearchImage",
            backend=str(getattr(self.backend, "name", "unknown_search")),
            input_summary=_query_summary(queries),
            output_summary=f"{len(references)} image references from {len(queries)} queries",
            details={
                "queries": queries,
                "top_k": candidate_limit,
                "references": references,
                "evidence": [item.to_dict() for item in evidence_items],
            },
            cost=backend_query_cost(self.backend, len(queries) if candidate_limit else 0),
            latency=time.perf_counter() - start,
        )
        return SearchResult(evidence=evidence_items, references=references, trace=trace)


def _normalize_queries(query: str | list[str]) -> list[str]:
    raw_items: list[Any]
    if isinstance(query, list):
        raw_items = query
    else:
        raw_items = [query]
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = " ".join(str(item or "").split()).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _query_summary(queries: list[str]) -> str:
    return " | ".join(queries)[:160]


def _backend_items(value: Any, search_type: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"Search backend {search_type}() must return list[dict]")
    return value


def _dedupe_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _evidence_key(item: dict[str, Any]) -> tuple[str, str]:
    kind = str(item.get("kind") or "")
    for field in ("image_url", "local_path", "source_url"):
        value = str(item.get(field) or "").strip()
        if value:
            return (kind, value.casefold())
    text_key = "|".join(
        str(item.get(field) or "").strip().casefold()
        for field in ("title", "snippet", "reference_text", "query")
    )
    return (kind, text_key)


def _text_to_evidence(item: dict[str, Any]) -> Evidence:
    return Evidence(
        kind="text",
        content=str(item.get("snippet") or item.get("title") or ""),
        source=str(item.get("source_url") or ""),
        payload={
            "title": str(item.get("title") or ""),
            "query": str(item.get("query") or ""),
            "source": str(item.get("source") or ""),
        },
    )


def _image_to_evidence(item: dict[str, Any], reference: str) -> Evidence:
    return Evidence(
        kind="image",
        content=reference,
        source=str(item.get("source_url") or ""),
        reference_path=str(item.get("local_path") or ""),
        payload={
            "title": str(item.get("title") or ""),
            "query": str(item.get("query") or ""),
            "image_url": str(item.get("image_url") or ""),
            "downloaded_image_url": str(item.get("downloaded_image_url") or ""),
            "source": str(item.get("source") or ""),
        },
    )


def _image_reference(index: int, item: dict[str, Any]) -> str:
    text = str(item.get("snippet") or item.get("title") or item.get("image_url") or "")
    parts = [f"Reference cue {index + 1}: {text}"]
    if item.get("source_url"):
        parts.append(f"Source: {item['source_url']}.")
    if item.get("image_url"):
        parts.append(f"Image URL: {item['image_url']}.")
    if item.get("local_path"):
        parts.append(f"Local path: {item['local_path']}.")
    return " ".join(parts)
