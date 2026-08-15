from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Protocol

from genrouter.schemas import TaskSignature


SIGNATURE_FIELDS = list(TaskSignature.__dataclass_fields__)


DEFAULT_SIGNATURE_WEIGHT = 0.7
MIN_SIMILARITY_WEIGHT = 0.001


class EmbeddingBackend(Protocol):
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


def _signature_vector(signature: TaskSignature | dict[str, Any]) -> list[float]:
    value = signature if isinstance(signature, TaskSignature) else TaskSignature.from_dict(signature)
    data = value.to_dict()
    return [float(data[field]) for field in SIGNATURE_FIELDS]


def signature_similarity(a: TaskSignature, b: TaskSignature | dict[str, Any]) -> float:
    av = _signature_vector(a)
    bv = _signature_vector(b)
    differences = [abs(x - y) / 5.0 for x, y in zip(av, bv)]
    mean_distance = sum(differences) / len(differences)
    peak_distance = max(differences)
    changed_ratio = sum(value > 0.0 for value in differences) / len(differences)
    distance = (mean_distance + peak_distance + changed_ratio) / 3.0
    similarity = max(0.0, min(1.0, 1.0 - distance))
    return similarity**2


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Embedding vectors must have the same non-zero length")
    dot = sum(x * y for x, y in zip(a, b))
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(y * y for y in b))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (an * bn)))


def combined_similarity(
    task_signature: TaskSignature,
    row: dict[str, Any],
    prompt_similarity: float,
    signature_weight: float = DEFAULT_SIGNATURE_WEIGHT,
) -> float:
    input_data = row["input"]
    weight = max(0.0, min(1.0, float(signature_weight)))
    return (
        (1.0 - weight) * prompt_similarity
        + weight * signature_similarity(
            task_signature,
            TaskSignature.from_dict(input_data["task_signature"]),
        )
    )


class ExperienceBank:
    def __init__(
        self,
        path: str | Path,
        embedding_backend: EmbeddingBackend | None = None,
        signature_weight: float = DEFAULT_SIGNATURE_WEIGHT,
    ) -> None:
        self.path = Path(path)
        self.embedding_backend = embedding_backend
        self.signature_weight = max(0.0, min(1.0, float(signature_weight)))

    def add(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def estimate(
        self,
        task_signature: TaskSignature,
        workflow: str,
        generator: str,
        prompt: str = "",
        top_k: int = 20,
    ) -> dict[str, Any] | None:
        retrieved = self.retrieve_prompts(
            task_signature,
            prompt=prompt,
            top_k=top_k,
        )
        weighted_rows: list[tuple[float, dict[str, Any]]] = []
        matched_prompts: set[str] = set()
        for item in retrieved:
            similarity = float(item["similarity"])
            for row in item["records"]:
                plan = row["plan"]
                if plan["workflow"] == workflow and plan["generator"] == generator:
                    weighted_rows.append((similarity, row))
                    matched_prompts.add(str(item["prompt_key"]))
        if not weighted_rows:
            return None
        denom = sum(max(score, MIN_SIMILARITY_WEIGHT) for score, _ in weighted_rows)
        if denom <= 0.0:
            return None
        score = _weighted_metric_value(weighted_rows, "score", denom)
        cost = _weighted_execution_value(weighted_rows, "cost", denom)
        latency = _weighted_execution_value(weighted_rows, "latency_seconds", denom)
        utility = _weighted_metric_value(weighted_rows, "utility", denom)
        return {
            "score": score,
            "cost": cost,
            "latency": latency,
            "utility": utility,
            "source": "trajectory",
            "num_records": len(weighted_rows),
            "num_prompts": len(matched_prompts),
        }

    def retrieve(
        self,
        task_signature: TaskSignature,
        workflow: str,
        generator: str,
        prompt: str = "",
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.records()
            if row["plan"]["workflow"] == workflow
            and row["plan"]["generator"] == generator
        ]
        prompt_scores = self._prompt_similarities(prompt, [str(row["input"]["prompt"]) for row in rows])
        ranked = sorted(
            zip(rows, prompt_scores),
            key=lambda item: combined_similarity(
                task_signature,
                item[0],
                item[1],
                signature_weight=self.signature_weight,
            ),
            reverse=True,
        )
        return [row for row, _ in ranked[: max(0, int(top_k))]]

    def retrieve_prompts(
        self,
        task_signature: TaskSignature,
        prompt: str = "",
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for row in self.records():
            key = _prompt_key(row)
            input_data = row["input"]
            item = grouped.setdefault(
                key,
                {
                    "prompt_key": key,
                    "prompt": str(input_data["prompt"]),
                    "task_signature": input_data["task_signature"],
                    "records": [],
                },
            )
            item["records"].append(row)

        items = list(grouped.values())
        prompt_scores = self._prompt_similarities(prompt, [item["prompt"] for item in items])
        ranked: list[dict[str, Any]] = []
        for item, prompt_score in zip(items, prompt_scores):
            representative = {
                "input": {
                    "prompt": item["prompt"],
                    "task_signature": item["task_signature"],
                }
            }
            similarity = combined_similarity(
                task_signature,
                representative,
                prompt_score,
                signature_weight=self.signature_weight,
            )
            ranked.append({**item, "similarity": similarity})
        ranked.sort(key=lambda item: float(item["similarity"]), reverse=True)
        return ranked[: max(0, int(top_k))]

    def _prompt_similarities(self, prompt: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        if self.embedding_backend is not None:
            embeddings = self.embedding_backend.embed_many([prompt, *candidates])
            return [cosine_similarity(embeddings[0], embedding) for embedding in embeddings[1:]]
        return [text_similarity(prompt, candidate) for candidate in candidates]


def _text_vector(text: str) -> dict[str, float]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    vector: dict[str, float] = {}
    for token in tokens:
        vector[token] = vector.get(token, 0.0) + 1.0
    return vector


def text_similarity(a: str, b: str) -> float:
    av = _text_vector(a)
    bv = _text_vector(b)
    tokens = sorted(set(av) | set(bv))
    return cosine_similarity(
        [av.get(token, 0.0) for token in tokens],
        [bv.get(token, 0.0) for token in tokens],
    ) if tokens else 0.0


def _prompt_key(row: dict[str, Any]) -> str:
    input_data = row["input"]
    return " ".join(str(input_data["prompt"]).split()).casefold()


def _weighted_metric_value(
    weighted_rows: list[tuple[float, dict[str, Any]]],
    key: str,
    denom: float,
) -> float:
    total = 0.0
    for sim, row in weighted_rows:
        metrics = row["metrics"]
        value = metrics[key]
        total += max(sim, MIN_SIMILARITY_WEIGHT) * float(value or 0.0)
    return total / denom


def _weighted_execution_value(
    weighted_rows: list[tuple[float, dict[str, Any]]],
    key: str,
    denom: float,
) -> float:
    total = 0.0
    for sim, row in weighted_rows:
        execution = row["execution"]
        value = execution[key]
        total += max(sim, MIN_SIMILARITY_WEIGHT) * float(value or 0.0)
    return total / denom
