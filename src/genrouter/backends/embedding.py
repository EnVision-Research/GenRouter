from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from genrouter.backends.http_client import exponential_retry_delay, request_with_retry


class EmbeddingRequestError(RuntimeError):
    """Raised when the configured embedding backend fails."""


@dataclass(frozen=True)
class EmbeddingBackendSpec:
    name: str
    base_url: str
    model: str
    api_key_env: str
    dimensions: int
    cache_path: str
    batch_size: int

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "EmbeddingBackendSpec":
        params = dict(data.get("default_params", {}) or {})
        return cls(
            name=str(data["default"]),
            base_url=str(data["base_url"]),
            model=str(data["model"]),
            api_key_env=str(data["api_key_env"]),
            dimensions=int(params.get("dimensions", 256)),
            cache_path=str(params.get("cache_path", "data/prompt_embeddings.jsonl")),
            batch_size=max(1, int(params.get("batch_size", 10))),
        )


@dataclass
class OpenAIEmbeddingBackend:
    spec: EmbeddingBackendSpec
    timeout: float = 120.0
    max_retries: int = 2
    _cache: dict[str, list[float]] = field(default_factory=dict, init=False)
    _cache_loaded: bool = field(default=False, init=False)

    @property
    def endpoint(self) -> str:
        return f"{self.spec.base_url.rstrip('/')}/embeddings"

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self._load_cache()
        keys = [self._cache_key(text) for text in texts]
        missing: list[str] = []
        seen: set[str] = set()
        for text, key in zip(texts, keys):
            if key not in self._cache and key not in seen:
                seen.add(key)
                missing.append(text)

        for start in range(0, len(missing), self.spec.batch_size):
            batch = missing[start : start + self.spec.batch_size]
            vectors = self._request(batch)
            for text, vector in zip(batch, vectors):
                self._cache[self._cache_key(text)] = vector
                self._append_cache(text, vector)
        return [self._cache[key] for key in keys]

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "input": texts,
            "dimensions": self.spec.dimensions,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

        def request() -> list[list[float]]:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list) or len(data) != len(texts):
                raise EmbeddingRequestError("Embedding response has invalid data")
            ordered = sorted(data, key=lambda item: int(item["index"]))
            vectors = [item.get("embedding") for item in ordered]
            if any(not isinstance(vector, list) or not vector for vector in vectors):
                raise EmbeddingRequestError("Embedding response has invalid vectors")
            return [[float(value) for value in vector] for vector in vectors]

        return request_with_retry(
            request,
            max_retries=self.max_retries,
            label=self.spec.name,
            error_cls=EmbeddingRequestError,
            retry_delay=exponential_retry_delay,
        )

    def _load_cache(self) -> None:
        if self._cache_loaded:
            return
        self._cache_loaded = True
        path = Path(self.spec.cache_path)
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("model") == self.spec.model:
                    self._cache[self._cache_key(str(item["text"]))] = [float(value) for value in item["embedding"]]

    def _append_cache(self, text: str, vector: list[float]) -> None:
        path = Path(self.spec.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"model": self.spec.model, "text": text, "embedding": vector},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _cache_key(self, text: str) -> str:
        return f"{self.spec.model}\n{text}"

    def _api_key(self) -> str:
        api_key = os.environ.get(self.spec.api_key_env)
        if not api_key:
            raise ValueError(f"Set {self.spec.api_key_env} before using embedding backend '{self.spec.name}'")
        return api_key


def build_embedding_backend(config: dict[str, Any]) -> OpenAIEmbeddingBackend:
    spec = EmbeddingBackendSpec.from_config(config)
    params = dict(config.get("default_params", {}) or {})
    return OpenAIEmbeddingBackend(
        spec=spec,
        timeout=float(params.get("timeout", 120.0)),
        max_retries=int(params.get("max_retries", 2)),
    )
