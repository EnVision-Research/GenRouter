from __future__ import annotations

from typing import Any, Protocol


class LLMBackend(Protocol):
    name: str

    def think(self, prompt: str) -> str:
        ...

    def think_json(self, prompt: str) -> dict[str, Any]:
        ...


class MLLMBackend(Protocol):
    name: str

    def think(self, prompt: str, images: list[bytes] | None = None) -> str:
        ...

    def think_json(self, prompt: str, images: list[bytes] | None = None) -> dict[str, Any]:
        ...


class GeneratorBackend(Protocol):
    name: str
    supports_reference: bool
    cost_per_call: float

    def generate(
        self,
        prompt: str,
        references: list[str] | None = None,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        ...


class ScorerBackend(Protocol):
    name: str

    def score(self, prompt: str, image: bytes | str) -> float:
        ...
