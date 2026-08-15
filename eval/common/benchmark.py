from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Case:
    case_id: str
    prompt: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenerationRecord:
    case_id: str
    prompt: str
    workflow: str
    generator: str
    image_path: Path
    result_path: Path
    task_signature: dict[str, Any]
    trace: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Score:
    case_id: str
    value: float
    metrics: dict[str, Any]


@runtime_checkable
class Benchmark(Protocol):
    name: str
    official_root: Path

    def load_cases(self) -> list[Case]:
        ...

    def image_path(self, case: Case, output_dir: Path) -> Path:
        ...

    def evaluate(
        self,
        records: list[GenerationRecord],
        work_dir: Path,
    ) -> list[Score]:
        ...


def require_official_checkout(
    root: str | Path,
    revision: str,
    required_paths: Iterable[str],
) -> Path:
    checkout = Path(root)
    if not checkout.is_dir():
        raise FileNotFoundError(f"Official benchmark checkout not found: {checkout}")
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != revision:
        raise RuntimeError(
            f"Official benchmark checkout is {actual}; expected revision {revision}: {checkout}"
        )
    missing = [
        str(checkout / relative)
        for relative in required_paths
        if not (checkout / relative).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Official benchmark files not found: {', '.join(missing)}"
        )
    return checkout.resolve()
