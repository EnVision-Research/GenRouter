from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.artimuse.evaluator import evaluate_records
from eval.common.benchmark import (
    Case,
    GenerationRecord,
    Score,
    require_official_checkout,
)


REVISION = "750d980b6b7e9d99da60a302dcdbcab14e01003f"


class Benchmark:
    name = "artimuse"

    def __init__(
        self,
        official_root: Path,
        prompts_path: Path,
        *,
        python: str,
        model_path: Path,
        device: str,
    ) -> None:
        self.official_root = Path(official_root)
        self.prompts_path = Path(prompts_path)
        self.python = python
        self.model_path = Path(model_path)
        self.device = device

    def load_cases(self) -> list[Case]:
        rows = [
            json.loads(line)
            for line in self.prompts_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        return [
            Case(f"{index:05d}", row["prompt"], row)
            for index, row in enumerate(rows)
        ]

    def image_path(self, case: Case, output_dir: Path) -> Path:
        return Path(output_dir) / f"{case.case_id}.png"

    def evaluate(
        self,
        records: list[GenerationRecord],
        work_dir: Path,
    ) -> list[Score]:
        return evaluate_records(
            records,
            Path(work_dir),
            official_root=self.official_root,
            python=self.python,
            model_path=self.model_path,
            device=self.device,
        )


def build_benchmark(config: Mapping[str, Any]) -> Benchmark:
    root = require_official_checkout(
        config["paths"]["official_root"],
        REVISION,
        ("src/eval/eval_dataset.py", "src/artimuse"),
    )
    model_path = Path(config["models"]["judge"])
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"ArtiMuse checkpoint directory not found: {model_path}"
        )
    evaluator = config["evaluator"]
    return Benchmark(
        root,
        config["paths"]["prompts"],
        python=str(evaluator["python"]),
        model_path=model_path,
        device=str(evaluator["device"]),
    )
