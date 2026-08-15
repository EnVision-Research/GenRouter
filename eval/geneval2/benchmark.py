from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.common.benchmark import (
    Case,
    GenerationRecord,
    Score,
    require_official_checkout,
)
from eval.geneval2.evaluator import evaluate_records


REVISION = "a6e82d2289e8d418f27f0adee77908b07060eea3"


class Benchmark:
    name = "geneval2"

    def __init__(
        self,
        official_root: Path,
        *,
        python: str,
        method: str,
        hf_home: Path,
    ) -> None:
        self.official_root = Path(official_root)
        self.python = python
        self.method = method
        self.hf_home = Path(hf_home)

    def load_cases(self) -> list[Case]:
        data_path = self.official_root / "geneval2_data.jsonl"
        rows = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
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
            method=self.method,
            hf_home=self.hf_home,
        )


def build_benchmark(config: Mapping[str, Any]) -> Benchmark:
    root = require_official_checkout(
        config["paths"]["official_root"],
        REVISION,
        ("geneval2_data.jsonl", "evaluation.py"),
    )
    evaluator = config["evaluator"]
    return Benchmark(
        root,
        python=str(evaluator["python"]),
        method=str(evaluator["method"]),
        hf_home=Path(evaluator["hf_home"]),
    )
