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
from eval.spatialgeneval.evaluator import evaluate_records


REVISION = "8b294b5fd0bca204fcfbf2cd74b75d9c359e40f6"


class Benchmark:
    name = "spatialgeneval"

    def __init__(
        self,
        official_root: Path,
        *,
        python: str,
        base_url: str,
        api_name: str,
        rollout: int,
        count: int,
        temperature: float,
        max_workers: int,
    ) -> None:
        self.official_root = Path(official_root)
        self.python = python
        self.base_url = base_url
        self.api_name = api_name
        self.rollout = rollout
        self.count = count
        self.temperature = temperature
        self.max_workers = max_workers

    def load_cases(self) -> list[Case]:
        path = self.official_root / "eval/SpatialGenEval_T2I_Prompts.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        return [Case(str(row["id"]), row["prompt"], row) for row in rows]

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
            base_url=self.base_url,
            api_name=self.api_name,
            rollout=self.rollout,
            count=self.count,
            temperature=self.temperature,
            max_workers=self.max_workers,
        )


def build_benchmark(config: Mapping[str, Any]) -> Benchmark:
    root = require_official_checkout(
        config["paths"]["official_root"],
        REVISION,
        (
            "eval/SpatialGenEval_T2I_Prompts.jsonl",
            "scripts/spatialgeneval_stage1_eval.py",
            "scripts/spatialgeneval_stage2_acc.py",
        ),
    )
    evaluator = config["evaluator"]
    service = config["services"]["judge"]
    return Benchmark(
        root,
        python=str(evaluator["python"]),
        base_url=str(service["base_url"]),
        api_name=str(service["model"]),
        rollout=int(evaluator["rollout"]),
        count=int(evaluator["count"]),
        temperature=float(evaluator["temperature"]),
        max_workers=int(evaluator["max_workers"]),
    )
