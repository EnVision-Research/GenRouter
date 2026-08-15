from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.common.benchmark import (
    Benchmark,
    Case,
    GenerationRecord,
    Score,
    require_official_checkout,
)
from eval.wise.evaluator import evaluate_wise


REVISION = "09b5539d64681bf11102fc5c87e63a387beaf71d"
REQUIRED_PATHS = (
    "data_verified/merge.json",
    "vllm_eval.py",
    "calculate_verified.py",
)


def load_cases(root: Path) -> list[Case]:
    rows = json.loads(
        (root / "data_verified" / "merge.json").read_text(encoding="utf-8")
    )
    return [
        Case(str(row["prompt_id"]), str(row["Prompt"]).strip(), dict(row))
        for row in rows
    ]


class WiseBenchmark:
    name = "wise"

    def __init__(
        self,
        official_root: Path,
        evaluator_config: Mapping[str, Any],
    ) -> None:
        self.official_root = official_root
        self._evaluator_config = dict(evaluator_config)

    def load_cases(self) -> list[Case]:
        return load_cases(self.official_root)

    def image_path(self, case: Case, output_dir: Path) -> Path:
        return output_dir / f"{case.case_id}.png"

    def evaluate(
        self,
        records: list[GenerationRecord],
        work_dir: Path,
    ) -> list[Score]:
        return evaluate_wise(
            records,
            self.official_root,
            work_dir,
            self._evaluator_config,
        )


def build_benchmark(config: Mapping[str, Any]) -> Benchmark:
    paths = dict(config.get("paths", {}))
    root = require_official_checkout(
        Path(paths.get("official_root", "eval/wise/wise")),
        REVISION,
        REQUIRED_PATHS,
    )
    evaluator_config = dict(config.get("evaluator", {}))
    evaluator_config["model"] = str(dict(config.get("models", {}))["judge"])
    evaluator_config["api_base"] = str(
        evaluator_config.pop("judge_base_url", "http://127.0.0.1:8000/v1")
    )
    return WiseBenchmark(root, evaluator_config)
