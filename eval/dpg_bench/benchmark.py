from __future__ import annotations

import re
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
from eval.dpg_bench.evaluator import evaluate_dpg


REVISION = "3c228f1dc6c4d3cad0a47493816151a419f14db3"
REQUIRED_PATHS = (
    "dpg_bench/prompts",
    "dpg_bench/dpg_bench.csv",
    "dpg_bench/compute_dpg_bench.py",
)


class DpgBenchmark:
    name = "dpg_bench"

    def __init__(
        self,
        official_root: Path,
        evaluator_config: Mapping[str, Any],
    ) -> None:
        self.official_root = official_root
        self._evaluator_config = dict(evaluator_config)

    def load_cases(self) -> list[Case]:
        prompt_dir = self.official_root / "dpg_bench" / "prompts"
        paths = sorted(
            prompt_dir.glob("*.txt"),
            key=lambda path: _natural_key(path.stem),
        )
        cases: list[Case] = []
        for path in paths:
            prompt = path.read_text(encoding="utf-8").strip()
            if prompt:
                cases.append(
                    Case(
                        path.stem,
                        prompt,
                        {"prompt_file": path.name},
                    )
                )
        return cases

    def image_path(self, case: Case, output_dir: Path) -> Path:
        return output_dir / f"{case.case_id}.png"

    def evaluate(
        self,
        records: list[GenerationRecord],
        work_dir: Path,
    ) -> list[Score]:
        return evaluate_dpg(
            records,
            self.official_root,
            work_dir,
            self._evaluator_config,
        )


def build_benchmark(config: Mapping[str, Any]) -> Benchmark:
    paths = dict(config.get("paths", {}))
    root = require_official_checkout(
        Path(paths.get("official_root", "eval/dpg_bench/ELLA")),
        REVISION,
        REQUIRED_PATHS,
    )
    return DpgBenchmark(root, dict(config.get("evaluator", {})))


def _natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.casefold())
        for token in re.split(r"(\d+)", value)
        if token
    )
