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
from eval.textbench.evaluator import evaluate_textbench


REVISION = "2b8237bb3789638c290eeda3e83ed81bd3652c3b"
REQUIRED_PATHS = (
    "textbench/text_prompts.jsonl",
    "textbench/text_prompts_zh.jsonl",
    "textbench/evaluate_text_reward.py",
    "textbench/summary_scores.py",
)


class TextBench:
    name = "textbench"

    def __init__(
        self,
        official_root: Path,
        evaluator_config: Mapping[str, Any],
    ) -> None:
        self.official_root = official_root
        self._evaluator_config = dict(evaluator_config)
        self.language = str(self._evaluator_config.get("language", "en")).lower()
        self.repeats = int(self._evaluator_config.get("repeats", 4))

    def load_cases(self) -> list[Case]:
        filename = (
            "text_prompts_zh.jsonl"
            if self.language == "zh"
            else "text_prompts.jsonl"
        )
        rows = [
            json.loads(line)
            for line in (
                self.official_root / "textbench" / filename
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cases: list[Case] = []
        for row in rows:
            prompt_id = int(row["prompt_id"])
            for repeat_id in range(1, self.repeats + 1):
                metadata = {
                    **row,
                    "language": self.language,
                    "repeat_id": repeat_id,
                }
                cases.append(
                    Case(
                        f"{self.language}:{prompt_id}:{repeat_id}",
                        str(row["prompt"]),
                        metadata,
                    )
                )
        return cases

    def image_path(self, case: Case, output_dir: Path) -> Path:
        prompt_id = int(case.metadata["prompt_id"])
        repeat_id = int(case.metadata["repeat_id"])
        return output_dir / f"{prompt_id:04d}_{repeat_id}.png"

    def evaluate(
        self,
        records: list[GenerationRecord],
        work_dir: Path,
    ) -> list[Score]:
        return evaluate_textbench(
            records,
            self.official_root,
            work_dir,
            self._evaluator_config,
        )


def build_benchmark(config: Mapping[str, Any]) -> Benchmark:
    paths = dict(config.get("paths", {}))
    root = require_official_checkout(
        Path(paths.get("official_root", "eval/textbench/X-Omni")),
        REVISION,
        REQUIRED_PATHS,
    )
    evaluator_config = dict(config.get("evaluator", {}))
    language = str(evaluator_config.get("language", "en")).lower()
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported TextBench language: {language}")
    repeats = int(evaluator_config.get("repeats", 4))
    if repeats != 4:
        raise ValueError("LongText-Bench release evaluation requires exactly 4 repeats")
    evaluator_config.update(language=language, repeats=repeats)
    return TextBench(root, evaluator_config)
