from __future__ import annotations

import csv
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
from eval.oneig.evaluator import evaluate_oneig


CATEGORY_DIRS = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
    "Multilingualism": "multilingualism",
}
REVISION = "41b49831e79e6dde5323618c164da1c4cf0f699d"
REQUIRED_PATHS = (
    "OneIG-Bench.csv",
    "OneIG-Bench-ZH.csv",
    "scripts/alignment/alignment_score.py",
    "scripts/style/style_score.py",
    "scripts/style/models/checkpoint.pth",
    "scripts/style/models/ViT-L-14.pt",
    "scripts/text/text_score.py",
    "scripts/reasoning/reasoning_score.py",
)


class OneigBenchmark:
    name = "oneig"

    def __init__(
        self,
        official_root: Path,
        evaluator_config: Mapping[str, Any],
    ) -> None:
        self.official_root = official_root
        self._evaluator_config = dict(evaluator_config)
        self.language = str(self._evaluator_config.get("language", "en")).lower()

    def load_cases(self) -> list[Case]:
        csv_name = "OneIG-Bench-ZH.csv" if self.language == "zh" else "OneIG-Bench.csv"
        prompt_field = "prompt_cn" if self.language == "zh" else "prompt_en"
        cases: list[Case] = []
        with (self.official_root / csv_name).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                category = str(row.get("category") or "").strip()
                local_id = str(row.get("id") or "").strip()
                prompt = str(row.get(prompt_field) or "").strip()
                if not category or not local_id or not prompt:
                    continue
                category_dir = CATEGORY_DIRS[category]
                metadata = {
                    **row,
                    "language": self.language,
                    "mode": self.language.upper(),
                    "category_dir": category_dir,
                    "local_id": local_id,
                }
                cases.append(
                    Case(
                        f"{self.language}:{category_dir}:{local_id}",
                        prompt,
                        metadata,
                    )
                )
        return cases

    def image_path(self, case: Case, output_dir: Path) -> Path:
        return output_dir / f"{case.case_id.replace(':', '__')}.png"

    def evaluate(
        self,
        records: list[GenerationRecord],
        work_dir: Path,
    ) -> list[Score]:
        return evaluate_oneig(
            records,
            self.official_root,
            work_dir,
            self._evaluator_config,
        )


def build_benchmark(config: Mapping[str, Any]) -> Benchmark:
    paths = dict(config.get("paths", {}))
    root = require_official_checkout(
        Path(paths.get("official_root", "eval/oneig/OneIG-Benchmark")),
        REVISION,
        REQUIRED_PATHS,
    )
    evaluator_config = dict(config.get("evaluator", {}))
    language = str(evaluator_config.get("language", "en")).lower()
    if language not in {"en", "zh"}:
        raise ValueError(f"Unsupported OneIG language: {language}")
    evaluator_config["language"] = language
    return OneigBenchmark(root, evaluator_config)
