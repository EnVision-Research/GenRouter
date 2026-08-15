from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.common.benchmark import GenerationRecord, Score


def evaluate_wise(
    records: list[GenerationRecord],
    official_root: Path,
    work_dir: Path,
    config: Mapping[str, Any],
) -> list[Score]:
    work_dir = work_dir.resolve()
    image_dir = work_dir / "images"
    output_dir = work_dir / "official_results"
    batch_json = work_dir / "verified_batch.json"
    image_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(dict(record.metadata.get("case_metadata") or {}))
        shutil.copy2(record.image_path, image_dir / f"{record.case_id}.png")
    batch_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(official_root / "vllm_eval.py"),
        "--json_path",
        str(batch_json),
        "--image_dir",
        str(image_dir),
        "--output_dir",
        str(output_dir),
        "--model",
        str(config["model"]),
        "--result_full",
        "full.json",
        "--result_scores",
        "scores.jsonl",
        "--api_base",
        str(config.get("api_base", "http://127.0.0.1:8000/v1")),
        "--max_workers",
        str(int(config.get("max_workers", 10))),
        "--timeout",
        str(int(config.get("timeout", 300))),
    ]
    subprocess.run(
        command,
        cwd=official_root,
        text=True,
        capture_output=True,
        check=True,
        timeout=int(config.get("command_timeout_seconds", 3600)),
    )

    by_case: dict[str, Score] = {}
    score_path = output_dir / "scores.jsonl"
    for line in score_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = float(row["score"])
        if value not in {0.0, 1.0}:
            raise RuntimeError(
                f"WISE returned non-binary score for case {row.get('prompt_id')}"
            )
        case_id = str(row["prompt_id"])
        by_case[case_id] = Score(
            case_id=case_id,
            value=value,
            metrics={
                "score": value,
                "subcategory": str(row.get("Subcategory") or ""),
            },
        )

    scores: list[Score] = []
    for record in records:
        try:
            scores.append(by_case[record.case_id])
        except KeyError as exc:
            raise RuntimeError(
                f"WISE returned no score for case {record.case_id}"
            ) from exc
    return scores
