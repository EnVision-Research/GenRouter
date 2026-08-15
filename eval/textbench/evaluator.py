from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.common.benchmark import GenerationRecord, Score


def evaluate_textbench(
    records: list[GenerationRecord],
    official_root: Path,
    work_dir: Path,
    config: Mapping[str, Any],
) -> list[Score]:
    language = str(config.get("language", "en")).lower()
    image_dir = work_dir / "images"
    output_dir = work_dir / "official_results"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_by_image: dict[tuple[int, int], GenerationRecord] = {}
    for record in records:
        metadata = dict(record.metadata.get("case_metadata") or {})
        prompt_id = int(metadata["prompt_id"])
        repeat_id = int(metadata["repeat_id"])
        shutil.copy2(
            record.image_path,
            image_dir / f"{prompt_id:04d}_{repeat_id}.png",
        )
        case_by_image[(prompt_id, repeat_id)] = record

    evaluate_command = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={int(config.get('nproc_per_node', 1))}",
        str(official_root / "textbench" / "evaluate_text_reward.py"),
        "--sample_dir",
        str(image_dir),
        "--output_dir",
        str(output_dir),
        "--mode",
        language,
    ]
    timeout = int(config.get("command_timeout_seconds", 3600))
    subprocess.run(
        evaluate_command,
        cwd=official_root / "textbench",
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as destination:
        for chunk in sorted(output_dir.glob("results_chunk*.jsonl")):
            destination.write(chunk.read_text(encoding="utf-8"))

    summary_command = [
        sys.executable,
        str(official_root / "textbench" / "summary_scores.py"),
        str(results_path),
        "--mode",
        language,
    ]
    subprocess.run(
        summary_command,
        cwd=official_root / "textbench",
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )

    by_case: dict[str, Score] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_stem, repeat_text = Path(row["image"]).stem.rsplit("_", 1)
        key = (int(prompt_stem), int(repeat_text))
        record = case_by_image.get(key)
        if record is None:
            continue
        metadata = dict(record.metadata.get("case_metadata") or {})
        by_case[record.case_id] = Score(
            case_id=record.case_id,
            value=float(row["text_accuray"]),
            metrics={
                "match_word_count": int(row["match_word_count"]),
                "gt_word_count": int(row["gt_word_count"]),
                "text_accuray": float(row["text_accuray"]),
                "ocr_gt": str(row.get("ocr_gt") or ""),
                "ocr_results": str(row.get("ocr_results") or ""),
                "category": str(metadata.get("category") or ""),
                "length": str(metadata.get("length") or ""),
                "repeat_id": key[1],
            },
        )

    scores: list[Score] = []
    for record in records:
        try:
            scores.append(by_case[record.case_id])
        except KeyError as exc:
            raise RuntimeError(
                f"LongText-Bench returned no score for case {record.case_id}"
            ) from exc
    return scores
