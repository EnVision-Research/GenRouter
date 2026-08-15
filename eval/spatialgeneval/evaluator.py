from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from eval.common.benchmark import GenerationRecord, Score


def stage_inputs(
    records: list[GenerationRecord],
    work_dir: Path,
) -> tuple[Path, Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    image_dir = work_dir / "images"
    image_dir.mkdir(exist_ok=True)
    input_json = work_dir / "input.jsonl"
    output = work_dir / "official_results.jsonl"
    if output.exists():
        output.unlink()
    input_json.write_text(
        "".join(
            json.dumps(
                dict(record.metadata.get("case_metadata") or {}),
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    for index, record in enumerate(records):
        target = image_dir / f"{index:06d}{record.image_path.suffix.lower()}"
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(record.image_path.resolve())
    return input_json, image_dir, output


def build_commands(
    *,
    python: str,
    official_root: Path,
    input_json: Path,
    image_dir: Path,
    output: Path,
    base_url: str,
    api_name: str,
    rollout: int,
    count: int,
    temperature: float,
    max_workers: int,
) -> tuple[list[str], list[str]]:
    stage1 = [
        python,
        str(
            (official_root / "scripts/spatialgeneval_stage1_eval.py").resolve()
        ),
        "--input_json",
        str(input_json.resolve()),
        "--image_pth",
        str(image_dir.resolve()),
        "--output_json",
        str(output.resolve()),
        "--base_url",
        base_url,
        "--api_name",
        api_name,
        "--rollout",
        str(rollout),
        "--count",
        str(count),
        "--temperature",
        str(temperature),
        "--max_workers",
        str(max_workers),
    ]
    stage2 = [
        python,
        str(
            (official_root / "scripts/spatialgeneval_stage2_acc.py").resolve()
        ),
        str(output.resolve()),
        "--min_count",
        str(count),
    ]
    return stage1, stage2


def parse_results(
    records: list[GenerationRecord],
    payload: list[dict[str, object]],
) -> list[Score]:
    by_id = {str(row["id"]): row for row in payload}
    expected = {record.case_id for record in records}
    if set(by_id) != expected:
        raise RuntimeError(
            "SpatialGenEval score IDs mismatch: "
            f"expected={sorted(expected)} actual={sorted(by_id)}"
        )
    scores: list[Score] = []
    for record in records:
        metadata = dict(record.metadata.get("case_metadata") or {})
        row = by_id[record.case_id]
        results = [bool(value) for value in row["true-or-false"]]
        if len(results) != 10:
            raise RuntimeError(
                "SpatialGenEval returned "
                f"{len(results)} question results for {record.case_id}"
            )
        scores.append(
            Score(
                record.case_id,
                sum(results) / 10.0,
                {
                    "question_results": results,
                    "basic_accuracy": sum(results[:2]) / 2.0,
                    "spatial_accuracy": sum(results[2:]) / 8.0,
                    "question_type": list(metadata.get("question_type", [])),
                    "model_preds_cot": row["model_preds_cot"],
                },
            )
        )
    return scores


def evaluate_records(
    records: list[GenerationRecord],
    work_dir: Path,
    *,
    official_root: Path,
    python: str,
    base_url: str,
    api_name: str,
    rollout: int,
    count: int,
    temperature: float,
    max_workers: int,
) -> list[Score]:
    input_json, image_dir, output = stage_inputs(records, work_dir)
    stage1, stage2 = build_commands(
        python=python,
        official_root=official_root,
        input_json=input_json,
        image_dir=image_dir,
        output=output,
        base_url=base_url,
        api_name=api_name,
        rollout=rollout,
        count=count,
        temperature=temperature,
        max_workers=max_workers,
    )
    environment = os.environ.copy()
    environment.setdefault("OPENAI_API_KEY", "EMPTY")
    with (Path(work_dir) / "stage1.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            stage1,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    scores = parse_results(records, rows)
    with (Path(work_dir) / "stage2.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            stage2,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return scores
