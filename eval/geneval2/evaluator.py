from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from eval.common.benchmark import GenerationRecord, Score


def stage_inputs(
    records: list[GenerationRecord],
    work_dir: Path,
) -> tuple[Path, Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    benchmark_data = work_dir / "benchmark.jsonl"
    image_map = work_dir / "image_paths.json"
    output = work_dir / "score_lists.json"
    benchmark_data.write_text(
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
    image_map.write_text(
        json.dumps(
            {
                record.prompt: str(record.image_path.resolve())
                for record in records
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return benchmark_data, image_map, output


def build_command(
    *,
    python: str,
    official_root: Path,
    benchmark_data: Path,
    image_map: Path,
    method: str,
    output: Path,
) -> list[str]:
    return [
        python,
        str((official_root / "evaluation.py").resolve()),
        "--benchmark_data",
        str(benchmark_data.resolve()),
        "--image_filepath_data",
        str(image_map.resolve()),
        "--method",
        method,
        "--output_file",
        str(output.resolve()),
    ]


def parse_scores(
    records: list[GenerationRecord],
    score_lists: list[list[float]],
    *,
    method: str,
) -> list[Score]:
    if len(records) != len(score_lists):
        raise RuntimeError(
            f"GenEval2 returned {len(score_lists)} scores for {len(records)} records"
        )
    parsed: list[Score] = []
    for record, atom_scores in zip(records, score_lists, strict=True):
        metadata = dict(record.metadata.get("case_metadata") or {})
        values = [float(value) for value in atom_scores]
        if not values:
            raise RuntimeError(
                f"GenEval2 returned no atom scores for {record.case_id}"
            )
        am = sum(values) / len(values)
        gm = (
            0.0
            if 0.0 in values
            else math.exp(sum(math.log(value) for value in values) / len(values))
        )
        value = gm if method == "soft_tifa_gm" else am
        parsed.append(
            Score(
                record.case_id,
                value,
                {
                    "method": method,
                    "atom_scores": values,
                    "soft_tifa_am": am,
                    "soft_tifa_gm": gm,
                    "skills": list(metadata.get("skills", [])),
                    "atom_count": metadata.get("atom_count"),
                },
            )
        )
    return parsed


def evaluate_records(
    records: list[GenerationRecord],
    work_dir: Path,
    *,
    official_root: Path,
    python: str,
    method: str,
    hf_home: Path,
) -> list[Score]:
    benchmark_data, image_map, output = stage_inputs(records, work_dir)
    command = build_command(
        python=python,
        official_root=official_root,
        benchmark_data=benchmark_data,
        image_map=image_map,
        method=method,
        output=output,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": str(Path(hf_home).resolve()),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    with (work_dir / "evaluator.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=official_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return parse_scores(
        records,
        json.loads(output.read_text(encoding="utf-8")),
        method=method,
    )
