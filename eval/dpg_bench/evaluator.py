from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.common.benchmark import GenerationRecord, Score


def evaluate_dpg(
    records: list[GenerationRecord],
    official_root: Path,
    work_dir: Path,
    config: Mapping[str, Any],
) -> list[Score]:
    image_dir = work_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    expected: dict[Path, str] = {}
    for record in records:
        staged = image_dir / f"{record.case_id}.png"
        shutil.copy2(record.image_path, staged)
        expected[staged.resolve()] = record.case_id

    result_path = work_dir / "results.txt"
    command = [
        sys.executable,
        str(official_root / "dpg_bench" / "compute_dpg_bench.py"),
        "--image-root-path",
        str(image_dir),
        "--resolution",
        str(int(config.get("resolution", 1328))),
        "--csv",
        str(official_root / "dpg_bench" / "dpg_bench.csv"),
        "--res-path",
        str(result_path),
        "--pic-num",
        str(int(config.get("pic_num", 1))),
        "--vqa-model",
        str(config.get("vqa_model", "mplug")),
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
    for line in result_path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        case_id = expected.get(Path(fields[0]).resolve())
        if case_id is None:
            continue
        try:
            values = [float(field) for field in fields[1:] if field]
        except ValueError:
            continue
        if not values:
            continue
        by_case[case_id] = Score(
            case_id=case_id,
            value=values[-1],
            metrics={"sample_scores": values[:-1]},
        )

    scores: list[Score] = []
    for record in records:
        try:
            scores.append(by_case[record.case_id])
        except KeyError as exc:
            raise RuntimeError(
                f"DPG-Bench returned no score for case {record.case_id}"
            ) from exc
    return scores
