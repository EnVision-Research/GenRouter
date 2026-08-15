from __future__ import annotations

import json
import subprocess
from pathlib import Path

from eval.common.benchmark import GenerationRecord, Score


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def stage_runtime(
    records: list[GenerationRecord],
    work_dir: Path,
    *,
    official_root: Path,
    model_path: Path,
) -> tuple[Path, Path]:
    runtime = Path(work_dir) / "official_runtime"
    images = runtime / "test_datasets/GenRouter/images"
    images.mkdir(parents=True, exist_ok=True)
    _replace_symlink(runtime / "src", Path(official_root) / "src")
    _replace_symlink(runtime / "checkpoints/ArtiMuse", Path(model_path))
    rows: list[dict[str, str]] = []
    for record in records:
        filename = f"{record.case_id}{record.image_path.suffix.lower()}"
        _replace_symlink(images / filename, record.image_path)
        rows.append({"image": filename})
    (runtime / "test_datasets/GenRouter/test.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return runtime, runtime / "results/dataset_results/GenRouter_ArtiMuse.json"


def build_command(
    *,
    python: str,
    official_root: Path,
    device: str,
) -> list[str]:
    return [
        python,
        str((official_root / "src/eval/eval_dataset.py").resolve()),
        "--model_name",
        "ArtiMuse",
        "--dataset",
        "GenRouter",
        "--device",
        device,
    ]


def parse_results(
    records: list[GenerationRecord],
    payload: list[dict[str, object]],
) -> list[Score]:
    by_id = {Path(str(row["image"])).stem: float(row["score"]) for row in payload}
    expected = {record.case_id for record in records}
    if set(by_id) != expected:
        raise RuntimeError(
            "ArtiMuse score IDs mismatch: "
            f"expected={sorted(expected)} actual={sorted(by_id)}"
        )
    return [
        Score(
            record.case_id,
            by_id[record.case_id] / 100.0,
            {"aesthetic_score": by_id[record.case_id]},
        )
        for record in records
    ]


def evaluate_records(
    records: list[GenerationRecord],
    work_dir: Path,
    *,
    official_root: Path,
    python: str,
    model_path: Path,
    device: str,
) -> list[Score]:
    runtime, output = stage_runtime(
        records,
        work_dir,
        official_root=official_root,
        model_path=model_path,
    )
    command = build_command(
        python=python,
        official_root=official_root,
        device=device,
    )
    with (Path(work_dir) / "evaluator.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=runtime,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return parse_results(
        records,
        json.loads(output.read_text(encoding="utf-8")),
    )
