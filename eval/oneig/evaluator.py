from __future__ import annotations

import ast
import csv
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.common.benchmark import GenerationRecord, Score


METRICS_BY_CATEGORY = {
    "anime": ("alignment", "style"),
    "human": ("alignment",),
    "object": ("alignment",),
    "text": ("text",),
    "reasoning": ("reasoning",),
    "multilingualism": ("alignment",),
}
TEXT_MAX_EDIT_DISTANCE = {"EN": 100.0, "ZH": 50.0}


def evaluate_oneig(
    records: list[GenerationRecord],
    official_root: Path,
    work_dir: Path,
    config: Mapping[str, Any],
) -> list[Score]:
    mode = str(config.get("language", "en")).upper()
    model_name = str(config.get("model_name", "genrouter"))
    image_grid = str(config.get("image_grid", "1,1"))
    image_root = work_dir / "images"
    work_dir.mkdir(parents=True, exist_ok=True)
    scripts_root = work_dir / "scripts"
    if scripts_root.is_symlink():
        scripts_root.unlink()
    elif scripts_root.exists():
        raise RuntimeError(f"OneIG work directory contains a real scripts path: {scripts_root}")
    scripts_root.symlink_to(
        (official_root / "scripts").resolve(),
        target_is_directory=True,
    )
    results_dir = work_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)

    configured_prefix = config.get("command_prefix")
    if configured_prefix is None:
        command_prefix = [sys.executable]
    elif isinstance(configured_prefix, (list, tuple)) and all(
        isinstance(part, str) and part for part in configured_prefix
    ):
        command_prefix = list(configured_prefix)
    else:
        raise ValueError("OneIG evaluator.command_prefix must be a non-empty argv list")
    if not command_prefix:
        raise ValueError("OneIG evaluator.command_prefix must be a non-empty argv list")

    metadata_by_case: dict[str, dict[str, Any]] = {}
    categories: list[str] = []
    for record in records:
        metadata = dict(record.metadata.get("case_metadata") or {})
        category = str(metadata["category_dir"])
        local_id = str(metadata["local_id"])
        target = image_root / category / model_name / f"{local_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.image_path, target)
        metadata_by_case[record.case_id] = metadata
        categories.append(category)

    present = list(dict.fromkeys(categories))
    required_metrics = {
        metric
        for category in present
        for metric in METRICS_BY_CATEGORY[category]
    }
    base = [
        "--mode",
        mode,
        "--model_names",
        model_name,
        "--image_grid",
        image_grid,
    ]
    commands: list[list[str]] = []
    alignment_categories = [
        category for category in present if "alignment" in METRICS_BY_CATEGORY[category]
    ]
    if "alignment" in required_metrics:
        commands.append(
            [
                *command_prefix,
                "-m",
                "scripts.alignment.alignment_score",
                *base,
                "--image_dirname",
                str(image_root),
                "--class_items",
                *alignment_categories,
            ]
        )
    if "style" in required_metrics:
        commands.append(
            [
                *command_prefix,
                "-m",
                "scripts.style.style_score",
                *base,
                "--image_dirname",
                str(image_root / "anime"),
            ]
        )
    if "text" in required_metrics:
        commands.append(
            [
                *command_prefix,
                "-m",
                "scripts.text.text_score",
                *base,
                "--image_dirname",
                str(image_root / "text"),
            ]
        )
    if "reasoning" in required_metrics:
        commands.append(
            [
                *command_prefix,
                "-m",
                "scripts.reasoning.reasoning_score",
                *base,
                "--image_dirname",
                str(image_root / "reasoning"),
            ]
        )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(official_root.resolve()) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    for command in commands:
        subprocess.run(
            command,
            cwd=work_dir,
            env=env,
            text=True,
            capture_output=True,
            check=True,
            timeout=int(config.get("command_timeout_seconds", 3600)),
        )

    metric_rows = {
        metric: _read_metric_rows(
            results_dir,
            metric,
            mode,
            model_name,
        )
        for metric in required_metrics
    }
    scores: list[Score] = []
    for record in records:
        metadata = metadata_by_case[record.case_id]
        category = str(metadata["category_dir"])
        local_id = str(metadata["local_id"])
        metrics: dict[str, float] = {}
        for metric in METRICS_BY_CATEGORY[category]:
            key = f"{category}_{local_id}" if metric == "alignment" else local_id
            try:
                raw = metric_rows[metric][key]
            except KeyError as exc:
                raise RuntimeError(
                    f"OneIG {metric} returned no score for case {record.case_id}"
                ) from exc
            if metric == "text":
                values = ast.literal_eval(raw)
                if not isinstance(values, (list, tuple)) or len(values) != 3:
                    raise RuntimeError(f"Invalid OneIG text score for case {record.case_id}")
                ed, cr, wac = (float(value) for value in values)
                if ed < 0 or not 0.0 <= cr <= 1.0 or not 0.0 <= wac <= 1.0:
                    raise RuntimeError(f"Invalid OneIG text score for case {record.case_id}")
                metrics.update(text_ed=ed, text_cr=cr, text_wac=wac)
            else:
                value = float(raw)
                if not 0.0 <= value <= 1.0:
                    raise RuntimeError(
                        f"Invalid OneIG {metric} score for case {record.case_id}"
                    )
                metrics[metric] = value

        if category == "anime":
            value = (metrics["alignment"] + metrics["style"]) / 2.0
        elif category == "text":
            ed = metrics["text_ed"]
            cr = metrics["text_cr"]
            wac = metrics["text_wac"]
            maximum = TEXT_MAX_EDIT_DISTANCE[mode]
            value = 1.0 - min(maximum, ed) * (1.0 - cr) * (1.0 - wac) / maximum
        elif category == "reasoning":
            value = metrics["reasoning"]
        else:
            value = metrics["alignment"]
        scores.append(Score(record.case_id, value, metrics))
    return scores


def _read_metric_rows(
    results_dir: Path,
    metric: str,
    mode: str,
    model_name: str,
) -> dict[str, str]:
    paths = list(results_dir.glob(f"{metric}_prompt_score_{mode}_*.csv"))
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one OneIG {metric} result for {mode}, found {len(paths)}"
        )
    with paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or model_name not in rows[0]:
        raise RuntimeError(f"OneIG {metric} result has no {model_name} column")
    model_index = rows[0].index(model_name)
    return {
        row[0].strip(): row[model_index].strip()
        for row in rows[1:]
        if row and len(row) > model_index and row[0].strip()
    }
