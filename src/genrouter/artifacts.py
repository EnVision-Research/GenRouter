from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from genrouter.schemas import TaskSignature, WorkflowResult

_LOCAL_PATH_RE = re.compile(r"Local path:\s*(?P<path>[^\n]+)")


def _image_extension(image: bytes) -> str:
    if image.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image.startswith(b"GIF87a") or image.startswith(b"GIF89a"):
        return ".gif"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def write_workflow_artifacts(
    result: WorkflowResult,
    output_dir: str | Path,
    *,
    prompt: str | None = None,
    task_signature: TaskSignature | None = None,
) -> dict[str, str]:
    run_dir = Path(output_dir) / result.prompt_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    trace_path = run_dir / "trace.jsonl"
    paths = {
        "run_dir": str(run_dir),
        "result_path": str(result_path),
        "trace_path": str(trace_path),
    }
    result_payload = result.to_dict()
    payload = {"prompt_id": result_payload.pop("prompt_id")}
    if prompt is not None:
        payload["prompt"] = prompt
    if task_signature is not None:
        payload["task_signature"] = task_signature.to_dict()
    payload.update(result_payload)
    if result.final_image:
        image_path = run_dir / f"final_image{_image_extension(result.final_image)}"
        image_path.write_bytes(result.final_image)
        payload["final_image_path"] = str(image_path)
        paths["image_path"] = str(image_path)
    sketch_paths = _copy_sketch_artifacts(result, run_dir)
    if sketch_paths:
        payload["sketch_artifacts"] = sketch_paths
        paths.update(sketch_paths)
    reference_paths = _copy_reference_artifacts(result, run_dir)
    if reference_paths:
        payload["reference_artifacts"] = reference_paths
        paths.update(reference_paths)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with trace_path.open("w", encoding="utf-8") as handle:
        for item in result.trace:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return paths


def _copy_sketch_artifacts(result: WorkflowResult, run_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for trace in result.trace:
        if trace.primitive != "Sketch":
            continue
        details = trace.details or {}
        code_path = Path(str(details.get("code_path") or ""))
        image_path = Path(str(details.get("sketch_image_path") or ""))
        if code_path.is_file():
            _remove_stale_sketch_code_files(run_dir)
            target = run_dir / f"sketch{code_path.suffix or '.txt'}"
            shutil.copyfile(code_path, target)
            copied["sketch_code_path"] = str(target)
        if image_path.is_file():
            target = run_dir / "sketch.png"
            shutil.copyfile(image_path, target)
            copied["sketch_image_path"] = str(target)
    return copied


def _copy_reference_artifacts(result: WorkflowResult, run_dir: Path) -> dict[str, str]:
    if result.workflow not in {"RefGen", "HybridGen"}:
        return {}
    copied: dict[str, str] = {}
    reference_index = 1
    seen: set[Path] = set()
    for trace in result.trace:
        if trace.primitive != "Generate":
            continue
        details = trace.details or {}
        references = details.get("references") or []
        if not isinstance(references, list):
            continue
        for reference in references:
            source = _reference_local_path(reference)
            if source is None:
                continue
            resolved = source.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            target = run_dir / f"reference{reference_index}{source.suffix.lower() or '.bin'}"
            shutil.copyfile(source, target)
            copied[f"reference{reference_index}_path"] = str(target)
            reference_index += 1
    return copied


def _reference_local_path(reference: object) -> Path | None:
    if not isinstance(reference, str):
        return None
    candidates = [reference.strip()]
    match = _LOCAL_PATH_RE.search(reference)
    if match:
        candidates.insert(0, match.group("path").strip().rstrip(").,;]"))
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and _is_image_path(path):
            return path
    return None


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def _remove_stale_sketch_code_files(run_dir: Path) -> None:
    for suffix in (".svg", ".html", ".py", ".json", ".txt"):
        path = run_dir / f"sketch{suffix}"
        if path.is_file():
            path.unlink()
