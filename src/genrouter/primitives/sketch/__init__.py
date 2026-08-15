"""Code-sketch primitive.

Ask an LLM for executable visual code, render it to a PNG reference, and pass
that reference to downstream image generation. The minimal contract is code;
records and render instructions are optional metadata.
"""

from __future__ import annotations

import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from genrouter.primitives.costing import backend_cost_details, backend_token_cost, backend_token_usage
from genrouter.primitives.sketch.render import render_sketch
from genrouter.schemas import AnalyzeTargets, PrimitiveTrace, SketchResult, TaskSignature

SUPPORTED_SKETCH_TYPES = ("svg", "html_css", "threejs")
_ARTIFACT_ROOT = "genrouter_codesketch"
_EXTENSIONS = {"svg": ".svg", "html_css": ".html", "threejs": ".html"}

_SYSTEM_PROMPT = """Given an image-generation prompt, create a simple executable code sketch as a strict
layout reference for a later image generator. Preserve requested entities, counts,
spatial relations, and exact visible text.

Choose one sketch type:
- "svg": precise 2D layouts, diagrams, charts, posters, icons.
- "html_css": text-heavy posters, UI, page layouts.
- "threejs": 3D, perspective, depth, physical scenes.

Return ONLY strict JSON:
{
  "reasoning": ["brief planning note", "..."],
  "sketch_type": "svg | html_css | threejs",
  "records": [
    {
      "id": "r1",
      "kind": "entity | text | relation",
      "name": "short description",
      "count": 1,
      "details": "what and where"
    }
  ],
  "code": "complete directly renderable code",
  "render_prompt": "Follow this sketch as a strict layout reference."
}

Rules:
- Output JSON only; it must parse with json.loads.
- Use brief reasoning notes, not long chain-of-thought.
- records must be non-empty and match the code.
- Do not invent facts, brands, names, or extra visible text.
- Use simple shapes/placeholders when exact visual identity is not required.
- For svg, output a complete <svg> element.
- For html_css, output a complete HTML document with inline CSS.
- For "threejs": output JavaScript module body code only. THREE is already imported.
Create the scene by assigning to the existing variable `scene`, e.g.
`scene = new THREE.Scene();`. Do not write import statements, HTML, renderer setup,
or declare `const scene` / `let scene`. You may optionally assign `camera`.
- No external assets or network dependencies.
"""

_REPAIR_NOTE = """The previous sketch failed to render:
{error}

Return corrected strict JSON with the same keys. Fix the code, preserve the original
prompt requirements, and output JSON only."""


class Sketch:
    name = "Sketch"

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm

    def __call__(
        self,
        prompt: str,
        signature: TaskSignature,
        config: dict[str, Any],
        prompt_id: str = "",
        targets: AnalyzeTargets | None = None,
    ) -> SketchResult:
        start = time.perf_counter()
        width = int(config.get("canvas_width", 1024))
        height = int(config.get("canvas_height", 1024))
        requested_type = str(config.get("sketch_type", "auto") or "auto")
        artifact_dir = _artifact_dir(prompt_id)
        max_repairs = max(0, int(config.get("max_repair_attempts", 3)))

        payload = self._ask(prompt, signature, requested_type, width, height, targets)
        token_usage = _token_usage_accumulator()
        token_usage.add(self.llm)
        cost = backend_token_cost(self.llm)
        attempts = 1
        error = ""
        while True:
            try:
                plan = _parse_payload(payload, requested_type, prompt)
                image_path = artifact_dir / "sketch.png"
                render_sketch(
                    plan["sketch_type"], plan["code"], image_path,
                    width=width, height=height, records=plan["records"],
                )
                break
            except Exception as exc:  # noqa: BLE001 - surfaced to the repair LLM
                error = f"{type(exc).__name__}: {exc}"
                if attempts > max_repairs:
                    raise
                payload = self._ask(
                    prompt, signature, requested_type, width, height, targets,
                    previous_code=str(payload.get("code", "")), error=error,
                )
                token_usage.add(self.llm)
                cost += backend_token_cost(self.llm)
                attempts += 1

        return self._finish(
            prompt=prompt, signature=signature, plan=plan, artifact_dir=artifact_dir,
            image_path=image_path, latency=time.perf_counter() - start,
            token_usage=token_usage.data, cost=cost,
        )

    def _ask(
        self,
        prompt: str,
        signature: TaskSignature,
        requested_type: str,
        width: int,
        height: int,
        targets: AnalyzeTargets | None,
        previous_code: str = "",
        error: str = "",
    ) -> dict[str, Any]:
        context = ""
        if targets is not None:
            hints: list[str] = []
            for item in targets.reason:
                hints.append(f"- reason: {item}")
            for item in targets.search_image:
                hints.append(f"- entity: {item}")
            if hints:
                context = "\nAnalysis hints:\n" + "\n".join(hints)
        type_line = (
            "Choose the sketch_type yourself."
            if requested_type == "auto"
            else f"Use sketch_type: {requested_type}."
        )
        repair = f"\n\n{_REPAIR_NOTE}\nPrevious code:\n{previous_code}\nRender error: {error}" if error else ""
        message = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Canvas: {width}x{height}. {type_line}\n"
            f"Task signature: {signature.to_dict()}{context}\n\n"
            f"Prompt: {prompt}{repair}"
        )
        payload = self.llm.think_json(message)
        if not isinstance(payload, dict):
            raise TypeError("Sketch LLM must return a JSON object")
        return payload

    def _finish(
        self,
        *,
        prompt: str,
        signature: TaskSignature,
        plan: dict[str, Any],
        artifact_dir: Path,
        image_path: Path,
        latency: float,
        token_usage: dict[str, int],
        cost: float,
    ) -> SketchResult:
        sketch_type = plan["sketch_type"]
        records = plan["records"]
        code = plan["code"]
        reasoning = plan["reasoning"]
        render_prompt = plan["render_prompt"]

        code_path = artifact_dir / f"sketch{_EXTENSIONS.get(sketch_type, '.txt')}"
        code_path.write_text(code, encoding="utf-8")

        trace = PrimitiveTrace(
            primitive=self.name,
            backend=str(getattr(self.llm, "name", "unknown_sketch")),
            input_summary=prompt[:160],
            output_summary=f"{sketch_type} sketch rendered",
            details={
                "prompt": prompt,
                "signature": signature.to_dict(),
                "sketch_type": sketch_type,
                "records": records,
                "reasoning": reasoning,
                "code": code,
                "code_path": str(code_path),
                "sketch_image_path": str(image_path),
                "render_prompt": render_prompt,
                **_cost_details(self.llm, token_usage),
            },
            cost=cost,
            latency=latency,
        )
        return SketchResult(
            sketch_type=sketch_type,
            code=code,
            image_path=str(image_path),
            code_path=str(code_path),
            records=records,
            render_prompt=render_prompt,
            reasoning=reasoning,
            trace=trace,
        )


def _parse_payload(payload: dict[str, Any], requested_type: str, prompt: str = "") -> dict[str, Any]:
    sketch_type = str(payload.get("sketch_type") or (requested_type if requested_type != "auto" else "svg"))
    if sketch_type not in SUPPORTED_SKETCH_TYPES:
        raise ValueError(f"Unsupported sketch_type: {sketch_type}")

    code = str(payload.get("code") or "").strip()
    if not code:
        raise ValueError("Sketch response must include non-empty code")

    return {
        "sketch_type": sketch_type,
        "records": _records(payload.get("records"), prompt),
        "code": code,
        "render_prompt": str(payload.get("render_prompt") or "Use this rendered code sketch as a strict layout reference.").strip(),
        "reasoning": _string_list(payload.get("reasoning")),
    }


def _records(value: Any, prompt: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        return [{"id": "sketch", "kind": "sketch", "count": 1, "source": "synthetic", "prompt": prompt}]
    records: list[dict[str, Any]] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            continue
        item = dict(record)
        item.setdefault("id", f"record_{index + 1}")
        item.setdefault("kind", "entity")
        records.append(item)
    return records or [{"id": "sketch", "kind": "sketch", "count": 1, "source": "synthetic", "prompt": prompt}]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _artifact_dir(prompt_id: str) -> Path:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", prompt_id or "").strip("-") or uuid.uuid4().hex
    path = Path(tempfile.gettempdir()) / _ARTIFACT_ROOT / slug
    if path.exists():
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


class _TokenUsageAccumulator:
    def __init__(self) -> None:
        self.data: dict[str, int] = {}

    def add(self, backend: Any | None) -> None:
        for key, value in backend_token_usage(backend).items():
            self.data[key] = self.data.get(key, 0) + int(value)


def _token_usage_accumulator() -> _TokenUsageAccumulator:
    return _TokenUsageAccumulator()


def _cost_details(backend: Any | None, token_usage: dict[str, int]) -> dict[str, Any]:
    details = backend_cost_details(backend)
    if token_usage:
        details = dict(details)
        details["token_usage"] = dict(token_usage)
    return details


__all__ = ["Sketch", "SUPPORTED_SKETCH_TYPES"]
