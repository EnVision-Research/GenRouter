from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genrouter.backends.chat import ChatBackendSpec, ChatCompletionBackend


@dataclass
class MLLMScorerBackend:
    chat: ChatCompletionBackend

    @property
    def name(self) -> str:
        return f"{self.chat.name}_scorer"

    @property
    def score_source(self) -> str:
        return "scorer"

    def score(self, prompt: str, image: bytes | str) -> float:
        images = [_image_bytes(image)]
        payload = self.chat.think_json(
            "Score the generated image for prompt adherence and visual quality. "
            "Return only JSON with schema: "
            '{"score":0.0,"rationale":"short reason"}. '
            "The score must be a number from 0 to 1.\n\n"
            f"Prompt: {prompt}",
            images=images,
        )
        for key in ("score", "image_score", "overall_score"):
            if key in payload:
                return _normalize_score(payload.get(key))
        return 0.0


@dataclass
class WiseScorerBackend:
    """WISE benchmark scorer using an OpenAI-compatible vision chat endpoint."""

    chat: ChatCompletionBackend
    data_path: str | Path
    last_score_details: dict[str, Any] = field(default_factory=dict)
    _by_prompt: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"{self.chat.name}_wise_scorer"

    @property
    def score_source(self) -> str:
        return "wise"

    def score(self, prompt: str, image: bytes | str) -> float:
        item = self._prompt_item(prompt)
        image_bytes = _image_bytes(image)
        eval_prompt = _wise_evaluation_prompt(item)
        raw = self.chat.think(eval_prompt, images=[image_bytes]).strip()
        score = _extract_wise_score(raw)
        self.last_score_details = {
            "benchmark": "wise",
            "prompt_id": item.get("prompt_id"),
            "category": item.get("Category"),
            "subcategory": item.get("Subcategory"),
            "explanation": item.get("Explanation"),
            "evaluation": raw,
            "score": score,
            "scorer_backend": self.name,
            "judge_model": self.chat.spec.model,
        }
        return score

    def _prompt_item(self, prompt: str) -> dict[str, Any]:
        if not self._by_prompt:
            self._by_prompt = _load_wise_items(self.data_path)
        key = prompt.strip()
        if key not in self._by_prompt:
            raise ValueError(f"WISE scorer cannot find prompt in {self.data_path}: {prompt[:120]}")
        return self._by_prompt[key]


def _load_wise_items(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"WISE scorer data must be a JSON list: {path}")
    by_prompt: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("Prompt") or item.get("prompt") or "").strip()
        if prompt:
            by_prompt[prompt] = dict(item)
    return by_prompt


def _wise_evaluation_prompt(item: dict[str, Any]) -> str:
    prompt = str(item["Prompt"])
    explanation = str(item["Explanation"])
    return f'''Please evaluate this generated image for the WISE benchmark and return ONLY one binary score.

# WISE Text-to-Image Evaluation Protocol

WISE is a knowledge-intensive text-to-image benchmark. Many prompts do not directly state the final visual answer. The image must use commonsense, cultural, scientific, spatial, or temporal knowledge to infer what should appear.

Judge these points:
1. Does the image contain the main objects or scene required by the PROMPT?
2. Does it satisfy the intended knowledge-based answer described in the EXPLANATION?
3. Are important relations correct, such as spatial layout, temporal state, physical effect, biological behavior, cultural object, or scientific phenomenon?
4. Is the image visually usable for judging, without obvious collapse, severe deformation, unreadable main objects, or major artifacts?

Score 1 only if the image is semantically correct according to both PROMPT and EXPLANATION and is visually usable. Score 0 if it misses the intended answer, follows only surface words, has wrong key objects/relations, is ambiguous, or has severe visual artifacts.

If there is serious doubt, return 0.

Output exactly one line and nothing else:
Score: 0
or
Score: 1

---

PROMPT: "{prompt}"
EXPLANATION: "{explanation}"
'''


def _extract_wise_score(text: str) -> float:
    match = re.search(r"\*{0,2}Score\*{0,2}\s*[::]?\s*([01])\b", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    nums = re.findall(r"(?m)^\s*([01])\s*$", text)
    if len(nums) == 1:
        return float(nums[0])
    raise ValueError(f"Could not extract WISE binary score from scorer output: {text[:200]!r}")


def _image_bytes(image: bytes | str) -> bytes:
    if isinstance(image, bytes):
        return image
    path = Path(str(image))
    if path.is_file():
        return path.read_bytes()
    raise ValueError("Scorer image input must be image bytes or a local image path")


def _normalize_score(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def build_scorer_backend(config: dict[str, Any]):
    spec = ChatBackendSpec.from_config(str(config.get("default", "scorer")), dict(config))
    timeout = float(spec.default_params.get("timeout", 120.0))
    max_retries = int(spec.default_params.get("max_retries", 1))
    chat = ChatCompletionBackend(spec=spec, timeout=timeout, max_retries=max_retries)
    if spec.backend == "wise":
        data_path = (
            config.get("data_path")
            or spec.default_params.get("data_path")
            or "eval/wise/wise/data_verified/merge.json"
        )
        return WiseScorerBackend(chat=chat, data_path=str(data_path))
    return MLLMScorerBackend(chat=chat)
