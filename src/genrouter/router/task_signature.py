from __future__ import annotations

from typing import Any

from genrouter.schemas import TaskSignature


SIGNATURE_FIELDS = tuple(TaskSignature.__dataclass_fields__)


TASK_SIGNATURE_SYSTEM_PROMPT = """Analyze the user's image generation prompt, briefly explain the scoring rationale, and then output a task signature as JSON.

Output format:

<reason>
Rewrite: score, a short explanation
Search_text: ...
Search_image: ...
Reason: ...
Skill: ...
Verify_refine: ...
Code_sketch: ...
Give the score and then briefly explain from each field. Keep the explanation concise and grounded in the given prompt.
</reason>

<json>
{"rewrite": "...", "search_text": "...", "search_image": "...", "reason": "...", "skill": "...", "verify_refine": "...", "code_sketch": "..."}
Each value must be a numeric score in 0, 1, 2, 3, 4, or 5.
</json>

Task signature definitions:

Rewrite:
Use Rewrite when the prompt requires clarification, disambiguation, or restructuring before it can be effectively used for image generation.

Common indicators include:
- ambiguous or underspecified visual entities,
- incomplete scene descriptions,
- unclear relationships between objects or attributes,
- disorganized or verbose instructions,
- prompts that benefit from clearer or more structured expression.

Rewrite focuses on improving the prompt expression. If deriving the correct scene requires inference beyond the given information, prefer Reason instead.
Do not assign a high score solely because aesthetic details are missing, and keep the score low for simple prompts that are already clear.

Search Text:
Use Search Text when the prompt requires external textual knowledge that is not provided in the input.

Common indicators include:
- current news or events,
- dates and timelines,
- historical or cultural facts,
- geographic knowledge,
- scientific knowledge,
- entity identification or selection,
- any other factual information needed to complete the prompt.

Search Image:
Use Search Image when the prompt requires visual references to faithfully preserve the visual identity of a specific entity or identity of a specific entity.

Common indicators include:
- people,
- fictional characters,
- products, brands or logos,
- landmarks or famous places,
- artworks,
- long-tail entities,
- any subject whose visual identity cannot be reliably inferred from text alone.

Do not assign a high score for generic categories (e.g., "a cat", "a castle", "a mountain") unless a specific identity must be preserved.

Reason:
Use Reason when the prompt requires reasoning to derive the correct visual scene before rewriting it into an image prompt. The input specifies facts or relationships rather than an explicit image description.

Common indicators include:
- temporal reasoning (before/after, time conversion),
- causal reasoning,
- physical or chemical processes,
- geographic inference,
- commonsense knowledge,
- object interactions,
- implicit spatial relationships,
- symbolic or semantic mapping.

Skill:
Use Skill and score high when the prompt requires specialized image-generation expertise.

Common indicators include:
- quantity counting (number appears in the prompt, such as "six", "7"),
- readable text rendering,
- precise spatial layouts and relationships,
- multi-object coordination,
- attribute binding,
- human anatomy and body coherence,
- material and texture consistency,
- aesthetic or creative execution.

Verify&Refine:
Use Verify&Refine when the generated image is likely to require post-generation verification and iterative refinement to ensure it satisfies all specified constraints.

Common indicators include:
- exact text rendering,
- exact object counts,
- left-right or relative spatial relationships,
- fine-grained object attributes,
- many interacting entities,
- any requirement where small visual errors are unacceptable.

Code_sketch:
Use Code Sketch when the prompt is better satisfied through a programmable intermediate representation before rendering.

Common indicators include:
- exact counts
- precise layouts
- charts
- geometry
- text rendering

Scoring discipline:

Score high when the prompt meets the above definitions.
Assign scores independently. A high score in one dimension should not imply high scores in others.

Score scale:

0: Unnecessary.
1: Slightly beneficial but usually unnecessary.
2: Helpful and improves reliability.
3: Important for reliable task completion.
4: Essential; performance would degrade substantially without it.
5: Indispensable; the task cannot be correctly completed without it.
"""


class TaskSignatureExtractor:
    def __init__(self, llm: Any | None = None, max_retries: int = 2) -> None:
        self.llm = llm
        self.max_retries = max(0, int(max_retries))

    def extract(self, prompt: str) -> TaskSignature:
        if self.llm is None:
            raise RuntimeError("TaskSignatureExtractor requires an LLM backend with think_json")
        last_payload: Any = None
        for attempt in range(self.max_retries + 1):
            if hasattr(self.llm, "think_json_messages"):
                payload = self.llm.think_json_messages(_signature_messages(prompt, attempt=attempt))
            elif hasattr(self.llm, "think_json"):
                payload = self.llm.think_json(_legacy_signature_prompt(prompt, attempt=attempt))
            else:
                raise RuntimeError("TaskSignatureExtractor requires an LLM backend with think_json")
            last_payload = payload
            if _has_complete_signature(payload):
                return TaskSignature.from_dict(payload)
        raise ValueError(f"LLM must return a complete GenRouter task signature after {self.max_retries + 1} attempts: {last_payload!r}")


def _signature_messages(prompt: str, attempt: int = 0) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": TASK_SIGNATURE_SYSTEM_PROMPT},
        {"role": "user", "content": _signature_user_prompt(prompt, attempt=attempt)},
    ]


def _signature_user_prompt(prompt: str, attempt: int = 0) -> str:
    instructions = ["Analyze this image generation prompt only."]
    if attempt:
        instructions.append(
            "Previous output was invalid or incomplete. Retry now and return exactly one compact JSON object "
            "with the eight required integer fields."
        )
    return (
        "\n".join(instructions)
        + "\n\n"
        "<image_prompt>\n"
        f"{prompt}\n"
        "</image_prompt>"
    )


def _legacy_signature_prompt(prompt: str, attempt: int = 0) -> str:
    return f"{TASK_SIGNATURE_SYSTEM_PROMPT}\n\n{_signature_user_prompt(prompt, attempt=attempt)}"


def _has_complete_signature(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    for field in SIGNATURE_FIELDS:
        if field not in payload:
            return False
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
            return False
    return True
