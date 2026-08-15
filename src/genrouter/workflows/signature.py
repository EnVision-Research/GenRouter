from __future__ import annotations

from typing import Any

from genrouter.schemas import TaskSignature


HYBRID_NEED_FIELDS = (
    "search_text",
    "search_image",
    "reason",
    "skill",
    "verify_refine",
    "code_sketch",
)


def signature_from_config(config: dict[str, Any]) -> TaskSignature:
    payload = config.get("task_signature")
    if not payload:
        raise ValueError(
            "Workflow requires config['task_signature']; the router's "
            "TaskSignatureExtractor must inject it before workflow.run."
        )
    return TaskSignature.from_dict(payload)


def requires_hybrid(signature: TaskSignature, threshold: int = 3) -> bool:
    """Return whether at least two independent workflow capabilities are needed.

    Rewrite is intentionally excluded because specialized workflows already
    rewrite after completing their primary capability.
    """
    return sum(getattr(signature, field) >= threshold for field in HYBRID_NEED_FIELDS) >= 2


def hybrid_branches(signature: TaskSignature) -> dict[str, bool]:
    branches = {
        "text_search": signature.search_text >= 3,
        "image_search": signature.search_image >= 3,
        "reasoning": signature.reason >= 3,
        "skills": signature.skill >= 3,
        "sketch": signature.code_sketch >= 4,
    }
    branches["requires_reference"] = branches["image_search"] or branches["sketch"]
    return branches
