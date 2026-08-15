from __future__ import annotations

from typing import Any, Protocol

from genrouter.schemas import WorkflowResult


class WorkflowExecutionError(RuntimeError):
    def __init__(self, message: str, partial_result: WorkflowResult) -> None:
        super().__init__(message)
        self.partial_result = partial_result


class BaseWorkflow(Protocol):
    name: str

    def run(self, prompt: str, generator: Any, config: dict[str, Any], prompt_id: str = "") -> WorkflowResult:
        ...
