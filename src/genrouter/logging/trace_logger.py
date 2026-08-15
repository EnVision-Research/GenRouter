from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from genrouter.schemas import PrimitiveTrace


class TraceLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: PrimitiveTrace | dict[str, Any]) -> None:
        payload = trace.to_dict() if isinstance(trace, PrimitiveTrace) else dict(trace)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
