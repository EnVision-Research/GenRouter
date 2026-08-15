from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from eval.common.benchmark import GenerationRecord, Score


@dataclass
class Manifest:
    run_id: str
    benchmark: str
    genrouter_revision: str
    official_revision: str
    config: dict[str, Any]
    generator_options: list[str]
    concrete_generators: dict[str, dict[str, Any]]
    seed: int
    cold_start_size: int
    batch_size: int
    completed_phases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Manifest":
        return cls(**payload)


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "manifest.json"

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def experience_path(self) -> Path:
        return self.root / "experience.jsonl"

    @property
    def route_memory_path(self) -> Path:
        return self.root / "route_memory.jsonl"

    def initialize(self, expected: Manifest) -> Manifest:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.is_file():
            self._write_json(self.manifest_path, expected.to_dict())
            return expected
        current = Manifest.from_dict(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )
        current_identity = {**current.to_dict(), "completed_phases": []}
        expected_identity = {**expected.to_dict(), "completed_phases": []}
        if current_identity != expected_identity:
            raise ValueError(
                f"Existing run manifest does not match requested run: {self.root}"
            )
        return current

    def manifest(self) -> Manifest:
        return Manifest.from_dict(
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
        )

    def is_complete(self, phase: str) -> bool:
        return phase in self.manifest().completed_phases

    def phase_dir(self, phase: str) -> Path:
        if phase == "cold_start":
            return self.root / "cold_start"
        prefix, batch_id = phase.split(":", 1)
        if prefix != "batch":
            raise ValueError(f"Unknown phase: {phase}")
        return self.root / "batches" / batch_id

    def write_phase(
        self,
        phase: str,
        records: list[GenerationRecord],
        scores: list[Score],
    ) -> None:
        phase_dir = self.phase_dir(phase)
        self._write_jsonl(
            phase_dir / "records.jsonl",
            (_record_dict(item) for item in records),
        )
        self._write_jsonl(
            phase_dir / "scores.jsonl",
            (asdict(item) for item in scores),
        )
        self._write_json(
            phase_dir / "prepared.json",
            {
                "phase": phase,
                "record_ids": [record.case_id for record in records],
                "score_ids": [score.case_id for score in scores],
            },
        )

    def read_prepared_phase(
        self,
        phase: str,
    ) -> tuple[list[GenerationRecord], list[Score]] | None:
        phase_dir = self.phase_dir(phase)
        marker_path = phase_dir / "prepared.json"
        if not marker_path.is_file():
            return None
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        record_rows = self.read_jsonl(phase_dir / "records.jsonl")
        score_rows = self.read_jsonl(phase_dir / "scores.jsonl")
        record_ids = [str(row.get("case_id") or "") for row in record_rows]
        score_ids = [str(row.get("case_id") or "") for row in score_rows]
        if (
            marker.get("phase") != phase
            or marker.get("record_ids") != record_ids
            or marker.get("score_ids") != score_ids
        ):
            raise RuntimeError(f"Prepared phase artifacts are inconsistent: {phase}")
        records = []
        for row in record_rows:
            payload = dict(row)
            payload["image_path"] = Path(payload["image_path"])
            payload["result_path"] = Path(payload["result_path"])
            records.append(GenerationRecord(**payload))
        scores = [Score(**row) for row in score_rows]
        return records, scores

    def refresh_final_indexes(self) -> None:
        phases = [
            phase
            for phase in self.manifest().completed_phases
            if phase.startswith("batch:")
        ]
        phases.sort()
        records: list[dict[str, Any]] = []
        scores: list[dict[str, Any]] = []
        for phase in phases:
            phase_dir = self.phase_dir(phase)
            records.extend(self.read_jsonl(phase_dir / "records.jsonl"))
            scores.extend(self.read_jsonl(phase_dir / "scores.jsonl"))
        self._write_jsonl(self.root / "records.jsonl", records)
        self._write_jsonl(self.root / "scores.jsonl", scores)

    def complete_phase(self, phase: str) -> None:
        manifest = self.manifest()
        if phase not in manifest.completed_phases:
            manifest.completed_phases.append(phase)
            self._write_json(self.manifest_path, manifest.to_dict())

    def write_summary(self, payload: dict[str, Any]) -> None:
        self._write_json(self.root / "summary.json", payload)

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        temporary.replace(path)


def _record_dict(record: GenerationRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["image_path"] = str(record.image_path)
    payload["result_path"] = str(record.result_path)
    return payload
