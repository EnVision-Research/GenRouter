from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SKILL_NAMES = [
    "spatial_layout",
    "aesthetic_drawing",
    "text_rendering",
    "creative_drawing",
    "anatomy_body_coherence",
    "attribute_binding",
    "physical_material_consistency",
    "quantity_counting",
]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    source_path: str


class SkillBank:
    def __init__(
        self,
        names: list[str] | None = None,
        skills_dir: str | Path | None = None,
        max_skills: int = 1,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.names = list(DEFAULT_SKILL_NAMES if names is None else names)
        if skills_dir:
            self.skills_dir = Path(skills_dir)
        else:
            self.skills_dir = Path(__file__).resolve().parent / "skills"
        self.max_skills = max(0, int(max_skills))
        self.skills = self._load()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SkillBank":
        values = dict(config.get("skills", config))
        return cls(
            names=[str(item) for item in values.get("names", DEFAULT_SKILL_NAMES)],
            skills_dir=values.get("skills_dir"),
            max_skills=int(values.get("max_skills", 1)),
            enabled=bool(values.get("enabled", True)),
        )

    def _load(self) -> dict[str, Skill]:
        if not self.enabled:
            return {}
        if not self.skills_dir.is_dir():
            raise FileNotFoundError(f"Skills directory not found: {self.skills_dir}")
        skills: dict[str, Skill] = {}
        for name in self.names:
            path = self.skills_dir / f"{name}.md"
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                description, instructions = _skill_sections(content)
                skills[name] = Skill(
                    name=name,
                    description=description,
                    instructions=instructions,
                    source_path=str(path),
                )
        return skills

    def available(self) -> list[str]:
        return sorted(self.skills.keys())

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def manifest(self) -> str:
        lines: list[str] = []
        for name in self.available():
            skill = self.skills[name]
            lines.append(f"- SKILL_ID: {name}\n  DESCRIPTION: {skill.description}")
        return "\n".join(lines)

    def select(self, names: list[str]) -> list[Skill]:
        selected: list[Skill] = []
        for name in names:
            if len(selected) >= self.max_skills:
                break
            skill = self.get(name)
            if skill:
                selected.append(skill)
        return selected


def _skill_sections(content: str) -> tuple[str, str]:
    description = _markdown_section(content, "Description")
    instructions = _markdown_section(content, "Instructions")
    if not description or not instructions:
        raise ValueError("Skill markdown requires ## Description and ## Instructions sections")
    return description, instructions


def _markdown_section(content: str, heading: str) -> str:
    marker = f"## {heading}"
    if marker not in content:
        return ""
    section = content.split(marker, 1)[1]
    if "\n## " in section:
        section = section.split("\n## ", 1)[0]
    return section.strip()
