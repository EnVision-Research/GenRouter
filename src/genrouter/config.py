from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a GenRouter configuration file is invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    default: dict[str, Any]
    workflows: dict[str, dict[str, Any]]
    generators: dict[str, dict[str, Any]]
    skills: dict[str, Any]
    config_dir: Path


def load_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw)
    except ModuleNotFoundError:
        loaded = _load_json_like(raw)

    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Config must be a mapping: {config_path}")
    return dict(loaded)


def _load_json_like(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", raw))


def _load_api_profiles(path: Path) -> dict[str, dict[str, str]] | None:
    if not path.is_file():
        return None

    api_config = load_mapping(path)
    profiles: dict[str, dict[str, str]] = {}
    for section_name in ("models", "services"):
        section = api_config.get(section_name, {})
        if not isinstance(section, Mapping):
            raise ConfigError(f"API config section must be a mapping: {section_name}")
        for profile_name, raw_profile in section.items():
            if not isinstance(raw_profile, Mapping):
                raise ConfigError(f"API profile must be a mapping: {section_name}.{profile_name}")
            name = str(profile_name).strip()
            if not name:
                raise ConfigError(f"API profile name must not be empty: {section_name}")
            if name in profiles:
                raise ConfigError(f"Duplicate API profile name: {name}")

            profile: dict[str, str] = {}
            for key, value in raw_profile.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ConfigError(f"API profile fields must be strings: {section_name}.{name}")
                profile[key] = value.strip()
            for required_key in ("api_type", "base_url", "api_key"):
                if not profile.get(required_key):
                    raise ConfigError(f"API profile requires {required_key}: {section_name}.{name}")
            env_name = profile.get("api_key_env", "")
            if env_name and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
                raise ConfigError(f"Invalid api_key_env: {section_name}.{name}")
            profiles[name] = profile
    return profiles


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith("<") and stripped.endswith(">")


def _set_api_key_environment(profile: Mapping[str, str], env_name: str) -> None:
    api_key = str(profile.get("api_key", "")).strip()
    if env_name and api_key and not _is_placeholder(api_key):
        os.environ.setdefault(env_name, api_key)


def _apply_api_profile(
    target: MutableMapping[str, Any],
    profiles: Mapping[str, Mapping[str, str]],
) -> None:
    profile_name = str(target.get("api_profile", "")).strip()
    if not profile_name:
        return
    if profile_name not in profiles:
        raise ConfigError(f"Unknown API profile: {profile_name}")

    profile = profiles[profile_name]
    target["base_url"] = profile["base_url"]
    env_name = str(target.get("api_key_env", "")).strip()
    _set_api_key_environment(profile, env_name)


def _apply_api_profiles(
    default: MutableMapping[str, Any],
    generators: MutableMapping[str, dict[str, Any]],
    profiles: Mapping[str, Mapping[str, str]],
) -> None:
    for value in default.values():
        if isinstance(value, MutableMapping):
            _apply_api_profile(value, profiles)
    for generator in generators.values():
        _apply_api_profile(generator, profiles)

    # Standalone services such as Hugging Face are not referenced by a backend
    # section, so their profile declares the environment variable explicitly.
    for profile in profiles.values():
        _set_api_key_environment(profile, str(profile.get("api_key_env", "")).strip())


def load_project_config(config_dir: str | Path = "configs") -> ProjectConfig:
    root = Path(config_dir)
    default = load_mapping(root / "default.yaml")
    config_paths = dict(default.get("configs", {})) if isinstance(default.get("configs"), Mapping) else {}

    workflows_path = Path(config_paths.get("workflows", root / "workflows.yaml"))
    generators_path = Path(config_paths.get("generators", root / "generators.yaml"))
    skills_path = Path(config_paths.get("skills", root / "skills.yaml"))

    workflows = load_mapping(workflows_path)
    generators = load_mapping(generators_path)
    skills = load_mapping(skills_path)
    api_profiles = _load_api_profiles(root / "api_config.yaml")
    if api_profiles is not None:
        _apply_api_profiles(default, generators, api_profiles)
    return ProjectConfig(
        default=default,
        workflows={str(k): dict(v) for k, v in workflows.items() if isinstance(v, Mapping)},
        generators={str(k): dict(v) for k, v in generators.items() if isinstance(v, Mapping)},
        skills=skills,
        config_dir=root,
    )
