from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit


class ServicePreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class _Probe:
    name: str
    url: str
    expected_model: str = ""


def _root_health_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _required_probes(config: Mapping[str, Any]) -> list[_Probe]:
    probes: list[_Probe] = []
    services = dict(config.get("services") or {})
    for name, value in services.items():
        if isinstance(value, Mapping):
            probes.append(
                _Probe(
                    name,
                    _models_url(str(value["base_url"])),
                    str(value.get("model") or ""),
                )
            )
        else:
            probes.append(_Probe(name, _root_health_url(str(value))))

    evaluator = dict(config.get("evaluator") or {})
    judge_base_url = str(evaluator.get("judge_base_url") or "")
    if judge_base_url:
        expected = str(dict(config.get("models") or {}).get("judge") or "")
        probes.append(_Probe("judge", _models_url(judge_base_url), expected))
    return probes


def check_required_services(
    config: Mapping[str, Any],
    *,
    timeout: float = 2.0,
) -> None:
    failures: list[str] = []
    for probe in _required_probes(config):
        try:
            request = urllib.request.Request(probe.url, method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if probe.expected_model:
                available = {
                    str(item.get("id"))
                    for item in payload.get("data", [])
                    if isinstance(item, Mapping)
                }
                if probe.expected_model not in available:
                    failures.append(
                        f"{probe.name}: expected model {probe.expected_model!r} "
                        f"at {probe.url}"
                    )
            elif payload.get("status") != "ok":
                failures.append(
                    f"{probe.name}: unhealthy response from {probe.url}"
                )
        except Exception as exc:
            failures.append(f"{probe.name}: {probe.url} ({exc})")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise ServicePreflightError(
            "Required model services are not ready:\n"
            f"{details}\n"
            "Run `bash scripts/serve.sh`, then retry."
        )
