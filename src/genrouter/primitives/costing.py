from __future__ import annotations

from typing import Any


def backend_token_usage(backend: Any | None) -> dict[str, int]:
    source = _usage_source(backend)
    raw = getattr(source, "last_usage", {}) if source is not None else {}
    if not isinstance(raw, dict):
        return {}
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        value = _int_value(raw.get(key))
        if value:
            usage[key] = value
    return usage


def backend_token_pricing(backend: Any | None) -> dict[str, Any]:
    source = _usage_source(backend)
    if source is None:
        return {}
    return {
        "input_token_price": _float_value(getattr(source, "input_token_price", 0.0)),
        "output_token_price": _float_value(getattr(source, "output_token_price", 0.0)),
    }


def backend_cost_details(backend: Any | None) -> dict[str, Any]:
    usage = backend_token_usage(backend)
    pricing = backend_token_pricing(backend)
    if not usage and not pricing:
        return {}
    return {
        "token_usage": usage,
        "token_pricing": pricing,
    }


def backend_call_cost(backend: Any | None, calls: int = 1) -> float:
    token_cost = backend_token_cost(backend)
    return token_cost * max(0, int(calls))


def backend_query_cost(backend: Any | None, queries: int = 1) -> float:
    raw = getattr(backend, "cost_per_query", getattr(backend, "cost_per_call", 0.0))
    try:
        return float(raw) * max(0, int(queries))
    except (TypeError, ValueError):
        return 0.0


def backend_token_cost(backend: Any | None) -> float:
    usage = backend_token_usage(backend)
    pricing = backend_token_pricing(backend)
    if not usage or not pricing:
        return 0.0
    input_price = float(pricing.get("input_token_price", 0.0) or 0.0)
    output_price = float(pricing.get("output_token_price", 0.0) or 0.0)
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0


def _usage_source(backend: Any | None) -> Any | None:
    if backend is None:
        return None
    chat = getattr(backend, "chat", None)
    return chat if chat is not None else backend


def _int_value(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _float_value(*values: Any) -> float:
    for value in values:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0
