from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def urlopen_with_retry(
    request: urllib.request.Request,
    *,
    timeout: float,
    max_retries: int,
    label: str,
    error_cls: type[Exception],
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[bytes, str]:
    attempts = max(1, int(max_retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with opener(request, timeout=timeout) as response:
                return response.read(), response.headers.get("content-type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = error_cls(f"{label} HTTP {exc.code}: {detail[:500]}")
            if exc.code < 500 or attempt == attempts - 1:
                raise last_error from exc
        except urllib.error.URLError as exc:
            last_error = error_cls(f"{label} request failed: {exc}")
            if attempt == attempts - 1:
                raise last_error from exc
    raise error_cls(f"{label} request failed: {last_error}")


def request_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int,
    label: str,
    error_cls: type[Exception],
    retry_delay: Callable[[int], float] | None = None,
    sleep: Callable[[float], Any] = time.sleep,
) -> T:
    attempts = max(1, int(max_retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts and retry_delay is not None:
                sleep(retry_delay(attempt))
    raise error_cls(f"{label} failed after {attempts} attempts: {last_error}")


def exponential_retry_delay(attempt: int) -> float:
    return min(8.0, 1.0 + 2.0 * attempt)
