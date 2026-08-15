from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from genrouter.backends.http_client import urlopen_with_retry
from genrouter.backends.image_payloads import (
    decode_base64_image,
    download_url,
    first_json_image,
    reference_image_urls,
)
from genrouter.schemas import GeneratorSpec


class GeneratorRequestError(RuntimeError):
    """Raised when a generator backend request fails."""


def _is_local_http_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return parsed.scheme in {"http", "https"} and host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


@dataclass
class HttpGeneratorBackend:
    spec: GeneratorSpec
    timeout: float = 600.0
    max_retries: int = 1

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def supports_reference(self) -> bool:
        return self.spec.supports_reference

    @property
    def cost_per_call(self) -> float:
        return self.spec.cost_per_call

    def generate(
        self,
        prompt: str,
        references: list[str] | None = None,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        if references and not self.supports_reference:
            raise ValueError(f"Generator '{self.name}' does not support reference images")
        if not self.spec.endpoint:
            raise ValueError(f"Generator '{self.name}' has no endpoint configured")

        merged_params = dict(self.spec.default_params)
        if params:
            merged_params.update(params)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "generator": self.name,
            "params": merged_params,
        }
        if references:
            payload["references"] = list(references)
        if seed is not None:
            payload["seed"] = int(seed)

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.spec.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw, content_type = self._post_with_retry(request)

        if content_type.startswith("image/"):
            return raw

        try:
            payload_json = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise GeneratorRequestError(
                f"{self.name} returned non-image and non-JSON response: {raw[:120]!r}"
            ) from exc
        decoded = first_json_image(payload_json, opener=urllib.request.urlopen)
        if decoded is None:
            raise GeneratorRequestError(f"{self.name} JSON response did not contain image data")
        return decoded

    def _post_with_retry(self, request: urllib.request.Request) -> tuple[bytes, str]:
        return urlopen_with_retry(
            request,
            timeout=self.timeout,
            max_retries=self.max_retries,
            label=self.name,
            error_cls=GeneratorRequestError,
            opener=urllib.request.urlopen,
        )


@dataclass
class ModelScopeImageGeneratorBackend:
    spec: GeneratorSpec
    timeout: float = 600.0
    max_retries: int = 1
    poll_interval: float = 5.0
    max_polls: int = 120

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def supports_reference(self) -> bool:
        return self.spec.supports_reference

    @property
    def cost_per_call(self) -> float:
        return self.spec.cost_per_call

    @property
    def endpoint(self) -> str:
        if self.spec.endpoint:
            return self.spec.endpoint
        if not self.spec.base_url:
            raise ValueError(f"Generator '{self.name}' has no base_url or endpoint configured")
        return f"{self.spec.base_url.rstrip('/')}/v1/images/generations"

    @property
    def task_base_url(self) -> str:
        if self.spec.base_url:
            return self.spec.base_url.rstrip("/")
        return self.endpoint.split("/v1/images/generations", 1)[0].rstrip("/")

    def generate(
        self,
        prompt: str,
        references: list[str] | None = None,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        if references and not self.supports_reference:
            raise ValueError(f"Generator '{self.name}' does not support reference images")
        if not self.spec.model:
            raise ValueError(f"Generator '{self.name}' has no ModelScope model id configured")

        merged_params = dict(self.spec.default_params)
        if params:
            merged_params.update(params)
        if references:
            reference_mode = str(merged_params.get("reference_mode", "prompt"))
            if reference_mode == "image":
                max_reference_size = int(merged_params.get("max_reference_size", 2048))
                image_urls = reference_image_urls(
                    references,
                    max_reference_size,
                    resize_error_cls=GeneratorRequestError,
                )
                if not image_urls:
                    raise ValueError(
                        f"Generator '{self.name}' reference_mode='image' requires local image reference paths"
                    )
                payload = self._payload(prompt=prompt, seed=seed, params=merged_params)
                payload["image_url"] = image_urls[0]
                request = urllib.request.Request(
                    self.endpoint,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers=self._headers(async_mode=bool(merged_params.get("async_mode", True))),
                    method="POST",
                )
                raw, content_type = self._urlopen_with_retry(request)
                if content_type.startswith("image/"):
                    return raw
                response = self._json_response(raw)
                decoded = first_json_image(response, opener=urllib.request.urlopen)
                if decoded is not None:
                    return decoded
                output_image = self._first_output_image(response)
                if output_image is not None:
                    return output_image

                task_id = response.get("task_id")
                if isinstance(task_id, str) and task_id:
                    return self._poll_task(task_id)
                raise GeneratorRequestError(f"{self.name} response did not contain image data or task_id")
            if reference_mode != "prompt":
                raise ValueError(
                    f"Generator '{self.name}' only supports reference_mode='prompt' for references"
                )
            prompt = self._prompt_with_references(prompt, references)
        payload = self._payload(prompt=prompt, seed=seed, params=merged_params)

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(async_mode=bool(merged_params.get("async_mode", True))),
            method="POST",
        )
        raw, content_type = self._urlopen_with_retry(request)
        if content_type.startswith("image/"):
            return raw

        response = self._json_response(raw)
        decoded = first_json_image(response, opener=urllib.request.urlopen)
        if decoded is not None:
            return decoded
        output_image = self._first_output_image(response)
        if output_image is not None:
            return output_image

        task_id = response.get("task_id")
        if isinstance(task_id, str) and task_id:
            return self._poll_task(task_id)
        raise GeneratorRequestError(f"{self.name} response did not contain image data or task_id")

    def _payload(self, prompt: str, seed: int | None, params: dict[str, Any]) -> dict[str, Any]:
        internal_keys = {
            "api_key",
            "api_key_env",
            "async_mode",
            "timeout",
            "max_retries",
            "poll_interval",
            "max_polls",
            "reference_mode",
            "max_reference_size",
        }
        payload = {"model": self.spec.model, "prompt": prompt}
        for key, value in params.items():
            if key not in internal_keys and value is not None:
                payload[key] = value
        if seed is not None:
            payload["seed"] = int(seed)
        return payload

    def _prompt_with_references(self, prompt: str, references: list[str]) -> str:
        reference_lines = "\n".join(f"- {item}" for item in references if str(item).strip())
        if not reference_lines:
            return prompt
        return (
            f"{prompt}\n\n"
            "Reference guidance to preserve in the image generation:\n"
            f"{reference_lines}\n"
            "Use these references as textual grounding cues. Do not render URLs or labels."
        )

    def _headers(self, async_mode: bool = False, task_poll: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self._optional_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif not _is_local_http_url(self.endpoint):
            headers["Authorization"] = f"Bearer {self._api_key()}"
        if async_mode:
            headers["X-ModelScope-Async-Mode"] = "true"
        if task_poll:
            headers["X-ModelScope-Task-Type"] = "image_generation"
        return headers

    def _optional_api_key(self) -> str:
        env_name = self.spec.api_key_env or str(self.spec.default_params.get("api_key_env") or "")
        api_key = os.environ.get(env_name) if env_name else ""
        return api_key or str(self.spec.default_params.get("api_key") or "")

    def _api_key(self) -> str:
        env_name = self.spec.api_key_env or str(self.spec.default_params.get("api_key_env") or "MODELSCOPE_API_KEY")
        api_key = os.environ.get(env_name) or str(self.spec.default_params.get("api_key") or "")
        if not api_key:
            raise ValueError(f"Set {env_name} before using generator '{self.name}' in real mode")
        return api_key

    def _poll_task(self, task_id: str) -> bytes:
        task_url = f"{self.task_base_url}/v1/tasks/{task_id}"
        total_polls = max(1, int(self.max_polls))
        for poll_index in range(total_polls):
            request = urllib.request.Request(
                task_url,
                headers=self._headers(task_poll=True),
                method="GET",
            )
            raw, content_type = self._urlopen_with_retry(request)
            if content_type.startswith("image/"):
                return raw
            response = self._json_response(raw)
            status = str(response.get("task_status", "")).upper()
            if status == "SUCCEED":
                decoded = first_json_image(response, opener=urllib.request.urlopen)
                if decoded is not None:
                    return decoded
                output_image = self._first_output_image(response)
                if output_image is not None:
                    return output_image
                raise GeneratorRequestError(f"{self.name} task {task_id} succeeded without image data")
            if status == "FAILED":
                if _is_transient_task_failure(response) and poll_index < total_polls - 1:
                    time.sleep(self.poll_interval)
                    continue
                raise GeneratorRequestError(f"{self.name} task {task_id} failed: {json.dumps(response)[:500]}")
            if poll_index < total_polls - 1:
                time.sleep(self.poll_interval)
        raise GeneratorRequestError(f"{self.name} task {task_id} timed out after {total_polls} polls")

    def _first_output_image(self, response: dict[str, Any]) -> bytes | None:
        output_images = response.get("output_images")
        if isinstance(output_images, list) and output_images:
            first = output_images[0]
            if isinstance(first, str) and first:
                if first.startswith("data:image"):
                    return decode_base64_image(first)
                if first.startswith("http://") or first.startswith("https://"):
                    return download_url(first, timeout=self.timeout, opener=urllib.request.urlopen)
                return decode_base64_image(first)
        return None

    def _urlopen_with_retry(self, request: urllib.request.Request) -> tuple[bytes, str]:
        return urlopen_with_retry(
            request,
            timeout=self.timeout,
            max_retries=self.max_retries,
            label=self.name,
            error_cls=GeneratorRequestError,
            opener=urllib.request.urlopen,
        )

    def _json_response(self, raw: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise GeneratorRequestError(f"{self.name} returned non-image and non-JSON response: {raw[:120]!r}") from exc
        if not isinstance(payload, dict):
            raise GeneratorRequestError(f"{self.name} JSON response must be an object")
        return payload


def _is_transient_task_failure(response: dict[str, Any]) -> bool:
    errors = response.get("errors")
    if not isinstance(errors, dict):
        return False
    message = str(errors.get("message") or "")
    return "dequeued" in message.casefold()


@dataclass
class ReferenceApiGeneratorBackend:
    """Interface for proprietary or SDK-backed reference-conditioned generators.

    The concrete provider client should be injected later. Keeping this adapter
    explicit makes unsupported real calls fail before silently producing bad data.
    """

    spec: GeneratorSpec
    client: Any | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def supports_reference(self) -> bool:
        return self.spec.supports_reference

    @property
    def cost_per_call(self) -> float:
        return self.spec.cost_per_call

    def generate(
        self,
        prompt: str,
        references: list[str] | None = None,
        seed: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> bytes:
        if references and not self.supports_reference:
            raise ValueError(f"Generator '{self.name}' does not support reference images")
        if self.client is None:
            raise NotImplementedError(
                f"Generator '{self.name}' requires a provider client before real API calls can run"
            )
        return self.client.generate(prompt=prompt, references=references or [], seed=seed, params=params or {})


def build_generator_backend(spec: GeneratorSpec):
    if spec.provider == "http":
        timeout = float(spec.default_params.get("timeout", 600.0))
        max_retries = int(spec.default_params.get("max_retries", 1))
        return HttpGeneratorBackend(spec=spec, timeout=timeout, max_retries=max_retries)
    if spec.provider == "modelscope":
        timeout = float(spec.default_params.get("timeout", 600.0))
        max_retries = int(spec.default_params.get("max_retries", 1))
        poll_interval = float(spec.default_params.get("poll_interval", 5.0))
        max_polls = int(spec.default_params.get("max_polls", 120))
        return ModelScopeImageGeneratorBackend(
            spec=spec,
            timeout=timeout,
            max_retries=max_retries,
            poll_interval=poll_interval,
            max_polls=max_polls,
        )
    if spec.provider in {"api", "placeholder"}:
        return ReferenceApiGeneratorBackend(spec=spec)
    raise ValueError(f"Unsupported generator provider for {spec.name}: {spec.provider}")
