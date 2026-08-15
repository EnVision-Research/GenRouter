from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from genrouter.backends.http_client import urlopen_with_retry
from genrouter.backends.image_payloads import image_media_type

class ChatRequestError(RuntimeError):
    """Raised when a chat backend request fails."""


def _is_local_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
        "0.0.0.0",
    }


@dataclass(frozen=True)
class ChatBackendSpec:
    name: str
    backend: str = "openai"
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    default_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, data: dict[str, Any]) -> "ChatBackendSpec":
        default_params = dict(data.get("default_params", {}) or {})
        return cls(
            name=name,
            backend=str(data.get("backend", "openai")),
            base_url=str(data.get("base_url", "")),
            model=str(data.get("model", "")),
            api_key_env=str(data.get("api_key_env", "")),
            default_params=default_params,
        )


@dataclass
class ChatCompletionBackend:
    spec: ChatBackendSpec
    timeout: float = 120.0
    max_retries: int = 1
    last_usage: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def input_token_price(self) -> float:
        return _config_float(self.spec.default_params, "input_token_price")

    @property
    def output_token_price(self) -> float:
        return _config_float(self.spec.default_params, "output_token_price")

    @property
    def endpoint(self) -> str:
        if not self.spec.base_url:
            raise ValueError(f"Chat backend '{self.name}' has no base_url configured")
        return f"{self.spec.base_url.rstrip('/')}/chat/completions"

    def think(self, prompt: str, images: list[bytes] | None = None) -> str:
        response = self._chat(prompt=prompt, images=images)
        return self._message_text(response)

    def think_json(self, prompt: str, images: list[bytes] | None = None) -> dict[str, Any]:
        text = self.think(prompt=prompt, images=images)
        return _json_payload(text)

    def think_messages(self, messages: list[dict[str, Any]]) -> str:
        response = self._chat_messages(messages)
        return self._message_text(response)

    def think_json_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return _json_payload(self.think_messages(messages))

    def _chat(self, prompt: str, images: list[bytes] | None) -> dict[str, Any]:
        return self._chat_messages(self._messages(prompt, images=images))

    def _chat_messages(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.spec.model:
            raise ValueError(f"Chat backend '{self.name}' has no model configured")
        payload = {
            "model": self.spec.model,
            "messages": messages,
        }
        for key, value in self.spec.default_params.items():
            if key not in _RESERVED_CHAT_PARAM_KEYS and value is not None:
                payload[key] = value

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        raw = self._post_with_retry(request)
        try:
            response = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ChatRequestError(f"{self.name} returned non-JSON response: {raw[:120]!r}") from exc
        if not isinstance(response, dict):
            raise ChatRequestError(f"{self.name} response must be a JSON object")
        self.last_usage = _parse_usage(response.get("usage"))
        return response

    def _messages(self, prompt: str, images: list[bytes] | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        system_prompt = str(self.spec.default_params.get("system_prompt") or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self._content(prompt, images=images)})
        return messages

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if not _is_local_http_url(self.spec.base_url):
            headers["Authorization"] = f"Bearer {self._api_key()}"
        return headers

    def _content(self, prompt: str, images: list[bytes] | None) -> str | list[dict[str, Any]]:
        if not images:
            return prompt
        content: list[dict[str, Any]] = []
        for image in images:
            media_type = image_media_type(image)
            encoded = base64.b64encode(image).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
        content.append({"type": "text", "text": prompt})
        return content

    def _api_key(self) -> str:
        env_name = self.spec.api_key_env
        api_key = os.environ.get(env_name) if env_name else ""
        if not api_key:
            name = env_name or "the configured API key environment variable"
            raise ValueError(f"Set {name} before using chat backend '{self.name}' in real mode")
        return api_key

    def _post_with_retry(self, request: urllib.request.Request) -> bytes:
        raw, _ = urlopen_with_retry(
            request,
            timeout=self.timeout,
            max_retries=self.max_retries,
            label=self.name,
            error_cls=ChatRequestError,
            opener=urllib.request.urlopen,
        )
        return raw

    def _message_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ChatRequestError(f"{self.name} response did not contain choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise ChatRequestError(f"{self.name} response choice must be an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ChatRequestError(f"{self.name} response choice did not contain a message")
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)


@dataclass
class LocalTransformersChatBackend:
    spec: ChatBackendSpec
    last_usage: dict[str, int] = field(default_factory=dict)
    _processor: Any = None
    _model: Any = None
    _device: str = ""

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def input_token_price(self) -> float:
        return _config_float(self.spec.default_params, "input_token_price")

    @property
    def output_token_price(self) -> float:
        return _config_float(self.spec.default_params, "output_token_price")

    def think(self, prompt: str, images: list[bytes] | None = None) -> str:
        if images:
            raise ValueError(f"Local chat backend '{self.name}' only supports text prompts")
        return self.think_messages(self._messages(prompt))

    def think_json(self, prompt: str, images: list[bytes] | None = None) -> dict[str, Any]:
        text = self.think(prompt=prompt, images=images)
        parsed = _parse_json_text(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        return {"response": text}

    def think_messages(self, messages: list[dict[str, str]]) -> str:
        processor, model, device = self._load()
        text = _apply_chat_template(
            processor,
            messages,
            enable_thinking=bool(self.spec.default_params.get("enable_thinking", False)),
        )
        inputs = processor(text=[text], return_tensors="pt")
        input_len = int(inputs["input_ids"].shape[-1])
        if device != "cpu":
            inputs = inputs.to(device)
        output = model.generate(**inputs, **self._generation_kwargs(processor))
        output_ids = output[0]
        generated_ids = output_ids[input_len:]
        text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = _sequence_length(generated_ids)
        self.last_usage = {
            "prompt_tokens": input_len,
            "completion_tokens": completion_tokens,
            "total_tokens": input_len + completion_tokens,
        }
        return text

    def think_json_messages(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        text = self.think_messages(messages)
        parsed = _parse_json_text(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        return {"response": text}

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        system_prompt = str(self.spec.default_params.get("system_prompt") or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _load(self):
        if self._processor is not None and self._model is not None and self._device:
            return self._processor, self._model, self._device
        if not self.spec.model:
            raise ValueError(f"Local chat backend '{self.name}' has no model path configured")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install torch and transformers before using local_transformers chat backend") from exc

        device_name = str(self.spec.default_params.get("device") or "auto")
        dtype_name = str(self.spec.default_params.get("dtype") or "auto")
        target_device, model_kwargs = _resolve_local_device(torch, device_name)
        self._processor = AutoProcessor.from_pretrained(self.spec.model, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.spec.model,
            dtype=_resolve_local_dtype(torch, dtype_name),
            trust_remote_code=True,
            **model_kwargs,
        )
        if not model_kwargs:
            self._model = self._model.to(target_device)
        self._model.eval()
        self._device = target_device
        return self._processor, self._model, self._device

    def _generation_kwargs(self, processor) -> dict[str, Any]:
        params = self.spec.default_params
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(params.get("max_new_tokens", params.get("max_tokens", 512))),
            "do_sample": bool(params.get("do_sample", params.get("sample", False))),
            "eos_token_id": processor.tokenizer.eos_token_id,
            "pad_token_id": processor.tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(params.get("temperature", 0.1))
            kwargs["top_p"] = float(params.get("top_p", 0.9))
            kwargs["top_k"] = int(params.get("top_k", 20))
        return kwargs


def _json_payload(text: str) -> dict[str, Any]:
    parsed = _parse_json_text(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    return {"response": text}


def _parse_json_text(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        try:
            return json.loads(stripped[object_start : object_end + 1])
        except json.JSONDecodeError:
            pass

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start >= 0 and array_end > array_start:
        try:
            return json.loads(stripped[array_start : array_end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _apply_chat_template(processor, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _resolve_local_dtype(torch, name: str):
    if name == "auto":
        return "auto"
    values = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"Unsupported local chat dtype: {name}")
    return values[name]


def _resolve_local_device(torch, device: str) -> tuple[str, dict[str, Any]]:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda", {"device_map": "auto"}
        return "cpu", {}
    if device == "cpu":
        return "cpu", {}
    if device.startswith("cuda"):
        return device, {"device_map": device}
    raise ValueError(f"Unsupported local chat device: {device}")


def _sequence_length(sequence: Any) -> int:
    shape = getattr(sequence, "shape", None)
    if shape:
        return int(shape[-1])
    return len(sequence)


def _config_float(data: dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            return float(data.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def _parse_usage(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    usage: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        try:
            value = max(0, int(raw.get(key, 0)))
        except (TypeError, ValueError):
            value = 0
        if value:
            usage[key] = value
    return usage




_RESERVED_CHAT_PARAM_KEYS = {
    "device",
    "do_sample",
    "dtype",
    "max_new_tokens",
    "max_retries",
    "sample",
    "system_prompt",
    "timeout",
    "input_token_price",
    "output_token_price",
}


def _build_chat_backend(config: dict[str, Any], default_name: str):
    spec = ChatBackendSpec.from_config(str(config.get("default", default_name)), dict(config))
    if spec.backend in {"local", "local_transformers", "transformers"}:
        return LocalTransformersChatBackend(spec=spec)
    timeout = float(spec.default_params.get("timeout", 120.0))
    max_retries = int(spec.default_params.get("max_retries", 1))
    return ChatCompletionBackend(spec=spec, timeout=timeout, max_retries=max_retries)


def build_llm_backend(config: dict[str, Any]):
    return _build_chat_backend(config, "llm")


def build_signature_llm_backend(config: dict[str, Any]):
    if "signature_llm" in config:
        return _build_chat_backend(dict(config.get("signature_llm", {})), "signature_llm")
    if "llm" in config:
        return _build_chat_backend(dict(config.get("llm", {})), "llm")
    return _build_chat_backend(config, "signature_llm")


def build_mllm_backend(config: dict[str, Any]):
    spec = ChatBackendSpec.from_config(str(config.get("default", "mllm")), dict(config))
    timeout = float(spec.default_params.get("timeout", 120.0))
    max_retries = int(spec.default_params.get("max_retries", 1))
    return ChatCompletionBackend(spec=spec, timeout=timeout, max_retries=max_retries)
