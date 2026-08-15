#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, Request

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genrouter.router.task_signature import TASK_SIGNATURE_SYSTEM_PROMPT

DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "Qwen3.5-4B")
DEFAULT_PORT = 8011
DEFAULT_SYSTEM_PROMPT = TASK_SIGNATURE_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local Qwen signature extraction LLM.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda:0", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--system", default=None)
    parser.add_argument("--system-file", default=None)
    parser.add_argument("--prompt", default=None, help="Run one prompt and exit instead of serving HTTP.")
    parser.add_argument("--no-debug", action="store_true")
    return parser.parse_args()


def system_prompt(args: argparse.Namespace) -> str:
    if args.system_file:
        return Path(args.system_file).read_text(encoding="utf-8")
    return args.system or DEFAULT_SYSTEM_PROMPT


def load_model(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    target_device, model_kwargs = resolve_device(torch, args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=resolve_dtype(torch, args.dtype),
        trust_remote_code=True,
        **model_kwargs,
    )
    if not model_kwargs:
        model = model.to(target_device)
    model.eval()

    print(f"[info] model path: {args.model_path}", file=sys.stderr)
    print(f"[info] target device: {target_device}", file=sys.stderr)
    print(f"[info] dtype: {args.dtype}", file=sys.stderr)
    return processor, model, target_device


def resolve_dtype(torch_module: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[dtype]


def resolve_device(torch_module: Any, device: str) -> tuple[str, dict[str, Any]]:
    if device == "auto":
        return ("cuda", {"device_map": "auto"}) if torch_module.cuda.is_available() else ("cpu", {})
    if device == "cpu":
        return "cpu", {}
    if device.startswith("cuda"):
        return device, {"device_map": device}
    raise ValueError(f"Unsupported device: {device}")


def generate_reply(
    processor: Any,
    model: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
    device: str,
    usage: dict[str, int] | None = None,
) -> str:
    text = apply_chat_template(processor, messages, args.enable_thinking)
    inputs = processor(text=[text], return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[-1])
    if not args.no_debug:
        print(f"[debug] input tokens: {input_len}", file=sys.stderr)
    if device != "cpu":
        inputs = inputs.to(device)

    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if args.sample:
        kwargs.update({"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k})

    output = model.generate(**inputs, **kwargs)[0]
    generated = output[input_len:]
    if usage is not None:
        completion_tokens = int(generated.shape[-1])
        usage.update(
            prompt_tokens=input_len,
            completion_tokens=completion_tokens,
            total_tokens=input_len + completion_tokens,
        )
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def apply_chat_template(processor: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def make_messages(system: str, prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Analyze this image generation prompt only:\n\n<image_prompt>\n{prompt}\n</image_prompt>"},
    ]


def create_app(processor: Any, model: Any, args: argparse.Namespace, device: str, system: str) -> FastAPI:
    app = FastAPI(title="GenRouter signature extraction LLM")
    generate_lock = Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_path": args.model_path, "device": device}

    async def completions(request: Request) -> dict[str, Any]:
        payload = await request.json()
        messages = request_messages(payload, system)
        usage: dict[str, int] = {}
        with generate_lock:
            content = generate_reply(processor, model, messages, args, device, usage)
        return {
            "id": "genrouter-signature-local",
            "object": "chat.completion",
            "model": payload.get("model") or args.model_path,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": usage,
        }

    app.post("/chat/completions")(completions)
    app.post("/v1/chat/completions")(completions)
    return app


def serve(processor: Any, model: Any, args: argparse.Namespace, device: str, system: str) -> None:
    import uvicorn

    uvicorn.run(create_app(processor, model, args, device, system), host=args.host, port=args.port)


def request_messages(payload: dict[str, Any], system: str = DEFAULT_SYSTEM_PROMPT) -> list[dict[str, str]]:
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if isinstance(item, dict):
                content = message_content(item.get("content"))
                if content:
                    messages.append({"role": str(item.get("role") or "user"), "content": content})
        return messages

    prompt = str(payload.get("prompt") or payload.get("input") or "").strip()
    return make_messages(system, prompt) if prompt else []


def message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
        parts += [item for item in content if isinstance(item, str)]
        return "\n".join(parts).strip()
    return str(content or "").strip()


def main() -> None:
    args = parse_args()
    system = system_prompt(args)
    processor, model, device = load_model(args)
    if args.prompt is not None:
        print(generate_reply(processor, model, make_messages(system, args.prompt), args, device))
        return
    serve(processor, model, args, device, system)


if __name__ == "__main__":
    main()
