from __future__ import annotations

import argparse
import asyncio
import base64
import io
import multiprocessing as mp
import os
import queue
import threading
import urllib.request
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, Request

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "Qwen-Image-Edit-2511")
DEFAULT_PORT = 8008

CONFIG = SimpleNamespace(
    model_path=DEFAULT_MODEL_PATH,
    device="cuda:0",
    dtype="bfloat16",
    timeout=600.0,
    persistent_pipeline=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if CONFIG.persistent_pipeline:
        _start_workers()
    yield


APP = FastAPI(title="Qwen Image Edit local server", lifespan=lifespan)
TASK_QUEUE: queue.Queue[tuple[Future[str], str, list[Any], dict[str, Any]]] = queue.Queue()
WORKERS_STARTED = False


@APP.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "pid": os.getpid(), "model_path": CONFIG.model_path, "device": CONFIG.device}


@APP.post("/v1/images/generations")
async def generate(request: Request) -> dict[str, Any]:
    payload = await request.json()
    prompt = str(payload.get("prompt") or "").strip()
    images = _image_inputs(payload)
    encoded = await _submit(prompt, images, payload)
    image = f"data:image/png;base64,{encoded}"
    return {"model": payload.get("model") or CONFIG.model_path, "images": [image], "output_images": [image]}


async def _submit(prompt: str, images: list[Any], payload: dict[str, Any]) -> str:
    _start_workers()
    result: Future[str] = Future()
    TASK_QUEUE.put((result, prompt, images, payload))
    try:
        return await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(result)),
            timeout=CONFIG.timeout,
        )
    except TimeoutError as exc:
        raise TimeoutError("Processing timeout") from exc


def _start_workers() -> None:
    global WORKERS_STARTED
    if WORKERS_STARTED:
        return
    threading.Thread(target=_worker, args=(CONFIG.device,), daemon=True).start()
    WORKERS_STARTED = True


def _worker(device: str) -> None:
    pipeline_state: tuple[Any, Any] | None = None
    if CONFIG.persistent_pipeline:
        pipeline_state = _load_pipeline(device)
    while True:
        result, prompt, images, payload = TASK_QUEUE.get()
        try:
            if CONFIG.persistent_pipeline:
                pipe, torch_module = pipeline_state
                encoded = _encode_png(_run_with_pipeline(pipe, torch_module, device, prompt, images, payload))
            else:
                encoded = _run_in_child(prompt, images, payload)
        except Exception as exc:  # noqa: BLE001 - expose local server/model errors to caller.
            result.set_exception(RuntimeError(f"{type(exc).__name__}: {exc}"))
        else:
            result.set_result(encoded)
        finally:
            TASK_QUEUE.task_done()


def _run_in_child(prompt: str, images: list[Any], payload: dict[str, Any]) -> str:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_child_generate, args=(prompt, images, payload, _config_snapshot(), result_queue))
    process.start()
    try:
        status, value = result_queue.get(timeout=float(CONFIG.timeout))
    except queue.Empty as exc:
        process.terminate()
        raise TimeoutError("Generation subprocess timed out") from exc
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    if status == "ok":
        return str(value)
    raise RuntimeError(str(value))


def _child_generate(prompt: str, images: list[Any], payload: dict[str, Any], config: dict[str, Any], result_queue: Any) -> None:
    try:
        _apply_config(config)
        pipe, torch_module = _load_pipeline(CONFIG.device)
        result_queue.put(("ok", _encode_png(_run_with_pipeline(pipe, torch_module, CONFIG.device, prompt, images, payload))))
    except Exception as exc:  # noqa: BLE001 - child returns errors to parent.
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _config_snapshot() -> dict[str, Any]:
    return dict(vars(CONFIG))


def _apply_config(config: dict[str, Any]) -> None:
    for key, value in config.items():
        setattr(CONFIG, key, value)


def _load_pipeline(device: str) -> tuple[Any, Any]:
    import torch
    from diffusers import QwenImageEditPlusPipeline

    model_path = Path(CONFIG.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(str(model_path), torch_dtype=_torch_dtype(torch)).to(device)
    pipe.set_progress_bar_config(disable=None)
    return pipe, torch


def _run_with_pipeline(
    pipe: Any,
    torch_module: Any,
    device: str,
    prompt: str,
    images: list[Any],
    payload: dict[str, Any],
) -> Any:
    inputs: dict[str, Any] = {
        "image": images,
        "prompt": prompt,
        "true_cfg_scale": float(payload.get("true_cfg_scale", payload.get("guidance", 4.0))),
        "negative_prompt": str(payload.get("negative_prompt", " ")),
        "num_inference_steps": int(payload.get("num_inference_steps") or payload.get("steps") or 40),
        "guidance_scale": float(payload.get("guidance_scale", 1.0)),
        "num_images_per_prompt": int(payload.get("num_images_per_prompt") or payload.get("n") or 1),
    }
    if payload.get("seed") is not None:
        inputs["generator"] = torch_module.Generator(device=device).manual_seed(int(payload["seed"]))

    with torch_module.inference_mode():
        return pipe(**inputs).images[0]


def _image_inputs(payload: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in ("image_url", "image_urls", "images", "image"):
        value = payload.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value is not None:
            values.append(value)
    return [_decode_image(value) for value in values]


def _decode_image(value: Any) -> Any:
    from PIL import Image

    if isinstance(value, dict):
        value = value.get("url") or value.get("image_url") or value.get("data")
    value = value.strip()
    if value.startswith("data:image") and "," in value:
        raw = base64.b64decode(value.split(",", 1)[1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=60) as response:  # noqa: S310 - local service input.
            raw = response.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(value).convert("RGB")


def _encode_png(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _torch_dtype(torch_module: Any) -> Any:
    return {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }[CONFIG.dtype.lower()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Qwen-Image-Edit-2511.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    parser.add_argument("--persistent-pipeline", action="store_true")
    return parser


def main() -> None:
    import uvicorn

    args = _build_parser().parse_args()
    CONFIG.model_path = args.model_path
    CONFIG.device = args.device
    CONFIG.dtype = args.dtype
    CONFIG.persistent_pipeline = args.persistent_pipeline
    uvicorn.run(APP, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
