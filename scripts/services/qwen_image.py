from __future__ import annotations

import argparse
import asyncio
import base64
import io
import multiprocessing as mp
import os
import queue
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "Qwen-Image-2512")
DEFAULT_PORT = 8009

CONFIG = SimpleNamespace(
    model_path=DEFAULT_MODEL_PATH,
    device="cuda:0",
    dtype="bfloat16",
    default_size="1328x1328",
    default_steps=50,
    default_true_cfg_scale=4.0,
    default_guidance_scale=None,
    default_negative_prompt=" ",
    timeout=600.0,
    persistent_pipeline=False,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if CONFIG.persistent_pipeline:
        _start_workers()
    yield


APP = FastAPI(title="Qwen Image local server", lifespan=lifespan)
TASK_QUEUE: queue.Queue[tuple[Future[bytes], dict[str, Any]]] = queue.Queue()
WORKERS_STARTED = False


@APP.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "pid": os.getpid(), "model_path": CONFIG.model_path, "devices": _devices()}


@APP.post("/generate")
async def generate_png(prompt: str) -> Response:
    return Response(content=await _submit({"prompt": prompt}), media_type="image/png")


@APP.post("/v1/images/generations")
async def generate(request: Request) -> dict[str, Any]:
    payload = await request.json()
    encoded = base64.b64encode(await _submit(payload)).decode("utf-8")
    image = f"data:image/png;base64,{encoded}"
    return {"model": payload.get("model") or CONFIG.model_path, "images": [image], "output_images": [image]}


async def _submit(payload: dict[str, Any]) -> bytes:
    _start_workers()
    result: Future[bytes] = Future()
    TASK_QUEUE.put((result, payload))
    try:
        return await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(result)),
            timeout=CONFIG.timeout,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Processing timeout") from exc


def _start_workers() -> None:
    global WORKERS_STARTED
    if WORKERS_STARTED:
        return
    for device in _devices():
        threading.Thread(target=_worker, args=(device,), daemon=True).start()
    WORKERS_STARTED = True


def _worker(device: str) -> None:
    pipeline_state: tuple[Any, Any] | None = None
    if CONFIG.persistent_pipeline:
        pipeline_state = _load_pipeline(device)
    while True:
        result, payload = TASK_QUEUE.get()
        try:
            if CONFIG.persistent_pipeline:
                pipe, torch_module = pipeline_state
                image = _run_with_pipeline(pipe, torch_module, device, payload)
            else:
                image = _run_in_child(payload, device)
        except Exception as exc:  # noqa: BLE001 - expose local server/model errors to caller.
            result.set_exception(RuntimeError(f"{type(exc).__name__}: {exc}"))
        else:
            result.set_result(image)
        finally:
            TASK_QUEUE.task_done()


def _run_in_child(payload: dict[str, Any], device: str) -> bytes:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_child_generate, args=(payload, device, _config_snapshot(), result_queue))
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
        return value
    raise RuntimeError(str(value))


def _child_generate(payload: dict[str, Any], device: str, config: dict[str, Any], result_queue: Any) -> None:
    try:
        _apply_config(config)
        pipe, torch_module = _load_pipeline(device)
        result_queue.put(("ok", _run_with_pipeline(pipe, torch_module, device, payload)))
    except Exception as exc:  # noqa: BLE001 - child returns errors to parent.
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _config_snapshot() -> dict[str, Any]:
    return dict(vars(CONFIG))


def _apply_config(config: dict[str, Any]) -> None:
    for key, value in config.items():
        setattr(CONFIG, key, value)


def _load_pipeline(device: str) -> tuple[Any, Any]:
    import torch
    from diffusers import DiffusionPipeline

    model_path = Path(CONFIG.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    pipe = DiffusionPipeline.from_pretrained(str(model_path), torch_dtype=_torch_dtype(torch)).to(device)
    pipe.set_progress_bar_config(disable=None)
    return pipe, torch


def _run_with_pipeline(pipe: Any, torch_module: Any, device: str, payload: dict[str, Any]) -> bytes:
    return _encode_png(_run_pipeline(pipe, torch_module, device, payload))


def _run_pipeline(pipe: Any, torch_module: Any, device: str, payload: dict[str, Any]) -> Any:
    width, height = _parse_size(payload.get("size", CONFIG.default_size))
    inputs: dict[str, Any] = {
        "prompt": str(payload["prompt"]),
        "num_inference_steps": int(payload.get("num_inference_steps") or payload.get("steps") or CONFIG.default_steps),
        "width": width,
        "height": height,
        "num_images_per_prompt": int(payload.get("num_images_per_prompt") or payload.get("n") or 1),
    }

    if payload.get("seed") is not None:
        inputs["generator"] = torch_module.Generator(device=device).manual_seed(int(payload["seed"]))
    if CONFIG.default_negative_prompt or payload.get("negative_prompt") is not None:
        inputs["negative_prompt"] = str(payload.get("negative_prompt", CONFIG.default_negative_prompt))
    if CONFIG.default_true_cfg_scale is not None or payload.get("true_cfg_scale") is not None or payload.get("guidance") is not None:
        inputs["true_cfg_scale"] = float(payload.get("true_cfg_scale", payload.get("guidance", CONFIG.default_true_cfg_scale)))
    if CONFIG.default_guidance_scale is not None or payload.get("guidance_scale") is not None:
        inputs["guidance_scale"] = float(payload.get("guidance_scale", CONFIG.default_guidance_scale))

    with torch_module.inference_mode():
        return pipe(**inputs).images[0]


def _encode_png(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _parse_size(value: Any) -> tuple[int, int]:
    width, height = str(value).lower().split("x", 1)
    return int(width), int(height)


def _torch_dtype(torch_module: Any) -> Any:
    return {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }[CONFIG.dtype.lower()]


def _devices() -> list[str]:
    return [device.strip() for device in CONFIG.device.split(",") if device.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Qwen-Image-2512.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda:0", help="Comma-separated devices, e.g. cuda:0,cuda:1.")
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
