from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Request


ROOT = Path(__file__).resolve().parents[2]
QWEN_IMAGE = ROOT / "scripts" / "services" / "qwen_image.py"
QWEN_EDIT = ROOT / "scripts" / "services" / "qwen_image_edit.py"
CHAT_QWEN35 = ROOT / "scripts" / "services" / "chat_qwen35.py"

APP = FastAPI(title="Qwen image/edit switch proxy")


@dataclass
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8260
    device: str = "cuda:0"
    timeout: float = 1800.0
    chat_host: str = "127.0.0.1"
    chat_port: int = 8011
    chat_device: str = ""
    chat_model_path: str = ""
    chat_dtype: str = "bfloat16"
    chat_enabled: bool = True
    persistent_pipeline: bool = True
    log_dir: str = "data/logs"


@dataclass
class BackendState:
    mode: str = ""
    process: Any = None
    port: int = 0

    def is_running(self, mode: str) -> bool:
        return self.mode == mode and self.process is not None and self.process.poll() is None

    def clear(self) -> None:
        self.mode = ""
        self.process = None
        self.port = 0


CONFIG = ServiceConfig()
ACTIVE = BackendState()
CHAT_PROCESS: Any = None


@APP.get("/health")
def health() -> dict[str, Any]:
    chat_ready = CONFIG.chat_enabled and _health_ok(CONFIG.chat_host, CONFIG.chat_port)
    return {
        "status": "ok",
        "active_mode": ACTIVE.mode,
        "active_port": ACTIVE.port,
        "chat_port": CONFIG.chat_port if CONFIG.chat_enabled else 0,
        "chat_ready": chat_ready,
        "chat_owned": CHAT_PROCESS is not None,
    }


@APP.post("/v1/images/generations")
async def generations(request: Request) -> dict[str, Any]:
    payload = await request.json()
    mode = "edit" if _has_image(payload) else "image"
    target_port = _ensure_backend(mode)
    response = requests.post(
        f"http://127.0.0.1:{target_port}/v1/images/generations",
        json=payload,
        timeout=CONFIG.timeout,
        proxies={"http": "", "https": ""},
    )
    response.raise_for_status()
    return response.json()


def _has_image(payload: dict[str, Any]) -> bool:
    return any(payload.get(key) for key in ("image", "images", "image_url", "image_urls"))


def _ensure_backend(mode: str) -> int:
    if ACTIVE.is_running(mode):
        return ACTIVE.port
    _stop_backend()
    target_port = CONFIG.port + (10 if mode == "image" else 11)
    if _health_ok("127.0.0.1", target_port):
        raise RuntimeError(f"port {target_port} already serves a backend not owned by this proxy")
    script = QWEN_IMAGE if mode == "image" else QWEN_EDIT
    command = [sys.executable, str(script), "--host", "127.0.0.1", "--port", str(target_port), "--device", CONFIG.device]
    if CONFIG.persistent_pipeline:
        command.append("--persistent-pipeline")
    process = _start_process(command, f"qwen_{mode}_{target_port}.log")
    ACTIVE.mode = mode
    ACTIVE.port = target_port
    ACTIVE.process = process
    try:
        _wait_health(target_port, process=process)
    except Exception:
        _stop_backend()
        raise
    return target_port


def _start_chat_background() -> None:
    global CHAT_PROCESS
    if not CONFIG.chat_enabled or _process_running(CHAT_PROCESS):
        return
    CHAT_PROCESS = None
    if _health_ok(CONFIG.chat_host, CONFIG.chat_port):
        return
    command = [
        sys.executable,
        str(CHAT_QWEN35),
        "--host",
        CONFIG.chat_host,
        "--port",
        str(CONFIG.chat_port),
        "--device",
        CONFIG.chat_device or CONFIG.device,
        "--dtype",
        CONFIG.chat_dtype,
        "--no-debug",
    ]
    if CONFIG.chat_model_path:
        command.extend(["--model-path", CONFIG.chat_model_path])
    CHAT_PROCESS = _start_process(command, f"chat_qwen35_{CONFIG.chat_port}.log")


def _start_process(command: list[str], log_name: str) -> Any:
    with _log_path(log_name).open("a", encoding="utf-8") as log:
        return subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _log_path(name: str) -> Path:
    path = ROOT / CONFIG.log_dir
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def _wait_health(port: int, host: str = "127.0.0.1", process: Any = None) -> None:
    deadline = time.monotonic() + CONFIG.timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"backend process exited before becoming healthy on port {port}")
        expected_pid = process.pid if process is not None else None
        if _health_ok(host, port, expected_pid=expected_pid):
            return
        time.sleep(1)
    raise RuntimeError(f"backend did not become healthy on port {port}")


def _health_ok(host: str, port: int, expected_pid: int | None = None) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as response:
            if response.status != 200:
                return False
            return expected_pid is None or json.loads(response.read()).get("pid") == expected_pid
    except OSError:
        return False


def _process_running(process: Any) -> bool:
    return process is not None and process.poll() is None


def _stop_process(process: Any) -> None:
    if not _process_running(process):
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def _stop_backend() -> None:
    try:
        _stop_process(ACTIVE.process)
    finally:
        ACTIVE.clear()


def _stop_chat() -> None:
    global CHAT_PROCESS
    try:
        _stop_process(CHAT_PROCESS)
    finally:
        CHAT_PROCESS = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8260)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--chat-host", default="127.0.0.1")
    parser.add_argument("--chat-port", type=int, default=8011)
    parser.add_argument("--chat-device", default="", help="Defaults to --device.")
    parser.add_argument("--chat-model-path", default="", help="Optional local model path for the signature server.")
    parser.add_argument("--chat-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--no-chat", action="store_true")
    parser.add_argument("--no-persistent-pipeline", action="store_true")
    parser.add_argument("--log-dir", default="data/logs")
    args = parser.parse_args()
    CONFIG.host = args.host
    CONFIG.port = args.port
    CONFIG.device = args.device
    CONFIG.timeout = args.timeout
    CONFIG.chat_host = args.chat_host
    CONFIG.chat_port = args.chat_port
    CONFIG.chat_device = args.chat_device
    CONFIG.chat_model_path = args.chat_model_path
    CONFIG.chat_dtype = args.chat_dtype
    CONFIG.chat_enabled = not args.no_chat
    CONFIG.persistent_pipeline = not args.no_persistent_pipeline
    CONFIG.log_dir = args.log_dir
    atexit.register(_stop_backend)
    atexit.register(_stop_chat)
    _start_chat_background()

    import uvicorn

    uvicorn.run(APP, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
