from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .qwen_image import APP, CONFIG
else:
    from qwen_image import APP, CONFIG

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "Z-Image-Turbo")
DEFAULT_PORT = 8010

CONFIG.model_path = DEFAULT_MODEL_PATH
CONFIG.default_size = "1024x1024"
CONFIG.default_steps = 9
CONFIG.default_true_cfg_scale = None
CONFIG.default_guidance_scale = 0.0
CONFIG.default_negative_prompt = ""
CONFIG.persistent_pipeline = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Z-Image-Turbo.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda:0", help="Comma-separated devices, e.g. cuda:0,cuda:1.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    return parser


def main() -> None:
    import uvicorn

    args = _build_parser().parse_args()
    CONFIG.model_path = args.model_path
    CONFIG.device = args.device
    CONFIG.dtype = args.dtype
    uvicorn.run(APP, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
