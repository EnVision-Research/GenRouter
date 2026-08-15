# SpatialGenEval

This adapter loads 1,230 official prompts and invokes the official stage 1 and
stage 2 evaluators. The pinned official checkout remains unmodified.

## Official Checkout

```bash
git clone https://github.com/AMAP-ML/SpatialGenEval.git eval/spatialgeneval/SpatialGenEval
git -C eval/spatialgeneval/SpatialGenEval checkout 8b294b5fd0bca204fcfbf2cd74b75d9c359e40f6
test "$(git -C eval/spatialgeneval/SpatialGenEval rev-parse HEAD)" = "8b294b5fd0bca204fcfbf2cd74b75d9c359e40f6"
```

## Required Assets and Models

The checkout must contain `eval/SpatialGenEval_T2I_Prompts.jsonl`,
`scripts/spatialgeneval_stage1_eval.py`, and
`scripts/spatialgeneval_stage2_acc.py`. Download the official judge model:

```bash
hf download Qwen/Qwen2.5-VL-72B-Instruct \
  --local-dir models/Qwen2.5-VL-72B-Instruct
```

Serve it with the API identity and port configured under `services.judge`:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS> vllm serve models/Qwen2.5-VL-72B-Instruct \
  --served-model-name Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size <GPU_COUNT> \
  --port 8001
```

Replace the GPU placeholders for the host. The shared generation services use
Z-Image on port `8010`, Qwen Image on `8009`, Qwen Image Edit on `8008`, and
the task-signature model on `8011`.

## Environment

Use the root `environment.yml` for GenRouter and the official evaluator. Run the
Qwen judge from a vLLM environment compatible with the installed CUDA stack.
The adapter owns the evaluator request settings; GPU allocation and service
lifetime remain operator-controlled.

## Run

With the judge and shared model services running:

```bash
PYTHONPATH=src python eval/run.py --config configs/eval/spatialgeneval.yaml
```

## Output

Artifacts are written to `data/benchmarks/spatialgeneval/<run-id>/`, including
the manifest, generations, official metrics, normalized scores, experience,
route memory, and summary. A failed official command does not advance the
manifest.
