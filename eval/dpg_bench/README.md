# DPG-Bench

This adapter loads DPG prompts and runs the official ELLA evaluator with mPLUG
visual question answering. The pinned official checkout remains unmodified.

## Official Checkout

```bash
git clone https://github.com/TencentQQGYLab/ELLA.git eval/dpg_bench/ELLA
git -C eval/dpg_bench/ELLA checkout 3c228f1dc6c4d3cad0a47493816151a419f14db3
test "$(git -C eval/dpg_bench/ELLA rev-parse HEAD)" = "3c228f1dc6c4d3cad0a47493816151a419f14db3"
```

## Required Assets and Models

The checkout must contain `dpg_bench/prompts/`, `dpg_bench/dpg_bench.csv`, and
`dpg_bench/compute_dpg_bench.py`. Its evaluator loads
`damo/mplug_visual-question-answering_coco_large_en` through ModelScope.

Keep the cache inside the ignored `models/modelscope` directory:

```bash
mkdir -p models/modelscope
MODELSCOPE_CACHE="$PWD/models/modelscope" \
  modelscope download damo/mplug_visual-question-answering_coco_large_en
```

ModelScope owns the internal cache layout below `models/modelscope`; use the
same `MODELSCOPE_CACHE` value when running. The shared generation services use
Z-Image on port `8010`, Qwen Image on `8009`, Qwen Image Edit on `8008`, and
the task-signature model on `8011`.

## Environment

Use the root `environment.yml`, which includes ModelScope and the evaluator's
runtime dependencies.

## Run

```bash
MODELSCOPE_CACHE="$PWD/models/modelscope" \
  PYTHONPATH=src python eval/run.py --config configs/eval/dpg_bench.yaml
```

## Output

Artifacts are written to `data/benchmarks/dpg_bench/<run-id>/`, including the
manifest, generations, official metrics, normalized scores, experience, route
memory, and summary. A failed official command does not advance the manifest.
