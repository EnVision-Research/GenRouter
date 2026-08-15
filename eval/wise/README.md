# WISE

This adapter loads WISE Verified cases and invokes the pinned local-Qwen
evaluator. The pinned official checkout remains unmodified.

## Official Checkout

```bash
git clone https://github.com/PKU-YuanGroup/WISE.git eval/wise/wise
git -C eval/wise/wise checkout 09b5539d64681bf11102fc5c87e63a387beaf71d
test "$(git -C eval/wise/wise rev-parse HEAD)" = "09b5539d64681bf11102fc5c87e63a387beaf71d"
```

## Required Assets and Models

The checkout must contain `data_verified/merge.json`, `vllm_eval.py`, and
`calculate_verified.py`. Download the configured judge if it was not already
downloaded during shared model setup:

```bash
hf download Qwen/Qwen3.5-35B-A3B \
  --local-dir models/Qwen3.5-35B-A3B
```

The committed service command is:

```bash
CUDA_VISIBLE_DEVICES=4,5 conda run -n genrouter-wise --no-capture-output \
  vllm serve models/Qwen3.5-35B-A3B \
  --served-model-name Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 \
  --port 8000
```

Edit GPU assignments in `scripts/serve.sh` for the host. The remaining shared
services are Z-Image on port `8010`, Qwen Image on `8009`, Qwen Image Edit on
`8008`, and the task-signature model on `8011`.

This configuration reports WISE Verified with the local Qwen judge.
`WISE_legacy` with `gpt-4o-2024-05-13` is the original paper protocol and is
not the score produced by `configs/eval/wise.yaml`.

## Environment

Use the root `environment.yml` for generation and routing. Create the isolated
vLLM judge environment once:

```bash
conda env create --file eval/wise/environment.yml
```

## Run

Start the configured services:

```bash
bash scripts/serve.sh
```

Run WISE in another terminal:

```bash
PYTHONPATH=src python eval/run.py --config configs/eval/wise.yaml
```

## Output

Artifacts are written to `data/benchmarks/wise/<run-id>/`, including the
manifest, generations, official metrics, normalized scores, experience, route
memory, and summary. A failed official command does not advance the manifest.
