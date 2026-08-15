# GenEval2

This adapter loads 800 official prompts and runs GenEval2 Soft-TIFA after each
completed protocol phase. The pinned official checkout remains unmodified.

## Official Checkout

```bash
git clone https://github.com/facebookresearch/GenEval2.git eval/geneval2/GenEval2
git -C eval/geneval2/GenEval2 checkout a6e82d2289e8d418f27f0adee77908b07060eea3
test "$(git -C eval/geneval2/GenEval2 rev-parse HEAD)" = "a6e82d2289e8d418f27f0adee77908b07060eea3"
```

## Required Assets and Models

The checkout must contain `geneval2_data.jsonl` and `evaluation.py`. Download
`Qwen/Qwen3-VL-8B-Instruct` into the cache configured by
`evaluator.hf_home: models/huggingface`:

```bash
HF_HOME="$PWD/models/huggingface" \
  hf download Qwen/Qwen3-VL-8B-Instruct
```

The adapter sets `HF_HOME=models/huggingface`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1` for evaluation, so the model must be cached before the
run. The shared generation services use Z-Image on port `8010`, Qwen Image on
`8009`, Qwen Image Edit on `8008`, and the task-signature model on `8011`.

## Environment

Use the root `environment.yml` and install the pinned GenEval2 dependencies in
the Python selected by `evaluator.python`.

## Run

```bash
PYTHONPATH=src python eval/run.py --config configs/eval/geneval2.yaml
```

## Output

Artifacts are written to `data/benchmarks/geneval2/<run-id>/`, including the
manifest, generations, official metrics, normalized scores, experience, route
memory, and summary. A failed official command does not advance the manifest.
