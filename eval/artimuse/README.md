# ArtiMuse

This adapter scores the 553-case GenRouter aesthetic prompt manifest with the
official ArtiMuse evaluator. The pinned official checkout remains unmodified.

## Official Checkout

```bash
git clone https://github.com/thunderbolt215/ArtiMuse.git eval/artimuse/ArtiMuse
git -C eval/artimuse/ArtiMuse checkout 750d980b6b7e9d99da60a302dcdbcab14e01003f
test "$(git -C eval/artimuse/ArtiMuse rev-parse HEAD)" = "750d980b6b7e9d99da60a302dcdbcab14e01003f"
```

## Required Assets and Models

The checkout must contain `src/eval/eval_dataset.py` and `src/artimuse/`.
Download the official `Thunderbolt215215/ArtiMuse` checkpoint to the path in
`configs/eval/artimuse.yaml`:

```bash
hf download Thunderbolt215215/ArtiMuse \
  --local-dir models/ArtiMuse
```

The evaluator reads `models/ArtiMuse`. The shared generation services use
Z-Image on port `8010`, Qwen Image on `8009`, Qwen Image Edit on `8008`, and
the task-signature model on `8011`.

## Environment

Use the root environment, install the pinned checkout requirements, and install
the CUDA-specific dependency separately:

```bash
pip install -r eval/artimuse/ArtiMuse/requirements.txt
pip install flash-attn --no-build-isolation
```

## Run

Start the shared model services, then run from the repository root:

```bash
PYTHONPATH=src python eval/run.py --config configs/eval/artimuse.yaml
```

## Output

Artifacts are written to `data/benchmarks/artimuse/<run-id>/`, including the
manifest, generations, official metrics, normalized scores, experience, route
memory, and summary. A failed official command does not advance the manifest.
