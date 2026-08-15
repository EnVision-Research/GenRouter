# Benchmark Evaluation

All seven paper benchmarks use the shared engine in `eval/common/engine.py` and one public entry point:

```bash
bash scripts/serve.sh
```

Then run the selected benchmark in another terminal:

```bash
PYTHONPATH=src python eval/run.py --config configs/eval/wise.yaml
```

Each benchmark config fixes its official checkout, evaluator, local service overrides, seed, cold-start size 10, and routed batch size 50. The ordered generator pool comes from `configs/default.yaml`. Generators declare `generation_modes` in `configs/generators.yaml`: `text2image`, `image2image`, or both. Workflows declare their compatible mode in `configs/workflows.yaml`, with HybridGen deriving it from the active task signature.

The default paper pool uses Z-Image on port `8010`, Qwen Image on `8009`, Qwen Image Edit on `8008`, and the task-signature service on `8011`. Cold start evaluates every compatible workflow-generator pair for the first 10 cases. GenRouter then regenerates those cases for final scoring and processes the rest in ordered batches of 50, refreshing experience and route memory only after a complete official evaluation.

The bundled service scripts load checkpoints from the ignored repository-local `models/` directory. GPU assignments are direct `CUDA_VISIBLE_DEVICES` values in `scripts/serve.sh`; edit or comment out those command lines as needed. The benchmark runner checks required endpoints before creating run artifacts.

To add GPT-Image, Flux, or another backend, register its provider and `generation_modes`, add its name to `generator.options` in `configs/default.yaml`, and optionally add a same-named model/service override to a benchmark config. No benchmark adapter or engine edit is required for an existing mode.

| Benchmark | Setup | Config |
| --- | --- | --- |
| DPG-Bench | [setup](dpg_bench/README.md) | [`dpg_bench.yaml`](../configs/eval/dpg_bench.yaml) |
| OneIG | [setup](oneig/README.md) | [`oneig.yaml`](../configs/eval/oneig.yaml) |
| WISE | [setup](wise/README.md) | [`wise.yaml`](../configs/eval/wise.yaml) |
| LongText-Bench | [setup](textbench/README.md) | [`textbench.yaml`](../configs/eval/textbench.yaml) |
| GenEval2 | [setup](geneval2/README.md) | [`geneval2.yaml`](../configs/eval/geneval2.yaml) |
| ArtiMuse | [setup](artimuse/README.md) | [`artimuse.yaml`](../configs/eval/artimuse.yaml) |
| SpatialGenEval | [setup](spatialgeneval/README.md) | [`spatialgeneval.yaml`](../configs/eval/spatialgeneval.yaml) |

Official source stays in the ignored nested directory named by each benchmark README. The release validates the pinned revision and required files, but it does not clone, patch, update, or delete official repositories.
