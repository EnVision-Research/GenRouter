<div align="center">

# GenRouter

<!-- Paper title, authors, affiliations, venue, and links will be added here. -->

</div>

GenRouter routes image-generation prompts to compatible workflow and generator
combinations, then turns official benchmark feedback into experience and route
memory for later decisions.

## Method

For each prompt, GenRouter extracts a task signature, retrieves prior scored
experience, filters incompatible workflow-generator pairs, and selects a
Pareto-efficient plan. The evaluation engine applies one protocol across seven
benchmarks: cold start, routed rerun, and batched feedback.

## Repository Layout

```text
.
|-- src/genrouter/       # Routing, workflows, backends, memory, and logging
|-- configs/            # Runtime, generator, workflow, and evaluation configs
|-- scripts/services/   # Local image and language model services
|-- eval/               # Shared benchmark runner and official adapters
|-- data/               # Ignored run artifacts; only .gitkeep is tracked
`-- models/             # Ignored local checkpoints; only .gitkeep is tracked
```

## Installation

```bash
conda env create --file environment.yml
conda activate genrouter
```

OneIG and WISE use additional isolated environments documented in their
benchmark setup guides.

## Model Setup

Download the shared local models to the paths expected by
`scripts/services/` and `scripts/serve.sh`:

```bash
hf download Tongyi-MAI/Z-Image-Turbo \
  --local-dir models/Z-Image-Turbo
hf download Qwen/Qwen-Image-2512 \
  --local-dir models/Qwen-Image-2512
hf download Qwen/Qwen-Image-Edit-2511 \
  --local-dir models/Qwen-Image-Edit-2511
hf download Qwen/Qwen3.5-4B \
  --local-dir models/Qwen3.5-4B
hf download Qwen/Qwen3.5-35B-A3B \
  --local-dir models/Qwen3.5-35B-A3B
```

| Service | Model directory | Port |
| --- | --- | ---: |
| Z-Image generator | `models/Z-Image-Turbo` | `8010` |
| Qwen Image generator | `models/Qwen-Image-2512` | `8009` |
| Qwen Image Edit generator | `models/Qwen-Image-Edit-2511` | `8008` |
| Task-signature model | `models/Qwen3.5-4B` | `8011` |
| WISE judge | `models/Qwen3.5-35B-A3B` | `8000` |

Evaluator-specific checkpoints are listed in each benchmark README; they are
not downloaded by `eval/run.py` unless the upstream evaluator explicitly uses
an online model cache.

## Configuration

For remote backends, create the ignored API configuration and fill only the
profiles you use:

```bash
cp configs/api_config.example.yaml configs/api_config.yaml
```

Credentials may also be supplied through `DASHSCOPE_API_KEY`,
`MODELSCOPE_API_KEY`, and `SERPER_API_KEY`. Generator capabilities and defaults
are defined in `configs/generators.yaml`; the routing pool is defined in
`configs/default.yaml`.

## Quick Start

List registered workflows, generators, and skills:

```bash
PYTHONPATH=src python run.py --list all
```

Run one explicit workflow-generator plan:

```bash
PYTHONPATH=src python run.py \
  --prompt 'A red cube on a table' \
  --workflow DirectGen \
  --generator qwen_image \
  --prompt-id demo \
  --output-dir data/runs
```

Omit `--workflow` and `--generator` to route from accumulated experience and
route memory.

## Benchmarks

Every benchmark uses the same entry point. Its setup guide owns the pinned
official checkout, evaluator model, cache, and environment contract.

| Benchmark | Setup | Run |
| --- | --- | --- |
| DPG-Bench | [guide](eval/dpg_bench/README.md) | `PYTHONPATH=src python eval/run.py --config configs/eval/dpg_bench.yaml` |
| OneIG | [guide](eval/oneig/README.md) | `PYTHONPATH=src python eval/run.py --config configs/eval/oneig.yaml` |
| WISE | [guide](eval/wise/README.md) | `PYTHONPATH=src python eval/run.py --config configs/eval/wise.yaml` |
| LongText-Bench | [guide](eval/textbench/README.md) | `PYTHONPATH=src python eval/run.py --config configs/eval/textbench.yaml` |
| GenEval2 | [guide](eval/geneval2/README.md) | `PYTHONPATH=src python eval/run.py --config configs/eval/geneval2.yaml` |
| ArtiMuse | [guide](eval/artimuse/README.md) | `PYTHONPATH=src python eval/run.py --config configs/eval/artimuse.yaml` |
| SpatialGenEval | [guide](eval/spatialgeneval/README.md) | `PYTHONPATH=src python eval/run.py --config configs/eval/spatialgeneval.yaml` |

## Reproduce the Paper

Edit GPU assignments or comment out unused services in `scripts/serve.sh`, then
start the services required by the selected benchmark:

```bash
bash scripts/serve.sh
```

Run the benchmark in another terminal using the command in the table above.
See [Benchmark Evaluation](eval/README.md) for the common protocol and service
configuration.

## Output Layout

Manual runs write one directory per prompt:

```text
data/runs/<prompt-id>/
|-- result.json
`-- trace.jsonl
```

Benchmark runs keep protocol state, scores, memory, and generated images under:

```text
data/benchmarks/<benchmark>/<run-id>/
|-- manifest.json
|-- cold_start/
|-- batches/
|-- records.jsonl
|-- scores.jsonl
|-- experience.jsonl
|-- route_memory.jsonl
|-- summary.json
`-- images/
```

The manifest advances only after a complete phase or batch succeeds.

## Citation

<!-- BibTeX citation will be added here. -->

## License and Third-Party Benchmarks

This repository does not currently contain a license file, so no project
license is asserted here. Official benchmark repositories, datasets,
evaluator models, and weights retain their own licenses and are not
redistributed by this repository.
