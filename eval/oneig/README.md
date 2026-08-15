# OneIG

This adapter loads the official English or Chinese OneIG cases and runs the
alignment, style, text, and reasoning metrics selected by category. The pinned
official checkout remains unmodified.

## Official Checkout

```bash
git clone https://github.com/OneIG-Bench/OneIG-Benchmark.git eval/oneig/OneIG-Benchmark
git -C eval/oneig/OneIG-Benchmark checkout 41b49831e79e6dde5323618c164da1c4cf0f699d
test "$(git -C eval/oneig/OneIG-Benchmark rev-parse HEAD)" = "41b49831e79e6dde5323618c164da1c4cf0f699d"
```

## Required Assets and Models

The checkout must contain `OneIG-Bench.csv`, `OneIG-Bench-ZH.csv`, and the
official metric scripts under `scripts/`. Set `evaluator.language` in
`configs/eval/oneig.yaml` to `en` or `zh`.

The style metric requires two manually placed files:

| Asset | Source | Destination |
| --- | --- | --- |
| CSD checkpoint | [Google Drive](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view?usp=sharing) | `eval/oneig/OneIG-Benchmark/scripts/style/models/checkpoint.pth` |
| CLIP ViT-L/14 | [OpenAI](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | `eval/oneig/OneIG-Benchmark/scripts/style/models/ViT-L-14.pt` |

Create the directory and download the directly accessible CLIP file:

```bash
mkdir -p eval/oneig/OneIG-Benchmark/scripts/style/models
curl -L \
  https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt \
  -o eval/oneig/OneIG-Benchmark/scripts/style/models/ViT-L-14.pt
```

Download the Hugging Face models used by the category metrics into one ignored
cache:

```bash
export HF_HOME="$PWD/models/huggingface"
hf download Qwen/Qwen2.5-VL-7B-Instruct
hf download xingpng/OneIG-StyleEncoder
hf download openai/clip-vit-large-patch14-336
hf download microsoft/LLM2CLIP-Openai-L-14-336
hf download microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned
```

Alignment and text use Qwen2.5-VL; style uses CSD and OneIG-StyleEncoder;
reasoning uses the three LLM2CLIP repositories. The shared generation services
use Z-Image on port `8010`, Qwen Image on `8009`, Qwen Image Edit on `8008`,
and the task-signature model on `8011`.

## Environment

GenRouter generation uses the root environment. Official OneIG metrics use the
isolated environment named by `evaluator.command_prefix`:

```bash
conda env create --file eval/oneig/environment.yml
```

## Run

Pass the same Hugging Face cache to the OneIG subprocesses:

```bash
HF_HOME="$PWD/models/huggingface" \
  PYTHONPATH=src python eval/run.py --config configs/eval/oneig.yaml
```

## Output

Artifacts are written to `data/benchmarks/oneig/<run-id>/`, including the
manifest, generations, official metrics, normalized scores, experience, route
memory, and summary. Retrying an incomplete phase clears that attempt's stale
metric CSV files before the official evaluator runs again.
