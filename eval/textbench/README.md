# LongText-Bench

This adapter expands each official prompt to four images and runs X-Omni's OCR
evaluator and summarizer. The pinned official checkout remains unmodified.

## Official Checkout

```bash
git clone https://github.com/X-Omni-Team/X-Omni.git eval/textbench/X-Omni
git -C eval/textbench/X-Omni checkout 2b8237bb3789638c290eeda3e83ed81bd3652c3b
test "$(git -C eval/textbench/X-Omni rev-parse HEAD)" = "2b8237bb3789638c290eeda3e83ed81bd3652c3b"
```

## Required Assets and Models

The checkout must contain `textbench/text_prompts.jsonl`,
`textbench/text_prompts_zh.jsonl`, `textbench/evaluate_text_reward.py`, and
`textbench/summary_scores.py`. The upstream evaluator hardcodes
`Qwen/Qwen2.5-VL-7B-Instruct`; download it into a repository-local cache:

```bash
HF_HOME="$PWD/models/huggingface" \
  hf download Qwen/Qwen2.5-VL-7B-Instruct
```

Set `evaluator.language` to `en` or `zh`. The release protocol requires four
repeats. The shared generation services use Z-Image on port `8010`, Qwen Image
on `8009`, Qwen Image Edit on `8008`, and the task-signature model on `8011`.

## Environment

Use the root `environment.yml`. The adapter starts the evaluator with
`torchrun --standalone`; `evaluator.nproc_per_node` controls its local worker
count.

## Run

Use the same cache root so the upstream `from_pretrained` call finds the model:

```bash
HF_HOME="$PWD/models/huggingface" \
  PYTHONPATH=src python eval/run.py --config configs/eval/textbench.yaml
```

## Output

Artifacts are written to `data/benchmarks/textbench/<run-id>/`, including the
manifest, generations, official metrics, normalized scores, experience, route
memory, and summary. A failed official command does not advance the manifest.
