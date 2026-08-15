from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genrouter.artifacts import write_workflow_artifacts
from genrouter.backends.chat import build_llm_backend, build_mllm_backend, build_signature_llm_backend
from genrouter.backends.generator import build_generator_backend
from genrouter.backends.scorer import build_scorer_backend
from genrouter.backends.search import build_search_backend
from genrouter.config import load_project_config
from genrouter.knowledge.skill_bank import SkillBank
from genrouter.memory.experience_bank import ExperienceBank
from genrouter.memory.experience_record import build_experience_record
from genrouter.registries import build_generator_registry, build_workflow_registry
from genrouter.router import (
    GenRouter,
    RouteMemoryBank,
    TaskSignatureExtractor,
    compatible_plan,
    construct_candidate_plans,
)
from genrouter.workflows.base import WorkflowExecutionError
from genrouter.workflows.factory import build_workflow


@dataclass
class RuntimeContext:
    config: Any
    workflows: Any
    generators: Any
    skills: SkillBank
    llm: Any
    signature_llm: Any
    mllm: Any
    search_backend: Any
    scorer: Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenRouter CLI")
    _add_common_config_args(parser)
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List registered resources.")
    _add_common_config_args(list_parser)
    list_parser.add_argument("target", nargs="?", choices=["all", "workflows", "generators", "skills"], default="all")

    run_parser = subparsers.add_parser("run", help="Run a manually selected workflow-generator plan.")
    _add_common_config_args(run_parser)
    _add_prompt_args(run_parser)
    run_parser.add_argument("--workflow", required=True)
    run_parser.add_argument("--generator", "--model", dest="generator", required=True)
    run_parser.add_argument("--output-dir", default="")

    route_parser = subparsers.add_parser("route", help="Select and run a plan with GenRouter.")
    _add_common_config_args(route_parser)
    _add_prompt_args(route_parser)
    route_parser.add_argument("--output-dir", default="")

    cold_parser = subparsers.add_parser("cold-start", help="Run all compatible plans for prompts and refresh route memory.")
    _add_common_config_args(cold_parser)
    cold_parser.add_argument("--prompt-file", required=True)
    cold_parser.add_argument("--output-dir", default="")
    cold_parser.add_argument("--prompt-id", default="")
    cold_parser.add_argument("--distill-every-records", type=int, default=50)

    distill_parser = subparsers.add_parser("distill-route-memory", help="Refresh route memory from trajectory records.")
    _add_common_config_args(distill_parser)
    distill_parser.add_argument("--experience-bank", default="")
    distill_parser.add_argument("--route-memory", default="")

    # Legacy flat options. If no subcommand is provided, main() maps these to list/run/route behavior.
    parser.add_argument("--mode", choices=["route"], default="route", help=argparse.SUPPRESS)
    parser.add_argument("--list", choices=["all", "workflows", "generators", "skills"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--prompt", default="", help=argparse.SUPPRESS)
    parser.add_argument("--prompt-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--workflow", default="", help=argparse.SUPPRESS)
    parser.add_argument("--generator", "--model", dest="generator", default="", help=argparse.SUPPRESS)
    parser.add_argument("--prompt-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="", help=argparse.SUPPRESS)
    return parser


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", default="configs", help="Directory containing default/workflows/generators/skills YAML files.")
    parser.add_argument("--config", default="", help="Path to default.yaml. The parent directory is used as the config directory.")


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", default="", help="Prompt to run through GenRouter.")
    parser.add_argument("--prompt-file", default="", help="Text file with one prompt per non-empty line.")
    parser.add_argument("--prompt-id", default="", help="Stable prompt id for artifact paths.")


def _config_dir(args: argparse.Namespace) -> str:
    if args.config:
        path = Path(args.config)
        return str(path.parent if path.suffix else path)
    return str(args.config_dir)


def _load_prompt_file(path: str) -> list[str]:
    prompt_path = Path(path)
    prompts = [line.strip() for line in prompt_path.read_text(encoding="utf-8").splitlines()]
    return [prompt for prompt in prompts if prompt]


def _prompt_ids(prompts: list[str], prompt_file: str, prompt_id: str) -> list[str]:
    if len(prompts) == 1:
        return [prompt_id or f"prompt-{uuid.uuid4().hex[:12]}"]
    prefix = prompt_id or Path(prompt_file).stem or "prompt"
    return [f"{prefix}-{index:04d}" for index in range(1, len(prompts) + 1)]


def _build_context(args: argparse.Namespace) -> RuntimeContext:
    config = load_project_config(_config_dir(args))
    workflows = build_workflow_registry(config.workflows)
    generators = build_generator_registry(config.generators)
    skills = SkillBank.from_config(config.skills)
    llm = build_llm_backend(config.default.get("llm", {}))
    signature_llm = build_signature_llm_backend(config.default) if "signature_llm" in config.default else llm
    return RuntimeContext(
        config=config,
        workflows=workflows,
        generators=generators,
        skills=skills,
        llm=llm,
        signature_llm=signature_llm,
        mllm=build_mllm_backend(config.default.get("mllm", {})),
        search_backend=build_search_backend(config.default.get("search", {})),
        scorer=build_scorer_backend(config.default.get("scorer", {})),
    )


def _signature_extractor(ctx: RuntimeContext) -> TaskSignatureExtractor:
    routing = dict(ctx.config.default.get("routing", {}))
    return TaskSignatureExtractor(ctx.signature_llm, max_retries=int(routing.get("signature_max_retries", 2)))


def _router(ctx: RuntimeContext) -> GenRouter:
    return GenRouter.from_config(ctx.workflows, ctx.generators, ctx.config.default, llm=ctx.signature_llm)


def _run_selected_plan(
    *,
    prompt: str,
    prompt_id: str,
    workflow_name: str,
    generator_name: str,
    output_dir: str,
    ctx: RuntimeContext,
    task_signature: Any,
    route_decision: Any = None,
    benchmark: str = "manual",
) -> dict[str, Any]:
    if workflow_name not in ctx.workflows:
        raise SystemExit(f"Unknown workflow: {workflow_name}")
    if generator_name not in ctx.generators:
        raise SystemExit(f"Unknown generator/model: {generator_name}")

    workflow_spec = ctx.workflows.get(workflow_name)
    generator_spec = ctx.generators.get(generator_name)
    if not compatible_plan(workflow_spec, generator_spec, task_signature):
        raise SystemExit(f"Incompatible plan: workflow={workflow_name}, generator={generator_name}")

    workflow = build_workflow(
        workflow_name,
        ctx.skills,
        llm=ctx.llm,
        mllm=ctx.mllm,
        search_backend=ctx.search_backend,
    )
    generator = build_generator_backend(generator_spec)
    workflow_config = workflow_spec.config.to_dict()
    workflow_config["task_signature"] = task_signature.to_dict()
    try:
        result = workflow.run(
            prompt=prompt,
            generator=generator,
            config=workflow_config,
            prompt_id=prompt_id,
        )
    except WorkflowExecutionError as exc:
        write_workflow_artifacts(
            exc.partial_result,
            output_dir,
            prompt=prompt,
            task_signature=task_signature,
        )
        raise

    image_score = float(ctx.scorer.score(prompt, result.final_image))
    routing = dict(ctx.config.default.get("routing", {}))
    result = replace(result, score=image_score).with_utility(
        lambda_c=float(routing.get("lambda_c", 0.0)),
        lambda_l=float(routing.get("lambda_l", 0.0)),
    )
    paths = write_workflow_artifacts(
        result,
        output_dir,
        prompt=prompt,
        task_signature=task_signature,
    )
    result_payload = result.to_dict()
    if "image_path" in paths:
        result_payload["final_image_path"] = paths["image_path"]
    sketch_artifacts = {key: value for key, value in paths.items() if key.startswith("sketch_")}
    if sketch_artifacts:
        result_payload["sketch_artifacts"] = sketch_artifacts
    reference_artifacts = {key: value for key, value in paths.items() if key.startswith("reference")}
    if reference_artifacts:
        result_payload["reference_artifacts"] = reference_artifacts

    score_source = str(getattr(ctx.scorer, "score_source", "scorer"))
    score_details = getattr(ctx.scorer, "last_score_details", {})
    workflow_data = {}
    if isinstance(score_details, dict) and score_details:
        workflow_data[score_source] = dict(score_details)

    record = build_experience_record(
        prompt_id=prompt_id,
        prompt=prompt,
        benchmark=benchmark,
        task_signature=task_signature,
        workflow_name=workflow_name,
        generator_name=generator_name,
        image_score=image_score,
        result=result,
        result_payload=result_payload,
        selected_by="route" if route_decision else "manual",
        lambda_c=float(routing.get("lambda_c", 0.0)),
        lambda_l=float(routing.get("lambda_l", 0.0)),
        score_source=score_source,
        workflow_data=workflow_data,
    )
    ExperienceBank(_experience_path(ctx)).add(record)
    _maybe_refresh_route_memory(ctx)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "selected_plan": {"workflow": workflow_name, "generator": generator_name},
        "final_image_path": result_payload.get("final_image_path", ""),
        "image_score": image_score,
        "score_source": score_source,
        "score_details": score_details if isinstance(score_details, dict) else {},
        "cost": result.cost,
        "latency": result.latency,
        "token_usage": result_payload.get("token_usage", {}),
        "route_decision": route_decision.to_dict() if route_decision else None,
        "result": result_payload,
        "artifacts": paths,
    }


def _run_prompt(args: argparse.Namespace, ctx: RuntimeContext, *, routed: bool) -> dict[str, Any]:
    workflow_name = "" if routed else args.workflow
    generator_name = "" if routed else args.generator
    route_decision = None
    if routed:
        route_decision = _router(ctx).select(args.prompt)
        workflow_name = route_decision.selected_plan.workflow
        generator_name = route_decision.selected_plan.generator
        task_signature = route_decision.task_signature
    else:
        if not workflow_name or not generator_name:
            raise SystemExit("Manual run requires both --workflow and --generator/--model")
        task_signature = _signature_extractor(ctx).extract(args.prompt)
    return _run_selected_plan(
        prompt=args.prompt,
        prompt_id=args.prompt_id or f"prompt-{uuid.uuid4().hex[:12]}",
        workflow_name=workflow_name,
        generator_name=generator_name,
        output_dir=args.output_dir,
        ctx=ctx,
        task_signature=task_signature,
        route_decision=route_decision,
    )


def _run_prompt_batch(args: argparse.Namespace, ctx: RuntimeContext, *, routed: bool) -> dict[str, Any]:
    prompts = _load_prompt_file(args.prompt_file)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompt_file}")
    prompt_ids = _prompt_ids(prompts, args.prompt_file, args.prompt_id)
    runs = []
    for prompt, prompt_id in zip(prompts, prompt_ids):
        item_args = argparse.Namespace(**vars(args))
        item_args.prompt = prompt
        item_args.prompt_id = prompt_id
        item_args.prompt_file = ""
        runs.append(_run_prompt(item_args, ctx, routed=routed))
    return {"prompt_file": str(Path(args.prompt_file).resolve()), "count": len(runs), "runs": runs}


def _run_or_route(args: argparse.Namespace, *, routed: bool) -> dict[str, Any]:
    if args.prompt and args.prompt_file:
        raise SystemExit("Use either --prompt or --prompt-file, not both")
    if not args.prompt and not args.prompt_file:
        raise SystemExit("Provide --prompt or --prompt-file")
    ctx = _build_context(args)
    args.output_dir = _output_dir(ctx, args.output_dir)
    if args.prompt_file:
        return _run_prompt_batch(args, ctx, routed=routed)
    return _run_prompt(args, ctx, routed=routed)


def _cold_start(args: argparse.Namespace) -> dict[str, Any]:
    prompts = _load_prompt_file(args.prompt_file)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompt_file}")
    ctx = _build_context(args)
    args.output_dir = _output_dir(ctx, args.output_dir, child="cold_start")
    extractor = _signature_extractor(ctx)
    runs = []
    evaluated_plans: set[tuple[str, str]] = set()
    for prompt_index, prompt in enumerate(prompts, start=1):
        task_signature = extractor.extract(prompt)
        candidates = construct_candidate_plans(
            ctx.workflows,
            ctx.generators,
            task_signature=task_signature,
            generator_options=_generator_options(ctx),
        )
        prompt_prefix = args.prompt_id or Path(args.prompt_file).stem or "cold-start"
        for candidate in candidates:
            evaluated_plans.add((candidate.workflow, candidate.generator))
            prompt_id = f"{prompt_prefix}-{prompt_index:04d}-{candidate.workflow}-{candidate.generator}".lower()
            runs.append(
                _run_selected_plan(
                    prompt=prompt,
                    prompt_id=prompt_id,
                    workflow_name=candidate.workflow,
                    generator_name=candidate.generator,
                    output_dir=args.output_dir,
                    ctx=ctx,
                    task_signature=task_signature,
                    benchmark="cold_start",
                )
            )
            if args.distill_every_records > 0 and len(runs) % args.distill_every_records == 0:
                _distill_route_memory(ctx)
    route_memory = _distill_route_memory(ctx)
    return {
        "prompt_file": str(Path(args.prompt_file).resolve()),
        "prompts": len(prompts),
        "candidate_plans": len(evaluated_plans),
        "records": len(runs),
        "route_memory_path": _route_memory_path(ctx),
        "route_memory_buckets": len(route_memory),
    }


def _distill_command(args: argparse.Namespace) -> dict[str, Any]:
    config = load_project_config(_config_dir(args))
    route_memory = _distill_route_memory(
        config,
        experience_path=args.experience_bank or None,
        route_memory_path=args.route_memory or None,
    )
    return {
        "experience_bank": args.experience_bank or _experience_path(config),
        "route_memory_path": args.route_memory or _route_memory_path(config),
        "route_memory_buckets": len(route_memory),
    }


def _distill_route_memory(
    ctx_or_config: RuntimeContext | Any,
    experience_path: str | None = None,
    route_memory_path: str | None = None,
) -> list[dict[str, Any]]:
    config = ctx_or_config.config if isinstance(ctx_or_config, RuntimeContext) else ctx_or_config
    routing = dict(config.default.get("routing", {}))
    records = ExperienceBank(experience_path or _experience_path(config)).records()
    route_memory = RouteMemoryBank.distill(
        records,
        lambda_c=float(routing.get("lambda_c", 0.0)),
        lambda_l=float(routing.get("lambda_l", 0.0)),
    )
    RouteMemoryBank(route_memory_path or _route_memory_path(config)).write(route_memory)
    return route_memory


def _maybe_refresh_route_memory(ctx: RuntimeContext) -> None:
    routing = dict(ctx.config.default.get("routing", {}))
    every = int(routing.get("distill_every_records", 0) or 0)
    if every <= 0:
        return
    record_count = len(ExperienceBank(_experience_path(ctx)).records())
    if record_count and record_count % every == 0:
        _distill_route_memory(ctx)


def _list_payload(args: argparse.Namespace, target: str) -> dict[str, object]:
    config = load_project_config(_config_dir(args))
    workflows = build_workflow_registry(config.workflows)
    generators = build_generator_registry(config.generators)
    skills = SkillBank.from_config(config.skills)
    payload: dict[str, object] = {}
    if target in {"all", "workflows"}:
        payload["workflows"] = workflows.names()
    if target in {"all", "generators"}:
        payload["generators"] = generators.names()
    if target in {"all", "skills"}:
        payload["skills"] = skills.available()
    return payload


def _experience_path(ctx_or_config: RuntimeContext | Any) -> str:
    config = ctx_or_config.config if isinstance(ctx_or_config, RuntimeContext) else ctx_or_config
    return str(dict(config.default.get("paths", {})).get("experience_bank", "data/experience_bank.jsonl"))


def _route_memory_path(ctx_or_config: RuntimeContext | Any) -> str:
    config = ctx_or_config.config if isinstance(ctx_or_config, RuntimeContext) else ctx_or_config
    paths = dict(config.default.get("paths", {}))
    return str(paths.get("route_memory", "data/route_memory.jsonl"))


def _output_dir(
    ctx_or_config: RuntimeContext | Any,
    requested: str,
    *,
    child: str = "",
) -> str:
    if requested:
        return requested
    config = ctx_or_config.config if isinstance(ctx_or_config, RuntimeContext) else ctx_or_config
    root = Path(str(dict(config.default.get("paths", {})).get("logs", "data/runs")))
    return str(root / child) if child else str(root)


def _generator_options(ctx: RuntimeContext) -> list[str]:
    return _string_list(dict(ctx.config.default.get("generator", {})).get("options"))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _legacy_command(args: argparse.Namespace) -> str:
    if args.list or (not args.prompt and not args.prompt_file):
        return "list"
    if args.workflow or args.generator:
        return "run"
    return "route"


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    args = _build_parser().parse_args()
    command = args.command or _legacy_command(args)
    if command == "list":
        _print(_list_payload(args, getattr(args, "target", None) or args.list or "all"))
        return
    if command == "run":
        _print(_run_or_route(args, routed=False))
        return
    if command == "route":
        _print(_run_or_route(args, routed=True))
        return
    if command == "cold-start":
        _print(_cold_start(args))
        return
    if command == "distill-route-memory":
        _print(_distill_command(args))
        return
    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
