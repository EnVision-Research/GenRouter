from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.common.benchmark import Benchmark
from eval.common.engine import EvaluationEngine, GenRouterEvaluationRuntime
from eval.common.experience import apply_scores_and_refresh_memory
from eval.common.preflight import ServicePreflightError, check_required_services
from eval.common.records import Manifest, RunStore
from eval.artimuse.benchmark import build_benchmark as build_artimuse
from eval.dpg_bench.benchmark import build_benchmark as build_dpg_bench
from eval.geneval2.benchmark import build_benchmark as build_geneval2
from eval.oneig.benchmark import build_benchmark as build_oneig
from eval.spatialgeneval.benchmark import build_benchmark as build_spatialgeneval
from eval.textbench.benchmark import build_benchmark as build_textbench
from eval.wise.benchmark import build_benchmark as build_wise
from genrouter.config import load_mapping


Factory = Callable[[Mapping[str, Any]], Benchmark]
BENCHMARK_FACTORIES: dict[str, Factory] = {
    "dpg_bench": build_dpg_bench,
    "wise": build_wise,
    "oneig": build_oneig,
    "textbench": build_textbench,
    "geneval2": build_geneval2,
    "artimuse": build_artimuse,
    "spatialgeneval": build_spatialgeneval,
}


def build_selected_benchmark(config: Mapping[str, Any]) -> Benchmark:
    name = str(config.get("benchmark") or "")
    try:
        factory = BENCHMARK_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {name}") from exc
    return factory(config)


def git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a GenRouter paper benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config_path = Path(args.config)
    config = load_mapping(config_path)
    benchmark = build_selected_benchmark(config)
    try:
        check_required_services(config)
    except ServicePreflightError as exc:
        raise SystemExit(str(exc)) from exc
    expected_revision = str(dict(config["official"])["revision"])
    paths = dict(config["paths"])
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    store = RunStore(Path(paths["output_root"]) / run_id)
    runtime = GenRouterEvaluationRuntime(
        config_dir=Path("configs"),
        benchmark_config=config,
        experience_path=store.experience_path,
        route_memory_path=store.route_memory_path,
    )
    generator_options = runtime.configured_generator_options()
    protocol = dict(config.get("protocol") or {})
    routing = dict(runtime.config.default.get("routing", {}) or {})
    manifest = Manifest(
        run_id=run_id,
        benchmark=benchmark.name,
        genrouter_revision=git_revision(REPO_ROOT),
        official_revision=expected_revision,
        config=dict(config),
        generator_options=generator_options,
        concrete_generators=runtime.concrete_generator_configs(
            generator_options
        ),
        seed=int(config.get("seed", 0) or 0),
        cold_start_size=int(protocol.get("cold_start_size", 10)),
        batch_size=int(protocol.get("batch_size", 50)),
        completed_phases=[],
    )
    store.initialize(manifest)

    def apply_phase(records, scores, *, selected_by: str):
        return apply_scores_and_refresh_memory(
            records,
            scores,
            benchmark=benchmark.name,
            selected_by=selected_by,
            experience_path=store.experience_path,
            route_memory_path=store.route_memory_path,
            lambda_c=float(routing.get("lambda_c", 0.0)),
            lambda_l=float(routing.get("lambda_l", 0.0)),
        )

    summary = EvaluationEngine(
        benchmark=benchmark,
        runtime=runtime,
        generator_options=generator_options,
        store=store,
        apply_phase=apply_phase,
    ).run(
        cold_start_size=manifest.cold_start_size,
        batch_size=manifest.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
