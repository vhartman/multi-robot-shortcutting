"""Run a small certified sum-cost comparison on frozen paths."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
SCALING = BENCHMARK / "scaling_2d"
for directory in (BENCHMARK, HERE, SCALING, REPO / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import ROOT, mode_runs, restore_mrbench_path  # noqa: E402
from expanded_lib import (  # noqa: E402
    assembly_case,
    build_mrbench_environment,
    cases,
    read_json,
    sha256,
    write_json,
)
import scalable_tree_group  # noqa: E402,F401
from mr_shortcutting import run  # noqa: E402


CASE_IDS = (
    "rai-triple-waypoints",
    "rai-cube-four",
    "assembly-cross-4r-openarm",
    "assembly-car-mini-4r-openarm",
)
POLICIES = ("tree_prefix", "scalable_tree_group")


def restore(case, source):
    if case["kind"] == "mrbench":
        env = build_mrbench_environment(case)
        path = restore_mrbench_path(env, source["states"])
    else:
        from assembly_loader import build, restore_records

        env, _, _ = build(assembly_case(case), ROOT)
        path = restore_records(env, source["states"])
    env.cost_metric = "euclidean"
    env.cost_reduction = "sum"
    return env, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--case", action="append", choices=CASE_IDS)
    parser.add_argument("--path-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "analysis" / "sum-cost-scalable-pilot",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scheduled = 0
    selected_cases = args.case or list(CASE_IDS)
    total = len(selected_cases) * len(POLICIES)
    for case_id in selected_cases:
        case = cases()[case_id]
        source_path = HERE / "inputs" / case_id / f"path{args.path_index}.json"
        source = read_json(source_path)
        for policy in POLICIES:
            destination = (
                args.output_root / case_id
                / f"{policy}-path{args.path_index}-seed{args.seed}.json"
            )
            if destination.is_file() and not args.force:
                print(f"[skip] {destination}", flush=True)
                continue

            env, input_path = restore(case, source)
            input_modes = mode_runs(input_path)
            start = np.asarray(input_path[0].q.state(), dtype=float).copy()
            end = np.asarray(input_path[-1].q.state(), dtype=float).copy()
            started = time.perf_counter()
            output_path, result = run(
                env,
                input_path,
                policy,
                args.seconds,
                args.seed,
                interpolation_resolution=0.5,
            )
            total_wall = time.perf_counter() - started

            strict = bool(env.is_path_collision_free(
                output_path,
                resolution=float(env.collision_resolution),
                tolerance=float(env.collision_tolerance),
                check_start_and_end=True,
                check_edges_in_order=True,
                binary_order=False,
            ))
            valid = (
                bool(env.is_valid_plan(output_path))
                if hasattr(env, "is_valid_plan") else None
            )
            modes_preserved = input_modes == mode_runs(output_path)
            endpoints_preserved = bool(
                np.allclose(output_path[0].q.state(), start, rtol=0.0, atol=1e-8)
                and np.allclose(output_path[-1].q.state(), end, rtol=0.0, atol=1e-8)
            )
            if not (
                strict and valid is not False and modes_preserved and endpoints_preserved
            ):
                raise RuntimeError(
                    f"certificate failed for {case_id} {policy}: strict={strict}, "
                    f"valid={valid}, modes={modes_preserved}, endpoints={endpoints_preserved}"
                )

            result.update({
                "schema": "mr-shortcut-sum-cost-scalable-pilot-v1",
                "case_id": case_id,
                "method_id": policy,
                "path_index": args.path_index,
                "shortcut_seed": args.seed,
                "cost": {"per_agent": "euclidean", "reduction": "sum"},
                "source_sha256": sha256(source_path),
                "total_wall_seconds": total_wall,
                "strict_collision_free": strict,
                "valid_plan": valid,
                "mode_runs_preserved": modes_preserved,
                "endpoints_preserved": endpoints_preserved,
            })
            write_json(destination, result)
            scheduled += 1
            print(
                f"[ok {scheduled}/{total}] "
                f"{case_id} {policy}: "
                f"{result['initial_cost']:.7g} -> {result['final_cost']:.7g}",
                flush=True,
            )

    print(json.dumps({"completed": scheduled, "output": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
