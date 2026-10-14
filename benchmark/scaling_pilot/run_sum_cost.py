"""Sum-cost cell worker; use run_sum_cost_study.py for supervised execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (REPO / "src", BENCHMARK, BENCHMARK / "all_scaling",
                  BENCHMARK / "scaling_2d", HERE):
    sys.path.insert(0, str(directory))

import study_lib as scaling  # noqa: E402
from common import environment_fingerprint, mode_runs, restore_mrbench_path  # noqa: E402
import scalable_tree_group  # noqa: E402,F401
from mr_shortcutting import run  # noqa: E402


CASES = (
    ("rai-disk-scaling-r12-g4", 15.0),
    ("rai-isolated-box-stacking-r9-b4", 30.0),
    ("rai-coupled-box-stacking-r6-b8", 30.0),
    ("rai-mobile-wall-scaling-r7-x5-z5", 30.0),
)
OUTPUT = HERE / "sum-cost" / "results"


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mains_connected():
    supplies = Path("/sys/class/power_supply")
    online = list(supplies.glob("*/online"))
    return not online or any(path.read_text().strip() == "1" for path in online)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--case", choices=[case_id for case_id, _ in CASES])
    parser.add_argument("--path-index", type=int, default=0)
    parser.add_argument("--shortcut-seed", type=int, default=0)
    parser.add_argument("--method")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol_path = HERE / "protocol.json"
    protocol = read_json(protocol_path)
    methods = protocol["methods"]
    if args.method:
        methods = [item for item in methods if item["id"] == args.method]
        if not methods:
            parser.error("unknown method")
    cases = [item for item in CASES if args.case is None or item[0] == args.case]
    expected = len(cases) * len(methods)
    if args.output and expected != 1:
        parser.error("--output requires a single --case and --method")
    completed = 0

    for case_id, seconds in cases:
        case = scaling.cases()[case_id]
        source_path = scaling.input_path(case_id, args.path_index).resolve()
        source = read_json(source_path)
        for selected in methods:
            destination = args.output or (OUTPUT / case_id / f"{selected['id']}-path{args.path_index}-seed{args.shortcut_seed}.json")
            if destination.is_file() and not args.force:
                print(f"[skip] {destination}", flush=True)
                continue
            if not mains_connected():
                raise RuntimeError("mains power disconnected before benchmark cell")

            env = scaling.build_environment(case)
            if environment_fingerprint(env) != source["environment_fingerprint"]:
                raise RuntimeError("restored environment differs from input")
            env.cost_metric = "euclidean"
            env.cost_reduction = "sum"
            input_path = restore_mrbench_path(env, source["states"])
            from multi_robot_multi_goal_planning.problems.util import path_cost

            initial_cost = float(path_cost(input_path, env.batch_config_cost))
            before = mode_runs(input_path)
            start = np.asarray(input_path[0].q.state(), dtype=float).copy()
            end = np.asarray(input_path[-1].q.state(), dtype=float).copy()
            wall_before = time.time()
            monotonic_before = time.monotonic()
            started = time.perf_counter()
            output_path, result = run(
                env, input_path, selected["policy"], seconds, args.shortcut_seed,
                interpolation_resolution=0.5,
            )
            elapsed = time.perf_counter() - started
            suspended = ((time.time() - wall_before)
                         - (time.monotonic() - monotonic_before)) > 2.0

            strict = bool(env.is_path_collision_free(
                output_path, resolution=float(env.collision_resolution),
                tolerance=float(env.collision_tolerance),
                check_start_and_end=True, check_edges_in_order=True,
                binary_order=False,
            ))
            valid = (bool(env.is_valid_plan(output_path))
                     if hasattr(env, "is_valid_plan") else None)
            terminal = bool(env.is_terminal_mode(output_path[-1].mode))
            modes = mode_runs(output_path)
            endpoints = bool(
                np.allclose(output_path[0].q.state(), start, rtol=0, atol=1e-8)
                and np.allclose(output_path[-1].q.state(), end, rtol=0, atol=1e-8)
            )
            final_cost = float(path_cost(output_path, env.batch_config_cost))
            if suspended or not (strict and valid is not False and terminal
                                 and modes == before and endpoints):
                raise RuntimeError(
                    f"certificate failed for {case_id}/{selected['id']}: "
                    f"suspended={suspended}, strict={strict}, valid={valid}, "
                    f"terminal={terminal}, modes={modes == before}, "
                    f"endpoints={endpoints}"
                )
            if abs(float(result["initial_cost"]) - initial_cost) > 1e-8 * max(1, initial_cost):
                raise RuntimeError("reported initial sum cost differs from restored cost")
            if abs(float(result["final_cost"]) - final_cost) > 1e-8 * max(1, final_cost):
                raise RuntimeError("reported final sum cost differs from certified path cost")

            result.update({
                "schema": "mr-shortcut-scaling-sum-cost-pilot-v1",
                "case_id": case_id,
                "family": case["family"],
                "method_id": selected["id"],
                "implementation_policy": selected["policy"],
                "path_index": args.path_index,
                "shortcut_seed": args.shortcut_seed,
                "budget_seconds": seconds,
                "total_wall_seconds": elapsed,
                "cost": {
                    "per_robot_metric": "euclidean",
                    "implementation_reduction": "sum",
                    "path_reduction": "sum_over_edges",
                },
                "source_path": str(source_path),
                "source_sha256": sha256(source_path),
                "protocol_sha256": sha256(protocol_path),
                "input_mode_runs": before,
                "output_mode_runs": modes,
                "endpoints_preserved": endpoints,
                "terminal_mode_preserved": terminal,
                "strict_collision_free": strict,
                "valid_plan": valid,
                "suspend_detected": suspended,
            })
            write_json(destination, result)
            completed += 1
            print(
                f"[ok {completed}/{expected}] {case_id} {selected['id']}: "
                f"{initial_cost:.6g} -> {final_cost:.6g}", flush=True,
            )
    print(json.dumps({"completed": completed, "expected": expected,
                      "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
