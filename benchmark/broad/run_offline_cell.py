"""Restore, shortcut, and certify one broad offline cell."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from assembly_loader import build as build_assembly, restore_records  # noqa: E402
from common import (  # noqa: E402
    ROOT,
    build_mrbench_environment,
    environment_fingerprint,
    mode_runs,
    restore_mrbench_path,
)
from broad_lib import (  # noqa: E402
    cases,
    cost_record,
    implementation_sha256,
    manifest_cases,
    methods,
    protocol,
    protocol_sha256,
    read_json,
    sha256,
    write_json,
)
from mr_shortcutting import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(cases()))
    parser.add_argument("--method", required=True, choices=tuple(methods()))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = protocol()
    if args.seed not in data["shortcut_seeds"]:
        raise ValueError("shortcut seed is outside broad protocol")
    case_protocol = cases()[args.case]
    if args.seconds != float(case_protocol["seconds"]):
        raise ValueError("shortcut budget differs from broad protocol")

    source_path = args.input.resolve()
    source = read_json(source_path)
    expected = {
        "schema": "mr-shortcut-broad-input-v1",
        "broad_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "cost": cost_record(),
        "pythonhashseed": int(case_protocol.get(
            "pythonhashseed", data["pythonhashseed"]
        )),
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise RuntimeError(f"input metadata mismatch for {key}")

    case = manifest_cases()[args.case]
    if case["kind"] == "mrbench":
        env = build_mrbench_environment(case)
    else:
        env, _, _ = build_assembly(case, ROOT)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    fingerprint = environment_fingerprint(env)
    if fingerprint != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from adopted input")

    path = (
        restore_mrbench_path(env, source["states"])
        if case["kind"] == "mrbench"
        else restore_records(env, source["states"])
    )
    from multi_robot_multi_goal_planning.problems.util import path_cost

    restored_cost = float(path_cost(path, env.batch_config_cost))
    tolerance = 1e-9 * max(1.0, abs(float(source["initial_cost"])))
    if abs(restored_cost - float(source["initial_cost"])) > tolerance:
        raise RuntimeError("restored input cost changed")

    method = methods()[args.method]
    started = time.perf_counter()
    output_path, result = run(
        env,
        path,
        method["policy"],
        args.seconds,
        args.seed,
        interpolation_resolution=0.5,
    )
    result["total_wall_seconds"] = time.perf_counter() - started
    start_preserved = bool(np.array_equal(output_path[0].q.state(), path[0].q.state()))
    end_preserved = bool(np.array_equal(output_path[-1].q.state(), path[-1].q.state()))
    strict = bool(env.is_path_collision_free(
        output_path,
        resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance),
        check_start_and_end=True,
        check_edges_in_order=True,
        binary_order=False,
    ))
    valid = bool(env.is_valid_plan(output_path)) if hasattr(env, "is_valid_plan") else None
    input_modes = mode_runs(path)
    output_modes = mode_runs(output_path)
    terminal = bool(env.is_terminal_mode(output_path[-1].mode))
    if not (
        input_modes == output_modes
        and start_preserved
        and end_preserved
        and strict
        and valid is not False
        and terminal
    ):
        raise RuntimeError(
            "broad output certificate failed: "
            f"modes={input_modes == output_modes}, start={start_preserved}, "
            f"end={end_preserved}, strict={strict}, valid={valid}, terminal={terminal}"
        )

    result.update({
        "schema": "mr-shortcut-broad-offline-result-v1",
        "broad_protocol_sha256": protocol_sha256(),
        "implementation_sha256": implementation_sha256(),
        "case_id": args.case,
        "kind": case["kind"],
        "method_id": args.method,
        "implementation_policy": method["policy"],
        "shortcut_seed": args.seed,
        "cost": cost_record(),
        "source_sha256": sha256(source_path),
        "environment_fingerprint": fingerprint,
        "input_mode_runs": input_modes,
        "output_mode_runs": output_modes,
        "endpoint_start_preserved": start_preserved,
        "endpoint_end_preserved": end_preserved,
        "terminal_mode_preserved": terminal,
        "strict_collision_free": strict,
        "valid_plan": valid,
    })
    write_json(args.output.resolve(), result)
    print(
        f"[ok] broad {args.case} {args.method} seed{args.seed}: "
        f"{result['initial_cost']:.9g} -> {result['final_cost']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

