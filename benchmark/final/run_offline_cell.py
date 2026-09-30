"""Restore one frozen input and run one final offline-shortcutting cell."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    ROOT,
    build_mrbench_environment,
    environment_fingerprint,
    mode_runs,
    restore_mrbench_path,
)
from protocol_lib import (  # noqa: E402
    cost_record,
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
    data = protocol()
    method_map = methods()
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", required=True, choices=tuple(method_map))
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--shortcut-seed", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.path_index not in data["design"]["path_indices"]:
        raise ValueError("path index is outside frozen protocol")
    if args.shortcut_seed not in data["design"]["shortcut_seeds"]:
        raise ValueError("shortcut seed is outside frozen protocol")
    protocol_case = next(case for case in data["cases"] if case["id"] == args.case)
    if args.seconds != float(protocol_case["offline_seconds"]):
        raise ValueError("offline budget differs from frozen protocol")

    source_path = args.input.resolve()
    source = read_json(source_path)
    expected_source = {
        "schema": "mr-shortcut-final-input-v1",
        "final_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "path_index": args.path_index,
        "cost": cost_record(),
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise RuntimeError(
                f"input metadata mismatch for {key}: {source.get(key)!r} != {expected!r}"
            )

    case = manifest_cases()[args.case]
    if case["kind"] == "mrbench":
        env = build_mrbench_environment(case)
    else:
        from assembly_loader import build

        env, _, _ = build(case, ROOT)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    fingerprint = environment_fingerprint(env)
    if fingerprint != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from frozen input")

    if case["kind"] == "mrbench":
        path = restore_mrbench_path(env, source["states"])
    else:
        from assembly_loader import restore_records

        path = restore_records(env, source["states"])

    from multi_robot_multi_goal_planning.problems.util import path_cost

    restored_cost = float(path_cost(path, env.batch_config_cost))
    tolerance = 1e-9 * max(1.0, abs(float(source["initial_cost"])))
    if abs(restored_cost - float(source["initial_cost"])) > tolerance:
        raise RuntimeError(
            f"restored input cost changed: {restored_cost} != {source['initial_cost']}"
        )

    policy = method_map[args.method]["offline_policy"]
    started = time.perf_counter()
    output_path, result = run(
        env,
        path,
        policy,
        args.seconds,
        args.shortcut_seed,
        interpolation_resolution=float(data["offline"]["interpolation_resolution"]),
    )
    result["total_wall_seconds"] = time.perf_counter() - started
    start_preserved = bool(np.allclose(
        output_path[0].q.state(), path[0].q.state(), rtol=0.0, atol=1e-8
    ))
    end_preserved = bool(np.allclose(
        output_path[-1].q.state(), path[-1].q.state(), rtol=0.0, atol=1e-8
    ))
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
    input_modes = mode_runs(path)
    output_modes = mode_runs(output_path)
    terminal = bool(env.is_terminal_mode(output_path[-1].mode))
    certified = (
        input_modes == output_modes
        and start_preserved
        and end_preserved
        and terminal
        and strict
        and valid is not False
    )
    if not certified:
        raise RuntimeError("offline output certificate failed")

    result.update({
        "schema": "mr-shortcut-final-offline-result-v1",
        "final_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "kind": case["kind"],
        "method_id": args.method,
        "implementation_policy": policy,
        "path_index": args.path_index,
        "shortcut_seed": args.shortcut_seed,
        "cost": cost_record(),
        "source_path": str(source_path.relative_to(REPO)),
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
        f"[ok] offline {args.case} {args.method} path{args.path_index} "
        f"seed{args.shortcut_seed}: {result['initial_cost']:.9g} -> "
        f"{result['final_cost']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
