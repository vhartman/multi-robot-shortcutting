"""Restore, shortcut, and certify one expanded offline cell."""
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    ROOT,
    environment_fingerprint,
    mode_runs,
    restore_mrbench_path,
)
from expanded_lib import (  # noqa: E402
    assembly_case,
    build_mrbench_environment,
    cases,
    cost_record,
    implementation_sha256,
    methods_for,
    protocol,
    protocol_sha256,
    read_json,
    robot_dofs,
    sha256,
    write_json,
)
from mr_shortcutting import load_shortcutter_plugins, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(cases()))
    parser.add_argument("--method", required=True)
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--shortcut-seed", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = protocol()
    load_shortcutter_plugins(data.get("policy_modules", []))
    case = cases()[args.case]
    method_map = methods_for(case)
    if args.method not in method_map:
        raise ValueError(f"{args.method} is not enabled for {case['track']}")
    if args.path_index not in case["path_indices"]:
        raise ValueError("path index is outside the case protocol")
    if args.shortcut_seed not in data["shortcut_seeds"]:
        raise ValueError("shortcut seed is outside the protocol")
    if args.seconds != float(case["seconds"]):
        raise ValueError("shortcut budget differs from the protocol")

    source_path = args.input.resolve()
    source = read_json(source_path)
    expected_source = {
        "schema": "mr-shortcut-expanded-input-v1",
        "expanded_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "path_index": args.path_index,
        "cost": cost_record(),
        "pythonhashseed": int(case["pythonhashseed"]),
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise RuntimeError(
                f"input metadata mismatch for {key}: "
                f"{source.get(key)!r} != {expected!r}"
            )

    if case["kind"] == "mrbench":
        env = build_mrbench_environment(case)
    else:
        from assembly_loader import build

        env, _, _ = build(assembly_case(case), ROOT)
        env.cost_reduction = "max"
        env.cost_metric = "euclidean"
    fingerprint = environment_fingerprint(env)
    if fingerprint != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from expanded input")

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
            f"restored input cost changed: {restored_cost} != "
            f"{source['initial_cost']}"
        )

    method = method_map[args.method]
    started = time.perf_counter()
    output_path, result = run(
        env,
        path,
        method["policy"],
        args.seconds,
        args.shortcut_seed,
        interpolation_resolution=0.5,
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
        if hasattr(env, "is_valid_plan")
        else None
    )
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
            "expanded output certificate failed: "
            f"modes={input_modes == output_modes}, start={start_preserved}, "
            f"end={end_preserved}, strict={strict}, valid={valid}, "
            f"terminal={terminal}"
        )

    dofs = robot_dofs(env)
    result.update({
        "schema": "mr-shortcut-expanded-offline-result-v1",
        "expanded_protocol_sha256": protocol_sha256(),
        "implementation_sha256": implementation_sha256(),
        "case_id": args.case,
        "track": case["track"],
        "kind": case["kind"],
        "method_id": args.method,
        "implementation_policy": method["policy"],
        "path_index": args.path_index,
        "shortcut_seed": args.shortcut_seed,
        "cost": cost_record(),
        "source_sha256": sha256(source_path),
        "environment_fingerprint": fingerprint,
        "robot_count": len(env.robots),
        "robot_dofs": dofs,
        "total_dof": sum(dofs),
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
        f"[ok] expanded {args.case} {args.method} "
        f"path{args.path_index} seed{args.shortcut_seed}: "
        f"{result['initial_cost']:.9g} -> {result['final_cost']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
