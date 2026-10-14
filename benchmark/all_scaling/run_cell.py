"""Restore, shortcut, and certify one all-scaling cell."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (REPO / "src", BENCHMARK, BENCHMARK / "scaling_2d", HERE):
    sys.path.insert(0, str(directory))

from common import environment_fingerprint, mode_runs, restore_mrbench_path  # noqa: E402
from study_lib import (  # noqa: E402
    build_environment,
    cases,
    input_path,
    methods,
    protocol,
    protocol_sha256,
    read_json,
    robot_dofs,
    sha256,
    write_json,
)
import scalable_tree_group  # noqa: E402,F401
from mr_shortcutting import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(cases()))
    parser.add_argument("--method", required=True, choices=tuple(methods()))
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--shortcut-seed", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = protocol()
    case = cases()[args.case]
    method = methods()[args.method]
    source_path = input_path(args.case, args.path_index).resolve()
    source = read_json(source_path)
    if args.path_index not in data["path_indices"]:
        raise ValueError("path index outside protocol")
    if args.shortcut_seed not in data["shortcut_seeds"]:
        raise ValueError("shortcut seed outside protocol")
    if args.seconds != float(case["seconds"]):
        raise ValueError("shortcut budget differs from protocol")

    env = build_environment(case)
    fingerprint = environment_fingerprint(env)
    if fingerprint != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from input")
    path = restore_mrbench_path(env, source["states"])
    from multi_robot_multi_goal_planning.problems.util import path_cost

    restored_cost = float(path_cost(path, env.batch_config_cost))
    tolerance = 1e-9 * max(1.0, abs(float(source["initial_cost"])))
    if abs(restored_cost - float(source["initial_cost"])) > tolerance:
        raise RuntimeError("restored input cost changed")

    started = time.perf_counter()
    output_path, result = run(
        env,
        path,
        method["policy"],
        args.seconds,
        args.shortcut_seed,
        interpolation_resolution=float(data["offline"]["interpolation_resolution"]),
    )
    result["total_wall_seconds"] = time.perf_counter() - started
    start_preserved = bool(np.allclose(
        output_path[0].q.state(), path[0].q.state(), rtol=0, atol=1e-8
    ))
    end_preserved = bool(np.allclose(
        output_path[-1].q.state(), path[-1].q.state(), rtol=0, atol=1e-8
    ))
    strict = bool(env.is_path_collision_free(
        output_path,
        resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance),
        check_start_and_end=True,
        check_edges_in_order=True,
        binary_order=False,
    ))
    valid = bool(env.is_valid_plan(output_path)) if hasattr(env, "is_valid_plan") else None
    terminal = bool(env.is_terminal_mode(output_path[-1].mode))
    before, after = mode_runs(path), mode_runs(output_path)
    certificate_passed = bool(
        start_preserved and end_preserved and strict and terminal
        and valid is not False and before == after
    )
    dofs = robot_dofs(env)
    result.update({
        "schema": "mr-shortcut-all-scaling-result-v1",
        "status": "ok" if certificate_passed else "failed_certificate",
        "all_scaling_protocol_sha256": protocol_sha256(),
        "execution_manifest_sha256": args.manifest_sha256,
        "case_id": args.case,
        "family": case["family"],
        "method_id": args.method,
        "implementation_policy": method["policy"],
        "path_index": args.path_index,
        "shortcut_seed": args.shortcut_seed,
        "cost": dict(data["cost"]),
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "environment_fingerprint": fingerprint,
        "robot_count": len(env.robots),
        "robot_dofs": dofs,
        "total_dof": sum(dofs),
        "input_mode_runs": before,
        "output_mode_runs": after,
        "endpoint_start_preserved": start_preserved,
        "endpoint_end_preserved": end_preserved,
        "terminal_mode_preserved": terminal,
        "strict_collision_free": strict,
        "valid_plan": valid,
    })
    write_json(args.output.resolve(), result)
    if not certificate_passed:
        raise RuntimeError(
            "output certificate failed: "
            f"modes={before == after}, start={start_preserved}, "
            f"end={end_preserved}, strict={strict}, valid={valid}, "
            f"terminal={terminal}"
        )
    print(
        f"[ok] {args.case} {args.method} path{args.path_index} "
        f"seed{args.shortcut_seed}: {result['initial_cost']:.9g} -> "
        f"{result['final_cost']:.9g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
