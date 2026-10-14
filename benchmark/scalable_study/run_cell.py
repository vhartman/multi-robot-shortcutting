"""Restore, shortcut, and certify one scalable-study cell."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (BENCHMARK, HERE, BENCHMARK / "scaling_2d", REPO / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import ROOT, environment_fingerprint, mode_runs, restore_mrbench_path  # noqa: E402
from study_lib import (  # noqa: E402
    assembly_definition,
    build_mrbench_environment,
    cases,
    cost_record,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(cases()))
    parser.add_argument("--method", required=True, choices=tuple(methods()))
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--shortcut-seed", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = protocol()
    case = cases()[args.case]
    method = methods()[args.method]
    if args.path_index not in data["path_indices"]:
        raise ValueError("path index is outside the study protocol")
    if args.shortcut_seed not in data["shortcut_seeds"]:
        raise ValueError("shortcut seed is outside the study protocol")
    if args.seconds != float(case["seconds"]):
        raise ValueError("shortcut budget differs from the study protocol")

    source_path = args.input.resolve()
    source = read_json(source_path)
    if source.get("case_id") != args.case or source.get("path_index") != args.path_index:
        raise RuntimeError("input case/path metadata mismatch")
    if not source.get("states"):
        raise RuntimeError("input path is empty")

    if case["kind"] == "mrbench":
        env = build_mrbench_environment(case)
        fingerprint = environment_fingerprint(env)
        path = restore_mrbench_path(env, source["states"])
    else:
        from assembly_loader import build, restore_records

        env, _, _ = build(assembly_definition(case), ROOT)
        env.cost_reduction = "max"
        env.cost_metric = "euclidean"
        fingerprint = environment_fingerprint(env)
        path = restore_records(env, source["states"])

    if fingerprint != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from input")

    from multi_robot_multi_goal_planning.problems.util import path_cost

    restored_cost = float(path_cost(path, env.batch_config_cost))
    tolerance = 1e-9 * max(1.0, abs(float(source["initial_cost"])))
    if abs(restored_cost - float(source["initial_cost"])) > tolerance:
        raise RuntimeError("restored input cost changed")

    input_certificate = source.get("certificate") or {}
    if (
        input_certificate.get("strict_full_edge") is not True
        or input_certificate.get("terminal_mode") is not True
        or input_certificate.get("valid_plan") is False
    ):
        raise RuntimeError(f"input does not carry a strict certificate: {input_certificate}")

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
            "output certificate failed: "
            f"modes={input_modes == output_modes}, start={start_preserved}, "
            f"end={end_preserved}, strict={strict}, valid={valid}, terminal={terminal}"
        )

    dofs = robot_dofs(env)
    result.update({
        "schema": "mr-shortcut-scalable-study-result-v1",
        "study_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "family": case["family"],
        "kind": case["kind"],
        "method_id": args.method,
        "implementation_policy": method["policy"],
        "path_index": args.path_index,
        "shortcut_seed": args.shortcut_seed,
        "cost": cost_record(),
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "environment_fingerprint": fingerprint,
        "input_certificate": input_certificate,
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
        f"[ok] {args.case} {args.method} path{args.path_index} "
        f"seed{args.shortcut_seed}: "
        f"{result['initial_cost']:.9g} -> {result['final_cost']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
