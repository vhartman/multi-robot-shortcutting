"""Run and certify one exploratory policy on one frozen expanded input."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
SCALING = BENCHMARK / "scaling_2d"
ROOT_REPO = BENCHMARK.parent
for directory in (BENCHMARK, HERE, SCALING, ROOT_REPO / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import ROOT, environment_fingerprint, mode_runs, restore_mrbench_path  # noqa: E402
from expanded_lib import (  # noqa: E402
    assembly_case,
    build_mrbench_environment,
    cases,
    cost_record,
    protocol,
    protocol_sha256,
    read_json,
    robot_dofs,
    sha256,
    write_json,
)
import scalable_tree_group  # noqa: E402,F401
import max_gain_ablation  # noqa: E402,F401
from mr_shortcutting import run  # noqa: E402


POLICIES = ("scalable_tree_group", "tree_robot_local_gain")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(cases()))
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument("--path-index", type=int, default=0)
    parser.add_argument("--shortcut-seed", type=int, default=0)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    case = cases()[args.case]
    source_path = args.input.resolve()
    source = read_json(source_path)
    expected = {
        "schema": "mr-shortcut-expanded-input-v1",
        "expanded_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "path_index": args.path_index,
        "cost": cost_record(),
        "pythonhashseed": int(case["pythonhashseed"]),
    }
    for key, wanted in expected.items():
        if source.get(key) != wanted:
            raise RuntimeError(
                f"input metadata mismatch for {key}: "
                f"{source.get(key)!r} != {wanted!r}"
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
        raise RuntimeError(
            "environment fingerprint differs from expanded input: "
            f"{fingerprint} != {source['environment_fingerprint']}"
        )

    if case["kind"] == "mrbench":
        path = restore_mrbench_path(env, source["states"])
    else:
        from assembly_loader import restore_records

        path = restore_records(env, source["states"])

    from multi_robot_multi_goal_planning.problems.util import path_cost

    restored_cost = float(path_cost(path, env.batch_config_cost))
    tolerance = 1e-9 * max(1.0, abs(float(source["initial_cost"])))
    if abs(restored_cost - float(source["initial_cost"])) > tolerance:
        raise RuntimeError("restored input cost changed")

    started = time.perf_counter()
    output_path, result = run(
        env,
        path,
        args.policy,
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
        if hasattr(env, "is_valid_plan") else None
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
            "trial output certificate failed: "
            f"modes={input_modes == output_modes}, start={start_preserved}, "
            f"end={end_preserved}, strict={strict}, valid={valid}, "
            f"terminal={terminal}"
        )

    dofs = robot_dofs(env)
    trial_source = (
        SCALING / "scalable_tree_group.py"
        if args.policy == "scalable_tree_group"
        else HERE / "max_gain_ablation.py"
    )
    result.update({
        "schema": "mr-shortcut-expanded-trial-v1",
        "expanded_protocol_sha256": protocol_sha256(),
        "trial_implementation_sha256": sha256(trial_source),
        "case_id": args.case,
        "track": case["track"],
        "kind": case["kind"],
        "method_id": args.policy,
        "implementation_policy": args.policy,
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
        f"[ok] {args.case} {args.policy}: "
        f"{result['initial_cost']:.9g} -> {result['final_cost']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
