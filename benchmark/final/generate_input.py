"""Generate and certify one deterministic unshortcutted first-solution path."""
from __future__ import annotations

import argparse
import os
import random
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
    build_mrbench_environment,
    environment_fingerprint,
)
from protocol_lib import (  # noqa: E402
    cost_record,
    git_revision,
    manifest_cases,
    mr_bench_checkout,
    planner_seed,
    protocol,
    protocol_cases,
    protocol_sha256,
    write_json,
)


def certificate(env, path) -> dict:
    strict = bool(env.is_path_collision_free(
        path,
        resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance),
        check_start_and_end=True,
        check_edges_in_order=True,
        binary_order=False,
    ))
    terminal = bool(env.is_terminal_mode(path[-1].mode))
    valid = bool(env.is_valid_plan(path)) if hasattr(env, "is_valid_plan") else None
    result = {
        "strict_full_edge": strict,
        "terminal_mode": terminal,
        "valid_plan": valid,
    }
    if not strict or not terminal or valid is False:
        raise RuntimeError(f"input certificate failed: {result}")
    return result


def generate_mrbench(case: dict, seed: int, seconds: float):
    from multi_robot_multi_goal_planning.planners.planner_birrtstar import (
        BidirectionalRRTstar,
    )
    from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
    from multi_robot_multi_goal_planning.planners.termination_conditions import (
        RuntimeTerminationCondition,
    )

    env = build_mrbench_environment(case)
    fingerprint = environment_fingerprint(env)
    np.random.seed(seed)
    random.seed(seed)
    config = BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.stepsize = 1.0
    config.shortcutting = False
    config.with_mode_validation = False
    planner = BidirectionalRRTstar(env, config=config)
    started = time.perf_counter()
    path, _ = planner.plan(
        ptc=RuntimeTerminationCondition(seconds),
        optimize=False,
    )
    elapsed = time.perf_counter() - started
    return env, path, elapsed, fingerprint, {
        "environment": case["environment"],
        "mr_bench_revision": git_revision(mr_bench_checkout()),
    }


def generate_assembly(case: dict, seed: int, seconds: float):
    from assembly_loader import build
    import mp_export

    env, _, replay = build(case, ROOT)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    fingerprint = environment_fingerprint(env)
    started = time.perf_counter()
    path, info = mp_export.run_birrt(
        env,
        runtime=seconds,
        shortcut=False,
        optimize=False,
        seed=seed,
        verbose=False,
    )
    elapsed = time.perf_counter() - started
    if not path:
        frontier = dict(info or {}).get("frontier_task_ids")
        raise RuntimeError(
            f"{case['id']}: no path after {elapsed:.3f}s; frontier={frontier}"
        )
    checkout = Path(replay["assembly_checkout"])
    return env, path, elapsed, fingerprint, {
        "assembly_revision": git_revision(checkout),
        "task_signature": replay["task_signature"],
    }


def robot_dofs(env) -> list[int]:
    dims = env.robot_dims
    if isinstance(dims, dict):
        return [int(dims[robot]) for robot in env.robots]
    return [int(value) for value in dims]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(protocol_cases()))
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = protocol()
    if args.path_index not in data["design"]["path_indices"]:
        raise ValueError(f"path index is outside frozen protocol: {args.path_index}")
    if os.environ.get("PYTHONHASHSEED") != str(data["rng"]["pythonhashseed"]):
        raise RuntimeError("PYTHONHASHSEED must be 0 before Python starts")

    protocol_case = protocol_cases()[args.case]
    expected_seconds = float(protocol_case["input_generation_seconds"])
    if args.seconds != expected_seconds:
        raise ValueError(
            f"generation limit differs from protocol: {args.seconds} != {expected_seconds}"
        )
    case = manifest_cases()[args.case]
    if not case.get("enabled", True):
        raise RuntimeError(f"disabled manifest case: {args.case}")

    seed = planner_seed(args.case, args.path_index)
    if case["kind"] == "mrbench":
        env, path, elapsed, fingerprint, extra = generate_mrbench(
            case, seed, args.seconds
        )
    elif case["kind"] == "assembly":
        env, path, elapsed, fingerprint, extra = generate_assembly(
            case, seed, args.seconds
        )
    else:
        raise ValueError(f"unsupported case kind: {case['kind']}")
    if not path:
        raise RuntimeError(f"{args.case}: no first solution after {elapsed:.3f}s")

    from multi_robot_multi_goal_planning.problems.util import path_cost

    cert = certificate(env, path)
    dofs = robot_dofs(env)
    payload = {
        "schema": "mr-shortcut-final-input-v1",
        "final_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "kind": case["kind"],
        "path_index": args.path_index,
        "environment_seed": int(case["environment_seed"]),
        "planner_seed": seed,
        "pythonhashseed": int(data["rng"]["pythonhashseed"]),
        "planning_seconds_limit": float(args.seconds),
        "planning_wall_seconds": float(elapsed),
        "planner": "BidirectionalRRTstar",
        "optimize": False,
        "shortcutting": False,
        "cost": cost_record(),
        "environment_fingerprint": fingerprint,
        "initial_cost": float(path_cost(path, env.batch_config_cost)),
        "robot_dofs": dofs,
        "total_dof": sum(dofs),
        "state_count": len(path),
        "mode_count": 1 + sum(
            path[index].mode != path[index - 1].mode
            for index in range(1, len(path))
        ),
        "certificate": cert,
        "states": [state.to_dict() for state in path],
        **extra,
    }
    write_json(args.output.resolve(), payload)
    print(
        f"[ok] {args.case} path{args.path_index}: seed={seed}, "
        f"states={len(path)}, cost={payload['initial_cost']:.9g}, "
        f"first={elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
