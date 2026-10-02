"""Generate and certify one isolated-box-stacking first-solution path."""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import mpl_toolkits

_user_toolkit = Path.home() / ".local/lib/python3.10/site-packages/mpl_toolkits"
if _user_toolkit.is_dir() and str(_user_toolkit) not in mpl_toolkits.__path__:
    mpl_toolkits.__path__.insert(0, str(_user_toolkit))


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import environment_fingerprint  # noqa: E402
from expanded_lib import (  # noqa: E402
    build_mrbench_environment,
    cases,
    cost_record,
    git_revision,
    protocol,
    protocol_sha256,
    robot_dofs,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(cases()))
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    case = cases()[args.case]
    if case["input"] != "generate":
        raise ValueError(f"{args.case} uses an adopted input")
    if args.path_index not in case["path_indices"]:
        raise ValueError("path index is outside the case protocol")
    if args.seconds != float(case["input_generation_seconds"]):
        raise ValueError("input-generation budget differs from protocol")
    expected_hashseed = str(case["pythonhashseed"])
    if os.environ.get("PYTHONHASHSEED") != expected_hashseed:
        raise RuntimeError(
            f"PYTHONHASHSEED must be {expected_hashseed} before Python starts"
        )

    env = build_mrbench_environment(case)
    fingerprint = environment_fingerprint(env)
    planner_seed = int(case["environment_seed"]) + int(args.path_index)
    np.random.seed(planner_seed)
    random.seed(planner_seed)

    from multi_robot_multi_goal_planning.planners.planner_birrtstar import (
        BidirectionalRRTstar,
    )
    from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
    from multi_robot_multi_goal_planning.planners.termination_conditions import (
        RuntimeTerminationCondition,
    )
    from multi_robot_multi_goal_planning.problems.util import path_cost

    config = BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.stepsize = 1.0
    config.shortcutting = False
    config.with_mode_validation = False
    planner = BidirectionalRRTstar(env, config=config)
    started = time.perf_counter()
    path, _ = planner.plan(
        ptc=RuntimeTerminationCondition(args.seconds),
        optimize=False,
    )
    elapsed = time.perf_counter() - started
    if not path:
        raise RuntimeError(f"{args.case}: no first solution after {elapsed:.3f}s")

    cert = certificate(env, path)
    dofs = robot_dofs(env)
    payload = {
        "schema": "mr-shortcut-expanded-input-v1",
        "expanded_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "track": case["track"],
        "kind": case["kind"],
        "path_index": args.path_index,
        "constructor": case["constructor"],
        "environment_kwargs": case["environment_kwargs"],
        "environment_seed": int(case["environment_seed"]),
        "planner_seed": planner_seed,
        "pythonhashseed": int(case["pythonhashseed"]),
        "planning_seconds_limit": float(args.seconds),
        "planning_wall_seconds": float(elapsed),
        "planner": "BidirectionalRRTstar",
        "optimize": False,
        "shortcutting": False,
        "cost": cost_record(),
        "environment_fingerprint": fingerprint,
        "mr_bench_revision": git_revision(args.mr_bench_repo.resolve()),
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
    }
    write_json(args.output.resolve(), payload)
    print(
        f"[ok] {args.case} path{args.path_index}: seed={planner_seed}, "
        f"states={len(path)}, cost={payload['initial_cost']:.9g}, "
        f"first={elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
