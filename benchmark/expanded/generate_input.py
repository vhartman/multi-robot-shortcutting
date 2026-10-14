"""Generate and certify one first-solution benchmark input path."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import mpl_toolkits

_user_toolkit = Path.home() / ".local/lib/python3.10/site-packages/mpl_toolkits"
if _user_toolkit.is_dir() and str(_user_toolkit) not in mpl_toolkits.__path__:
    # ``mpl_toolkits`` is commonly a namespace package.  Its ``_NamespacePath``
    # is iterable but does not implement list.insert() on all supported Python
    # installations, so replace it with an explicitly ordered search path.
    mpl_toolkits.__path__ = [str(_user_toolkit), *list(mpl_toolkits.__path__)]


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
    sha256,
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
    parser.add_argument("--stepsize", type=float, default=1.0)
    parser.add_argument("--home-bias", type=float, default=0.0)
    parser.add_argument("--sampler", choices=("auto", "joint", "per_robot", "gibbs", "gibbs_random_scan"), default="auto")
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    case = cases()[args.case]
    if case["input"] != "generate":
        raise ValueError(f"{args.case} uses an adopted input")
    if args.path_index not in case["path_indices"]:
        raise ValueError("path index is outside the case protocol")
    if args.seconds < float(case["input_generation_seconds"]):
        raise ValueError("input-generation budget is below the protocol minimum")
    if args.stepsize <= 0 or not 0 <= args.home_bias <= 1:
        raise ValueError("invalid input-planner configuration")
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
    config.stepsize = args.stepsize
    config.p_home_bias = args.home_bias
    # The native dispatcher routes Gibbs variants through collision_free_sampler.
    config.sampler = "gibbs" if args.sampler == "gibbs_random_scan" else args.sampler
    config.shortcutting = False
    config.with_mode_validation = False
    planner = BidirectionalRRTstar(env, config=config)
    sampling_strategy = {"name": args.sampler}
    if args.sampler == "gibbs_random_scan":
        from input_sampling import RandomScanGibbsSampler

        planner.collision_free_sampler = RandomScanGibbsSampler(
            env, planner.collision_free_sampler.config, planner._get_seed_for_mode,
        )
        sampling_strategy["implementation_sha256"] = sha256(HERE / "input_sampling.py")
    planner._sample_config_fn = planner._timed_sample_configuration
    planner._sample_transition_config_fn = planner._timed_sample_transition_configuration
    planner._check_edge_cf = planner._timed_edge_collision_free
    planner._check_config_cf = planner._timed_collision_free
    started = time.perf_counter()

    def planning_timing():
        return {
            "sampling_seconds": planner._sampling_time,
            "configuration_check_seconds": planner._coll_checking_time,
            "valid_edge_check_seconds": planner._edge_check_time_success,
            "invalid_edge_check_seconds": planner._edge_check_time_failure,
        }

    class PlanningDeadline(RuntimeTerminationCondition):
        last_report = 0.0

        def should_terminate(self, current_iterations=None, current_time=None):
            elapsed = time.perf_counter() - started
            if elapsed - self.last_report >= 30:
                print(f"[progress] seconds={elapsed:.1f} modes={len(planner.modes)} "
                      f"timing={planning_timing()}", flush=True)
                self.last_report = elapsed
            return elapsed >= self.max_runtime_in_s

    path, _ = planner.plan(
        ptc=PlanningDeadline(args.seconds),
        optimize=False,
    )
    elapsed = time.perf_counter() - started
    write_json(args.output.with_suffix(".diagnostics.json"), {
        "case_id": args.case,
        "path_index": args.path_index,
        "planner_config": asdict(config),
        "sampling_strategy": sampling_strategy,
        "planning_seconds_limit": args.seconds,
        "planning_wall_seconds": elapsed,
        "planning_timing": planning_timing(),
        "modes_reached": len(planner.modes),
        "found_path": bool(path),
    })
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
        "planner_config": asdict(config),
        "sampling_strategy": sampling_strategy,
        "input_generator_sha256": sha256(Path(__file__)),
        "planning_timing": planning_timing(),
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
    if hasattr(env, "scene_metadata"):
        payload["scene"] = env.scene_metadata
    write_json(args.output.resolve(), payload)
    print(
        f"[ok] {args.case} path{args.path_index}: seed={planner_seed}, "
        f"states={len(path)}, cost={payload['initial_cost']:.9g}, "
        f"first={elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
