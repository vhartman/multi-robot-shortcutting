"""Generate one certified scaling input; invoked by prepare_inputs.py.

This fresh-checkout worker does not change or re-certify archived input paths.
It uses the published scene definitions and records its own generator hash.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

import run as pilot
from common import environment_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=tuple(pilot.cases()))
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--planner-seed", required=True, type=int)
    parser.add_argument("--seconds", type=float, default=200)
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.path_index not in pilot.path_indices("offline") or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("invalid path index or non-positive planning cap")
    if args.output.exists():
        parser.error("output already exists; inspect it before continuing")
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("PYTHONHASHSEED=0 must be set before Python starts")
    if pilot.scaling.git_revision(args.mr_bench_repo) != pilot.scaling.protocol()["software"]["mr_bench_revision"]:
        parser.error("mr_bench revision differs from the scaling protocol")
    if ".".join(map(str, sys.version_info[:3])) != pilot.scaling.protocol()["software"]["python"]:
        parser.error("Python version differs from the scaling protocol")

    from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
    from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
    from multi_robot_multi_goal_planning.planners.termination_conditions import RuntimeTerminationCondition
    from multi_robot_multi_goal_planning.problems.util import path_cost

    case = pilot.cases()[args.case]
    env = pilot.scaling.build_environment(case)
    fingerprint = environment_fingerprint(env)
    np.random.seed(args.planner_seed)
    random.seed(args.planner_seed)
    config = BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.stepsize = 0.5
    config.p_home_bias = 0.1
    config.shortcutting = False
    config.with_mode_validation = False
    config.sampler = "gibbs" if case["family"] == "disk_scaling" else "auto"
    planner = BidirectionalRRTstar(env, config=config)
    sampling = {"name": config.sampler}
    if case["family"] == "disk_scaling":
        sys.path.insert(0, str(pilot.BENCHMARK / "expanded"))
        from input_sampling import RandomScanGibbsSampler
        planner.collision_free_sampler = RandomScanGibbsSampler(
            env, planner.collision_free_sampler.config, planner._get_seed_for_mode
        )
        sampling = {"name": "gibbs_random_scan", "implementation_sha256": pilot.sha256(
            pilot.BENCHMARK / "expanded" / "input_sampling.py"
        )}
    started = time.perf_counter()
    path, _ = planner.plan(ptc=RuntimeTerminationCondition(args.seconds), optimize=False)
    elapsed = time.perf_counter() - started
    if not path:
        raise RuntimeError(f"no first solution within {args.seconds:g}s")
    cert = {
        "strict_full_edge": bool(env.is_path_collision_free(
            path, resolution=float(env.collision_resolution),
            tolerance=float(env.collision_tolerance), check_start_and_end=True,
            check_edges_in_order=True, binary_order=False,
        )),
        "terminal_mode": bool(env.is_terminal_mode(path[-1].mode)),
        "valid_plan": bool(env.is_valid_plan(path)) if hasattr(env, "is_valid_plan") else None,
    }
    if not cert["strict_full_edge"] or not cert["terminal_mode"] or cert["valid_plan"] is False:
        raise RuntimeError(f"input certification failed: {cert}")
    dofs = pilot.scaling.robot_dofs(env)
    value = {
        "schema": "mr-shortcut-scalable-study-input-v1",
        "case_id": args.case, "family": case["family"], "kind": "mrbench",
        "path_index": args.path_index, "environment_seed": int(case["environment_seed"]),
        "planner_seed": args.planner_seed, "pythonhashseed": 0,
        "planning_seconds_limit": args.seconds, "planning_wall_seconds": elapsed,
        "planner": "BidirectionalRRTstar", "planner_config": asdict(config),
        "sampling_strategy": sampling, "optimize": False, "shortcutting": False,
        "input_generator_sha256": pilot.sha256(Path(__file__)),
        "mr_bench_revision": pilot.scaling.git_revision(args.mr_bench_repo),
        "cost": pilot.scaling.protocol()["cost"], "environment_fingerprint": fingerprint,
        "initial_cost": float(path_cost(path, env.batch_config_cost)),
        "robot_dofs": dofs, "total_dof": sum(dofs), "state_count": len(path),
        "mode_count": 1 + sum(a.mode != b.mode for a, b in zip(path, path[1:])),
        "certificate": cert, "states": [state.to_dict() for state in path],
    }
    if hasattr(env, "scene_metadata"):
        value["scene"] = env.scene_metadata
    pilot.write_json(args.output, value)
    print(f"[certified] {args.case} path{args.path_index}: cost={value['initial_cost']:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
