"""Generate and certify one unshortcutted first-solution input path."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (BENCHMARK, BENCHMARK / "expanded", REPO / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import ROOT, environment_fingerprint  # noqa: E402
from study_lib import (  # noqa: E402
    assembly_definition,
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


def generate_mrbench(case, seed, settings):
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
    config.stepsize = float(settings["stepsize"])
    config.p_home_bias = float(settings["home_bias"])
    sampler = settings["sampler"]
    config.sampler = "gibbs" if sampler == "gibbs_random_scan" else sampler
    config.shortcutting = False
    config.with_mode_validation = False
    planner = BidirectionalRRTstar(env, config=config)
    sampling_strategy = {"name": sampler}
    if sampler == "gibbs_random_scan":
        from input_sampling import RandomScanGibbsSampler

        planner.collision_free_sampler = RandomScanGibbsSampler(
            env,
            planner.collision_free_sampler.config,
            planner._get_seed_for_mode,
        )
        sampling_strategy["implementation_sha256"] = sha256(
            BENCHMARK / "expanded" / "input_sampling.py"
        )
    planner._sample_config_fn = planner._timed_sample_configuration
    planner._sample_transition_config_fn = planner._timed_sample_transition_configuration
    planner._check_edge_cf = planner._timed_edge_collision_free
    planner._check_config_cf = planner._timed_collision_free
    started = time.perf_counter()
    path, _ = planner.plan(
        ptc=RuntimeTerminationCondition(float(settings["seconds"])),
        optimize=False,
    )
    elapsed = time.perf_counter() - started
    timing = {
        "sampling_seconds": planner._sampling_time,
        "configuration_check_seconds": planner._coll_checking_time,
        "valid_edge_check_seconds": planner._edge_check_time_success,
        "invalid_edge_check_seconds": planner._edge_check_time_failure,
    }
    return env, path, elapsed, fingerprint, asdict(config), sampling_strategy, timing


def generate_assembly(case, seed, settings):
    from assembly_loader import build
    import mp_export

    definition = assembly_definition(case)
    env, _, replay = build(definition, ROOT)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    fingerprint = environment_fingerprint(env)
    started = time.perf_counter()
    path, info = mp_export.run_birrt(
        env,
        runtime=float(settings["seconds"]),
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
    extra = {
        "assembly_revision": git_revision(Path(replay["assembly_checkout"])),
        "task_signature": replay["task_signature"],
    }
    return env, path, elapsed, fingerprint, {}, {"name": "assembly_native"}, extra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(cases()))
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--planner-seed", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--sampler", choices=("auto", "gibbs_random_scan"))
    parser.add_argument("--home-bias", type=float)
    args = parser.parse_args()

    data = protocol()
    case = cases()[args.case]
    if args.path_index not in data["path_indices"]:
        raise ValueError("path index is outside the study protocol")
    if os.environ.get("PYTHONHASHSEED") != str(data["pythonhashseed"]):
        raise RuntimeError("PYTHONHASHSEED must be fixed before Python starts")

    seed = (
        int(args.planner_seed)
        if args.planner_seed is not None
        else int(case["environment_seed"]) + args.path_index
    )
    settings = dict(case["input_generation"])
    if args.seconds is not None:
        settings["seconds"] = float(args.seconds)
    if args.sampler is not None:
        settings["sampler"] = args.sampler
    if args.home_bias is not None:
        settings["home_bias"] = float(args.home_bias)
    if case["kind"] == "mrbench":
        env, path, elapsed, fingerprint, planner_config, sampling, timing = (
            generate_mrbench(case, seed, settings)
        )
        extra = {"mr_bench_revision": git_revision(args.mr_bench_repo.resolve())}
    elif case["kind"] == "assembly":
        env, path, elapsed, fingerprint, planner_config, sampling, extra = (
            generate_assembly(case, seed, settings)
        )
        timing = {}
    else:
        raise ValueError(f"unsupported case kind: {case['kind']}")
    if not path:
        raise RuntimeError(f"{args.case}: no first solution after {elapsed:.3f}s")

    from multi_robot_multi_goal_planning.problems.util import path_cost

    cert = certificate(env, path)
    dofs = robot_dofs(env)
    payload = {
        "schema": "mr-shortcut-scalable-study-input-v1",
        "study_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "family": case["family"],
        "kind": case["kind"],
        "path_index": args.path_index,
        "environment_seed": int(case["environment_seed"]),
        "planner_seed": seed,
        "pythonhashseed": int(data["pythonhashseed"]),
        "planning_seconds_limit": float(settings["seconds"]),
        "planning_wall_seconds": float(elapsed),
        "planner": "BidirectionalRRTstar",
        "planner_config": planner_config,
        "sampling_strategy": sampling,
        "planning_timing": timing,
        "input_generator_sha256": sha256(Path(__file__)),
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
    if hasattr(env, "scene_metadata"):
        payload["scene"] = env.scene_metadata
    write_json(args.output.resolve(), payload)
    print(
        f"[ok] {args.case} path{args.path_index}: seed={seed}, "
        f"states={len(path)}, cost={payload['initial_cost']:.9g}, "
        f"first={elapsed:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
