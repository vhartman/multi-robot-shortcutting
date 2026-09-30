"""Run BiRRT* and a selected shortcutter in one shared anytime loop.

Every incumbent shortcut call uses an explicit collision-check budget. The
default preserves the upstream 100-check contract.  Shortcut randomness is private, so changing policy does not advance
the planner's NumPy or stdlib random streams.  There is no uncharged final shortcut
pass: planning, shortcutting, and reinsertion all consume the same wall-clock budget.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

# The host has a user matplotlib paired with a distro mpl_toolkits.  Put the
# matching user toolkit first before importing mr_bench's planner package.

from common import (
    ROOT,
    build_mrbench_environment,
    case_by_id,
    environment_fingerprint,
    git_revision,
    write_json,
)
from mr_shortcutting import available_shortcutters, run_checks

from multi_robot_multi_goal_planning.planners import shortcutting as shortcut_module
from multi_robot_multi_goal_planning.planners.planner_birrtstar import (
    BidirectionalRRTstar,
)
from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
from multi_robot_multi_goal_planning.planners.termination_conditions import (
    RuntimeTerminationCondition,
)
from multi_robot_multi_goal_planning.problems.util import path_cost


POLICIES = ("planner_only", "upstream_tree", *available_shortcutters())

PROTOCOL = "mr-shortcut-planner-loop-v2-exact-standard"


def _scalar(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError(f"expected scalar cost, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def _mode_runs(path) -> list[list[int]]:
    runs = []
    for state in path:
        signature = [int(value) for value in np.atleast_1d(state.mode.task_ids)]
        if not runs or signature != runs[-1]:
            runs.append(signature)
    return runs


class ShortcutPatch:
    """Replace every loaded alias of robot_mode_shortcut for one planner run."""

    def __init__(
        self, policy: str, seed: int, run_started: float, deadline: float,
        checks_per_call: int = 100,
    ):
        self.policy = policy
        self.seed = int(seed)
        self.run_started = float(run_started)
        self.deadline = float(deadline)
        self.checks_per_call = int(checks_per_call)
        if self.checks_per_call <= 0:
            raise ValueError("checks_per_call must be positive")
        self.original = shortcut_module.robot_mode_shortcut
        self.calls = []
        self.restores = []

    def __enter__(self):
        patch = self
        original = self.original

        def integrated(
            env,
            path,
            max_iter=1000,
            resolution=0.001,
            tolerance=0.01,
            interpolation_resolution=0.5,
            max_attempts=None,
            **_extra,
        ):
            call_index = len(patch.calls)
            call_seed = patch.seed + 104729 * call_index
            call_started = time.perf_counter()
            remaining = max(0.0, patch.deadline - call_started)
            applied_checks = patch.checks_per_call
            if patch.policy == "upstream_tree":
                numpy_state = np.random.get_state()
                python_state = random.getstate()
                np.random.seed(call_seed)
                random.seed(call_seed)
                try:
                    out, upstream_trace = original(
                        env,
                        path,
                        max_iter=applied_checks,
                        resolution=float(resolution),
                        tolerance=float(tolerance),
                        interpolation_resolution=float(interpolation_resolution),
                        max_attempts=max_attempts,
                        run_choice="tree",
                    )
                finally:
                    np.random.set_state(numpy_state)
                    random.setstate(python_state)
                costs = [float(value) for value in upstream_trace[0]]
                times = [float(value) for value in upstream_trace[1]]
                details = {
                    "wall_seconds": time.perf_counter() - call_started,
                    "initial_cost": costs[0],
                    "final_cost": costs[-1],
                    "trace": [
                        {
                            "time": elapsed,
                            "cost": cost,
                            "collision_checks": index,
                        }
                        for index, (elapsed, cost) in enumerate(zip(times, costs))
                    ],
                    "stats": {
                        "implementation": "landed robot_mode_shortcut(run_choice=tree)",
                        "recorded_collision_checks": max(0, len(costs) - 1),
                        "planner_rng_state_restored": True,
                    },
                }
            else:
                out, details = run_checks(
                    env,
                    path,
                    patch.policy,
                    max_collision_checks=applied_checks,
                    seed=call_seed,
                    interpolation_resolution=float(interpolation_resolution),
                    seconds=remaining,
                    max_attempts=max_attempts,
                )
            offset = call_started - patch.run_started
            patch.calls.append({
                "index": call_index,
                "seed": call_seed,
                "planner_requested_checks": int(max_iter),
                "applied_checks": applied_checks,
                "resolution": float(resolution),
                "tolerance": float(tolerance),
                "start_elapsed_seconds": float(offset),
                "wall_seconds": float(details["wall_seconds"]),
                "input_cost": float(details["initial_cost"]),
                "output_cost": float(details["final_cost"]),
                "input_states": len(path),
                "output_states": len(out),
                "input_mode_runs": _mode_runs(path),
                "output_mode_runs": _mode_runs(out),
                "trace": [
                    {
                        "time": float(offset + point["time"]),
                        "cost": float(point["cost"]),
                        "collision_checks": int(point["collision_checks"]),
                    }
                    for point in details["trace"]
                ],
                "stats": details["stats"],
            })
            return out, [
                [float(point["cost"]) for point in details["trace"]],
                [float(point["time"]) for point in details["trace"]],
            ]

        self.integrated = integrated
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("multi_robot_multi_goal_planning"):
                continue
            if getattr(module, "robot_mode_shortcut", None) is original:
                setattr(module, "robot_mode_shortcut", integrated)
                self.restores.append((module, original))
        if shortcut_module.robot_mode_shortcut is original:
            shortcut_module.robot_mode_shortcut = integrated
            self.restores.append((shortcut_module, original))
        return self

    def __exit__(self, exc_type, exc, tb):
        for module, original in reversed(self.restores):
            if getattr(module, "robot_mode_shortcut", None) is self.integrated:
                setattr(module, "robot_mode_shortcut", original)
        return False


def _generic_plan(
    env, runtime: float, optimize: bool = True, shortcutting: bool = True
):
    config = BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.stepsize = 1.0
    config.shortcutting = bool(optimize and shortcutting)
    config.with_mode_validation = False
    planner = BidirectionalRRTstar(env, config=config)
    return planner.plan(
        ptc=RuntimeTerminationCondition(float(runtime)),
        optimize=bool(optimize),
    )


def _assembly_plan(
    env, runtime: float, seed: int, optimize: bool = True,
    shortcutting: bool = True,
):
    """Use assembly's problem-specific BiRRT* setup but enable its internal hook."""
    import mp_export
    from multi_robot_multi_goal_planning.planners import rrtstar_base

    original_config = rrtstar_base.BaseRRTConfig

    class InLoopConfig(original_config):
        def __setattr__(self, name, value):
            if name == "shortcutting":
                value = bool(optimize and shortcutting)
            super().__setattr__(name, value)

    rrtstar_base.BaseRRTConfig = InLoopConfig
    try:
        return mp_export.run_birrt(
            env,
            runtime=float(runtime),
            shortcut=False,
            optimize=bool(optimize),
            verbose=False,
            seed=int(seed),
        )
    finally:
        rrtstar_base.BaseRRTConfig = original_config


def _full_curve(info, calls):
    events = []
    for elapsed, cost in zip(info.get("times", []), info.get("costs", [])):
        events.append((float(elapsed), _scalar(cost), "planner"))
    for call in calls:
        for point in call["trace"]:
            events.append((float(point["time"]), float(point["cost"]), "shortcut"))
    events.sort(key=lambda item: (item[0], item[1]))
    curve = []
    best = float("inf")
    for elapsed, cost, source in events:
        if cost < best - 1e-12:
            best = cost
            curve.append({"time": elapsed, "cost": cost, "source": source})
    return curve


def _certificate(env, path):
    strict = bool(env.is_path_collision_free(
        path,
        resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance),
        check_start_and_end=True,
        check_edges_in_order=True,
        binary_order=False,
    ))
    valid = bool(env.is_valid_plan(path)) if hasattr(env, "is_valid_plan") else None
    return {
        "strict_full_edge": strict,
        "valid_plan": valid,
        "terminal_mode": bool(env.is_terminal_mode(path[-1].mode)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--runtime", type=float, default=100.0)
    parser.add_argument("--shortcut-checks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planner-seed-offset", type=int)
    parser.add_argument("--shortcut-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--stop-at-first", action="store_true")
    args = parser.parse_args()

    case = case_by_id(args.case)
    if case["kind"] == "mrbench":
        env = build_mrbench_environment(case)
        replay = None
    else:
        from assembly_loader import build

        env, _, replay = build(case, ROOT)
    fingerprint = environment_fingerprint(env)
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    planner_seed_offset = (
        int(args.seed) if args.planner_seed_offset is None
        else int(args.planner_seed_offset)
    )
    planner_seed = int(case["environment_seed"]) + planner_seed_offset
    shortcut_seed = planner_seed if args.shortcut_seed is None else int(args.shortcut_seed)
    np.random.seed(planner_seed)
    random.seed(planner_seed)
    run_started = time.perf_counter()
    deadline = run_started + float(args.runtime)
    shortcutting_enabled = not args.stop_at_first and args.policy != "planner_only"

    with ShortcutPatch(
        args.policy, shortcut_seed, run_started, deadline, args.shortcut_checks
    ) as patch:
        sink = open(os.devnull, "w") if args.quiet else None
        try:
            redirect = contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext()
            with redirect:
                if case["kind"] == "mrbench":
                    path, info = _generic_plan(
                        env, args.runtime, optimize=not args.stop_at_first,
                        shortcutting=shortcutting_enabled,
                    )
                else:
                    path, info = _assembly_plan(
                        env, args.runtime, planner_seed,
                        optimize=not args.stop_at_first,
                        shortcutting=shortcutting_enabled,
                    )
        finally:
            if sink:
                sink.close()
    total_wall = time.perf_counter() - run_started
    info = dict(info or {})

    result = {
        "schema": "mr-shortcut-planner-loop-result-v1",
        "protocol": PROTOCOL,
        "case_id": case["id"],
        "kind": case["kind"],
        "policy": args.policy,
        "run_id": int(args.seed),
        "planner_seed": planner_seed,
        "planner_seed_offset": planner_seed_offset,
        "shortcut_seed": shortcut_seed,
        "runtime_seconds": float(args.runtime),
        "total_wall_seconds": float(total_wall),
        "environment_fingerprint": fingerprint,
        "source_path": None,
        "cost": {"per_agent": "euclidean", "reduction": "max"},
        "rng_protocol": {
            "planner_seed": planner_seed,
            "shortcut_rng_isolated_from_planner": True,
            "shortcut_seed": shortcut_seed,
            "shortcut_rng_isolation": (
                "save/seed/restore global NumPy and Python RNG state"
                if args.policy == "upstream_tree"
                else "private NumPy Generator"
            ),
            "shortcut_seed_stride": 104729,
        },
        "planner": {
            "name": "BidirectionalRRTstar",
            "optimize": not args.stop_at_first,
            "shortcutting_in_loop": shortcutting_enabled,
            "returned_path": bool(path),
            "info_times": [float(value) for value in info.get("times", [])],
            "info_costs": [_scalar(value) for value in info.get("costs", [])],
        },
        "shortcut": {
            "calls": patch.calls,
            "call_count": len(patch.calls),
            "requested_checks_per_call": int(args.shortcut_checks),
            "planner_default_checks_per_call": 100,
        },
        "assembly": {
            "checkout": replay["assembly_checkout"],
            "revision": git_revision(Path(replay["assembly_checkout"])),
        } if replay else None,
    }

    if path:
        certificate = _certificate(env, path)
        required = certificate["strict_full_edge"] and certificate["terminal_mode"]
        required = required and certificate["valid_plan"] is not False
        if not required:
            raise RuntimeError(f"strict output certificate failed: {certificate}")
        result.update({
            "output": {
                "states": len(path),
                "cost": _scalar(path_cost(path, env.batch_config_cost)),
                "mode_runs": _mode_runs(path),
            },
            "certificate": certificate,
            "full_curve": _full_curve(info, patch.calls),
        })
    else:
        result.update({"output": None, "certificate": None, "full_curve": []})

    write_json(args.output.resolve(), result)
    status = (
        f"cost {result['output']['cost']:.6f}"
        if result["output"] else "no path"
    )
    print(
        f"[done] {case['id']} {args.policy} seed {planner_seed}: {status}; "
        f"{len(patch.calls)} shortcut calls; {total_wall:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
