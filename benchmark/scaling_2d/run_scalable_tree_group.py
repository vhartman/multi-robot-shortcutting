"""Run certified scalable tree/group cells on saved disk-scaling paths."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
ROOT = BENCHMARK.parent
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK / "expanded"))
sys.path.insert(0, str(ROOT / "src"))
os.environ["MR_SHORTCUT_PROTOCOL"] = str(HERE / "protocol.json")
os.environ["MR_SHORTCUT_OUTPUT_ROOT"] = str(HERE)

from common import environment_fingerprint, mode_runs, restore_mrbench_path  # noqa: E402
from expanded_lib import build_mrbench_environment, cases, read_json, sha256, write_json  # noqa: E402
import scalable_tree_group  # noqa: E402,F401
from mr_shortcutting import run  # noqa: E402


POLICIES = ("scalable_tree_group",)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robots", type=int, action="append", required=True)
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--path-index", type=int, action="append", default=None)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--output-root", type=Path, default=HERE / "analysis" / "group-gain-trials-v2")
    args = parser.parse_args()

    policies = args.policy or list(POLICIES)
    paths = args.path_index or [0]
    seeds = args.seed or [0]
    scheduled = 0
    for robots in args.robots:
        case_id = f"rai-disk-scaling-r{robots}-g4"
        case = cases()[case_id]
        for path_index in paths:
            source_path = HERE / "inputs" / case_id / f"path{path_index}.json"
            source = read_json(source_path)
            for policy in policies:
                for seed in seeds:
                    output = args.output_root / case_id / f"{policy}-path{path_index}-seed{seed}.json"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    env = build_mrbench_environment(case)
                    if environment_fingerprint(env) != source["environment_fingerprint"]:
                        raise RuntimeError("environment fingerprint mismatch")
                    path = restore_mrbench_path(env, source["states"])
                    started = time.perf_counter()
                    output_path, result = run(
                        env, path, policy, args.seconds, seed,
                        interpolation_resolution=0.5,
                    )
                    total = time.perf_counter() - started
                    strict = bool(env.is_path_collision_free(
                        output_path,
                        resolution=float(env.collision_resolution),
                        tolerance=float(env.collision_tolerance),
                        check_start_and_end=True,
                        check_edges_in_order=True,
                        binary_order=False,
                    ))
                    valid = bool(env.is_valid_plan(output_path))
                    modes_preserved = mode_runs(path) == mode_runs(output_path)
                    if not strict or not valid or not modes_preserved:
                        raise RuntimeError(
                            f"certificate failed: strict={strict}, valid={valid}, "
                            f"modes={modes_preserved}"
                        )
                    trial_source = HERE / "scalable_tree_group.py"
                    trial_hash = sha256(trial_source)
                    result.update({
                        "schema": "mr-shortcut-group-gain-trial-v1",
                        "case_id": case_id,
                        "robot_count": robots,
                        "total_dof": 2 * robots,
                        "path_index": path_index,
                        "shortcut_seed": seed,
                        "source_sha256": sha256(source_path),
                        "trial_implementation_sha256": trial_hash,
                        "total_wall_seconds": total,
                        "strict_collision_free": strict,
                        "valid_plan": valid,
                        "mode_runs_preserved": modes_preserved,
                    })
                    write_json(output, result)
                    scheduled += 1
                    print(
                        f"[ok {scheduled}] r{robots} {policy} path{path_index} "
                        f"seed{seed}: {result['initial_cost']:.6g} -> "
                        f"{result['final_cost']:.6g}, wall={result['wall_seconds']:.3f}s",
                        flush=True,
                    )
    print(json.dumps({"scheduled": scheduled}, indent=2))


if __name__ == "__main__":
    main()
