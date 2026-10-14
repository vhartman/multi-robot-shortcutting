"""Deterministically reconstruct and certify a saved wall-clock shortcut result."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (REPO / "src", BENCHMARK, BENCHMARK / "expanded", HERE):
    sys.path.insert(0, str(directory))
os.environ["MR_SHORTCUT_PROTOCOL"] = str(HERE / "protocol.json")
os.environ["MR_SHORTCUT_OUTPUT_ROOT"] = str(HERE)

from common import environment_fingerprint, mode_runs, restore_mrbench_path  # noqa: E402
from expanded_lib import build_mrbench_environment, cases, read_json, write_json  # noqa: E402
import scalable_tree_group  # noqa: E402,F401
from mr_shortcutting import run_checks  # noqa: E402


def certificate(env, path):
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
        raise RuntimeError(f"reconstructed shortcut failed certification: {result}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "inputs" / "rai-disk-scaling-r12-g4" / "path0.json",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=(
            BENCHMARK / "scalable_study" / "offline" / "results"
            / "rai-disk-scaling-r12-g4"
            / "scalable_tree_group-path0-seed0.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            HERE / "analysis"
            / "disk-scaling-r12-path0-scalable-tree-group-seed0.json"
        ),
    )
    args = parser.parse_args()

    source = read_json(args.input.resolve())
    reference = read_json(args.reference.resolve())
    case = cases()[source["case_id"]]
    expected = {
        "case_id": source["case_id"],
        "path_index": source["path_index"],
        "method_id": "scalable_tree_group",
        "implementation_policy": "scalable_tree_group",
        "shortcut_seed": 0,
    }
    for key, value in expected.items():
        if reference.get(key) != value:
            raise RuntimeError(
                f"reference mismatch for {key}: {reference.get(key)!r} != {value!r}"
            )

    np.random.seed(int(case["environment_seed"]))
    random.seed(int(case["environment_seed"]))
    env = build_mrbench_environment(case)
    if environment_fingerprint(env) != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from the certified input")
    path = restore_mrbench_path(env, source["states"])
    checks = int(reference["stats"]["collision_checks"])
    output_path, replay = run_checks(
        env,
        path,
        "scalable_tree_group",
        max_collision_checks=checks,
        seed=int(reference["shortcut_seed"]),
        interpolation_resolution=float(reference["interpolation_resolution"]),
    )
    expected_cost = float(reference["final_cost"])
    actual_cost = float(replay["final_cost"])
    tolerance = 1e-10 * max(1.0, abs(expected_cost))
    if abs(actual_cost - expected_cost) > tolerance:
        raise RuntimeError(
            f"deterministic replay cost differs: {actual_cost} != {expected_cost}"
        )
    if replay["stats"]["accepted_lineage"] != reference["stats"]["accepted_lineage"]:
        # Timing fields in lineage are machine-dependent, so compare the
        # proposal identity and geometry separately below.
        wanted = [
            {key: item[key] for key in ("id", "first", "last", "robots", "source")}
            for item in reference["stats"]["accepted_lineage"]
        ]
        actual = [
            {key: item[key] for key in ("id", "first", "last", "robots", "source")}
            for item in replay["stats"]["accepted_lineage"]
        ]
        if actual != wanted:
            raise RuntimeError("deterministic replay accepted a different shortcut sequence")

    cert = certificate(env, output_path)
    if mode_runs(output_path) != reference["output_mode_runs"]:
        raise RuntimeError("reconstructed shortcut changed the reference mode sequence")
    payload = dict(source)
    payload.update({
        "schema": "mr-shortcut-render-path-v1",
        "state_count": len(output_path),
        "mode_count": len(mode_runs(output_path)),
        "initial_cost": actual_cost,
        "certificate": cert,
        "states": [state.to_dict() for state in output_path],
        "shortcut": {
            "method_id": reference["method_id"],
            "policy": reference["implementation_policy"],
            "seed": int(reference["shortcut_seed"]),
            "reference_budget_seconds": float(reference["budget_seconds"]),
            "replayed_collision_checks": checks,
            "original_cost": float(reference["initial_cost"]),
            "final_cost": actual_cost,
            "relative_cost": actual_cost / float(reference["initial_cost"]),
            "accepted_shortcuts": len(reference["stats"]["accepted_lineage"]),
            "reference_result": str(args.reference.resolve()),
        },
    })
    write_json(args.output.resolve(), payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "collision_checks": checks,
        "accepted_shortcuts": len(reference["stats"]["accepted_lineage"]),
        "original_cost": float(reference["initial_cost"]),
        "final_cost": actual_cost,
        "relative_cost": actual_cost / float(reference["initial_cost"]),
        "certificate": cert,
    }, indent=2))


if __name__ == "__main__":
    main()
