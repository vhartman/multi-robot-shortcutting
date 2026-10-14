"""Strictly recertify a historical input path for the scalable study."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (BENCHMARK, HERE, REPO / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import environment_fingerprint, restore_mrbench_path  # noqa: E402
from generate_input import certificate  # noqa: E402
from study_lib import (  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(cases()))
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = protocol()
    case = cases()[args.case]
    if case["kind"] != "mrbench":
        raise ValueError("historical recertification currently supports mrbench inputs")
    if args.path_index not in data["path_indices"]:
        raise ValueError("path index is outside the study protocol")
    if os.environ.get("PYTHONHASHSEED") != str(data["pythonhashseed"]):
        raise RuntimeError("PYTHONHASHSEED must be fixed before Python starts")

    source_path = args.input.resolve()
    source = read_json(source_path)
    if source.get("case_id") != args.case or not source.get("states"):
        raise RuntimeError("historical input metadata does not match the requested case")

    env = build_mrbench_environment(case)
    fingerprint = environment_fingerprint(env)
    if source.get("environment_fingerprint") != fingerprint:
        raise RuntimeError("environment fingerprint differs from historical input")
    path = restore_mrbench_path(env, source["states"])

    from multi_robot_multi_goal_planning.problems.util import path_cost

    initial_cost = float(path_cost(path, env.batch_config_cost))
    expected_cost = float(source["initial_cost"])
    tolerance = 1e-9 * max(1.0, abs(expected_cost))
    if abs(initial_cost - expected_cost) > tolerance:
        raise RuntimeError("restored historical input cost changed")

    cert = certificate(env, path)
    dofs = robot_dofs(env)
    payload = {
        **source,
        "schema": "mr-shortcut-scalable-study-input-v1",
        "study_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "family": case["family"],
        "kind": case["kind"],
        "path_index": args.path_index,
        "environment_seed": int(case["environment_seed"]),
        "pythonhashseed": int(data["pythonhashseed"]),
        "cost": cost_record(),
        "environment_fingerprint": fingerprint,
        "initial_cost": initial_cost,
        "robot_dofs": dofs,
        "total_dof": sum(dofs),
        "state_count": len(path),
        "mode_count": 1 + sum(
            path[index].mode != path[index - 1].mode
            for index in range(1, len(path))
        ),
        "certificate": cert,
        "states": [state.to_dict() for state in path],
        "recertification": {
            "source_path": str(source_path),
            "source_sha256": sha256(source_path),
            "certifier_sha256": sha256(Path(__file__)),
        },
    }
    write_json(args.output.resolve(), payload)
    print(
        f"[ok] {args.case} path{args.path_index}: states={len(path)}, "
        f"cost={initial_cost:.9g}, certificate={cert}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
