"""Run and validate one fixed-first-path planning-in-the-loop cell."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol_lib import (  # noqa: E402
    cost_record,
    methods,
    online_runtime,
    protocol,
    protocol_cases,
    protocol_sha256,
    read_json,
    sha256,
    write_json,
)

INNER_RUNNER = BENCHMARK / "online.py"


def main() -> int:
    data = protocol()
    method_map = methods()
    choices = tuple(method_map) + ("planner_only",)
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=tuple(protocol_cases()))
    parser.add_argument("--method", required=True, choices=choices)
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--shortcut-seed", type=int)
    parser.add_argument("--runtime", required=True, type=float)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.path_index not in data["design"]["path_indices"]:
        raise ValueError("path index is outside frozen protocol")
    if args.method == "planner_only":
        if args.shortcut_seed is not None:
            raise ValueError("planner_only does not take a shortcut seed")
        policy = "planner_only"
        shortcut_seed = 0
    else:
        if args.shortcut_seed not in data["design"]["shortcut_seeds"]:
            raise ValueError("shortcut seed is outside frozen protocol")
        policy = method_map[args.method]["online_policy"]
        shortcut_seed = int(args.shortcut_seed)

    case_settings = protocol_cases()[args.case]
    if not case_settings["online"]:
        raise ValueError(f"{args.case} is offline-only in the frozen protocol")

    source_path = args.input.resolve()
    source = read_json(source_path)
    expected_source = {
        "schema": "mr-shortcut-final-input-v1",
        "final_protocol_sha256": protocol_sha256(),
        "case_id": args.case,
        "path_index": args.path_index,
        "cost": cost_record(),
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise RuntimeError(
                f"input metadata mismatch for {key}: {source.get(key)!r} != {expected!r}"
            )
    expected_runtime = online_runtime(source)
    if args.runtime != expected_runtime:
        raise ValueError(
            f"online runtime differs from protocol: {args.runtime} != {expected_runtime}"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mr-shortcut-final-online-") as temporary:
        candidate = Path(temporary) / "candidate.json"
        command = [
            sys.executable,
            str(INNER_RUNNER),
            "--case", args.case,
            "--policy", policy,
            "--runtime", str(args.runtime),
            "--shortcut-checks", str(data["online"]["shortcut_checks_per_call"]),
            "--seed", str(args.path_index),
            "--planner-seed-offset", str(args.path_index),
            "--shortcut-seed", str(shortcut_seed),
            "--quiet",
            "--output", str(candidate),
        ]
        completed = subprocess.run(command, env=os.environ.copy())
        if completed.returncode:
            return completed.returncode
        result = read_json(candidate)

    curve = result.get("full_curve") or []
    if not curve:
        raise RuntimeError("online cell did not find a solution")
    expected_seed = int(source["planner_seed"])
    expected_cost = float(source["initial_cost"])
    actual_cost = float(curve[0]["cost"])
    tolerance = 1e-9 * max(1.0, abs(expected_cost))
    certificate = result.get("certificate") or {}
    certified = (
        certificate.get("strict_full_edge") is True
        and certificate.get("terminal_mode") is True
        and certificate.get("valid_plan") is not False
    )
    available_refinement = float(result["runtime_seconds"]) - float(curve[0]["time"])
    required_refinement = float(data["online"]["analysis_horizon_seconds"])
    errors = []
    if int(result["planner_seed"]) != expected_seed:
        errors.append(f"planner seed {result['planner_seed']} != {expected_seed}")
    if abs(actual_cost - expected_cost) > tolerance:
        errors.append(f"first cost {actual_cost} != {expected_cost}")
    if result["environment_fingerprint"] != source["environment_fingerprint"]:
        errors.append("environment fingerprint changed")
    if result.get("cost", {}).get("per_agent") != "euclidean":
        errors.append("cost metric changed")
    if result.get("cost", {}).get("reduction") != "max":
        errors.append("cost reduction changed")
    if available_refinement + 1e-9 < required_refinement:
        errors.append(
            f"only {available_refinement:.6f}s available after first solution; "
            f"need {required_refinement:.6f}s"
        )
    if not certified:
        errors.append(f"strict output certificate failed: {certificate}")
    if args.method != "planner_only":
        checks = int(data["online"]["shortcut_checks_per_call"])
        wrong = [
            call.get("applied_checks")
            for call in result.get("shortcut", {}).get("calls", [])
            if call.get("applied_checks") != checks
        ]
        if wrong:
            errors.append(f"online shortcut calls used non-protocol budgets: {wrong}")
    if errors:
        raise RuntimeError("; ".join(errors))

    result.update({
        "inner_schema": result.get("schema"),
        "inner_protocol": result.get("protocol"),
        "schema": "mr-shortcut-final-online-result-v1",
        "final_protocol_sha256": protocol_sha256(),
        "method_id": args.method,
        "implementation_policy": policy,
        "path_index": args.path_index,
        "shortcut_seed": None if args.method == "planner_only" else shortcut_seed,
        "cost": cost_record(),
        "source_path": str(source_path.relative_to(REPO)),
        "source_sha256": sha256(source_path),
        "fixed_first_path_validated": True,
        "first_solution_cost": actual_cost,
        "first_solution_time": float(curve[0]["time"]),
        "available_refinement_seconds": available_refinement,
        "analysis_time_origin": "first_solution",
        "analysis_horizon_seconds": required_refinement,
        "pythonhashseed": int(source["pythonhashseed"]),
    })
    write_json(output, result)
    print(
        f"[ok] online {args.case} {args.method} path{args.path_index} "
        f"seed={args.shortcut_seed}: first={actual_cost:.9g}, "
        f"final={result['output']['cost']:.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
