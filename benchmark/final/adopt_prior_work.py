"""Validate and adopt the completed 3x3 prior-work cells without rerunning them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol_lib import (  # noqa: E402
    REPO,
    cost_record,
    input_path,
    methods,
    offline_result_path,
    online_result_path,
    planner_seed,
    protocol,
    protocol_cases,
    protocol_sha256,
    read_json,
    sha256,
    write_json,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def certified(certificate: dict | None) -> bool:
    certificate = certificate or {}
    return bool(
        certificate.get("strict_full_edge") is True
        and certificate.get("terminal_mode") is True
        and certificate.get("valid_plan") is not False
    )


def adopt_input(source: Path, case_id: str, path_index: int, write: bool) -> dict:
    source_path = source / "inputs" / case_id / f"path{path_index}.json"
    data = read_json(source_path)
    require(data.get("schema") == "mr-shortcut-paper-comparison-input-v1", f"unexpected input schema: {source_path}")
    require(data.get("case_id") == case_id, f"case mismatch: {source_path}")
    require(int(data.get("planner_seed")) == planner_seed(case_id, path_index), f"planner seed mismatch: {source_path}")
    require(int(data.get("pythonhashseed")) == int(protocol()["rng"]["pythonhashseed"]), f"hash seed mismatch: {source_path}")
    require(certified(data.get("certificate")), f"input certificate failed: {source_path}")
    require(data.get("mr_bench_revision") == protocol()["software"]["mr_bench_revision"] or data.get("kind") == "assembly", f"mr_bench revision mismatch: {source_path}")
    adopted = dict(data)
    adopted.update({
        "schema": "mr-shortcut-final-input-v1",
        "final_protocol_sha256": protocol_sha256(),
        "path_index": path_index,
        "cost": cost_record(),
        "adopted_prior_work": {
            "source_schema": data["schema"],
            "source_sha256": sha256(source_path),
        },
    })
    destination = input_path(case_id, path_index)
    if write:
        write_json(destination, adopted)
    return adopted


def adopt_offline(
    source: Path, case_id: str, method_id: str, path_index: int, seed: int,
    frozen_input: dict, write: bool,
) -> None:
    method = methods()[method_id]
    policy = method["offline_policy"]
    source_path = source / "shortcut-only" / "results" / case_id / f"{policy}-path{path_index}-seed{seed}.json"
    data = read_json(source_path)
    expected_seconds = float(protocol_cases()[case_id]["offline_seconds"])
    require(data.get("schema") == "mr-shortcut-paper-comparison-offline-v1", f"unexpected offline schema: {source_path}")
    require(data.get("case_id") == case_id and data.get("policy") == policy, f"offline identity mismatch: {source_path}")
    require(int(data.get("seed")) == seed, f"offline seed mismatch: {source_path}")
    require(float(data.get("budget_seconds")) == expected_seconds, f"offline budget mismatch: {source_path}")
    require(data.get("environment_fingerprint") == frozen_input["environment_fingerprint"], f"offline fingerprint mismatch: {source_path}")
    tolerance = 1e-9 * max(1.0, abs(float(frozen_input["initial_cost"])))
    require(abs(float(data.get("initial_cost")) - float(frozen_input["initial_cost"])) <= tolerance, f"offline initial cost mismatch: {source_path}")
    require(data.get("strict_collision_free") is True, f"offline collision certificate failed: {source_path}")
    require(data.get("valid_plan") is not False, f"offline native certificate failed: {source_path}")
    require(data.get("terminal_mode_preserved") is True, f"offline terminal mode failed: {source_path}")
    require(data.get("endpoint_start_preserved") is True and data.get("endpoint_end_preserved") is True, f"offline endpoints changed: {source_path}")
    require(data.get("input_mode_runs") == data.get("output_mode_runs"), f"offline modes changed: {source_path}")
    destination = offline_result_path(case_id, method_id, path_index, seed)
    adopted = dict(data)
    adopted.update({
        "schema": "mr-shortcut-final-offline-result-v1",
        "final_protocol_sha256": protocol_sha256(),
        "method_id": method_id,
        "implementation_policy": policy,
        "path_index": path_index,
        "shortcut_seed": seed,
        "cost": cost_record(),
        "source_path": str(input_path(case_id, path_index).relative_to(REPO)),
        "source_sha256": sha256(input_path(case_id, path_index)) if write else None,
        "adopted_prior_work": {
            "source_schema": data["schema"],
            "source_sha256": sha256(source_path),
        },
    })
    if write:
        write_json(destination, adopted)


def validate_online(
    data: dict, source_path: Path, case_id: str, policy: str,
    path_index: int, seed: int | None, frozen_input: dict,
) -> tuple[float, float]:
    require(data.get("schema") == "mr-shortcut-planner-loop-result-v1", f"unexpected online schema: {source_path}")
    require(data.get("case_id") == case_id and data.get("policy") == policy, f"online identity mismatch: {source_path}")
    require(int(data.get("planner_seed")) == planner_seed(case_id, path_index), f"online planner seed mismatch: {source_path}")
    if seed is not None:
        require(int(data.get("shortcut_seed")) == seed, f"online shortcut seed mismatch: {source_path}")
        checks = int(data.get("shortcut", {}).get("requested_checks_per_call"))
        require(checks == int(protocol()["online"]["shortcut_checks_per_call"]), f"online check budget mismatch: {source_path}")
    require(data.get("environment_fingerprint") == frozen_input["environment_fingerprint"], f"online fingerprint mismatch: {source_path}")
    require(certified(data.get("certificate")), f"online certificate failed: {source_path}")
    curve = data.get("full_curve") or []
    require(bool(curve), f"online curve missing: {source_path}")
    first_time = float(curve[0]["time"])
    first_cost = float(curve[0]["cost"])
    tolerance = 1e-9 * max(1.0, abs(float(frozen_input["initial_cost"])))
    require(abs(first_cost - float(frozen_input["initial_cost"])) <= tolerance, f"online first cost mismatch: {source_path}")
    available = float(data["runtime_seconds"]) - first_time
    require(available + 1e-9 >= float(protocol()["online"]["analysis_horizon_seconds"]), f"online horizon too short: {source_path}")
    return first_time, available


def adopt_online(
    source: Path, case_id: str, method_id: str, path_index: int,
    seed: int | None, frozen_input: dict, write: bool,
) -> None:
    if method_id == "planner_only":
        policy = "planner_only"
        filename = f"planner_only-path{path_index}.json"
    else:
        policy = methods()[method_id]["online_policy"]
        filename = f"{policy}-path{path_index}-seed{seed}.json"
    source_path = source / "planner-loop" / "results" / case_id / filename
    data = read_json(source_path)
    first_time, available = validate_online(
        data, source_path, case_id, policy, path_index, seed, frozen_input
    )
    destination = online_result_path(case_id, method_id, path_index, seed)
    adopted = dict(data)
    adopted.update({
        "inner_schema": data["schema"],
        "inner_protocol": data.get("protocol"),
        "schema": "mr-shortcut-final-online-result-v1",
        "final_protocol_sha256": protocol_sha256(),
        "method_id": method_id,
        "implementation_policy": policy,
        "path_index": path_index,
        "shortcut_seed": seed,
        "cost": cost_record(),
        "source_path": str(input_path(case_id, path_index).relative_to(REPO)),
        "source_sha256": sha256(input_path(case_id, path_index)) if write else None,
        "fixed_first_path_validated": True,
        "first_solution_cost": float((data.get("full_curve") or [])[0]["cost"]),
        "first_solution_time": first_time,
        "available_refinement_seconds": available,
        "analysis_time_origin": "first_solution",
        "analysis_horizon_seconds": float(protocol()["online"]["analysis_horizon_seconds"]),
        "pythonhashseed": int(frozen_input["pythonhashseed"]),
        "adopted_prior_work": {
            "source_schema": data["schema"],
            "source_sha256": sha256(source_path),
        },
    })
    if write:
        write_json(destination, adopted)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    existing = tuple(protocol()["reuse"]["existing_method_ids"])
    counts = {"inputs": 0, "offline": 0, "online_methods": 0, "planner_only": 0}
    for case_id in protocol_cases():
        for path_index in protocol()["design"]["path_indices"]:
            frozen_input = adopt_input(source, case_id, path_index, args.write)
            counts["inputs"] += 1
            for method_id in existing:
                for seed in protocol()["design"]["shortcut_seeds"]:
                    adopt_offline(source, case_id, method_id, path_index, seed, frozen_input, args.write)
                    adopt_online(source, case_id, method_id, path_index, seed, frozen_input, args.write)
                    counts["offline"] += 1
                    counts["online_methods"] += 1
            adopt_online(source, case_id, "planner_only", path_index, None, frozen_input, args.write)
            counts["planner_only"] += 1
    print({"mode": "write" if args.write else "validate-only", **counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
