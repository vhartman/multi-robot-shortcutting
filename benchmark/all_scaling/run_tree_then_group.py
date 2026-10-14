"""Run the additive tree-then-compatible-group scaling comparison serially."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (REPO / "src", BENCHMARK, BENCHMARK / "scaling_2d", HERE):
    sys.path.insert(0, str(directory))

from common import environment_fingerprint, mode_runs, restore_mrbench_path  # noqa: E402
from study_lib import (  # noqa: E402
    build_environment,
    cases,
    input_path,
    protocol,
    protocol_sha256,
    read_json,
    robot_dofs,
    sha256,
    write_json,
)
import scalable_tree_group  # noqa: E402,F401
from mr_shortcutting import run  # noqa: E402


METHOD_ID = "tree_then_compatible_group"
POLICY = "tree_then_compatible_group"
OUTPUT_ROOT = HERE / "offline" / "tree-then-compatible-group"
AUDIT_PATH = OUTPUT_ROOT / "execution-audit.jsonl"
IMPLEMENTATION = BENCHMARK / "scaling_2d" / "scalable_tree_group.py"
FAMILY_ORDER = {
    "disk_scaling": 0,
    "isolated_box_stacking": 1,
    "coupled_box_stacking": 2,
    "mobile_wall_scaling": 3,
}


def mains_connected() -> bool:
    indicators, batteries = [], []
    for supply in Path("/sys/class/power_supply").glob("*"):
        kind = supply / "type"
        if kind.exists() and kind.read_text().strip() == "Battery":
            batteries.append(supply)
        elif (supply / "online").exists():
            indicators.append(supply / "online")
    return any(path.read_text().strip() == "1" for path in indicators) if indicators else not batteries


def output_path(case_id: str, path_index: int, seed: int) -> Path:
    return OUTPUT_ROOT / "results" / case_id / f"{METHOD_ID}-path{path_index}-seed{seed}.json"


def output_errors(path: Path, case: dict, path_index: int, seed: int) -> list[str]:
    if not path.is_file():
        return ["missing"]
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    source = input_path(case["id"], path_index)
    expected = {
        "schema": "mr-shortcut-tree-then-group-scaling-v1",
        "case_id": case["id"],
        "family": case["family"],
        "method_id": METHOD_ID,
        "implementation_policy": POLICY,
        "path_index": path_index,
        "shortcut_seed": seed,
        "source_sha256": sha256(source),
        "implementation_sha256": sha256(IMPLEMENTATION),
        "all_scaling_protocol_sha256": protocol_sha256(),
    }
    errors = [key for key, wanted in expected.items() if value.get(key) != wanted]
    if float(value.get("budget_seconds", -1)) != float(case["seconds"]):
        errors.append("budget_seconds")
    if value.get("cost") != protocol()["cost"]:
        errors.append("cost")
    if value.get("input_mode_runs") != value.get("output_mode_runs"):
        errors.append("mode sequence")
    for key in (
        "endpoint_start_preserved",
        "endpoint_end_preserved",
        "strict_collision_free",
        "terminal_mode_preserved",
    ):
        if value.get(key) is not True:
            errors.append(key)
    if value.get("valid_plan") is False:
        errors.append("valid_plan")
    return errors


def run_cell(case: dict, path_index: int, seed: int):
    source_path = input_path(case["id"], path_index).resolve()
    source = read_json(source_path)
    env = build_environment(case)
    if environment_fingerprint(env) != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from input")
    path = restore_mrbench_path(env, source["states"])
    from multi_robot_multi_goal_planning.problems.util import path_cost

    restored_cost = float(path_cost(path, env.batch_config_cost))
    tolerance = 1e-9 * max(1.0, abs(float(source["initial_cost"])))
    if abs(restored_cost - float(source["initial_cost"])) > tolerance:
        raise RuntimeError("restored input cost changed")
    started = time.perf_counter()
    output, result = run(
        env,
        path,
        POLICY,
        float(case["seconds"]),
        seed,
        interpolation_resolution=float(protocol()["offline"]["interpolation_resolution"]),
    )
    result["total_wall_seconds"] = time.perf_counter() - started
    start_preserved = bool(np.allclose(output[0].q.state(), path[0].q.state(), rtol=0, atol=1e-8))
    end_preserved = bool(np.allclose(output[-1].q.state(), path[-1].q.state(), rtol=0, atol=1e-8))
    strict = bool(env.is_path_collision_free(
        output,
        resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance),
        check_start_and_end=True,
        check_edges_in_order=True,
        binary_order=False,
    ))
    valid = bool(env.is_valid_plan(output)) if hasattr(env, "is_valid_plan") else None
    terminal = bool(env.is_terminal_mode(output[-1].mode))
    before, after = mode_runs(path), mode_runs(output)
    if not (
        start_preserved and end_preserved and strict and terminal
        and valid is not False and before == after
    ):
        raise RuntimeError(
            "output certificate failed: "
            f"modes={before == after}, start={start_preserved}, end={end_preserved}, "
            f"strict={strict}, valid={valid}, terminal={terminal}"
        )
    result.update({
        "schema": "mr-shortcut-tree-then-group-scaling-v1",
        "case_id": case["id"],
        "family": case["family"],
        "method_id": METHOD_ID,
        "implementation_policy": POLICY,
        "path_index": path_index,
        "shortcut_seed": seed,
        "cost": dict(protocol()["cost"]),
        "source_path": str(source_path),
        "source_sha256": sha256(source_path),
        "environment_fingerprint": source["environment_fingerprint"],
        "robot_count": len(env.robots),
        "robot_dofs": robot_dofs(env),
        "total_dof": sum(robot_dofs(env)),
        "input_mode_runs": before,
        "output_mode_runs": after,
        "endpoint_start_preserved": start_preserved,
        "endpoint_end_preserved": end_preserved,
        "terminal_mode_preserved": terminal,
        "strict_collision_free": strict,
        "valid_plan": valid,
        "implementation_sha256": sha256(IMPLEMENTATION),
        "all_scaling_protocol_sha256": protocol_sha256(),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "execute"))
    parser.add_argument("--family", action="append", choices=tuple(FAMILY_ORDER))
    parser.add_argument("--case", action="append", choices=tuple(cases()))
    parser.add_argument("--path-index", action="append", type=int, choices=tuple(protocol()["path_indices"]))
    parser.add_argument("--seed", action="append", type=int, choices=tuple(protocol()["shortcut_seeds"]))
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    selected = list(cases().values())
    if args.family:
        selected = [case for case in selected if case["family"] in set(args.family)]
    if args.case:
        selected = [case for case in selected if case["id"] in set(args.case)]
    selected.sort(key=lambda case: (FAMILY_ORDER[case["family"]], case["id"]))
    paths = args.path_index or protocol()["path_indices"]
    seeds = args.seed or protocol()["shortcut_seeds"]
    cells = [(case, path_index, seed) for case in selected for path_index in paths for seed in seeds]
    pending = [cell for cell in cells if output_errors(output_path(cell[0]["id"], cell[1], cell[2]), *cell)]
    invalid = [
        (cell, output_errors(output_path(cell[0]["id"], cell[1], cell[2]), *cell))
        for cell in cells
        if output_errors(output_path(cell[0]["id"], cell[1], cell[2]), *cell) not in ([], ["missing"])
    ]
    summary = {
        "cases": len(selected),
        "cells": len(cells),
        "complete": len(cells) - len(pending),
        "pending": len(pending),
        "pending_nominal_hours": sum(float(cell[0]["seconds"]) for cell in pending) / 3600,
        "invalid": [(cell[0]["id"], cell[1], cell[2], errors) for cell, errors in invalid],
        "mains_connected": mains_connected(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.command == "plan":
        return
    if not args.confirm:
        parser.error("execute requires --confirm")
    if invalid:
        raise RuntimeError("invalid existing outputs; move them aside before resuming")
    if not mains_connected():
        raise RuntimeError("mains disconnected")

    run_started = time.monotonic()
    wall_minus_mono = time.time() - run_started
    deadline = run_started + args.max_hours * 3600
    for index, (case, path_index, seed) in enumerate(pending, 1):
        if not mains_connected():
            raise RuntimeError("mains disconnected; stopped before next cell")
        if abs((time.time() - time.monotonic()) - wall_minus_mono) > 2:
            raise RuntimeError("suspend/clock jump detected")
        if time.monotonic() + float(case["seconds"]) + 120 > deadline:
            print("[stopped] six-hour deadline reserve reached", flush=True)
            break
        started = time.monotonic()
        result = run_cell(case, path_index, seed)
        elapsed = time.monotonic() - started
        destination = output_path(case["id"], path_index, seed)
        write_json(destination, result)
        errors = output_errors(destination, case, path_index, seed)
        if errors:
            raise RuntimeError(f"written output failed validation: {errors}")
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("a") as stream:
            stream.write(json.dumps({
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "case_id": case["id"],
                "family": case["family"],
                "path_index": path_index,
                "shortcut_seed": seed,
                "seconds": float(case["seconds"]),
                "elapsed_seconds": elapsed,
                "final_cost": result["final_cost"],
                "success": True,
            }) + "\n")
        print(
            f"[ok {index}/{len(pending)}] {case['id']} path{path_index} seed{seed}: "
            f"{result['initial_cost']:.6g} -> {result['final_cost']:.6g} ({elapsed:.2f}s)",
            flush=True,
        )

    remaining = [cell for cell in cells if output_errors(output_path(cell[0]["id"], cell[1], cell[2]), *cell)]
    print(json.dumps({"scheduled": len(cells), "remaining": len(remaining)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
