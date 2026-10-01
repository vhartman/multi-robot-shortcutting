"""Plan, preflight, and serially execute the frozen final experiment matrix.

Nothing runs without the explicit ``execute --confirm`` command.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol_lib import (  # noqa: E402
    LOCK_PATH,
    REPO,
    assembly_checkout,
    cost_record,
    git_revision,
    input_path,
    manifest_cases,
    methods,
    mr_bench_checkout,
    offline_result_path,
    online_result_path,
    online_runtime,
    protocol,
    protocol_cases,
    protocol_sha256,
    read_json,
    sha256,
    tracked_changes,
    validate_protocol_definition,
    verify_content_lock,
)

INPUT_CELL = HERE / "generate_input.py"
OFFLINE_CELL = HERE / "run_offline_cell.py"
ONLINE_CELL = HERE / "run_online_cell.py"
AUDIT_PATH = HERE / "execution-audit.jsonl"


@dataclass(frozen=True)
class Cell:
    phase: str
    case_id: str
    path_index: int
    output: Path
    log: Path
    command: tuple[str, ...]
    method_id: str | None = None
    shortcut_seed: int | None = None


def selected(values: list | None, defaults: Iterable) -> list:
    return list(values) if values else list(defaults)


def cells(
    phase: str,
    case_filter: list[str] | None = None,
    method_filter: list[str] | None = None,
    path_filter: list[int] | None = None,
    seed_filter: list[int] | None = None,
) -> list[Cell]:
    data = protocol()
    case_map = protocol_cases()
    method_map = methods()
    case_ids = selected(case_filter, case_map)
    method_ids = selected(method_filter, method_map)
    paths = selected(path_filter, data["design"]["path_indices"])
    seeds = selected(seed_filter, data["design"]["shortcut_seeds"])
    scheduled: list[Cell] = []

    for case_id in case_ids:
        case = case_map[case_id]
        for path_index in paths:
            source = input_path(case_id, path_index)
            if phase == "inputs":
                output = source
                log = HERE / "inputs" / "logs" / case_id / f"path{path_index}.log"
                command = (
                    sys.executable,
                    str(INPUT_CELL),
                    "--case", case_id,
                    "--path-index", str(path_index),
                    "--seconds", str(case["input_generation_seconds"]),
                    "--output", str(output),
                )
                scheduled.append(Cell(phase, case_id, path_index, output, log, command))
                continue

            if phase == "offline":
                for method_id in method_ids:
                    for seed in seeds:
                        output = offline_result_path(
                            case_id, method_id, path_index, seed
                        )
                        log = (
                            HERE / "offline" / "logs" / case_id
                            / f"{method_id}-path{path_index}-seed{seed}.log"
                        )
                        command = (
                            sys.executable,
                            str(OFFLINE_CELL),
                            "--case", case_id,
                            "--method", method_id,
                            "--path-index", str(path_index),
                            "--shortcut-seed", str(seed),
                            "--seconds", str(case["offline_seconds"]),
                            "--input", str(source),
                            "--output", str(output),
                        )
                        scheduled.append(Cell(
                            phase, case_id, path_index, output, log, command,
                            method_id, seed,
                        ))
                continue

            if phase != "online":
                raise ValueError(f"unknown phase: {phase}")
            if not case["online"]:
                continue
            runtime = online_runtime(read_json(source)) if source.exists() else None
            planner_output = online_result_path(
                case_id, "planner_only", path_index, None
            )
            planner_log = (
                HERE / "online" / "logs" / case_id
                / f"planner_only-path{path_index}.log"
            )
            planner_command = (
                sys.executable,
                str(ONLINE_CELL),
                "--case", case_id,
                "--method", "planner_only",
                "--path-index", str(path_index),
                "--runtime", str(runtime) if runtime is not None else "<requires-input>",
                "--input", str(source),
                "--output", str(planner_output),
            )
            scheduled.append(Cell(
                phase, case_id, path_index, planner_output, planner_log,
                planner_command, "planner_only", None,
            ))
            for method_id in method_ids:
                for seed in seeds:
                    output = online_result_path(
                        case_id, method_id, path_index, seed
                    )
                    log = (
                        HERE / "online" / "logs" / case_id
                        / f"{method_id}-path{path_index}-seed{seed}.log"
                    )
                    command = (
                        sys.executable,
                        str(ONLINE_CELL),
                        "--case", case_id,
                        "--method", method_id,
                        "--path-index", str(path_index),
                        "--shortcut-seed", str(seed),
                        "--runtime", str(runtime) if runtime is not None else "<requires-input>",
                        "--input", str(source),
                        "--output", str(output),
                    )
                    scheduled.append(Cell(
                        phase, case_id, path_index, output, log, command,
                        method_id, seed,
                    ))
    return scheduled


def artifact_errors(cell: Cell) -> list[str]:
    if not cell.output.exists():
        return ["missing"]
    try:
        result = read_json(cell.output)
    except (OSError, json.JSONDecodeError) as error:
        return [f"unreadable JSON: {error}"]
    errors = []
    expected_schema = {
        "inputs": "mr-shortcut-final-input-v1",
        "offline": "mr-shortcut-final-offline-result-v1",
        "online": "mr-shortcut-final-online-result-v1",
    }[cell.phase]
    expected = {
        "schema": expected_schema,
        "final_protocol_sha256": protocol_sha256(),
        "case_id": cell.case_id,
        "path_index": cell.path_index,
        "cost": cost_record(),
    }
    if cell.method_id is not None:
        expected["method_id"] = cell.method_id
    if cell.method_id != "planner_only" and cell.shortcut_seed is not None:
        expected["shortcut_seed"] = cell.shortcut_seed
    for key, value in expected.items():
        if result.get(key) != value:
            errors.append(f"{key} mismatch")
    if cell.phase == "inputs":
        cert = result.get("certificate") or {}
        if not (
            cert.get("strict_full_edge") is True
            and cert.get("terminal_mode") is True
            and cert.get("valid_plan") is not False
        ):
            errors.append("input certificate failed")
    else:
        source = input_path(cell.case_id, cell.path_index)
        if not source.exists() or result.get("source_sha256") != sha256(source):
            errors.append("source hash mismatch")
        if cell.phase == "offline":
            if not result.get("strict_collision_free"):
                errors.append("strict collision certificate failed")
            if result.get("valid_plan") is False:
                errors.append("native plan certificate failed")
        else:
            cert = result.get("certificate") or {}
            if not (
                cert.get("strict_full_edge") is True
                and cert.get("terminal_mode") is True
                and cert.get("valid_plan") is not False
            ):
                errors.append("online certificate failed")
            if not result.get("fixed_first_path_validated"):
                errors.append("first path was not validated")
    return errors


def preflight(case_ids: Iterable[str]) -> list[str]:
    errors = validate_protocol_definition() + verify_content_lock()
    expected_python = protocol()["software"]["python"]
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != expected_python:
        errors.append(f"Python {actual_python} != required {expected_python}")
    checkout = mr_bench_checkout()
    wanted = protocol()["software"]["mr_bench_revision"]
    actual = git_revision(checkout)
    if actual != wanted:
        errors.append(f"mr_bench revision {actual} != required {wanted}")
    dirty = tracked_changes(checkout)
    if dirty:
        errors.append(f"mr_bench has tracked changes: {dirty}")
    for case_id in case_ids:
        if protocol_cases()[case_id].get("assembly_revision"):
            try:
                assembly_checkout(case_id)
            except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
                errors.append(str(error))
    return errors


def append_audit(cell: Cell, status: str, elapsed: float, returncode: int) -> None:
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": cell.phase,
        "case_id": cell.case_id,
        "path_index": cell.path_index,
        "method_id": cell.method_id,
        "shortcut_seed": cell.shortcut_seed,
        "status": status,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "output": str(cell.output.relative_to(REPO)),
        "log": str(cell.log.relative_to(REPO)),
        "protocol_sha256": protocol_sha256(),
    }
    with AUDIT_PATH.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def execute(scheduled: list[Cell], force: bool) -> int:
    failures = []
    for index, cell in enumerate(scheduled, start=1):
        existing = artifact_errors(cell)
        if existing != ["missing"] and not existing and not force:
            print(f"[skip {index}/{len(scheduled)}] {cell.output}", flush=True)
            continue
        if existing != ["missing"] and existing and not force:
            failures.append([str(cell.output), existing])
            print(f"[invalid] {cell.output}: {existing}", flush=True)
            continue
        if "<requires-input>" in cell.command:
            failures.append([str(cell.output), ["required input is missing"]])
            print(f"[blocked] {cell.output}: required input is missing", flush=True)
            continue

        cell.output.parent.mkdir(parents=True, exist_ok=True)
        cell.log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(protocol()["rng"]["pythonhashseed"])
        checkout = assembly_checkout(cell.case_id)
        if checkout is not None:
            env["ASSEMBLY_REPO"] = str(checkout)
        started = time.perf_counter()
        with cell.log.open("w") as stream:
            completed = subprocess.run(
                list(cell.command), stdout=stream, stderr=subprocess.STDOUT, env=env
            )
        elapsed = time.perf_counter() - started
        after = artifact_errors(cell) if completed.returncode == 0 else ["cell failed"]
        ok = completed.returncode == 0 and not after
        status = "ok" if ok else "failed"
        append_audit(cell, status, elapsed, completed.returncode)
        print(
            f"[{status} {index}/{len(scheduled)}] {cell.output} ({elapsed:.3f}s)",
            flush=True,
        )
        if not ok:
            failures.append([str(cell.output), after])
    print(json.dumps({"scheduled": len(scheduled), "failures": failures}, indent=2))
    return int(bool(failures))


def add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mr-bench-repo", type=Path,
        help="mr_bench checkout; otherwise discover it from the installed package",
    )
    parser.add_argument(
        "--assembly-repo", type=Path,
        help="assembly checkout containing all required revision worktrees",
    )
    parser.add_argument("--phase", choices=("inputs", "offline", "online", "all"), default="all")
    parser.add_argument("--case", action="append", choices=tuple(protocol_cases()))
    parser.add_argument("--method", action="append", choices=tuple(methods()))
    parser.add_argument(
        "--pending-only", action="store_true",
        help="select the four methods without adopted prior-work data",
    )
    parser.add_argument("--path-index", action="append", type=int, choices=tuple(protocol()["design"]["path_indices"]))
    parser.add_argument("--shortcut-seed", action="append", type=int, choices=tuple(protocol()["design"]["shortcut_seeds"]))


def schedule_for(args, phase: str) -> list[Cell]:
    method_filter = (
        protocol()["reuse"]["pending_method_ids"]
        if args.pending_only else args.method
    )
    return cells(
        phase,
        args.case,
        method_filter,
        args.path_index,
        args.shortcut_seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="print the matrix without running it")
    add_filters(plan_parser)
    preflight_parser = subparsers.add_parser("preflight", help="validate code and revisions")
    add_filters(preflight_parser)
    execute_parser = subparsers.add_parser("execute", help="run cells serially")
    add_filters(execute_parser)
    execute_parser.add_argument("--confirm", action="store_true", help="required safety acknowledgement")
    execute_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.pending_only and args.method:
        parser.error("--pending-only cannot be combined with --method")
    if args.mr_bench_repo is not None:
        os.environ["MR_BENCH_REPO"] = str(args.mr_bench_repo.expanduser().resolve())
    if args.assembly_repo is not None:
        os.environ["ASSEMBLY_REPO"] = str(args.assembly_repo.expanduser().resolve())

    phases = ("inputs", "offline", "online") if args.phase == "all" else (args.phase,)
    case_ids = selected(args.case, protocol_cases())
    if args.command == "plan":
        by_phase = {}
        scheduled_by_phase = {phase: schedule_for(args, phase) for phase in phases}
        for phase, phase_cells in scheduled_by_phase.items():
            states = {"valid": 0, "missing": 0, "invalid": 0}
            for cell in phase_cells:
                errors = artifact_errors(cell)
                key = "valid" if not errors else ("missing" if errors == ["missing"] else "invalid")
                states[key] += 1
            by_phase[phase] = {"expected": len(phase_cells), **states}
        missing_offline = [
            cell for cell in scheduled_by_phase.get("offline", [])
            if artifact_errors(cell) == ["missing"]
        ]
        missing_online = [
            cell for cell in scheduled_by_phase.get("online", [])
            if artifact_errors(cell) == ["missing"]
        ]
        offline_seconds = sum(
            float(protocol_cases()[cell.case_id]["offline_seconds"])
            for cell in missing_offline
        )
        online_refinement_seconds = (
            len(missing_online)
            * float(protocol()["online"]["refinement_seconds_after_first_solution"])
        )
        print(json.dumps({
            "protocol": str(LOCK_PATH.relative_to(REPO)),
            "protocol_sha256": protocol_sha256(),
            "cells": by_phase,
            "expected_total": sum(value["expected"] for value in by_phase.values()),
            "missing_total": sum(value["missing"] for value in by_phase.values()),
            "missing_offline_nominal_hours": offline_seconds / 3600.0,
            "missing_online_refinement_only_hours": online_refinement_seconds / 3600.0,
            "note": (
                "Online total also includes first-solution replay and a "
                f"{protocol()['online']['replay_margin_seconds']}-second margin per missing cell."
            ),
        }, indent=2))
        return 0

    errors = preflight(case_ids)
    if errors:
        print(json.dumps({"preflight": "failed", "errors": errors}, indent=2))
        return 1
    print(json.dumps({"preflight": "ok", "cases": case_ids}, indent=2))
    if args.command == "preflight":
        return 0
    if not args.confirm:
        parser.error("execute requires --confirm; no experiments were started")

    for phase in phases:
        result = execute(schedule_for(args, phase), args.force)
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
