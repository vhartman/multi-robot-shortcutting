"""Plan, generate inputs, preflight, and run the scalable study serially."""
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

from study_lib import (
    BENCHMARK,
    HERE,
    REPO,
    cases,
    checkout_for_revision,
    git_revision,
    input_path,
    local_input_path,
    log_path,
    methods,
    protocol,
    protocol_sha256,
    read_json,
    result_path,
    sha256,
    validate_definition,
)


INPUT_GENERATOR = HERE / "generate_input.py"
CELL_RUNNER = HERE / "run_cell.py"
AUDIT_PATH = HERE / "execution-audit.jsonl"


@dataclass(frozen=True)
class InputCell:
    case_id: str
    path_index: int
    output: Path
    log: Path


@dataclass(frozen=True)
class TrialCell:
    case_id: str
    method_id: str
    path_index: int
    shortcut_seed: int
    seconds: float
    output: Path
    log: Path


def selected_cases(case_filter, family_filter):
    output = list(cases().values())
    if case_filter:
        wanted = set(case_filter)
        output = [case for case in output if case["id"] in wanted]
    if family_filter:
        wanted = set(family_filter)
        output = [case for case in output if case["family"] in wanted]
    return output


def input_cells(selected, path_filter=None):
    wanted = set(path_filter or protocol()["path_indices"])
    return [
        InputCell(
            case["id"],
            path_index,
            local_input_path(case["id"], path_index),
            HERE / "offline" / "logs" / "input-generation"
            / f"{case['id']}-path{path_index}.log",
        )
        for case in selected
        for path_index in case["path_indices"]
        if path_index in wanted
    ]


def trial_cells(selected, method_filter=None, path_filter=None, seed_filter=None):
    wanted_methods = set(method_filter or methods())
    wanted_paths = set(path_filter or protocol()["path_indices"])
    wanted_seeds = set(seed_filter or protocol()["shortcut_seeds"])
    return [
        TrialCell(
            case["id"],
            method_id,
            path_index,
            seed,
            float(case["seconds"]),
            result_path(case["id"], method_id, path_index, seed),
            log_path(case["id"], method_id, path_index, seed),
        )
        for case in selected
        for path_index in case["path_indices"]
        if path_index in wanted_paths
        for method_id in methods()
        if method_id in wanted_methods
        for seed in protocol()["shortcut_seeds"]
        if seed in wanted_seeds
    ]


def input_errors(cell: InputCell) -> list[str]:
    source = input_path(cell.case_id, cell.path_index)
    if not source.exists():
        return ["missing"]
    try:
        value = read_json(source)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    errors = []
    if value.get("case_id") != cell.case_id:
        errors.append("case_id")
    if value.get("path_index") != cell.path_index:
        errors.append("path_index")
    if not value.get("states"):
        errors.append("states")
    certificate = value.get("certificate") or {}
    if certificate.get("strict_full_edge") is not True:
        errors.append("strict certificate")
    if certificate.get("terminal_mode") is not True:
        errors.append("terminal certificate")
    if certificate.get("valid_plan") is False:
        errors.append("native certificate")
    if source == cell.output:
        if value.get("schema") != "mr-shortcut-scalable-study-input-v1":
            errors.append("schema")
        if value.get("study_protocol_sha256") != protocol_sha256():
            errors.append("protocol hash")
    return errors


def artifact_errors(cell: TrialCell) -> list[str]:
    if not cell.output.exists():
        return ["missing"]
    try:
        value = read_json(cell.output)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    source = input_path(cell.case_id, cell.path_index)
    expected = {
        "schema": "mr-shortcut-scalable-study-result-v1",
        "study_protocol_sha256": protocol_sha256(),
        "case_id": cell.case_id,
        "method_id": cell.method_id,
        "path_index": cell.path_index,
        "shortcut_seed": cell.shortcut_seed,
        "source_sha256": sha256(source) if source.exists() else None,
    }
    errors = [key for key, wanted in expected.items() if value.get(key) != wanted]
    for key in (
        "endpoint_start_preserved",
        "endpoint_end_preserved",
        "terminal_mode_preserved",
        "strict_collision_free",
    ):
        if value.get(key) is not True:
            errors.append(key)
    if value.get("input_mode_runs") != value.get("output_mode_runs"):
        errors.append("mode sequence")
    if value.get("valid_plan") is False:
        errors.append("native plan")
    return errors


def preflight_errors(selected, mr_bench: Path, assembly: Path | None):
    errors = validate_definition()
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != protocol()["software"]["python"]:
        errors.append(f"Python version differs: {actual_python}")
    if git_revision(mr_bench) != protocol()["software"]["mr_bench_revision"]:
        errors.append("mr_bench revision differs from protocol")
    assembly_cases = [case for case in selected if case["kind"] == "assembly"]
    if assembly_cases and assembly is None:
        errors.append("--assembly-repo is required for assembly cases")
    elif assembly is not None:
        for revision in sorted({case["assembly_revision"] for case in assembly_cases}):
            try:
                checkout_for_revision(assembly, revision)
            except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
                errors.append(str(error))
    return errors


def child_env(case, mr_bench: Path, assembly: Path | None):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(protocol()["pythonhashseed"])
    paths = [str(REPO / "src"), str(BENCHMARK), str(mr_bench / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["MR_BENCH_REPO"] = str(mr_bench)
    if case["kind"] == "assembly":
        if assembly is None:
            raise RuntimeError("assembly repository is required")
        env["ASSEMBLY_REPO"] = str(
            checkout_for_revision(assembly, case["assembly_revision"])
        )
    return env


def append_audit(phase, cell, status, elapsed, returncode):
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": phase,
        "case_id": cell.case_id,
        "path_index": cell.path_index,
        "status": status,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "output": str(cell.output.relative_to(REPO)),
        "log": str(cell.log.relative_to(REPO)),
        "study_protocol_sha256": protocol_sha256(),
    }
    if isinstance(cell, TrialCell):
        record.update({
            "method_id": cell.method_id,
            "shortcut_seed": cell.shortcut_seed,
        })
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def summarize(items, validator):
    counts = {"valid": 0, "missing": 0, "invalid": 0}
    pending = []
    for item in items:
        errors = validator(item)
        state = "valid" if not errors else ("missing" if errors == ["missing"] else "invalid")
        counts[state] += 1
        if errors:
            pending.append((item, errors))
    return counts, pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "preflight", "generate-inputs", "execute"))
    parser.add_argument("--case", action="append", choices=tuple(cases()))
    parser.add_argument(
        "--family",
        action="append",
        choices=("broad", "isolated_box_stacking", "disk_scaling", "coupled_box_stacking"),
    )
    parser.add_argument("--method", action="append", choices=tuple(methods()))
    parser.add_argument("--path-index", action="append", type=int, choices=tuple(protocol()["path_indices"]))
    parser.add_argument("--shortcut-seed", action="append", type=int, choices=tuple(protocol()["shortcut_seeds"]))
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--assembly-repo", type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = selected_cases(args.case, args.family)
    mr_bench = args.mr_bench_repo.expanduser().resolve()
    assembly = args.assembly_repo.expanduser().resolve() if args.assembly_repo else None
    errors = preflight_errors(selected, mr_bench, assembly)
    inputs = input_cells(selected, args.path_index)
    trials = trial_cells(selected, args.method, args.path_index, args.shortcut_seed)
    input_counts, input_pending = summarize(inputs, input_errors)
    trial_counts, trial_pending = summarize(trials, artifact_errors)

    if args.command == "plan":
        nominal = sum(cell.seconds for cell, _ in trial_pending)
        by_family = {}
        for family in sorted({case["family"] for case in selected}):
            family_cases = [case for case in selected if case["family"] == family]
            family_trials = trial_cells(
                family_cases, args.method, args.path_index, args.shortcut_seed
            )
            by_family[family] = {
                "cases": len(family_cases),
                "cells": len(family_trials),
                "nominal_shortcut_hours": sum(cell.seconds for cell in family_trials) / 3600,
            }
        print(json.dumps({
            "protocol_sha256": protocol_sha256(),
            "preflight_errors": errors,
            "cases": len(selected),
            "families": by_family,
            "inputs": {"expected": len(inputs), **input_counts},
            "shortcut_cells": {"expected": len(trials), **trial_counts},
            "missing_or_invalid_nominal_shortcut_hours": nominal / 3600,
        }, indent=2))
        return int(bool(errors))

    if errors:
        print(json.dumps({"preflight": "failed", "errors": errors}, indent=2))
        return 1
    if args.command == "preflight":
        print(json.dumps({
            "preflight": "ok",
            "cases": len(selected),
            "inputs": {"expected": len(inputs), **input_counts},
            "shortcut_cells": {"expected": len(trials), **trial_counts},
        }, indent=2))
        return 0

    if not args.confirm:
        parser.error(f"{args.command} requires --confirm")
    if args.command == "generate-inputs":
        failures = []
        for index, (cell, existing) in enumerate(input_pending, start=1):
            source = input_path(cell.case_id, cell.path_index)
            replacing_local = source == cell.output and source.exists()
            if existing != ["missing"] and replacing_local and not args.force:
                failures.append([str(input_path(cell.case_id, cell.path_index)), existing])
                print(f"[invalid {index}/{len(input_pending)}] {cell.case_id} path{cell.path_index}: {existing}", flush=True)
                continue
            case = cases()[cell.case_id]
            cell.output.parent.mkdir(parents=True, exist_ok=True)
            cell.log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(INPUT_GENERATOR),
                "--case", cell.case_id,
                "--path-index", str(cell.path_index),
                "--output", str(cell.output),
                "--mr-bench-repo", str(mr_bench),
            ]
            started = time.perf_counter()
            with cell.log.open("w") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=child_env(case, mr_bench, assembly),
                )
            elapsed = time.perf_counter() - started
            after = input_errors(cell) if completed.returncode == 0 else ["cell failed"]
            ok = completed.returncode == 0 and not after
            append_audit("input-generation", cell, "ok" if ok else "failed", elapsed, completed.returncode)
            print(f"[{'ok' if ok else 'failed'} {index}/{len(input_pending)}] {cell.case_id} path{cell.path_index} ({elapsed:.2f}s)", flush=True)
            if not ok:
                failures.append([str(cell.output), after])
        print(json.dumps({"scheduled": len(input_pending), "failures": failures}, indent=2))
        return int(bool(failures))

    input_failures = [
        [cell.case_id, cell.path_index, issue]
        for cell in inputs
        if (issue := input_errors(cell))
    ]
    if input_failures:
        print(json.dumps({"execute": "blocked", "input_errors": input_failures}, indent=2))
        return 1

    failures = []
    for index, cell in enumerate(trials, start=1):
        existing = artifact_errors(cell)
        if not existing and not args.force:
            print(f"[skip {index}/{len(trials)}] {cell.output}", flush=True)
            continue
        if existing != ["missing"] and not args.force:
            failures.append([str(cell.output), existing])
            print(f"[invalid {index}/{len(trials)}] {cell.output}: {existing}", flush=True)
            continue
        case = cases()[cell.case_id]
        source = input_path(cell.case_id, cell.path_index)
        cell.output.parent.mkdir(parents=True, exist_ok=True)
        cell.log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(CELL_RUNNER),
            "--case", cell.case_id,
            "--method", cell.method_id,
            "--path-index", str(cell.path_index),
            "--shortcut-seed", str(cell.shortcut_seed),
            "--seconds", str(cell.seconds),
            "--input", str(source),
            "--output", str(cell.output),
        ]
        started = time.perf_counter()
        with cell.log.open("w") as stream:
            completed = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=child_env(case, mr_bench, assembly),
            )
        elapsed = time.perf_counter() - started
        after = artifact_errors(cell) if completed.returncode == 0 else ["cell failed"]
        ok = completed.returncode == 0 and not after
        append_audit("offline", cell, "ok" if ok else "failed", elapsed, completed.returncode)
        print(f"[{'ok' if ok else 'failed'} {index}/{len(trials)}] {cell.case_id} {cell.method_id} path{cell.path_index} seed{cell.shortcut_seed} ({elapsed:.2f}s)", flush=True)
        if not ok:
            failures.append([str(cell.output), after])
    print(json.dumps({"scheduled": len(trials), "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
