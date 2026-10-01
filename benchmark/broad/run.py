"""Plan, preflight, and execute the broad offline first pass serially."""
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

from broad_lib import (
    HERE,
    REPO,
    cases,
    cost_record,
    git_revision,
    implementation_sha256,
    input_path,
    log_path,
    manifest_cases,
    methods,
    protocol,
    protocol_sha256,
    read_json,
    result_path,
    sha256,
)


CELL_RUNNER = HERE / "run_offline_cell.py"
AUDIT_PATH = HERE / "execution-audit.jsonl"


@dataclass(frozen=True)
class Cell:
    case_id: str
    method_id: str
    seed: int
    output: Path
    log: Path
    command: tuple[str, ...]


def selected(values, available):
    return list(values) if values else list(available)


def cells(case_filter=None, method_filter=None, seed_filter=None) -> list[Cell]:
    output = []
    for case_id in selected(case_filter, cases()):
        seconds = float(cases()[case_id]["seconds"])
        for method_id in selected(method_filter, methods()):
            for seed in selected(seed_filter, protocol()["shortcut_seeds"]):
                result = result_path(case_id, method_id, seed)
                output.append(Cell(
                    case_id,
                    method_id,
                    seed,
                    result,
                    log_path(case_id, method_id, seed),
                    (
                        sys.executable,
                        str(CELL_RUNNER),
                        "--case", case_id,
                        "--method", method_id,
                        "--seed", str(seed),
                        "--seconds", str(seconds),
                        "--input", str(input_path(case_id)),
                        "--output", str(result),
                    ),
                ))
    return output


def artifact_errors(cell: Cell) -> list[str]:
    if not cell.output.exists():
        return ["missing"]
    try:
        result = read_json(cell.output)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    expected = {
        "schema": "mr-shortcut-broad-offline-result-v1",
        "broad_protocol_sha256": protocol_sha256(),
        "implementation_sha256": implementation_sha256(),
        "case_id": cell.case_id,
        "method_id": cell.method_id,
        "shortcut_seed": cell.seed,
        "cost": cost_record(),
        "source_sha256": sha256(input_path(cell.case_id))
        if input_path(cell.case_id).exists() else None,
    }
    errors = [key for key, value in expected.items() if result.get(key) != value]
    if result.get("strict_collision_free") is not True:
        errors.append("strict collision certificate")
    if result.get("endpoint_start_preserved") is not True:
        errors.append("start endpoint")
    if result.get("endpoint_end_preserved") is not True:
        errors.append("end endpoint")
    if result.get("terminal_mode_preserved") is not True:
        errors.append("terminal mode")
    if result.get("input_mode_runs") != result.get("output_mode_runs"):
        errors.append("mode sequence")
    if result.get("valid_plan") is False:
        errors.append("native plan")
    return errors


def assembly_checkout(source: Path) -> Path:
    wanted = protocol()["software"]["assembly_cross_revision"]
    text = subprocess.check_output(
        ["git", "-C", str(source), "worktree", "list", "--porcelain"],
        text=True,
    )
    matches = []
    current = {}
    for line in text.splitlines() + [""]:
        if line:
            key, _, value = line.partition(" ")
            current[key] = value
        elif current:
            if current.get("HEAD") == wanted:
                matches.append(Path(current["worktree"]).resolve())
            current = {}
    if len(matches) != 1:
        raise RuntimeError(f"expected one assembly worktree at {wanted}; found {matches}")
    return matches[0]


def preflight(case_ids: list[str], mr_bench: Path, assembly: Path) -> list[str]:
    errors = []
    if protocol()["execution"] != "strictly_serial":
        errors.append("protocol is not serial")
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != protocol()["software"]["python"]:
        errors.append(f"Python {actual_python} differs from protocol")
    wanted = protocol()["software"]["mr_bench_revision"]
    if git_revision(mr_bench) != wanted:
        errors.append(f"mr_bench revision differs from {wanted}")
    known = manifest_cases()
    for case_id in case_ids:
        if case_id not in known:
            errors.append(f"unknown manifest case: {case_id}")
        path = input_path(case_id)
        if not path.exists():
            errors.append(f"missing adopted input: {path}")
            continue
        source = read_json(path)
        if source.get("broad_protocol_sha256") != protocol_sha256():
            errors.append(f"stale adopted input: {path}")
    if "assembly-cross-4r-openarm" in case_ids:
        try:
            assembly_checkout(assembly)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            errors.append(str(error))
    return errors


def append_audit(cell: Cell, status: str, elapsed: float, returncode: int) -> None:
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "case_id": cell.case_id,
        "method_id": cell.method_id,
        "shortcut_seed": cell.seed,
        "status": status,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "output": str(cell.output.relative_to(REPO)),
        "log": str(cell.log.relative_to(REPO)),
        "broad_protocol_sha256": protocol_sha256(),
        "implementation_sha256": implementation_sha256(),
    }
    with AUDIT_PATH.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "preflight", "execute"))
    parser.add_argument("--case", action="append", choices=tuple(cases()))
    parser.add_argument("--method", action="append", choices=tuple(methods()))
    parser.add_argument(
        "--seed", action="append", type=int,
        choices=tuple(protocol()["shortcut_seeds"]),
    )
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--assembly-repo", required=True, type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    case_ids = selected(args.case, cases())
    scheduled = cells(args.case, args.method, args.seed)
    errors = preflight(
        case_ids,
        args.mr_bench_repo.expanduser().resolve(),
        args.assembly_repo.expanduser().resolve(),
    )
    if errors:
        print(json.dumps({"preflight": "failed", "errors": errors}, indent=2))
        return 1
    if args.command == "preflight":
        print(json.dumps({"preflight": "ok", "cases": case_ids}, indent=2))
        return 0

    states = {"valid": 0, "missing": 0, "invalid": 0}
    missing_or_invalid = []
    for cell in scheduled:
        cell_errors = artifact_errors(cell)
        key = "valid" if not cell_errors else (
            "missing" if cell_errors == ["missing"] else "invalid"
        )
        states[key] += 1
        if cell_errors:
            missing_or_invalid.append(cell)
    nominal = sum(float(cases()[cell.case_id]["seconds"]) for cell in missing_or_invalid)
    if args.command == "plan":
        print(json.dumps({
            "protocol_sha256": protocol_sha256(),
            "implementation_sha256": implementation_sha256(),
            "cells": {"expected": len(scheduled), **states},
            "missing_or_invalid_nominal_hours": nominal / 3600.0,
        }, indent=2))
        return 0

    if not args.confirm:
        parser.error("execute requires --confirm")
    assembly_worktree = assembly_checkout(args.assembly_repo.expanduser().resolve())
    failures = []
    for index, cell in enumerate(scheduled, start=1):
        existing = artifact_errors(cell)
        if not existing and not args.force:
            print(f"[skip {index}/{len(scheduled)}] {cell.output}", flush=True)
            continue
        if existing != ["missing"] and not args.force:
            failures.append([str(cell.output), existing])
            print(f"[invalid {index}/{len(scheduled)}] {cell.output}: {existing}", flush=True)
            continue
        cell.output.parent.mkdir(parents=True, exist_ok=True)
        cell.log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(
            cases()[cell.case_id].get(
                "pythonhashseed", protocol()["pythonhashseed"]
            )
        )
        env["MR_BENCH_REPO"] = str(args.mr_bench_repo.expanduser().resolve())
        if manifest_cases()[cell.case_id]["kind"] == "assembly":
            env["ASSEMBLY_REPO"] = str(assembly_worktree)
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
        print(f"[{status} {index}/{len(scheduled)}] {cell.output} ({elapsed:.3f}s)", flush=True)
        if not ok:
            failures.append([str(cell.output), after])
    print(json.dumps({"scheduled": len(scheduled), "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

