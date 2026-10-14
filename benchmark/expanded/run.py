"""Plan, generate inputs, preflight, and run the expanded benchmark serially."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from expanded_lib import (
    ARTIFACT_ROOT,
    HERE,
    REPO,
    cases,
    checkout_for_revision,
    cost_record,
    generation_log_path,
    git_revision,
    implementation_sha256,
    input_path,
    log_path,
    methods,
    methods_for,
    protocol,
    protocol_sha256,
    read_json,
    result_path,
    sha256,
    validate_protocol,
)


CELL_RUNNER = HERE / "run_offline_cell.py"
INPUT_GENERATOR = HERE / "generate_input.py"
AUDIT_PATH = ARTIFACT_ROOT / "execution-audit.jsonl"


@dataclass(frozen=True)
class Cell:
    case_id: str
    method_id: str
    path_index: int
    shortcut_seed: int
    seconds: float
    output: Path
    log: Path


@dataclass(frozen=True)
class InputCell:
    case_id: str
    path_index: int
    seconds: float
    output: Path
    log: Path


def selected_cases(case_filter, track_filter):
    output = list(cases().values())
    if case_filter:
        wanted = set(case_filter)
        output = [case for case in output if case["id"] in wanted]
    if track_filter:
        output = [case for case in output if case["track"] == track_filter]
    return output


def offline_cells(
    selected,
    method_filter=None,
    path_filter=None,
    seed_filter=None,
) -> list[Cell]:
    output = []
    wanted_methods = set(method_filter or methods())
    wanted_paths = set(path_filter or (0, 1, 2))
    wanted_seeds = set(seed_filter or protocol()["shortcut_seeds"])
    for case in selected:
        for method_id in methods_for(case):
            if method_id not in wanted_methods:
                continue
            for path_index in case["path_indices"]:
                if path_index not in wanted_paths:
                    continue
                for seed in protocol()["shortcut_seeds"]:
                    if seed not in wanted_seeds:
                        continue
                    output.append(Cell(
                        case["id"],
                        method_id,
                        path_index,
                        seed,
                        float(case["seconds"]),
                        result_path(case["id"], method_id, path_index, seed),
                        log_path(case["id"], method_id, path_index, seed),
                    ))
    return output


def input_cells(selected, path_filter=None) -> list[InputCell]:
    wanted_paths = set(path_filter or (0, 1, 2))
    output = []
    for case in selected:
        if case["input"] != "generate":
            continue
        for path_index in case["path_indices"]:
            if path_index not in wanted_paths:
                continue
            output.append(InputCell(
                case["id"],
                path_index,
                float(case["input_generation_seconds"]),
                input_path(case["id"], path_index),
                generation_log_path(case["id"], path_index),
            ))
    return output


def input_errors(case_id: str, path_index: int) -> list[str]:
    path = input_path(case_id, path_index)
    if not path.exists():
        return ["missing"]
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    case = cases()[case_id]
    expected = {
        "schema": "mr-shortcut-expanded-input-v1",
        "expanded_protocol_sha256": protocol_sha256(),
        "case_id": case_id,
        "path_index": path_index,
        "cost": cost_record(),
        "pythonhashseed": int(case["pythonhashseed"]),
    }
    errors = [key for key, wanted in expected.items() if value.get(key) != wanted]
    if not value.get("states"):
        errors.append("empty states")
    certificate = value.get("certificate") or {}
    if certificate:
        if certificate.get("strict_full_edge") is not True:
            errors.append("strict input certificate")
        if certificate.get("terminal_mode") is not True:
            errors.append("terminal input certificate")
        if certificate.get("valid_plan") is False:
            errors.append("native input certificate")
    return errors


def artifact_errors(cell: Cell) -> list[str]:
    if not cell.output.exists():
        return ["missing"]
    try:
        value = read_json(cell.output)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    expected = {
        "schema": "mr-shortcut-expanded-offline-result-v1",
        "expanded_protocol_sha256": protocol_sha256(),
        "implementation_sha256": implementation_sha256(),
        "case_id": cell.case_id,
        "method_id": cell.method_id,
        "path_index": cell.path_index,
        "shortcut_seed": cell.shortcut_seed,
        "cost": cost_record(),
        "source_sha256": (
            sha256(input_path(cell.case_id, cell.path_index))
            if input_path(cell.case_id, cell.path_index).exists()
            else None
        ),
    }
    errors = [key for key, wanted in expected.items() if value.get(key) != wanted]
    for key in (
        "strict_collision_free",
        "endpoint_start_preserved",
        "endpoint_end_preserved",
        "terminal_mode_preserved",
    ):
        if value.get(key) is not True:
            errors.append(key)
    if value.get("input_mode_runs") != value.get("output_mode_runs"):
        errors.append("mode sequence")
    if value.get("valid_plan") is False:
        errors.append("native plan")
    return errors


def checkout_errors(selected, mr_bench: Path, assembly: Path | None):
    errors = validate_protocol()
    for relative_path, expected_digest in protocol().get("source_hashes", {}).items():
        source_path = REPO / relative_path
        if not source_path.is_file() or sha256(source_path) != expected_digest:
            errors.append(f"benchmark source differs from protocol: {relative_path}")
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != protocol()["software"]["python"]:
        errors.append(
            f"Python {actual_python} differs from "
            f"{protocol()['software']['python']}"
        )
    wanted = protocol()["software"]["mr_bench_revision"]
    if git_revision(mr_bench) != wanted:
        errors.append(f"mr_bench revision differs from {wanted}")
    assembly_cases = [case for case in selected if case["kind"] == "assembly"]
    if assembly_cases and assembly is None:
        errors.append("--assembly-repo is required for selected assembly cases")
    elif assembly is not None:
        for revision in sorted({
            case["assembly_revision"] for case in assembly_cases
        }):
            try:
                checkout_for_revision(assembly, revision)
            except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
                errors.append(str(error))
    return errors


def child_env(case, mr_bench: Path, assembly: Path | None = None):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(case["pythonhashseed"])
    paths = [str(REPO / "src"), str(mr_bench / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["MR_BENCH_REPO"] = str(mr_bench)
    if case["kind"] == "assembly":
        if assembly is None:
            raise RuntimeError("assembly repository not configured")
        env["ASSEMBLY_REPO"] = str(
            checkout_for_revision(assembly, case["assembly_revision"])
        )
    return env


def append_audit(kind: str, cell, status: str, elapsed: float, returncode: int):
    record = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": kind,
        "case_id": cell.case_id,
        "path_index": cell.path_index,
        "status": status,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "output": str(cell.output.relative_to(REPO)),
        "log": str(cell.log.relative_to(REPO)),
        "expanded_protocol_sha256": protocol_sha256(),
        "implementation_sha256": implementation_sha256(),
    }
    if isinstance(cell, Cell):
        record.update({
            "method_id": cell.method_id,
            "shortcut_seed": cell.shortcut_seed,
        })
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def state_counts(items, validator):
    counts = {"valid": 0, "missing": 0, "invalid": 0}
    pending = []
    for item in items:
        errors = validator(item)
        key = "valid" if not errors else (
            "missing" if errors == ["missing"] else "invalid"
        )
        counts[key] += 1
        if errors:
            pending.append((item, errors))
    return counts, pending


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("plan", "preflight", "generate-inputs", "execute")
    )
    parser.add_argument("--case", action="append", choices=tuple(cases()))
    parser.add_argument("--track", choices=("broad", "scaling"))
    parser.add_argument("--method", action="append", choices=tuple(methods()))
    parser.add_argument("--path-index", action="append", type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--shortcut-seed",
        action="append",
        type=int,
        choices=tuple(protocol()["shortcut_seeds"]),
    )
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--assembly-repo", type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--input-seconds", type=float)
    parser.add_argument("--input-stepsize", type=float)
    parser.add_argument("--input-home-bias", type=float)
    parser.add_argument("--input-sampler", choices=("auto", "joint", "per_robot", "gibbs", "gibbs_random_scan"))
    args = parser.parse_args()

    selected = selected_cases(args.case, args.track)
    wanted_paths = set(args.path_index or (0, 1, 2))
    mr_bench = args.mr_bench_repo.expanduser().resolve()
    assembly = (
        args.assembly_repo.expanduser().resolve()
        if args.assembly_repo is not None
        else None
    )
    errors = checkout_errors(selected, mr_bench, assembly)
    generated = input_cells(selected, args.path_index)
    if args.input_seconds is not None:
        generated = [replace(cell, seconds=max(cell.seconds, args.input_seconds)) for cell in generated]
    offline = offline_cells(
        selected,
        args.method,
        args.path_index,
        args.shortcut_seed,
    )
    input_counts, input_pending = state_counts(
        generated,
        lambda cell: input_errors(cell.case_id, cell.path_index),
    )
    offline_counts, offline_pending = state_counts(offline, artifact_errors)

    if args.command == "plan":
        missing_adopted = []
        for case in selected:
            if case["input"] != "archive":
                continue
            for path_index in case["path_indices"]:
                if path_index not in wanted_paths:
                    continue
                issue = input_errors(case["id"], path_index)
                if issue:
                    missing_adopted.append({
                        "case": case["id"],
                        "path_index": path_index,
                        "errors": issue,
                    })
        nominal = sum(cell.seconds for cell, _ in offline_pending)
        print(json.dumps({
            "protocol_sha256": protocol_sha256(),
            "implementation_sha256": implementation_sha256(),
            "checkout_errors": errors,
            "cases": len(selected),
            "inputs_to_generate": {
                "expected": len(generated),
                **input_counts,
            },
            "missing_or_invalid_adopted_inputs": missing_adopted,
            "offline_cells": {
                "expected": len(offline),
                **offline_counts,
                "missing_or_invalid_nominal_hours": nominal / 3600.0,
            },
        }, indent=2))
        return int(bool(errors))

    if errors:
        print(json.dumps({"preflight": "failed", "errors": errors}, indent=2))
        return 1

    if args.command == "generate-inputs":
        if not args.confirm:
            parser.error("generate-inputs requires --confirm")
        failures = []
        for index, (cell, existing) in enumerate(input_pending, start=1):
            if existing != ["missing"] and not args.force:
                failures.append([str(cell.output), existing])
                print(
                    f"[invalid {index}/{len(input_pending)}] "
                    f"{cell.output}: {existing}",
                    flush=True,
                )
                continue
            case = cases()[cell.case_id]
            sampler = args.input_sampler or case.get("input_sampler", "auto")
            stepsize = (
                args.input_stepsize
                if args.input_stepsize is not None
                else float(case.get("input_stepsize", 1.0))
            )
            home_bias = (
                args.input_home_bias
                if args.input_home_bias is not None
                else float(case.get("input_home_bias", 0.0))
            )
            cell.output.parent.mkdir(parents=True, exist_ok=True)
            cell.log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(INPUT_GENERATOR),
                "--case", cell.case_id,
                "--path-index", str(cell.path_index),
                "--seconds", str(cell.seconds),
                "--stepsize", str(stepsize),
                "--home-bias", str(home_bias),
                "--sampler", sampler,
                "--mr-bench-repo", str(mr_bench),
                "--output", str(cell.output),
            ]
            started = time.perf_counter()
            with cell.log.open("w") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=child_env(case, mr_bench),
                )
            elapsed = time.perf_counter() - started
            after = (
                input_errors(cell.case_id, cell.path_index)
                if completed.returncode == 0
                else ["cell failed"]
            )
            ok = completed.returncode == 0 and not after
            append_audit(
                "input-generation",
                cell,
                "ok" if ok else "failed",
                elapsed,
                completed.returncode,
            )
            print(
                f"[{'ok' if ok else 'failed'} {index}/{len(input_pending)}] "
                f"{cell.output} ({elapsed:.3f}s)",
                flush=True,
            )
            if not ok:
                failures.append([str(cell.output), after])
        print(json.dumps({
            "scheduled": len(input_pending),
            "failures": failures,
        }, indent=2))
        return int(bool(failures))

    all_input_errors = []
    for case in selected:
        for path_index in case["path_indices"]:
            if path_index not in wanted_paths:
                continue
            issue = input_errors(case["id"], path_index)
            if issue:
                all_input_errors.append({
                    "case": case["id"],
                    "path_index": path_index,
                    "errors": issue,
                })
    if all_input_errors:
        print(json.dumps({
            "preflight": "failed",
            "input_errors": all_input_errors,
        }, indent=2))
        return 1
    if args.command == "preflight":
        print(json.dumps({
            "preflight": "ok",
            "cases": [case["id"] for case in selected],
            "offline_cells": len(offline),
        }, indent=2))
        return 0

    if not args.confirm:
        parser.error("execute requires --confirm")
    failures = []
    for index, cell in enumerate(offline, start=1):
        existing = artifact_errors(cell)
        if not existing and not args.force:
            print(f"[skip {index}/{len(offline)}] {cell.output}", flush=True)
            continue
        if existing != ["missing"] and not args.force:
            failures.append([str(cell.output), existing])
            print(
                f"[invalid {index}/{len(offline)}] {cell.output}: {existing}",
                flush=True,
            )
            continue
        case = cases()[cell.case_id]
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
            "--input", str(input_path(cell.case_id, cell.path_index)),
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
        after = (
            artifact_errors(cell)
            if completed.returncode == 0
            else ["cell failed"]
        )
        ok = completed.returncode == 0 and not after
        append_audit(
            "offline",
            cell,
            "ok" if ok else "failed",
            elapsed,
            completed.returncode,
        )
        print(
            f"[{'ok' if ok else 'failed'} {index}/{len(offline)}] "
            f"{cell.output} ({elapsed:.3f}s)",
            flush=True,
        )
        if not ok:
            failures.append([str(cell.output), after])

    print(json.dumps({
        "scheduled": len(offline),
        "failures": failures,
    }, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
