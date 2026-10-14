"""Plan, preflight, and execute the frozen all-scaling comparison serially."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (REPO / "src", BENCHMARK, HERE):
    sys.path.insert(0, str(directory))

from study_lib import (  # noqa: E402
    Cell,
    cases,
    cells,
    definition_errors,
    failure_errors,
    input_path,
    local_failure_path,
    local_result_path,
    log_path,
    methods,
    protocol,
    protocol_sha256,
    read_json,
    result_errors,
    reusable_result,
    sha256,
    source_hashes,
    write_json,
)


MANIFEST_PATH = HERE / "execution-manifest.json"
AUDIT_PATH = HERE / "execution-audit.jsonl"
LOCK_PATH = HERE / "offline" / "execution.lock"
CELL_RUNNER = HERE / "run_cell.py"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def mains_connected() -> bool:
    indicators, batteries = [], []
    for supply in Path("/sys/class/power_supply").glob("*"):
        kind = supply / "type"
        if kind.exists() and kind.read_text().strip() == "Battery":
            batteries.append(supply)
        elif (supply / "online").exists():
            indicators.append(supply / "online")
    return any(path.read_text().strip() == "1" for path in indicators) if indicators else not batteries


def selected_cells(args) -> list[Cell]:
    selected = cells()
    if args.family:
        wanted = set(args.family)
        selected = [cell for cell in selected if cell.family in wanted]
    if args.case:
        wanted = set(args.case)
        selected = [cell for cell in selected if cell.case_id in wanted]
    if args.method:
        wanted = set(args.method)
        selected = [cell for cell in selected if cell.method_id in wanted]
    if args.path_index:
        wanted = set(args.path_index)
        selected = [cell for cell in selected if cell.path_index in wanted]
    if args.shortcut_seed:
        wanted = set(args.shortcut_seed)
        selected = [cell for cell in selected if cell.shortcut_seed in wanted]
    return selected


def make_definition() -> dict:
    records = []
    for cell in cells():
        reused = reusable_result(cell)
        records.append({
            "case_id": cell.case_id,
            "family": cell.family,
            "method_id": cell.method_id,
            "path_index": cell.path_index,
            "shortcut_seed": cell.shortcut_seed,
            "seconds": cell.seconds,
            "source_sha256": sha256(input_path(cell.case_id, cell.path_index)),
            "reused_result": str(reused.relative_to(REPO)) if reused else None,
            "reused_result_sha256": sha256(reused) if reused else None,
        })
    return {
        "schema": "mr-shortcut-all-scaling-execution-v1",
        "protocol_sha256": protocol_sha256(),
        "source_hashes": source_hashes(),
        "cells": records,
    }


def record_map(definition):
    return {
        (record["case_id"], record["method_id"], record["path_index"], record["shortcut_seed"]): record
        for record in definition["cells"]
    }


def selected_status(selected, definition, manifest_hash):
    records = record_map(definition)
    status = []
    for cell in selected:
        record = records[(cell.case_id, cell.method_id, cell.path_index, cell.shortcut_seed)]
        local = local_result_path(cell.case_id, cell.method_id, cell.path_index, cell.shortcut_seed)
        failure = local_failure_path(cell.case_id, cell.method_id, cell.path_index, cell.shortcut_seed)
        local_errors = result_errors(local, cell, manifest_hash)
        if not local_errors:
            state, path = "local", local
        elif record["reused_result"]:
            path = REPO / record["reused_result"]
            state = "reused" if not result_errors(path, cell) and sha256(path) == record["reused_result_sha256"] else "invalid"
        elif local_errors != ["missing"]:
            state, path = "invalid", local
        elif not failure_errors(failure, cell, manifest_hash):
            state, path = "failed", failure
        elif failure_errors(failure, cell, manifest_hash) != ["missing"]:
            state, path = "invalid", failure
        else:
            state, path = "pending", local
        status.append((cell, state, path, local_errors))
    return status


def child_env(mr_bench: Path):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(protocol()["pythonhashseed"])
    env["MPLCONFIGDIR"] = "/tmp/mr-shortcutting-matplotlib"
    paths = [str(REPO / "src"), str(BENCHMARK), str(mr_bench / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def supervised(command, env, log, timeout):
    if not mains_connected():
        raise RuntimeError("mains disconnected: execution stopped")
    started = time.monotonic()
    wall_minus_mono = time.time() - started
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if not mains_connected():
                    raise RuntimeError("mains disconnected during cell")
                if abs((time.time() - time.monotonic()) - wall_minus_mono) > 2:
                    raise RuntimeError("suspend/clock jump during cell")
                if time.monotonic() - started > timeout:
                    raise RuntimeError("cell exceeded watchdog limit")
                time.sleep(0.5)
            if not mains_connected():
                raise RuntimeError("mains disconnected at cell completion")
            if abs((time.time() - time.monotonic()) - wall_minus_mono) > 2:
                raise RuntimeError("suspend/clock jump at cell completion")
            return process.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "preflight", "execute"))
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--family", action="append", choices=tuple(protocol()["expected_case_counts"]))
    parser.add_argument("--case", action="append", choices=tuple(cases()))
    parser.add_argument("--method", action="append", choices=tuple(methods()))
    parser.add_argument("--path-index", action="append", type=int, choices=tuple(protocol()["path_indices"]))
    parser.add_argument("--shortcut-seed", action="append", type=int, choices=tuple(protocol()["shortcut_seeds"]))
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    mr_bench = args.mr_bench_repo.resolve()
    definition = make_definition()
    manifest_hash = digest(definition)
    selected = selected_cells(args)
    errors = definition_errors(mr_bench)
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    if actual_python != protocol()["software"]["python"]:
        errors.append(f"Python version differs: {actual_python}")
    statuses = selected_status(selected, definition, manifest_hash)
    invalid = [(cell, path, issues) for cell, state, path, issues in statuses if state == "invalid"]
    if invalid:
        errors.extend(f"invalid result: {path}" for _, path, _ in invalid)
    counts = Counter(state for _, state, _, _ in statuses)
    pending = [cell for cell, state, _, _ in statuses if state == "pending"]
    summary = {
        "protocol_sha256": protocol_sha256(),
        "execution_manifest_sha256": manifest_hash,
        "cases": len({cell.case_id for cell in selected}),
        "cells": len(selected),
        "states": dict(counts),
        "pending_nominal_hours": sum(cell.seconds for cell in pending) / 3600,
        "pending_by_family": dict(Counter(cell.family for cell in pending)),
        "pending_by_method": dict(Counter(cell.method_id for cell in pending)),
        "preflight_errors": errors,
        "mains_connected": mains_connected(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if errors or args.command != "execute":
        return int(bool(errors))
    if not args.confirm:
        parser.error("execute requires --confirm")
    if MANIFEST_PATH.exists():
        if read_json(MANIFEST_PATH) != definition:
            raise RuntimeError("execution manifest changed; refusing mixed resume")
    else:
        write_json(MANIFEST_PATH, definition)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, cell in enumerate(pending, 1):
            output = local_result_path(
                cell.case_id, cell.method_id, cell.path_index, cell.shortcut_seed
            )
            failure = local_failure_path(
                cell.case_id, cell.method_id, cell.path_index, cell.shortcut_seed
            )
            log = log_path(
                cell.case_id, cell.method_id, cell.path_index, cell.shortcut_seed
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            log.parent.mkdir(parents=True, exist_ok=True)
            candidate = output.with_suffix(".candidate.json")
            if candidate.exists():
                candidate.rename(candidate.with_name(candidate.name + f".previous-{time.time_ns()}"))
            command = [
                sys.executable,
                "-B",
                str(CELL_RUNNER),
                "--case", cell.case_id,
                "--method", cell.method_id,
                "--path-index", str(cell.path_index),
                "--shortcut-seed", str(cell.shortcut_seed),
                "--seconds", str(cell.seconds),
                "--manifest-sha256", manifest_hash,
                "--output", str(candidate),
            ]
            started = time.monotonic()
            code = supervised(command, child_env(mr_bench), log, timeout=900)
            elapsed = time.monotonic() - started
            success = code == 0 and candidate.exists()
            issues = result_errors(candidate, cell, manifest_hash) if success else ["cell failed"]
            success = success and not issues
            if success:
                candidate.replace(output)
            certificate_failure = (
                code != 0
                and candidate.exists()
                and not failure_errors(candidate, cell, manifest_hash)
            )
            if certificate_failure:
                failure.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(failure)
            with AUDIT_PATH.open("a") as stream:
                stream.write(json.dumps({
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "case_id": cell.case_id,
                    "family": cell.family,
                    "method_id": cell.method_id,
                    "path_index": cell.path_index,
                    "shortcut_seed": cell.shortcut_seed,
                    "seconds": cell.seconds,
                    "elapsed_seconds": elapsed,
                    "returncode": code,
                    "success": success,
                    "certificate_failure": certificate_failure,
                    "issues": issues,
                    "execution_manifest_sha256": manifest_hash,
                }) + "\n")
            print(
                f"[{'ok' if success else 'certificate-failed' if certificate_failure else 'failed'} "
                f"{index}/{len(pending)}] "
                f"{cell.case_id} {cell.method_id} path{cell.path_index} "
                f"seed{cell.shortcut_seed} ({elapsed:.2f}s)",
                flush=True,
            )
            if not success and not certificate_failure:
                print(f"[stopped] inspect {log}: {issues}", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
