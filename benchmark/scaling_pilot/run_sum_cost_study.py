"""Replicate the sum-cost comparison serially, preserving the initial pilot."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
import time
from pathlib import Path

import run as pilot

HERE = Path(__file__).resolve().parent
ROOT = HERE / "sum-cost"
PROTOCOL = HERE / "sum_cost_protocol.json"
WORKER = HERE / "run_sum_cost.py"


def definition():
    data = pilot.read_json(PROTOCOL)
    methods = pilot.methods()
    cells = []
    for case_id in data["case_ids"]:
        family = pilot.cases()[case_id]["family"]
        for path_index in data["path_indices"]:
            source = pilot.input_path(case_id, path_index)
            for seed in data["shortcut_seeds"]:
                for method_id, method in methods.items():
                    cells.append({
                        "case_id": case_id, "method_id": method_id,
                        "implementation_policy": method["policy"],
                        "path_index": path_index, "shortcut_seed": seed,
                        "source_sha256": pilot.sha256(source),
                        "budget_seconds": pilot.protocol()["offline"]["seconds_by_family"][family],
                        "cost": data["cost"],
                    })
    files = (PROTOCOL, WORKER, Path(__file__), pilot.CORE, pilot.SCALABLE,
             pilot.BENCHMARK / "all_scaling" / "study_lib.py",
             pilot.BENCHMARK / "common.py")
    return {"schema": data["schema"], "cells": cells,
            "source_hashes": {str(path.relative_to(pilot.REPO)): pilot.sha256(path) for path in files}}


def output_path(cell):
    return (ROOT / "results" / cell["case_id"] /
            f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json")


def result_errors(path, cell):
    if not path.is_file():
        return ["missing"]
    try:
        result = pilot.read_json(path)
    except (OSError, ValueError) as error:
        return [str(error)]
    errors = [key for key, value in cell.items() if result.get(key) != value]
    for key in ("strict_collision_free", "endpoints_preserved", "terminal_mode_preserved"):
        if result.get(key) is not True:
            errors.append(key)
    if result.get("suspend_detected") is not False:
        errors.append("suspend_detected")
    if result.get("valid_plan") is False:
        errors.append("valid_plan")
    if result.get("input_mode_runs") != result.get("output_mode_runs"):
        errors.append("mode sequence")
    initial, final = float(result.get("initial_cost", -1)), float(result.get("final_cost", -1))
    if not (math.isfinite(initial) and math.isfinite(final) and 0 <= final <= initial + 1e-7 * max(1, initial)):
        errors.append("cost")
    if not result.get("trace"):
        errors.append("trace")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "execute"))
    parser.add_argument("--mr-bench-repo", type=Path, required=True)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--max-hours", type=float, default=3)
    args = parser.parse_args()
    manifest = definition()
    errors, pending, reused = [], [], []
    frozen = ROOT / "replication-manifest.json"
    if frozen.exists() and pilot.read_json(frozen) != manifest:
        errors.append("frozen sum-cost replication manifest changed")
    data = pilot.read_json(PROTOCOL)
    for case_id in data["case_ids"]:
        for index in data["path_indices"]:
            errors.extend(f"{case_id}/path{index}: {issue}" for issue in pilot.scaling.input_errors(case_id, index))
    if pilot.scaling.git_revision(args.mr_bench_repo) != pilot.scaling.protocol()["software"]["mr_bench_revision"]:
        errors.append("mr-bench revision differs from saved inputs")
    if ".".join(map(str, sys.version_info[:3])) != pilot.scaling.protocol()["software"]["python"]:
        errors.append("Python version differs from saved-input protocol")
    for cell in manifest["cells"]:
        path = output_path(cell)
        issues = result_errors(path, cell)
        if issues == ["missing"]:
            pending.append(cell)
        elif issues:
            errors.append(f"{path}: {issues}")
        else:
            reused.append(path)
    print(json.dumps({"expected": len(manifest["cells"]), "reused": len(reused),
                      "pending": len(pending), "pending_nominal_hours": sum(c["budget_seconds"] for c in pending) / 3600,
                      "errors": errors, "mains_connected": pilot.mains_connected()}, indent=2), flush=True)
    if errors or args.command == "plan":
        return int(bool(errors))
    if not args.confirm:
        parser.error("execute requires --confirm")
    with (HERE / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not pilot.mains_connected():
            raise RuntimeError("mains disconnected")
        if not frozen.exists():
            pilot.write_json(frozen, manifest)
        deadline = time.monotonic() + args.max_hours * 3600
        for index, cell in enumerate(pending, 1):
            if time.monotonic() + cell["budget_seconds"] + 120 > deadline:
                print("[stopped] execution limit reserve reached", flush=True)
                return 2
            output = output_path(cell)
            candidate = output.with_suffix(".candidate.json")
            if candidate.exists():
                candidate.rename(candidate.with_name(candidate.name + f".previous-{time.time_ns()}"))
            command = [sys.executable, "-B", str(WORKER), "--case", cell["case_id"],
                       "--method", cell["method_id"], "--path-index", str(cell["path_index"]),
                       "--shortcut-seed", str(cell["shortcut_seed"]), "--output", str(candidate)]
            log = ROOT / "logs" / cell["case_id"] / output.with_suffix(".log").name
            started = time.monotonic()
            code = pilot.supervised(command, pilot.child_env(args.mr_bench_repo), log, timeout=900)
            issues = result_errors(candidate, cell) if code == 0 else [f"exit {code}"]
            if not issues:
                value = pilot.read_json(candidate)
                value["replication_manifest_sha256"] = pilot.digest(manifest)
                value["cell_wall_seconds"] = time.monotonic() - started
                pilot.write_json(candidate, value)
                candidate.replace(output)
            with (ROOT / "replication-audit.jsonl").open("a") as stream:
                stream.write(json.dumps({"cell": cell, "errors": issues, "elapsed_seconds": time.monotonic() - started}) + "\n")
            print(f"[{'failed' if issues else 'ok'} {index}/{len(pending)}] {output.name} {cell['case_id']}", flush=True)
            if issues:
                raise RuntimeError(f"cell failed: {issues}; inspect {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
