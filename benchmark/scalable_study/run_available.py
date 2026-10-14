"""Run every currently certified scalable-study path strictly serially."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run import input_cells, input_errors, preflight_errors  # noqa: E402
from study_lib import cases, protocol  # noqa: E402


def mains_connected() -> bool:
    roots = (Path("/sys/class/power_supply").glob("AC*/online"),
             Path("/sys/class/power_supply").glob("ADP*/online"))
    indicators = [path for group in roots for path in group]
    return not indicators or any(path.read_text().strip() == "1" for path in indicators)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--assembly-repo", required=True, type=Path)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("execution requires --confirm")

    mr_bench = args.mr_bench_repo.expanduser().resolve()
    assembly = args.assembly_repo.expanduser().resolve()
    selected = list(cases().values())
    errors = preflight_errors(selected, mr_bench, assembly)
    if errors:
        raise RuntimeError(f"preflight failed: {errors}")

    scheduled = 0
    skipped_inputs = []
    failures = []
    total = sum(len(case["path_indices"]) for case in selected)
    index = 0
    for case in selected:
        for path_index in case["path_indices"]:
            index += 1
            cell = input_cells([case], [path_index])[0]
            issues = input_errors(cell)
            if issues:
                skipped_inputs.append((case["id"], path_index, issues))
                print(
                    f"[unavailable {index}/{total}] {case['id']} path{path_index}: {issues}",
                    flush=True,
                )
                continue
            if not mains_connected():
                print("[stopped] mains power is disconnected", flush=True)
                return 2
            command = [
                sys.executable,
                str(HERE / "run.py"),
                "execute",
                "--confirm",
                "--case", case["id"],
                "--path-index", str(path_index),
                "--mr-bench-repo", str(mr_bench),
                "--assembly-repo", str(assembly),
            ]
            print(f"[run {index}/{total}] {case['id']} path{path_index}", flush=True)
            completed = subprocess.run(command)
            scheduled += len(protocol()["shortcut_seeds"]) * len(protocol()["methods"])
            if completed.returncode != 0:
                failures.append((case["id"], path_index, completed.returncode))
    print({
        "scheduled_cells": scheduled,
        "unavailable_inputs": skipped_inputs,
        "failed_path_runs": failures,
    }, flush=True)
    if not mains_connected():
        return 2
    completed = subprocess.run([sys.executable, str(HERE / "analyze.py"), "--allow-partial"])
    return completed.returncode or int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
