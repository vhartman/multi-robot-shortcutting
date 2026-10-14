"""Wait for the current job, then attempt remaining cells serially by a deadline.

Failures remain failures, with original logs and candidate paths retained. This
orchestrator continues other cells; it never relaxes certification or retries a
cell within its own invocation. Existing frozen worker/method definitions apply.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import sys
import time

import run as pilot
import run_sum_cost_study as summed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--mr-bench-repo", type=Path, required=True)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("--confirm is required")
    deadline = dt.datetime.fromisoformat(args.deadline).timestamp()
    wrapper = pilot.HERE / "capture_failed_cell.py"
    with (pilot.HERE / "execution.lock").open("a") as lock:
        print("[queued] Waiting for the active serial job.", flush=True)
        while True:
            if time.time() > deadline - 180:
                return 2
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(10)
        failures = []
        for track in ("sum", "max"):
            manifest = summed.definition() if track == "sum" else pilot.definition("offline")
            frozen = (summed.ROOT / "replication-manifest.json" if track == "sum"
                      else pilot.HERE / "offline" / "manifest.json")
            if pilot.read_json(frozen) != manifest:
                raise RuntimeError(f"{track}: frozen definitions changed")
            if track == "sum":
                records = []
                for cell in manifest["cells"]:
                    path = summed.output_path(cell)
                    issues = summed.result_errors(path, cell)
                    records.append((cell, "pending" if issues == ["missing"] else "invalid" if issues else "local", path))
            else:
                records = pilot.status("offline", manifest, pilot.digest(manifest))
            if any(state in ("invalid", "missing_input") for _, state, _ in records):
                raise RuntimeError(f"{track}: invalid existing data; inspect before continuing")
            pending = [(cell, path) for cell, state, path in records if state == "pending"]
            # Known rejected cells go last, so ordinary missing cells finish first.
            known = {("rai-isolated-box-stacking-r5-b4", 2, 0, "tree_group_gain"),
                     ("rai-isolated-box-stacking-r9-b4", 2, 0, "tree_then_compatible_group")}
            pending.sort(key=lambda pair: (pair[0]["case_id"], pair[0]["path_index"], pair[0]["shortcut_seed"], pair[0]["method_id"]) in known)
            print(f"[start {track}] {len(pending)} remaining cells", flush=True)
            for index, (cell, output) in enumerate(pending, 1):
                seconds = cell.get("seconds", cell.get("budget_seconds"))
                if time.time() + seconds + 180 > deadline:
                    print("[stopped] Overall deadline reserve reached.", flush=True)
                    return 2
                candidate = output.with_suffix(".candidate.json")
                if candidate.exists():
                    candidate.rename(candidate.with_name(candidate.name + f".previous-{time.time_ns()}"))
                root = summed.ROOT if track == "sum" else pilot.HERE / "offline"
                log = root / "logs" / cell["case_id"] / output.with_suffix(".log").name
                if log.exists():
                    log.rename(log.with_name(log.name + f".previous-{time.time_ns()}"))
                diagnostic = root / "diagnostics" / cell["case_id"] / f"{output.stem}-{time.time_ns()}.json"
                worker = summed.WORKER if track == "sum" else pilot.CELL_RUNNER
                options = [] if track == "sum" else ["offline", "--seconds", str(seconds)]
                command = [sys.executable, "-B", str(wrapper), str(worker), str(diagnostic), *options,
                           "--case", cell["case_id"], "--method", cell["method_id"],
                           "--path-index", str(cell["path_index"]), "--shortcut-seed", str(cell["shortcut_seed"]),
                           "--output", str(candidate)]
                started = time.monotonic()
                code = pilot.supervised(command, pilot.child_env(args.mr_bench_repo), log,
                                        timeout=min(900, max(1, deadline - time.time() - 30)))
                issues = ((summed.result_errors(candidate, cell) if track == "sum"
                           else pilot.result_errors(candidate, cell, "offline")) if code == 0 else [f"exit {code}"])
                if not issues:
                    value = pilot.read_json(candidate)
                    value["replication_manifest_sha256" if track == "sum" else "manifest_sha256"] = pilot.digest(manifest)
                    value["diagnostic_wrapper_sha256"] = pilot.sha256(wrapper)
                    value["continuation_runner_sha256"] = pilot.sha256(Path(__file__))
                    value["cell_wall_seconds"] = time.monotonic() - started
                    pilot.write_json(candidate, value)
                    candidate.replace(output)
                else:
                    failures.append({"track": track, "cell": cell, "issues": issues, "log": str(log), "diagnostic": str(diagnostic)})
                audit_path = root / ("replication-audit.jsonl" if track == "sum" else "audit.jsonl")
                with audit_path.open("a") as stream:
                    stream.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                             "cell": cell, "success": not issues, "errors": issues,
                                             "returncode": code, "elapsed_seconds": time.monotonic() - started,
                                             "diagnostic": str(diagnostic) if diagnostic.exists() else None}) + "\n")
                print(f"[{'failed' if issues else 'ok'} {track} {index}/{len(pending)}] {cell['case_id']} {output.stem}", flush=True)
        pilot.write_json(pilot.HERE / "analysis" / "continuation-failures.json", failures)
        print(json.dumps({"failed_cells": len(failures), "failures": failures}, indent=2), flush=True)
        return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
