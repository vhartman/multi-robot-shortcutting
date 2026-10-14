"""Prepare or explicitly execute additive offline/online comparisons, serially.

No planner or timed shortcut invocation is made by `plan` or `preflight`.
Existing base-study cells are only read. A separate manifest freezes each track.
"""
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
from pathlib import Path

import run as base
from study_lib import (HERE, REPO, cases, input_path, protocol_sha256, read_json,
                       sha256, write_json)

SETTINGS = HERE / "followups.json"
CELL_RUNNER = HERE / "followup_cells.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def mains_connected():
    indicators = []
    batteries = []
    for supply in Path("/sys/class/power_supply").glob("*"):
        kind = supply / "type"
        if kind.exists() and kind.read_text().strip() == "Battery":
            batteries.append(supply)
        elif (supply / "online").exists():
            indicators.append(supply / "online")
    # Desktops need not expose an AC adapter; laptops with unknown power fail closed.
    return any(p.read_text().strip() == "1" for p in indicators) if indicators else not batteries


def definition(track):
    settings = read_json(SETTINGS)
    if settings["base_protocol_sha256"] != protocol_sha256():
        raise RuntimeError("base protocol differs from follow-up definition")
    cells, unavailable = [], []
    if track == "offline":
        for ref in base.trial_cells(list(cases().values()), ["scalable_tree_group"]):
            issues = base.artifact_errors(ref)
            if issues:
                unavailable.append([ref.case_id, ref.path_index, ref.shortcut_seed, issues])
                continue
            cells.append({"case_id": ref.case_id, "path_index": ref.path_index,
                          "shortcut_seed": ref.shortcut_seed, "seconds": ref.seconds,
                          "method_id": "random_compatible_group",
                          "reference_sha256": sha256(ref.output)})
    else:
        for name in settings["online"]["cases"]:
            for index in settings["online"]["path_indices"]:
                input_cell = base.input_cells([cases()[name]], [index])[0]
                issues = base.input_errors(input_cell)
                if issues:
                    unavailable.append([name, index, issues])
                    continue
                source = read_json(input_path(name, index))
                for method in settings["online"]["methods"]:
                    cells.append({"case_id": name, "path_index": index,
                                  "shortcut_seed": settings["online"]["shortcut_seed"],
                                  "method_id": method,
                                  "planner_seed": source["planner_seed"],
                                  "estimated_seconds": source["planning_wall_seconds"]
                                  + settings["online"]["refinement_seconds"]})
    for cell in cells:
        cell["source_sha256"] = sha256(input_path(cell["case_id"], cell["path_index"]))
    files = [SETTINGS, CELL_RUNNER, Path(__file__), HERE / "run_cell.py",
             HERE / "run.py", HERE / "study_lib.py", HERE.parent / "online.py",
             HERE.parent / "common.py", HERE.parent / "assembly_loader.py",
             HERE.parent / "expanded" / "input_sampling.py"]
    files += sorted((REPO / "src" / "mr_shortcutting").glob("*.py"))
    return {"schema": "mr-shortcut-followup-manifest-v1", "track": track,
            "settings": settings, "study_protocol_sha256": protocol_sha256(),
            "extension_source_hashes": {str(p.relative_to(REPO)): sha256(p) for p in files},
            "cells": cells, "unavailable": unavailable}


def track_root(track):
    return HERE / "offline" / "followups" / ("online-v5" if track == "online" else track)


def output_path(track, cell):
    return track_root(track) / "results" / cell["case_id"] / (
        f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json")


def valid_output(path, manifest_hash, cell, track):
    if not path.exists():
        return False
    result = read_json(path)
    if (result.get("followup_manifest_sha256") != manifest_hash
            or result.get("source_sha256") != cell["source_sha256"]
            or any(result.get(k) != cell[k] for k in
                   ("case_id", "method_id", "path_index", "shortcut_seed"))):
        raise RuntimeError(f"stale/mismatched output, will not overwrite: {path}")
    if track == "offline":
        good = all(result.get(k) is True for k in
                   ("strict_collision_free", "endpoint_start_preserved",
                    "endpoint_end_preserved", "terminal_mode_preserved"))
        good = good and result.get("valid_plan") is not False
    else:
        cert = result.get("certificate") or {}
        pairing = result.get("first_path_pairing") or {}
        paired = (pairing.get("role") == "reference" if cell["method_id"] == "upstream_tree"
                  else pairing.get("validated") is True)
        good = (result.get("schema") == "mr-shortcut-scalable-online-pilot-v5" and paired
                and bool(result.get("first_solution", {}).get("states")) and
                cert.get("strict_full_edge") is True and cert.get("terminal_mode") is True
                and cert.get("valid_plan") is not False and bool(result.get("comparison_curve")))
    if not good:
        raise RuntimeError(f"invalid result, will not overwrite: {path}")
    return True


def supervised(command, env, log, timeout):
    if not mains_connected():
        raise RuntimeError("mains disconnected: execution is stopped, not queued")
    start = time.monotonic()
    wall_minus_mono = time.time() - start
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if not mains_connected():
                    raise RuntimeError("mains disconnected during cell; candidate is invalid")
                if abs((time.time() - time.monotonic()) - wall_minus_mono) > 2:
                    raise RuntimeError("suspend/clock jump during cell; candidate is invalid")
                if time.monotonic() - start > timeout:
                    raise RuntimeError("cell exceeded watchdog limit")
                time.sleep(0.5)
            if not mains_connected() or abs((time.time() - time.monotonic()) - wall_minus_mono) > 2:
                raise RuntimeError("power/suspend check failed at cell completion")
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
    parser.add_argument("--track", required=True, choices=("offline", "online"))
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--assembly-repo", required=True, type=Path)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--smoke-pair", action="store_true",
                        help="Run only the first online case/seed pair, with full budgets")
    args = parser.parse_args()
    if args.smoke_pair and args.track != "online":
        parser.error("--smoke-pair is only available for the online comparison")
    mrbench, assembly = args.mr_bench_repo.resolve(), args.assembly_repo.resolve()
    manifest = definition(args.track)
    selected = [cases()[name] for name in sorted({c["case_id"] for c in manifest["cells"]})]
    errors = base.preflight_errors(selected, mrbench, assembly)
    # Also freeze the actual external source files, not just their git revision.
    external = sorted((mrbench / "src" / "multi_robot_multi_goal_planning").rglob("*.py"))
    manifest["mrbench_source_hashes"] = {
        str(p.relative_to(mrbench)): sha256(p) for p in external
    }
    assembly_checkouts = {
        base.child_env(case, mrbench, assembly)["ASSEMBLY_REPO"]
        for case in selected if case["kind"] == "assembly"
    } if not errors else set()
    manifest["assembly_source_hashes"] = {
        str(p.relative_to(Path(checkout))): sha256(p)
        for checkout in sorted(assembly_checkouts)
        for p in (Path(checkout) / "mp_export.py",)
    }
    if args.track == "online" and manifest["unavailable"]:
        errors.append("online pilot has unavailable paired inputs")
    manifest_hash = digest(manifest)
    pending = [cell for cell in manifest["cells"]
               if not valid_output(output_path(args.track, cell), manifest_hash, cell, args.track)]
    if args.smoke_pair:
        first_cell = manifest["cells"][0]
        pending = [c for c in pending if (c["case_id"], c["path_index"]) ==
                   (first_cell["case_id"], first_cell["path_index"])]
    nominal = sum(c.get("seconds", c.get("estimated_seconds", 0)) for c in pending)
    print(json.dumps({"track": args.track, "cells": len(manifest["cells"]),
                      "pending": len(pending), "unavailable": manifest["unavailable"],
                      "estimated_hours_excluding_setup_and_certification": nominal / 3600,
                      "preflight_errors": errors, "mains_connected": mains_connected()}, indent=2), flush=True)
    if errors or args.command != "execute":
        return int(bool(errors))
    if not args.confirm:
        parser.error("execution requires --confirm")
    if not mains_connected():
        raise RuntimeError("mains disconnected; nothing started")
    root = HERE / "offline" / "followups"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = track_root(args.track) / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise RuntimeError("frozen follow-up manifest changed; refusing mixed resume")
        if not manifest_path.exists():
            write_json(manifest_path, manifest)
        failures = []
        for index, cell in enumerate(pending, 1):
            output = output_path(args.track, cell)
            output.parent.mkdir(parents=True, exist_ok=True)
            candidate = output.with_suffix(".candidate.json")
            if candidate.exists():
                # Preserve previous failed candidate rather than treating it as success.
                candidate.rename(candidate.with_name(candidate.name + f".previous-{time.time_ns()}"))
            command = [sys.executable, "-B", str(CELL_RUNNER), args.track,
                       "--case", cell["case_id"], "--method", cell["method_id"],
                       "--input", str(input_path(cell["case_id"], cell["path_index"])),
                       "--output", str(candidate)]
            if args.track == "offline":
                command += ["--path-index", str(cell["path_index"]),
                            "--shortcut-seed", str(cell["shortcut_seed"]),
                            "--seconds", str(cell["seconds"])]
            elif cell["method_id"] != "upstream_tree":
                reference_cell = dict(cell, method_id="upstream_tree")
                reference_path = output_path(args.track, reference_cell)
                if not valid_output(reference_path, manifest_hash, reference_cell, args.track):
                    raise RuntimeError("paired online baseline must finish first")
                command += ["--reference", str(reference_path)]
            env = base.child_env(cases()[cell["case_id"]], mrbench, assembly)
            start = time.monotonic()
            code = supervised(command, env, output.with_suffix(".log"), timeout=600)
            success = code == 0 and candidate.exists()
            if success:
                result = read_json(candidate)
                result.update(followup_manifest_sha256=manifest_hash,
                              source_sha256=cell["source_sha256"],
                              study_protocol_sha256=protocol_sha256(),
                              cell_wall_seconds=time.monotonic() - start)
                write_json(candidate, result)
                valid_output(candidate, manifest_hash, cell, args.track)
                candidate.replace(output)
            else:
                failures.append(cell)
            with (track_root(args.track) / "audit.jsonl").open("a") as stream:
                stream.write(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                         "cell": cell, "returncode": code, "success": success,
                                         "elapsed_seconds": time.monotonic() - start}) + "\n")
            print(f"[{'ok' if success else 'failed'} {index}/{len(pending)}] {output}", flush=True)
            if not success:
                print("[stopped] First failed cell; inspect its log before resuming.", flush=True)
                return 1
        return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
