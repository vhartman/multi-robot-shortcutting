"""Plan, preflight, and execute the frozen scaling pilot serially."""
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
for directory in (REPO / "src", BENCHMARK, BENCHMARK / "all_scaling",
                  BENCHMARK / "scaling_2d", HERE):
    sys.path.insert(0, str(directory))

import study_lib as scaling  # noqa: E402

CELL_RUNNER = HERE / "cell.py"
PROTOCOL_PATH = HERE / "protocol.json"
CORE = REPO / "src" / "mr_shortcutting" / "shortcutters.py"
SCALABLE = BENCHMARK / "scaling_2d" / "scalable_tree_group.py"
REFERENCE_ROOTS = (
    BENCHMARK / "all_scaling" / "offline" / "results",
    BENCHMARK / "all_scaling" / "offline" / "tree-then-compatible-group" / "results",
    BENCHMARK / "scalable_study" / "offline" / "results",
    BENCHMARK / "expanded" / "offline" / "results",
    BENCHMARK / "scaling_2d" / "offline" / "results",
    BENCHMARK / "scaling_mobile_wall" / "offline" / "results",
)
RETIRED_CASE_IDS = {"rai-disk-scaling-r1-g4"}
ALLOWED_EXTENSION_SOURCE_TRANSITIONS = {
    # The completed 24-case run predates the separately frozen online RNG-order
    # diagnostic. Retain that run under its archived manifest, while requiring
    # the exact post-diagnostic cell runner for the three added coupled cases.
    "benchmark/scaling_pilot/cell.py": {
        (
            "dd6a4493b57ef27532796722b7fcb1b1b25f47a5309908ad5bdb96057d982014",
            "878eb8314fd10eb7314869edaad8f7137f2fe8fd26e34ec2a62e692d4aa8f127",
        ),
    },
}


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cell_digest(cell):
    return digest(cell)


def protocol():
    return read_json(PROTOCOL_PATH)


def methods():
    return {item["id"]: item for item in protocol()["methods"]}


def path_indices(track):
    data = protocol()
    return data[track].get("path_indices", data["path_indices"])


def cases():
    available = scaling.cases()
    return {case_id: available[case_id] for case_id in protocol()["case_ids"]}


def input_path(case_id, path_index=0):
    return scaling.input_path(case_id, path_index)


def output_path(track, cell):
    return (HERE / track / "results" / cell["case_id"] /
            f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json")


def existing_output_paths(track, cell):
    """Return the canonical path plus any frozen diagnostic continuation."""
    paths = [output_path(track, cell)]
    if track != "online" or cell["family"] != "mobile_wall_scaling":
        return paths
    marker = cell["case_id"].split("-r", 1)[1].split("-", 1)[0]
    count = int(marker)
    if 3 <= count <= 5:
        root = HERE / "online-wall-replay" / "results"
    elif count >= 6:
        root = HERE / "online-wall-generator-order" / "results"
    else:
        return paths
    paths.append(
        root / cell["case_id"] /
        f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json"
    )
    return paths


def log_path(track, cell):
    return (HERE / track / "logs" / cell["case_id"] /
            f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.log")


def result_errors(path, cell, track, manifest_hash=None):
    path = Path(path)
    if not path.is_file():
        return ["missing"]
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    expected = {
        "case_id": cell["case_id"], "method_id": cell["method_id"],
        "implementation_policy": cell["policy"], "path_index": cell["path_index"],
        "shortcut_seed": cell["shortcut_seed"], "source_sha256": cell["source_sha256"],
    }
    errors = [key for key, wanted in expected.items() if value.get(key) != wanted]
    if manifest_hash is not None and value.get("manifest_sha256") != manifest_hash:
        errors.append("manifest_sha256")
    if track == "offline":
        if float(value.get("budget_seconds", -1)) != float(cell["seconds"]):
            errors.append("budget_seconds")
        if value.get("input_mode_runs") != value.get("output_mode_runs"):
            errors.append("mode sequence")
        for key in ("endpoint_start_preserved", "endpoint_end_preserved",
                    "strict_collision_free", "terminal_mode_preserved"):
            if value.get(key) is not True:
                errors.append(key)
        if value.get("valid_plan") is False:
            errors.append("valid_plan")
    else:
        pairing = value.get("first_path_pairing") or {}
        paired = (pairing.get("role") == "reference" if cell["method_id"] == protocol()["online"]["reference_method"]
                  else pairing.get("validated") is True)
        certificate = value.get("certificate") or {}
        if not paired:
            errors.append("first_path_pairing")
        for key in ("strict_full_edge", "terminal_mode"):
            if certificate.get(key) is not True:
                errors.append(key)
        if certificate.get("valid_plan") is False:
            errors.append("valid_plan")
        if not value.get("comparison_curve"):
            errors.append("comparison_curve")
    return errors


def reusable_result(cell):
    filename = f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json"
    for root in REFERENCE_ROOTS:
        candidate = root / cell["case_id"] / filename
        if not result_errors(candidate, cell, "offline"):
            return candidate
    return None


def cells(track):
    data, selected_methods = protocol(), methods()
    output = []
    for case_id, case in cases().items():
        for path_index in path_indices(track):
            source = input_path(case_id, path_index)
            source_data = read_json(source) if source.is_file() else {}
            for seed in data["shortcut_seeds"]:
                for method_id, selected in selected_methods.items():
                    record = {
                        "case_id": case_id, "family": case["family"],
                        "method_id": method_id, "policy": selected["policy"],
                        "path_index": path_index, "shortcut_seed": seed,
                        "source_sha256": sha256(source) if source.is_file() else None,
                    }
                    if track == "offline":
                        record["seconds"] = float(data["offline"]["seconds_by_family"][case["family"]])
                        reuse = reusable_result(record) if source_data else None
                        record["reuse_path"] = str(reuse.relative_to(REPO)) if reuse else None
                        record["reuse_sha256"] = sha256(reuse) if reuse else None
                    else:
                        record["planner_seed"] = source_data.get("planner_seed")
                        record["estimated_seconds"] = (
                            float(source_data.get("planning_wall_seconds", 0))
                            + float(data["online"]["refinement_seconds"])
                        )
                    output.append(record)
    if track == "online":
        reference = data["online"]["reference_method"]
        output.sort(key=lambda item: (
            data["case_ids"].index(item["case_id"]), item["path_index"],
            item["shortcut_seed"], item["method_id"] != reference,
            list(selected_methods).index(item["method_id"]),
        ))
    return output


def definition(track):
    files = [PROTOCOL_PATH, CELL_RUNNER, Path(__file__), CORE, SCALABLE,
             BENCHMARK / "online.py", BENCHMARK / "all_scaling" / "study_lib.py",
             BENCHMARK / "common.py"]
    return {
        "schema": "mr-shortcut-scaling-pilot-manifest-v1", "track": track,
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_hashes": {str(path.relative_to(REPO)): sha256(path) for path in files},
        "cells": cells(track),
    }


def mains_connected():
    indicators, batteries = [], []
    for supply in Path("/sys/class/power_supply").glob("*"):
        kind = supply / "type"
        if kind.exists() and kind.read_text().strip() == "Battery":
            batteries.append(supply)
        elif (supply / "online").exists():
            indicators.append(supply / "online")
    return any(path.read_text().strip() == "1" for path in indicators) if indicators else not batteries


def preflight(mr_bench, track):
    errors = []
    if scaling.git_revision(mr_bench) != scaling.protocol()["software"]["mr_bench_revision"]:
        errors.append("mr-bench revision differs from saved inputs")
    if ".".join(str(value) for value in sys.version_info[:3]) != scaling.protocol()["software"]["python"]:
        errors.append("Python version differs from saved-input protocol")
    available = scaling.cases()
    for case_id in protocol()["case_ids"]:
        if case_id not in available:
            errors.append(f"unknown case: {case_id}")
            continue
        for path_index in path_indices(track):
            issues = scaling.input_errors(case_id, path_index)
            if issues:
                errors.append(f"{case_id} path{path_index}: {issues}")
    return errors


def status(track, manifest, manifest_hash):
    records = []
    for cell in manifest["cells"]:
        if cell["source_sha256"] is None:
            records.append((cell, "missing_input", input_path(cell["case_id"], cell["path_index"])))
            continue
        candidates = existing_output_paths(track, cell)
        # A certified result belongs to its exact cell, not to the surrounding
        # manifest. This permits a strictly additive protocol extension without
        # rerunning cells that are unchanged. The online wall continuations are
        # retained in their separately frozen diagnostic directories.
        checked = [(path, result_errors(path, cell, track)) for path in candidates]
        valid = next((path for path, issues in checked if not issues), None)
        invalid = next((path for path, issues in checked if issues != ["missing"]), None)
        if valid is not None:
            records.append((cell, "local", valid))
        elif invalid is not None:
            records.append((cell, "invalid", invalid))
        elif track == "offline" and cell.get("reuse_path"):
            reused = REPO / cell["reuse_path"]
            valid = not result_errors(reused, cell, track) and sha256(reused) == cell["reuse_sha256"]
            records.append((cell, "reused" if valid else "invalid", reused))
        else:
            records.append((cell, "pending", candidates[0]))
    return records


def manifest_extension_errors(previous, current):
    """Return reasons why ``current`` is not a safe additive extension."""
    errors = []
    for key in ("schema", "track"):
        if previous.get(key) != current.get(key):
            errors.append(key)

    previous_cell_map = {
        cell_digest(cell): cell for cell in previous.get("cells", [])
    }
    previous_cells = set(previous_cell_map)
    current_cells = {cell_digest(cell) for cell in current.get("cells", [])}
    removed = previous_cells - current_cells
    unexpected_removed = [
        previous_cell_map[key] for key in removed
        if previous_cell_map[key].get("case_id") not in RETIRED_CASE_IDS
    ]
    if unexpected_removed:
        errors.append(
            f"{len(unexpected_removed)} existing cells changed or were removed"
        )

    # The protocol and orchestration runner necessarily change when cases are
    # added. All code that constructs or shortcuts paths must remain identical.
    allowed_changes = {
        str(PROTOCOL_PATH.relative_to(REPO)),
        str(Path(__file__).relative_to(REPO)),
    }
    old_sources = previous.get("source_hashes", {})
    new_sources = current.get("source_hashes", {})
    for path, old_hash in old_sources.items():
        new_hash = new_sources.get(path)
        permitted = (old_hash, new_hash) in ALLOWED_EXTENSION_SOURCE_TRANSITIONS.get(path, set())
        if path not in allowed_changes and new_hash != old_hash and not permitted:
            errors.append(f"source changed: {path}")
    return errors


def freeze_manifest_extension(manifest_path, manifest):
    """Freeze a new manifest, retaining any safely superseded manifest."""
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
        return
    previous = read_json(manifest_path)
    if previous == manifest:
        return
    errors = manifest_extension_errors(previous, manifest)
    if errors:
        raise RuntimeError(
            f"frozen {manifest['track']} manifest is not an additive extension: "
            + "; ".join(errors)
        )
    archive = manifest_path.with_name(
        f"{manifest_path.stem}.previous-{digest(previous)[:12]}{manifest_path.suffix}"
    )
    if archive.exists() and read_json(archive) != previous:
        raise RuntimeError(f"manifest archive collision: {archive}")
    if not archive.exists():
        write_json(archive, previous)
    write_json(manifest_path, manifest)


def child_env(mr_bench):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(protocol()["pythonhashseed"])
    env["MPLCONFIGDIR"] = "/tmp/mr-shortcutting-matplotlib"
    paths = [str(REPO / "src"), str(BENCHMARK), str(BENCHMARK / "all_scaling"),
             str(BENCHMARK / "scaling_2d"), str(mr_bench / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def supervised(command, env, log, timeout):
    if not mains_connected():
        raise RuntimeError("mains disconnected")
    started = time.monotonic()
    wall_minus_mono = time.time() - started
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if not mains_connected():
                    raise RuntimeError("mains disconnected during cell")
                if abs((time.time() - time.monotonic()) - wall_minus_mono) > 2:
                    raise RuntimeError("suspend/clock jump during cell")
                if time.monotonic() - started > timeout:
                    raise RuntimeError("cell exceeded watchdog")
                time.sleep(0.5)
            return process.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def execute(track, mr_bench, selected_status, manifest, manifest_hash, max_hours):
    manifest_path = HERE / track / "manifest.json"
    freeze_manifest_extension(manifest_path, manifest)
    pending = [cell for cell, state, _ in selected_status if state == "pending"]
    started_all = time.monotonic()
    deadline = started_all + max_hours * 3600
    reference_id = protocol()["online"]["reference_method"]
    for index, cell in enumerate(pending, 1):
        reserve = cell.get("seconds", cell.get("estimated_seconds", 0)) + 120
        if time.monotonic() + reserve > deadline:
            print(f"[stopped] {max_hours:g} hour execution limit reserve reached", flush=True)
            return 2
        output = output_path(track, cell)
        candidate = output.with_suffix(".candidate.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        if candidate.exists():
            candidate.rename(candidate.with_name(candidate.name + f".previous-{time.time_ns()}"))
        command = [sys.executable, "-B", str(CELL_RUNNER), track,
                   "--case", cell["case_id"], "--method", cell["method_id"],
                   "--path-index", str(cell["path_index"]),
                   "--shortcut-seed", str(cell["shortcut_seed"]),
                   "--output", str(candidate)]
        if track == "offline":
            command += ["--seconds", str(cell["seconds"])]
        elif cell["method_id"] != reference_id:
            reference = dict(cell, method_id=reference_id, policy=methods()[reference_id]["policy"])
            reference_path = output_path(track, reference)
            if result_errors(reference_path, reference, track):
                raise RuntimeError("paired online reference must finish first")
            command += ["--reference", str(reference_path)]
        cell_started = time.monotonic()
        code = supervised(command, child_env(mr_bench), log_path(track, cell), timeout=900)
        success = code == 0 and candidate.exists()
        if success:
            value = read_json(candidate)
            value["manifest_sha256"] = manifest_hash
            value["cell_wall_seconds"] = time.monotonic() - cell_started
            write_json(candidate, value)
            success = not result_errors(candidate, cell, track, manifest_hash)
        if success:
            candidate.replace(output)
        with (HERE / track / "audit.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "cell": cell,
                "returncode": code, "success": success,
                "elapsed_seconds": time.monotonic() - cell_started,
            }) + "\n")
        print(f"[{'ok' if success else 'failed'} {index}/{len(pending)}] "
              f"{cell['case_id']} {cell['method_id']} seed{cell['shortcut_seed']}", flush=True)
        if not success:
            print(f"[stopped] inspect {log_path(track, cell)}", flush=True)
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "preflight", "execute"))
    parser.add_argument("--track", required=True, choices=("offline", "online"))
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--max-hours", type=float, default=9.0)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    mr_bench = args.mr_bench_repo.resolve()
    manifest = definition(args.track)
    manifest_hash = digest(manifest)
    records = status(args.track, manifest, manifest_hash)
    errors = preflight(mr_bench, args.track)
    frozen = HERE / args.track / "manifest.json"
    if frozen.is_file():
        errors.extend(manifest_extension_errors(read_json(frozen), manifest))
    invalid = [(cell, str(path)) for cell, state, path in records if state == "invalid"]
    errors.extend(f"invalid existing result: {path}" for _, path in invalid)
    counts = Counter(state for _, state, _ in records)
    pending = [cell for cell, state, _ in records if state == "pending"]
    awaiting_inputs = [cell for cell, state, _ in records if state == "missing_input"]
    nominal = sum(cell.get("seconds", cell.get("estimated_seconds", 0)) for cell in pending)
    print(json.dumps({
        "track": args.track, "cases": len(cases()), "cells": len(records),
        "states": dict(counts), "pending_nominal_hours": nominal / 3600,
        "missing_inputs": sorted({str(path.relative_to(REPO)) for _, state, path in records
                                  if state == "missing_input"}),
        "missing_input_nominal_hours": sum(cell.get("seconds", 0) for cell in awaiting_inputs) / 3600,
        "pending_by_method": dict(Counter(cell["method_id"] for cell in pending)),
        "preflight_errors": errors, "mains_connected": mains_connected(),
        "manifest_sha256": manifest_hash,
    }, indent=2), flush=True)
    if errors or args.command != "execute":
        return int(bool(errors))
    if not args.confirm:
        parser.error("execute requires --confirm")
    if not mains_connected():
        raise RuntimeError("mains disconnected; nothing started")
    lock_path = HERE / "execution.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(args.track, mr_bench, records, manifest, manifest_hash, args.max_hours)


if __name__ == "__main__":
    raise SystemExit(main())
