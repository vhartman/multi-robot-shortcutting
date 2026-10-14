"""Run one-seed assembly hybrid policies in the actual planner loop."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from expanded_lib import REPO, cases, checkout_for_revision, read_json, sha256


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
RUNNER = BENCHMARK / "online_assembly_hybrid_trial.py"
SCALING = BENCHMARK / "scaling_2d"
OUTPUT = HERE / "analysis" / "assembly-hybrid-online"

NEW_POLICIES = ("scalable_tree_group",)
BASELINES = ("upstream_tree", "full_tree_gain_round_robin")
CASE_IDS = (
    "assembly-cross-4r-openarm",
    "assembly-car-mini-4r-openarm",
)


def source_path(case_id):
    if case_id == "assembly-car-mini-4r-openarm":
        return BENCHMARK / "final" / "inputs" / case_id / "path0.json"
    return HERE / "inputs" / case_id / "path0.json"


def runtime(case_id, source):
    if case_id == "assembly-car-mini-4r-openarm":
        return 104.0
    # The archived cross input was found in 124 s, but exact seeded replays
    # have exceeded the derived 159 s allowance before producing any path.
    # Preserve the previously used 200 s planning allowance plus 30 s of
    # post-solution refinement.
    return 230.0


def plugin_hash(policy):
    if policy == "scalable_tree_group":
        return sha256(SCALING / "scalable_tree_group.py")
    return None


def validate(result, source, policy, expected_runtime):
    errors = []
    curve = result.get("full_curve") or []
    if not curve:
        errors.append("missing solution curve")
    else:
        expected_cost = float(source["initial_cost"])
        actual_cost = float(curve[0]["cost"])
        tolerance = 1e-9 * max(1.0, abs(expected_cost))
        if abs(actual_cost - expected_cost) > tolerance:
            errors.append(f"first cost {actual_cost} != {expected_cost}")
        available = expected_runtime - float(curve[0]["time"])
        if available + 1e-9 < 30.0:
            errors.append(f"only {available:.6f}s after first solution")
    if int(result.get("planner_seed", -1)) != int(source["planner_seed"]):
        errors.append("planner seed changed")
    if result.get("environment_fingerprint") != source["environment_fingerprint"]:
        errors.append("environment fingerprint changed")
    certificate = result.get("certificate") or {}
    if not (
        certificate.get("strict_full_edge") is True
        and certificate.get("terminal_mode") is True
        and certificate.get("valid_plan") is not False
    ):
        errors.append(f"certificate failed: {certificate}")
    calls = result.get("shortcut", {}).get("calls", [])
    wrong = [call.get("applied_checks") for call in calls
             if call.get("applied_checks") != 100]
    if wrong:
        errors.append(f"non-100 check calls: {wrong}")
    if result.get("policy") != policy:
        errors.append("policy mismatch")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=CASE_IDS)
    parser.add_argument(
        "--policy", action="append", choices=NEW_POLICIES + BASELINES
    )
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--assembly-repo", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    case_ids = args.case or list(CASE_IDS)
    policies = args.policy or list(NEW_POLICIES)
    mr_bench = args.mr_bench_repo.expanduser().resolve()
    assembly = args.assembly_repo.expanduser().resolve()
    failures = []
    completed = 0
    cells = [(case_id, policy) for case_id in case_ids for policy in policies]
    for index, (case_id, policy) in enumerate(cells, start=1):
        output = OUTPUT / case_id / f"{policy}-path0-seed0.json"
        if output.exists() and not args.force:
            print(f"[skip {index}/{len(cells)}] {output}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        source_file = source_path(case_id)
        source = read_json(source_file)
        duration = runtime(case_id, source)
        case = cases()[case_id]
        checkout = checkout_for_revision(assembly, case["assembly_revision"])
        child = os.environ.copy()
        child["PYTHONHASHSEED"] = str(source["pythonhashseed"])
        child["PYTHONPATH"] = os.pathsep.join((
            str(REPO / "src"),
            str(mr_bench / "src"),
            child.get("PYTHONPATH", ""),
        ))
        child["MR_BENCH_REPO"] = str(mr_bench)
        child["ASSEMBLY_REPO"] = str(checkout)
        candidate = output.with_suffix(".candidate.json")
        log = output.with_suffix(".log")
        command = [
            sys.executable,
            str(RUNNER),
            "--case", case_id,
            "--policy", policy,
            "--runtime", str(duration),
            "--shortcut-checks", "100",
            "--seed", "0",
            "--planner-seed-offset", "0",
            "--shortcut-seed", "0",
            "--quiet",
            "--output", str(candidate),
        ]
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                env=child,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            failures.append([case_id, policy, f"exit {result.returncode}"])
            print(f"[failed {index}/{len(cells)}] {case_id} {policy}", flush=True)
            continue
        value = read_json(candidate)
        errors = validate(value, source, policy, duration)
        if errors:
            failures.append([case_id, policy, errors])
            print(
                f"[invalid {index}/{len(cells)}] {case_id} {policy}: {errors}",
                flush=True,
            )
            continue
        value.update({
            "schema": "mr-shortcut-assembly-hybrid-online-trial-v1",
            "trial_implementation_sha256": plugin_hash(policy),
            "source_path": str(source_file.relative_to(REPO)),
            "source_sha256": sha256(source_file),
            "path_index": 0,
            "shortcut_seed": 0,
            "analysis_time_origin": "first_solution",
            "analysis_horizon_seconds": 30.0,
        })
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(output)
        candidate.unlink()
        completed += 1
        first = float(value["full_curve"][0]["time"])
        final = float(value["output"]["cost"])
        print(
            f"[ok {index}/{len(cells)}] {case_id} {policy}: "
            f"first={first:.3f}s final={final:.6g}",
            flush=True,
        )
    print(json.dumps({"completed": completed, "failures": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
