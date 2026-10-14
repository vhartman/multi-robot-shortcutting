"""Plan or generate only the missing offline scaling inputs, serially."""
from __future__ import annotations

import argparse
import fcntl
import json
import shlex
import sys

from run import (HERE, child_env, input_path, mains_connected,
                 path_indices, protocol, read_json, scaling, supervised)


def missing_inputs(case_ids=None):
    targets = []
    for case_id in case_ids or protocol()["case_ids"]:
        used_seeds = set()
        reference = None
        for index in path_indices("offline"):
            source = input_path(case_id, index)
            if source.exists():
                issues = scaling.input_errors(case_id, index)
                if issues:
                    raise RuntimeError(f"invalid existing input {source}: {issues}")
                value = read_json(source)
                used_seeds.add(int(value["planner_seed"]))
                if reference is None:
                    reference = value
                elif any(value[key] != reference[key]
                         for key in ("environment_fingerprint", "cost")):
                    raise RuntimeError(f"existing inputs disagree on environment/cost: {source}")
        for index in path_indices("offline"):
            output = input_path(case_id, index)
            if output.exists():
                continue
            seed = int(scaling.cases()[case_id]["environment_seed"]) + index
            while seed in used_seeds:
                seed += 1000
            used_seeds.add(seed)
            targets.append((case_id, index, seed, output, reference))
    return targets


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "execute"))
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=200)
    parser.add_argument("--case", action="append", choices=protocol()["case_ids"])
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    targets = missing_inputs(args.case)
    commands = []
    for case_id, index, seed, output, reference in targets:
        candidate = output.with_suffix(".candidate.json")
        command = [sys.executable, str(HERE / "generate_input.py"),
                   "--case", case_id, "--path-index", str(index),
                   "--planner-seed", str(seed), "--seconds", str(args.seconds),
                   "--output", str(candidate), "--mr-bench-repo", str(args.mr_bench_repo.resolve())]
        commands.append((command, output, candidate, reference))
    print(json.dumps({"missing_inputs": len(targets), "commands": [shlex.join(c[0]) for c in commands]}, indent=2))
    if args.command == "plan":
        return 0
    if not args.confirm:
        parser.error("execute requires --confirm")
    if not mains_connected():
        raise RuntimeError("mains disconnected; nothing started")
    with (HERE / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        references = {}
        for command, output, candidate, reference in commands:
            if output.exists() or candidate.exists():
                raise RuntimeError(f"output or candidate already exists; inspect before continuing: {output}")
            log = HERE / "offline" / "logs" / "input-generation" / f"{output.parent.name}-{output.stem}.log"
            code = supervised(command, child_env(args.mr_bench_repo.resolve()), log, args.seconds + 180)
            if code or not candidate.exists():
                raise RuntimeError(f"input generation failed; inspect {log}")
            value = read_json(candidate)
            reference = references.get(output.parent, reference)
            if reference is not None:
                for key in ("environment_fingerprint", "cost"):
                    if value[key] != reference[key]:
                        raise RuntimeError(f"generated input differs in {key}; retained for inspection: {candidate}")
            cert = value.get("certificate", {})
            if not cert.get("strict_full_edge") or not cert.get("terminal_mode") or cert.get("valid_plan") is False:
                raise RuntimeError(f"uncertified candidate: {candidate}")
            candidate.replace(output)
            references[output.parent] = value
            print(f"[certified] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
