"""Run certified scalable-policy trials serially on expanded scenarios."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from expanded_lib import (
    REPO,
    cases,
    checkout_for_revision,
    input_path,
    protocol,
)


HERE = Path(__file__).resolve().parent
CELL_RUNNER = HERE / "run_trial_cell.py"
POLICIES = ("scalable_tree_group",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=tuple(cases()))
    parser.add_argument(
        "--track", action="append", choices=("broad", "scaling")
    )
    parser.add_argument("--policy", action="append", choices=POLICIES)
    parser.add_argument("--path-index", type=int, action="append")
    parser.add_argument("--shortcut-seed", type=int, action="append")
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--assembly-repo", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "analysis" / "scalable-group-trials",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected_tracks = set(args.track or ["broad"])
    selected = [
        case for case in cases().values()
        if case["track"] in selected_tracks
        and (not args.case or case["id"] in set(args.case))
    ]
    policies = args.policy or list(POLICIES)
    mr_bench = args.mr_bench_repo.expanduser().resolve()
    assembly = args.assembly_repo.expanduser().resolve()
    env = os.environ.copy()
    paths = [str(REPO / "src"), str(mr_bench / "src")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["MR_BENCH_REPO"] = str(mr_bench)

    cells = []
    for case in selected:
        path_indices = args.path_index or case["path_indices"]
        seeds = args.shortcut_seed or protocol()["shortcut_seeds"]
        for path_index in path_indices:
            for policy in policies:
                for seed in seeds:
                    cells.append((case, path_index, policy, seed))
    failures = []
    completed_count = 0
    for index, (case, path_index, policy, seed) in enumerate(cells, start=1):
        output = (
            args.output_root / case["id"]
            / f"{policy}-path{path_index}-seed{seed}.json"
        )
        if output.exists() and not args.force:
            print(f"[skip {index}/{len(cells)}] {output}", flush=True)
            continue
        source = input_path(case["id"], path_index)
        if not source.exists():
            failures.append((case["id"], policy, "missing input"))
            print(f"[missing {index}/{len(cells)}] {source}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        child = env.copy()
        child["PYTHONHASHSEED"] = str(case["pythonhashseed"])
        if case["kind"] == "assembly":
            child["ASSEMBLY_REPO"] = str(checkout_for_revision(
                assembly, case["assembly_revision"]
            ))
        command = [
            sys.executable,
            str(CELL_RUNNER),
            "--case", case["id"],
            "--policy", policy,
            "--path-index", str(path_index),
            "--shortcut-seed", str(seed),
            "--seconds", str(case["seconds"]),
            "--input", str(source),
            "--output", str(output),
        ]
        started = time.perf_counter()
        log = output.with_suffix(".log")
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                env=child,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        elapsed = time.perf_counter() - started
        if result.returncode:
            failures.append((case["id"], policy, result.returncode))
            status = "failed"
        else:
            completed_count += 1
            status = "ok"
        print(
            f"[{status} {index}/{len(cells)}] {case['id']} {policy} "
            f"path{path_index} seed{seed} "
            f"({elapsed:.2f}s)",
            flush=True,
        )
    print({"completed": completed_count, "failures": failures})
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
