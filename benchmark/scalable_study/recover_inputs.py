"""Recover difficult scalable-study inputs with short serial seed sweeps."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run import child_env, preflight_errors  # noqa: E402
from study_lib import cases, local_input_path, read_json  # noqa: E402


TARGETS = (
    ("rai-mobile-strut", 0, 11000),
    ("rai-mobile-strut", 1, 12000),
    ("rai-mobile-strut", 2, 13000),
    ("rai-unassigned-cleanup", 0, 21000),
    ("rai-unassigned-cleanup", 1, 22000),
    ("rai-unassigned-cleanup", 2, 23000),
    ("assembly-cross-4r-openarm", 1, 31000),
    ("assembly-cross-4r-openarm", 2, 32000),
)


def certified(path: Path) -> bool:
    if not path.exists():
        return False
    certificate = read_json(path).get("certificate") or {}
    return (
        certificate.get("strict_full_edge") is True
        and certificate.get("terminal_mode") is True
        and certificate.get("valid_plan") is not False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    parser.add_argument("--assembly-repo", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=200.0)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("recovery requires --confirm")
    if args.seconds <= 0 or args.attempts <= 0:
        parser.error("--seconds and --attempts must be positive")

    mr_bench = args.mr_bench_repo.expanduser().resolve()
    assembly = args.assembly_repo.expanduser().resolve()
    selected = [cases()[case_id] for case_id, _, _ in TARGETS]
    errors = preflight_errors(selected, mr_bench, assembly)
    if errors:
        raise RuntimeError(f"preflight failed: {errors}")

    log_root = HERE / "offline" / "logs" / "input-seed-sweep"
    log_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for target_index, (case_id, path_index, seed_base) in enumerate(TARGETS, start=1):
        case = cases()[case_id]
        output = local_input_path(case_id, path_index)
        if certified(output):
            print(f"[skip {target_index}/{len(TARGETS)}] {case_id} path{path_index}", flush=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        solved = False
        for attempt in range(args.attempts):
            seed = seed_base + attempt
            command = [
                sys.executable,
                str(HERE / "generate_input.py"),
                "--case", case_id,
                "--path-index", str(path_index),
                "--planner-seed", str(seed),
                "--seconds", str(args.seconds),
                "--output", str(output),
                "--mr-bench-repo", str(mr_bench),
            ]
            sampler = None
            if case["kind"] == "mrbench":
                sampler = "auto" if attempt % 2 == 0 else "gibbs_random_scan"
                command.extend(("--sampler", sampler))
            log = log_root / f"{case_id}-path{path_index}-seed{seed}.log"
            print(
                f"[start {target_index}/{len(TARGETS)}] {case_id} path{path_index} "
                f"attempt={attempt + 1}/{args.attempts} seed={seed} sampler={sampler or 'native'}",
                flush=True,
            )
            started = time.perf_counter()
            with log.open("w") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=child_env(case, mr_bench, assembly),
                )
            elapsed = time.perf_counter() - started
            solved = completed.returncode == 0 and certified(output)
            print(
                f"[{'ok' if solved else 'failed'} {target_index}/{len(TARGETS)}] "
                f"{case_id} path{path_index} seed={seed} ({elapsed:.2f}s)",
                flush=True,
            )
            if solved:
                break
        if not solved:
            failures.append((case_id, path_index))
    print({"targets": len(TARGETS), "failures": failures}, flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
