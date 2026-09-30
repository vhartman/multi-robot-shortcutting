"""Audit completeness and metadata of the frozen final result matrix."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol_lib import (  # noqa: E402
    REPO,
    protocol_cases,
    protocol_sha256,
    write_json,
)
from run import artifact_errors, cells, preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", action="append", choices=("inputs", "offline", "online")
    )
    parser.add_argument("--mr-bench-repo", type=Path)
    parser.add_argument("--assembly-repo", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=HERE / "analysis" / "validation.json"
    )
    args = parser.parse_args()
    if args.mr_bench_repo is not None:
        os.environ["MR_BENCH_REPO"] = str(args.mr_bench_repo.expanduser().resolve())
    if args.assembly_repo is not None:
        os.environ["ASSEMBLY_REPO"] = str(args.assembly_repo.expanduser().resolve())

    phases = args.phase or ["inputs", "offline", "online"]
    report = {
        "schema": "mr-shortcut-final-validation-v1",
        "protocol_sha256": protocol_sha256(),
        "preflight_errors": preflight(protocol_cases()),
        "phases": {},
    }
    invalid_total = 0
    missing_total = 0
    for phase in phases:
        expected = cells(phase)
        records = []
        valid = 0
        missing = 0
        invalid = 0
        for cell in expected:
            errors = artifact_errors(cell)
            if not errors:
                valid += 1
                continue
            if errors == ["missing"]:
                missing += 1
            else:
                invalid += 1
            records.append({
                "output": str(cell.output.relative_to(REPO)),
                "case_id": cell.case_id,
                "path_index": cell.path_index,
                "method_id": cell.method_id,
                "shortcut_seed": cell.shortcut_seed,
                "errors": errors,
            })
        report["phases"][phase] = {
            "expected": len(expected),
            "valid": valid,
            "missing": missing,
            "invalid": invalid,
            "problems": records,
        }
        missing_total += missing
        invalid_total += invalid

    report["complete"] = (
        not report["preflight_errors"]
        and missing_total == 0
        and invalid_total == 0
    )
    if not args.no_write:
        write_json(args.output.resolve(), report)
    compact = {
        "protocol_sha256": report["protocol_sha256"],
        "preflight_errors": report["preflight_errors"],
        "phases": {
            phase: {
                key: value
                for key, value in details.items()
                if key != "problems"
            }
            for phase, details in report["phases"].items()
        },
        "complete": report["complete"],
        "report": None if args.no_write else str(args.output.resolve()),
    }
    print(json.dumps(compact, indent=2))
    if invalid_total or report["preflight_errors"]:
        return 1
    if missing_total and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
