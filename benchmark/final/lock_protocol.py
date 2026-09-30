"""Create or inspect the content-addressed lock for the final protocol."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from protocol_lib import (  # noqa: E402
    LOCKED_FILES,
    LOCK_PATH,
    REPO,
    git_revision,
    mr_bench_checkout,
    protocol,
    protocol_sha256,
    sha256,
    tracked_changes,
    validate_protocol_definition,
    write_json,
)


def payload() -> dict:
    missing = [relative for relative in LOCKED_FILES if not (REPO / relative).exists()]
    if missing:
        raise FileNotFoundError(f"cannot lock missing files: {missing}")
    definition_errors = validate_protocol_definition()
    if definition_errors:
        raise RuntimeError(f"invalid protocol: {definition_errors}")
    mr_bench = mr_bench_checkout()
    return {
        "schema": "mr-shortcut-final-content-lock-v1",
        "protocol_sha256": protocol_sha256(),
        "files": {
            relative: sha256(REPO / relative)
            for relative in LOCKED_FILES
        },
        "external": {
            "mr_bench": {
                "revision": git_revision(mr_bench),
                "required_revision": protocol()["software"]["mr_bench_revision"],
                "tracked_changes": tracked_changes(mr_bench),
            },
            "assembly_revisions": {
                case["id"]: case["assembly_revision"]
                for case in protocol()["cases"]
                if "assembly_revision" in case
            },
        },
        "python": platform.python_version(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = payload()
    if args.write:
        write_json(LOCK_PATH, data)
        print(LOCK_PATH)
    else:
        print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
