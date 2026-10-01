"""Validate and copy the eleven archived one-path inputs for the broad pass."""
from __future__ import annotations

import argparse
from pathlib import Path

from broad_lib import (
    cases,
    cost_record,
    input_path,
    manifest_cases,
    protocol,
    protocol_sha256,
    read_json,
    sha256,
    write_json,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    source_root = args.source.expanduser().resolve()
    manifest = manifest_cases()
    count = 0

    for case_id in cases():
        source_path = source_root / f"{case_id}.json"
        require(source_path.is_file(), f"missing source input: {source_path}")
        source = read_json(source_path)
        case = manifest[case_id]
        expected_schema = cases()[case_id].get(
            "source_schema", "mr-shortcut-paper-comparison-input-v1"
        )
        require(
            source.get("schema") == expected_schema,
            f"unexpected source schema: {source_path}",
        )
        require(source.get("case_id") == case_id, f"case mismatch: {source_path}")
        require(source.get("kind") == case["kind"], f"kind mismatch: {source_path}")
        require(
            int(source.get("environment_seed")) == int(case["environment_seed"]),
            f"environment seed mismatch: {source_path}",
        )
        expected_hashseed = int(cases()[case_id].get(
            "pythonhashseed", protocol()["pythonhashseed"]
        ))
        actual_hashseed = int(source.get("pythonhashseed", expected_hashseed))
        require(actual_hashseed == expected_hashseed, f"hash seed mismatch: {source_path}")
        certificate = source.get("certificate") or {}
        if expected_schema == "mr-shortcut-paper-comparison-input-v1":
            require(certificate.get("strict_full_edge") is True, f"bad certificate: {source_path}")
            require(certificate.get("terminal_mode") is True, f"bad terminal mode: {source_path}")
            require(certificate.get("valid_plan") is not False, f"bad native plan: {source_path}")
        else:
            require(
                source.get("mr_bench_revision")
                == protocol()["software"]["mr_bench_revision"],
                f"legacy source revision mismatch: {source_path}",
            )
        require(bool(source.get("states")), f"empty path: {source_path}")

        adopted = dict(source)
        adopted.update({
            "schema": "mr-shortcut-broad-input-v1",
            "broad_protocol_sha256": protocol_sha256(),
            "cost": cost_record(),
            "pythonhashseed": expected_hashseed,
            "source_certificate_status": (
                "recorded-valid" if certificate else "recertify-before-shortcutting"
            ),
            "source_provenance": {
                "original_schema": source["schema"],
                "source_name": source_path.name,
                "source_sha256": sha256(source_path),
            },
        })
        if args.write:
            write_json(input_path(case_id), adopted)
        count += 1

    print({"mode": "write" if args.write else "validate-only", "inputs": count})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

