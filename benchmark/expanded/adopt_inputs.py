"""Validate and adopt the original 30 certified inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from expanded_lib import (
    cases,
    cost_record,
    input_path,
    protocol,
    protocol_sha256,
    read_json,
    sha256,
    write_json,
)


SOURCE_SCHEMAS = {
    "mr-shortcut-paper-comparison-input-v1",
    "mr-shortcut-benchmark-path-v1",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    source_root = args.source.expanduser().resolve()
    selected = [case for case in cases().values() if case["input"] == "archive"]

    for case in selected:
        case_id = case["id"]
        source_path = source_root / f"{case_id}.json"
        require(source_path.is_file(), f"missing source input: {source_path}")
        source = read_json(source_path)
        require(
            source.get("schema") in SOURCE_SCHEMAS,
            f"unexpected source schema: {source_path}",
        )
        require(source.get("case_id") == case_id, f"case mismatch: {source_path}")
        require(source.get("kind") == case["kind"], f"kind mismatch: {source_path}")
        require(
            int(source.get("environment_seed")) == int(case["environment_seed"]),
            f"environment seed mismatch: {source_path}",
        )
        actual_hashseed = int(
            source.get("pythonhashseed", case["pythonhashseed"])
        )
        require(
            actual_hashseed == int(case["pythonhashseed"]),
            f"hash seed mismatch: {source_path}",
        )
        if case["kind"] == "mrbench":
            require(
                source.get("mr_bench_revision")
                == protocol()["software"]["mr_bench_revision"],
                f"mr_bench revision mismatch: {source_path}",
            )
            require(
                source.get("environment") == case["environment"],
                f"environment mismatch: {source_path}",
            )
        else:
            require(
                source.get("assembly_revision") == case["assembly_revision"],
                f"assembly revision mismatch: {source_path}",
            )
        certificate = source.get("certificate") or {}
        if certificate:
            require(
                certificate.get("strict_full_edge") is True,
                f"bad strict certificate: {source_path}",
            )
            require(
                certificate.get("terminal_mode") is True,
                f"bad terminal certificate: {source_path}",
            )
            require(
                certificate.get("valid_plan") is not False,
                f"bad native certificate: {source_path}",
            )
        require(bool(source.get("states")), f"empty path: {source_path}")

        adopted = dict(source)
        adopted.update({
            "schema": "mr-shortcut-expanded-input-v1",
            "expanded_protocol_sha256": protocol_sha256(),
            "case_id": case_id,
            "path_index": 0,
            "cost": cost_record(),
            "pythonhashseed": int(case["pythonhashseed"]),
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
            write_json(input_path(case_id, 0), adopted)

    print({
        "mode": "write" if args.write else "validate-only",
        "inputs": len(selected),
        "protocol_sha256": protocol_sha256(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
