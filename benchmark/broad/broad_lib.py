from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
PROTOCOL_PATH = HERE / "protocol.json"
MANIFEST_PATH = BENCHMARK / "manifest.json"
IMPLEMENTATION_PATH = REPO / "src" / "mr_shortcutting" / "shortcutters.py"


def read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol() -> dict[str, Any]:
    return read_json(PROTOCOL_PATH)


def protocol_sha256() -> str:
    return sha256(PROTOCOL_PATH)


def implementation_sha256() -> str:
    return sha256(IMPLEMENTATION_PATH)


def manifest_cases() -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in read_json(MANIFEST_PATH)["cases"]}


def cases() -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in protocol()["cases"]}


def methods() -> dict[str, dict[str, Any]]:
    return {method["id"]: method for method in protocol()["methods"]}


def cost_record() -> dict[str, Any]:
    return dict(protocol()["cost"])


def input_path(case_id: str) -> Path:
    return HERE / "inputs" / f"{case_id}.json"


def result_path(case_id: str, method_id: str, seed: int) -> Path:
    return HERE / "results" / case_id / f"{method_id}-seed{seed}.json"


def log_path(case_id: str, method_id: str, seed: int) -> Path:
    return HERE / "logs" / case_id / f"{method_id}-seed{seed}.log"


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

