"""Shared, dependency-light helpers for the frozen final protocol."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
PROTOCOL_PATH = HERE / "protocol.json"
LOCK_PATH = HERE / "protocol.lock.json"
MANIFEST_PATH = BENCHMARK / "manifest.json"

LOCKED_FILES = (
    "pyproject.toml",
    "benchmark/manifest.json",
    "benchmark/common.py",
    "benchmark/assembly_loader.py",
    "benchmark/fixtures/assembly/cross-4r-openarm-plan.json",
    "benchmark/fixtures/assembly/cross-4r-openarm-path.json",
    "benchmark/fixtures/assembly/car-mini-4r-openarm-plan.json",
    "benchmark/fixtures/assembly/car-mini-4r-openarm-path.json",
    "benchmark/online.py",
    "src/mr_shortcutting/api.py",
    "src/mr_shortcutting/shortcutters.py",
    "src/mr_shortcutting/__init__.py",
    "benchmark/final/protocol.json",
    "benchmark/final/protocol_lib.py",
    "benchmark/final/generate_input.py",
    "benchmark/final/run_offline_cell.py",
    "benchmark/final/run_online_cell.py",
    "benchmark/final/run.py",
    "benchmark/final/validate.py",
    "benchmark/final/analyze.py",
    "benchmark/final/lock_protocol.py",
)


def read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
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


def manifest() -> dict[str, Any]:
    return read_json(MANIFEST_PATH)


def manifest_cases() -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in manifest()["cases"]}


def protocol_cases() -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in protocol()["cases"]}


def methods() -> dict[str, dict[str, Any]]:
    return {method["id"]: method for method in protocol()["methods"]}


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tracked_changes(path: Path) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--short", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ["not a readable Git worktree"]
    return [line for line in output.splitlines() if line]


def mr_bench_checkout() -> Path:
    configured = os.environ.get("MR_BENCH_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    spec = importlib.util.find_spec("multi_robot_multi_goal_planning")
    if spec is not None and spec.origin is not None:
        for candidate in Path(spec.origin).resolve().parents:
            if (candidate / ".git").exists():
                return candidate
    raise RuntimeError(
        "cannot locate mr_bench; set MR_BENCH_REPO=/path/to/mr_bench"
    )


def assembly_source() -> Path:
    configured = os.environ.get("ASSEMBLY_REPO")
    if not configured:
        raise RuntimeError(
            "assembly benchmarks require ASSEMBLY_REPO=/path/to/assembly"
        )
    return Path(configured).expanduser().resolve()


def _assembly_worktrees() -> list[dict[str, str]]:
    output = subprocess.check_output(
        ["git", "-C", str(assembly_source()), "worktree", "list", "--porcelain"],
        text=True,
    )
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in output.splitlines() + [""]:
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    return records


def assembly_checkout(case_id: str) -> Path | None:
    case = protocol_cases()[case_id]
    wanted = case.get("assembly_revision")
    if wanted is None:
        return None
    matches = [
        Path(record["worktree"]).resolve()
        for record in _assembly_worktrees()
        if record.get("HEAD") == wanted
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{case_id}: expected exactly one assembly worktree at {wanted}; "
            f"found {matches}"
        )
    return matches[0]


def input_path(case_id: str, path_index: int) -> Path:
    return HERE / "inputs" / case_id / f"path{path_index}.json"


def offline_result_path(
    case_id: str, method_id: str, path_index: int, shortcut_seed: int
) -> Path:
    return (
        HERE / "offline" / "results" / case_id
        / f"{method_id}-path{path_index}-seed{shortcut_seed}.json"
    )


def online_result_path(
    case_id: str, method_id: str, path_index: int, shortcut_seed: int | None
) -> Path:
    if method_id == "planner_only":
        name = f"planner_only-path{path_index}.json"
    else:
        name = f"{method_id}-path{path_index}-seed{shortcut_seed}.json"
    return HERE / "online" / "results" / case_id / name


def source_sha256(case_id: str, path_index: int) -> str:
    return sha256(input_path(case_id, path_index))


def planner_seed(case_id: str, path_index: int) -> int:
    case = manifest_cases()[case_id]
    return int(case["environment_seed"]) + int(path_index)


def online_runtime(source: dict[str, Any]) -> float:
    settings = protocol()["online"]
    return float(
        math.ceil(float(source["planning_wall_seconds"]))
        + int(settings["refinement_seconds_after_first_solution"])
        + int(settings["replay_margin_seconds"])
    )


def cost_record() -> dict[str, Any]:
    settings = protocol()["cost"]
    return {
        "per_agent": settings["implementation_metric"],
        "reduction": settings["implementation_reduction"],
        "regularization_weight": settings["regularization_weight"],
        "edge_formula": settings["edge_formula"],
    }


def validate_protocol_definition() -> list[str]:
    errors: list[str] = []
    data = protocol()
    case_ids = [case["id"] for case in data["cases"]]
    method_ids = [method["id"] for method in data["methods"]]
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate case id")
    if len(method_ids) != len(set(method_ids)):
        errors.append("duplicate method id")
    known_cases = manifest_cases()
    for case in data["cases"]:
        entry = known_cases.get(case["id"])
        if entry is None:
            errors.append(f"unknown case: {case['id']}")
        elif not entry.get("enabled", True):
            errors.append(f"disabled case in final protocol: {case['id']}")
    try:
        from mr_shortcutting import available_shortcutters

        offline_policies = set(available_shortcutters())
        online_policies = offline_policies | {"upstream_tree"}
        for method in data["methods"]:
            if method["offline_policy"] not in offline_policies:
                errors.append(f"unknown offline policy: {method['offline_policy']}")
            if method["online_policy"] not in online_policies:
                errors.append(f"unknown online policy: {method['online_policy']}")
    except (OSError, SyntaxError, ValueError, RuntimeError) as error:
        errors.append(f"policy registry check failed: {error}")
    if data["design"]["execution"] != "strictly_serial":
        errors.append("final execution must remain strictly serial")
    if data["online"]["shortcut_checks_per_call"] != 250:
        errors.append("final online check budget must remain 250")
    return errors


def verify_content_lock() -> list[str]:
    if not LOCK_PATH.exists():
        return [f"missing content lock: {LOCK_PATH}"]
    lock = read_json(LOCK_PATH)
    errors: list[str] = []
    expected = lock.get("files", {})
    for relative in LOCKED_FILES:
        path = REPO / relative
        if not path.exists():
            errors.append(f"missing locked file: {relative}")
            continue
        actual = sha256(path)
        if expected.get(relative) != actual:
            errors.append(f"content lock mismatch: {relative}")
    extras = sorted(set(expected) - set(LOCKED_FILES))
    if extras:
        errors.append(f"content lock has unexpected files: {extras}")
    return errors
