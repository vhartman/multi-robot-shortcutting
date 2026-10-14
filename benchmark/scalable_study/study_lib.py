from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
PROTOCOL_PATH = HERE / "protocol.json"
MANIFEST_PATH = BENCHMARK / "manifest.json"


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


def manifest_cases() -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in read_json(MANIFEST_PATH)["cases"]}


def methods() -> dict[str, dict[str, Any]]:
    return {method["id"]: method for method in protocol()["methods"]}


def _resolved_case_sets() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    data = protocol()
    for case_set in data["case_sets"]:
        source = read_json(REPO / case_set["source_protocol"])
        selected = [
            case for case in source["cases"]
            if case["track"] == case_set["source_track"]
        ]
        for source_case in selected:
            case = dict(source_case)
            case["family"] = case_set["family"]
            case["path_indices"] = list(data["path_indices"])
            case["existing_input_root"] = case_set["existing_input_root"]
            case["seconds"] = float(
                case_set.get("seconds_overrides", {}).get(
                    case["id"], case["seconds"]
                )
            )
            if "input_generation_seconds_default" in case_set:
                case["input_generation_seconds"] = float(
                    case_set["input_generation_seconds_overrides"].get(
                        case["id"], case_set["input_generation_seconds_default"]
                    )
                )
            generation = dict(data["input_generation_defaults"])
            generation["seconds"] = float(case["input_generation_seconds"])
            generation.update(
                case_set.get("input_generation_overrides", {}).get(case["id"], {})
            )
            case["input_generation"] = generation
            output.append(case)
    for source_case in data["explicit_cases"]:
        case = dict(source_case)
        case["path_indices"] = list(data["path_indices"])
        generation = dict(data["input_generation_defaults"])
        generation["seconds"] = float(case["input_generation_seconds"])
        case["input_generation"] = generation
        output.append(case)
    return output


def cases() -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in _resolved_case_sets()}


def cost_record() -> dict[str, Any]:
    return dict(protocol()["cost"])


def local_input_path(case_id: str, path_index: int) -> Path:
    return HERE / "inputs" / case_id / f"path{path_index}.json"


def input_path(case_id: str, path_index: int) -> Path:
    local = local_input_path(case_id, path_index)
    if local.exists():
        return local
    case = cases()[case_id]
    root = case.get("existing_input_root")
    if root:
        imported = REPO / root / case_id / f"path{path_index}.json"
        if imported.exists():
            return imported
    final_input = BENCHMARK / "final" / "inputs" / case_id / f"path{path_index}.json"
    if final_input.exists():
        return final_input
    return local


def result_path(case_id: str, method_id: str, path_index: int, seed: int) -> Path:
    return (
        HERE / "offline" / "results" / case_id
        / f"{method_id}-path{path_index}-seed{seed}.json"
    )


def log_path(case_id: str, method_id: str, path_index: int, seed: int) -> Path:
    return (
        HERE / "offline" / "logs" / case_id
        / f"{method_id}-path{path_index}-seed{seed}.log"
    )


def generation_log_path(case_id: str, path_index: int) -> Path:
    return HERE / "offline" / "logs" / "input-generation" / f"{case_id}-path{path_index}.log"


def build_mrbench_environment(case: dict[str, Any]):
    seed = int(case["environment_seed"])
    np.random.seed(seed)
    random.seed(seed)
    constructor = case.get("constructor")
    if constructor == "rai_isolated_arm_box_stack_env":
        from multi_robot_multi_goal_planning.problems.rai.rai_envs import (
            rai_isolated_arm_box_stack_env,
        )

        env = rai_isolated_arm_box_stack_env(**case["environment_kwargs"])
    elif constructor == "rai_disk_scaling":
        from scaling_2d.environment import DiskScalingEnvironment

        env = DiskScalingEnvironment(**case["environment_kwargs"])
    elif constructor == "coupled_box_stacking_env":
        from scaling_coupled.environment import coupled_box_stacking_env

        env = coupled_box_stacking_env(**case["environment_kwargs"])
    else:
        from multi_robot_multi_goal_planning.problems import get_env_by_name

        env = get_env_by_name(case["environment"])
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    return env


def robot_dofs(env) -> list[int]:
    if isinstance(env.robot_dims, dict):
        return [int(env.robot_dims[robot]) for robot in env.robots]
    return [int(value) for value in env.robot_dims]


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkout_for_revision(source: Path, revision: str) -> Path:
    text = subprocess.check_output(
        ["git", "-C", str(source), "worktree", "list", "--porcelain"],
        text=True,
    )
    matches = []
    current: dict[str, str] = {}
    for line in text.splitlines() + [""]:
        if line:
            key, _, value = line.partition(" ")
            current[key] = value
        elif current:
            if current.get("HEAD") == revision:
                matches.append(Path(current["worktree"]).resolve())
            current = {}
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one assembly worktree at {revision}; found {matches}"
        )
    return matches[0]


def assembly_definition(case: dict[str, Any]) -> dict[str, Any]:
    manifest = manifest_cases().get(case["id"])
    if manifest is None:
        raise KeyError(f"assembly case is not in benchmark/manifest.json: {case['id']}")
    return manifest


def validate_definition() -> list[str]:
    data = protocol()
    errors = []
    if data.get("execution") != "strictly_serial":
        errors.append("execution must be strictly_serial")
    resolved = list(cases())
    if len(resolved) != len(set(resolved)):
        errors.append("duplicate case IDs")
    counts = Counter(case["family"] for case in cases().values())
    if dict(counts) != data["expected_case_counts"]:
        errors.append(
            f"case-family counts differ: {dict(counts)} != "
            f"{data['expected_case_counts']}"
        )
    for case_set in data["case_sets"]:
        source = REPO / case_set["source_protocol"]
        if not source.is_file() or sha256(source) != case_set["source_protocol_sha256"]:
            errors.append(f"source protocol changed: {case_set['source_protocol']}")
    for relative, expected in data["source_hashes"].items():
        source = REPO / relative
        if not source.is_file() or sha256(source) != expected:
            errors.append(f"source changed: {relative}")
    return errors
