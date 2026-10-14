from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
PROTOCOL_PATH = Path(os.environ.get("MR_SHORTCUT_PROTOCOL", HERE / "protocol.json"))
ARTIFACT_ROOT = Path(os.environ.get("MR_SHORTCUT_OUTPUT_ROOT", HERE))
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


def _expanded_case_grids(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand compact Cartesian case grids used by scaling protocols.

    A scalar axis value is added to ``environment_kwargs`` under the axis
    name.  A mapping-valued axis is merged into those kwargs, which keeps
    coupled dimensions such as a wall's x/z shape together.
    """
    output = []
    for grid in data.get("case_grids", []):
        axis_names = list(grid["axes"])
        axis_values = [grid["axes"][name] for name in axis_names]
        for values in itertools.product(*axis_values):
            case = dict(grid["case"])
            kwargs = dict(case.get("environment_kwargs", {}))
            context = {}
            for name, value in zip(axis_names, values):
                if isinstance(value, dict):
                    kwargs.update(value)
                    context.update(value)
                else:
                    kwargs[name] = value
                    context[name] = value
            case["environment_kwargs"] = kwargs
            case["id"] = grid["id_format"].format(**context)
            output.append(case)
    return output


def cases() -> dict[str, dict[str, Any]]:
    data = protocol()
    resolved = list(data.get("cases", [])) + _expanded_case_grids(data)
    return {case["id"]: case for case in resolved}


def methods() -> dict[str, dict[str, Any]]:
    return {method["id"]: method for method in protocol()["methods"]}


def methods_for(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        method_id: method
        for method_id, method in methods().items()
        if case["track"] in method["tracks"]
    }


def cost_record() -> dict[str, Any]:
    return dict(protocol()["cost"])


def input_path(case_id: str, path_index: int) -> Path:
    return ARTIFACT_ROOT / "inputs" / case_id / f"path{path_index}.json"


def result_path(case_id: str, method_id: str, path_index: int, seed: int) -> Path:
    return (
        ARTIFACT_ROOT / "offline" / "results" / case_id
        / f"{method_id}-path{path_index}-seed{seed}.json"
    )


def log_path(case_id: str, method_id: str, path_index: int, seed: int) -> Path:
    return (
        ARTIFACT_ROOT / "offline" / "logs" / case_id
        / f"{method_id}-path{path_index}-seed{seed}.log"
    )


def generation_log_path(case_id: str, path_index: int) -> Path:
    return (
        ARTIFACT_ROOT / "offline" / "logs" / "input-generation"
        / f"{case_id}-path{path_index}.log"
    )


def manifest_cases() -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in read_json(MANIFEST_PATH)["cases"]}


def assembly_case(case: dict[str, Any]) -> dict[str, Any]:
    return manifest_cases()[case["id"]]


def build_mrbench_environment(case: dict[str, Any]):
    np.random.seed(int(case["environment_seed"]))
    random.seed(int(case["environment_seed"]))
    if case.get("constructor") == "rai_disk_scaling":
        from scaling_2d.environment import DiskScalingEnvironment

        env = DiskScalingEnvironment(**case["environment_kwargs"])
    elif case.get("constructor") == "rai_mobile_wall_scaling_env":
        from scaling_mobile_wall.environment import mobile_wall_scaling_env

        env = mobile_wall_scaling_env(**case["environment_kwargs"])
    elif "constructor" in case:
        from multi_robot_multi_goal_planning.problems.rai.rai_envs import (
            rai_isolated_arm_box_stack_env,
        )

        constructors = {
            "rai_isolated_arm_box_stack_env": rai_isolated_arm_box_stack_env,
        }
        env = constructors[case["constructor"]](**case.get("environment_kwargs", {}))
    else:
        from multi_robot_multi_goal_planning.problems import get_env_by_name

        env = get_env_by_name(case["environment"])
    env.cost_reduction = "max"
    env.cost_metric = "euclidean"
    return env


def robot_dofs(env) -> list[int]:
    dims = env.robot_dims
    if isinstance(dims, dict):
        return [int(dims[robot]) for robot in env.robots]
    return [int(value) for value in dims]


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
        ["git", "-C", str(source), "worktree", "list", "--porcelain"], text=True
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


def validate_protocol() -> list[str]:
    data = protocol()
    errors = []
    if data.get("execution") != "strictly_serial":
        errors.append("execution must be strictly_serial")
    resolved_cases = list(cases().values())
    case_ids = [case["id"] for case in resolved_cases]
    method_ids = [method["id"] for method in data["methods"]]
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate case id")
    expected_case_count = data.get("expected_case_count")
    if expected_case_count is not None and len(case_ids) != expected_case_count:
        errors.append(
            f"case count differs: {len(case_ids)} != {expected_case_count}"
        )
    if len(method_ids) != len(set(method_ids)):
        errors.append("duplicate method id")
    tracks = {case["track"] for case in resolved_cases}
    for method in data["methods"]:
        if not set(method["tracks"]) <= tracks:
            errors.append(f"unknown method track: {method['id']}")
    for case in resolved_cases:
        if not case.get("path_indices"):
            errors.append(f"no paths: {case['id']}")
        if case.get("constructor") == "rai_isolated_arm_box_stack_env":
            kwargs = case.get("environment_kwargs", {})
            if kwargs.get("num_boxes") != 4:
                errors.append(
                    f"scaling case does not use four boxes per robot: {case['id']}"
                )
    return errors
