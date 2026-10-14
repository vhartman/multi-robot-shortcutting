from __future__ import annotations

import hashlib
import itertools
import json
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
PROTOCOL_PATH = HERE / "protocol.json"
CORE_IMPLEMENTATION = REPO / "src" / "mr_shortcutting" / "shortcutters.py"
SCALABLE_IMPLEMENTATION = BENCHMARK / "scaling_2d" / "scalable_tree_group.py"


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


def source_hashes() -> dict[str, str]:
    files = [
        Path(__file__),
        HERE / "run.py",
        HERE / "run_cell.py",
        CORE_IMPLEMENTATION,
        REPO / "src" / "mr_shortcutting" / "api.py",
        SCALABLE_IMPLEMENTATION,
        BENCHMARK / "scaling_2d" / "environment.py",
        BENCHMARK / "scaling_coupled" / "environment.py",
        BENCHMARK / "scaling_mobile_wall" / "environment.py",
        BENCHMARK / "common.py",
    ]
    return {str(path.relative_to(REPO)): sha256(path) for path in files}


def cases() -> dict[str, dict[str, Any]]:
    output = []
    data = protocol()
    for grid in data["case_grids"]:
        names = list(grid["axes"])
        for values in itertools.product(*(grid["axes"][name] for name in names)):
            case = dict(grid["case"])
            kwargs = dict(case.get("environment_kwargs", {}))
            context = {}
            for name, value in zip(names, values):
                if isinstance(value, dict):
                    kwargs.update(value)
                    context.update(value)
                else:
                    kwargs[name] = value
                    context[name] = value
            case["environment_kwargs"] = kwargs
            case["id"] = grid["id_format"].format(**context)
            case["path_indices"] = list(data["path_indices"])
            output.append(case)
    return {case["id"]: case for case in output}


def methods() -> dict[str, dict[str, Any]]:
    return {method["id"]: method for method in protocol()["methods"]}


def input_path(case_id: str, path_index: int) -> Path:
    case = cases()[case_id]
    return REPO / case["input_root"] / case_id / f"path{path_index}.json"


def local_result_path(case_id: str, method_id: str, path_index: int, seed: int) -> Path:
    return (
        HERE / "offline" / "results" / case_id
        / f"{method_id}-path{path_index}-seed{seed}.json"
    )


def local_failure_path(case_id: str, method_id: str, path_index: int, seed: int) -> Path:
    return (
        HERE / "offline" / "failures" / case_id
        / f"{method_id}-path{path_index}-seed{seed}.json"
    )


def log_path(case_id: str, method_id: str, path_index: int, seed: int) -> Path:
    return (
        HERE / "offline" / "logs" / case_id
        / f"{method_id}-path{path_index}-seed{seed}.log"
    )


@dataclass(frozen=True)
class Cell:
    case_id: str
    family: str
    method_id: str
    path_index: int
    shortcut_seed: int
    seconds: float


def cells() -> list[Cell]:
    data = protocol()
    return [
        Cell(
            case["id"], case["family"], method_id, path_index, seed,
            float(case["seconds"]),
        )
        for case in cases().values()
        for method_id in methods()
        for path_index in data["path_indices"]
        for seed in data["shortcut_seeds"]
    ]


def input_errors(case_id: str, path_index: int) -> list[str]:
    path = input_path(case_id, path_index)
    if not path.is_file():
        return ["missing"]
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    errors = []
    if value.get("case_id") != case_id:
        errors.append("case_id")
    if value.get("path_index") != path_index:
        errors.append("path_index")
    if not value.get("states"):
        errors.append("states")
    if value.get("cost") != protocol()["cost"]:
        errors.append("cost")
    certificate = value.get("certificate") or {}
    if certificate.get("strict_full_edge") is not True:
        errors.append("strict certificate")
    if certificate.get("terminal_mode") is not True:
        errors.append("terminal certificate")
    if certificate.get("valid_plan") is False:
        errors.append("native certificate")
    return errors


def result_errors(path: Path, cell: Cell, manifest_sha256: str | None = None) -> list[str]:
    if not path.is_file():
        return ["missing"]
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    expected = {
        "case_id": cell.case_id,
        "method_id": cell.method_id,
        "path_index": cell.path_index,
        "shortcut_seed": cell.shortcut_seed,
        "implementation_policy": methods()[cell.method_id]["policy"],
        "source_sha256": sha256(input_path(cell.case_id, cell.path_index)),
    }
    errors = [key for key, wanted in expected.items() if value.get(key) != wanted]
    if float(value.get("budget_seconds", -1)) != cell.seconds:
        errors.append("budget_seconds")
    if value.get("cost") != protocol()["cost"]:
        errors.append("cost")
    if value.get("input_mode_runs") != value.get("output_mode_runs"):
        errors.append("mode sequence")
    for key in (
        "strict_collision_free",
        "endpoint_start_preserved",
        "endpoint_end_preserved",
        "terminal_mode_preserved",
    ):
        if value.get(key) is not True:
            errors.append(key)
    if value.get("valid_plan") is False:
        errors.append("valid_plan")
    if manifest_sha256 is not None and value.get("execution_manifest_sha256") != manifest_sha256:
        errors.append("execution_manifest_sha256")
    implementation = value.get("implementation_sha256")
    compatible_implementations = set(
        protocol()["software"]["compatible_core_implementation_sha256"]
    )
    compatible_implementations.add(sha256(CORE_IMPLEMENTATION))
    if implementation is not None and implementation not in compatible_implementations:
        errors.append("implementation_sha256")
    return errors


def failure_errors(path: Path, cell: Cell, manifest_sha256: str) -> list[str]:
    if not path.is_file():
        return ["missing"]
    try:
        value = read_json(path)
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    expected = {
        "status": "failed_certificate",
        "case_id": cell.case_id,
        "method_id": cell.method_id,
        "path_index": cell.path_index,
        "shortcut_seed": cell.shortcut_seed,
        "implementation_policy": methods()[cell.method_id]["policy"],
        "source_sha256": sha256(input_path(cell.case_id, cell.path_index)),
        "execution_manifest_sha256": manifest_sha256,
    }
    errors = [key for key, wanted in expected.items() if value.get(key) != wanted]
    if float(value.get("budget_seconds", -1)) != cell.seconds:
        errors.append("budget_seconds")
    if value.get("cost") != protocol()["cost"]:
        errors.append("cost")
    if value.get("input_mode_runs") != value.get("output_mode_runs"):
        errors.append("mode sequence")
    certificate_keys = (
        "strict_collision_free",
        "endpoint_start_preserved",
        "endpoint_end_preserved",
        "terminal_mode_preserved",
    )
    if all(value.get(key) is True for key in certificate_keys) and value.get("valid_plan") is not False:
        errors.append("certificate did not fail")
    return errors


def reusable_result(cell: Cell) -> Path | None:
    for relative in protocol()["reuse_result_roots"]:
        candidate = (
            REPO / relative / cell.case_id
            / f"{cell.method_id}-path{cell.path_index}-seed{cell.shortcut_seed}.json"
        )
        if candidate.exists() and not result_errors(candidate, cell):
            return candidate
    return None


def build_environment(case: dict[str, Any]):
    np.random.seed(int(case["environment_seed"]))
    random.seed(int(case["environment_seed"]))
    constructor = case["constructor"]
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
    elif constructor == "rai_mobile_wall_scaling_env":
        from scaling_mobile_wall.environment import mobile_wall_scaling_env

        env = mobile_wall_scaling_env(**case["environment_kwargs"])
    else:
        raise ValueError(f"unknown constructor: {constructor}")
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


def definition_errors(mr_bench: Path) -> list[str]:
    data = protocol()
    errors = []
    counts = Counter(case["family"] for case in cases().values())
    if dict(counts) != data["expected_case_counts"]:
        errors.append(f"case counts differ: {dict(counts)}")
    if len(cases()) != len(set(cases())):
        errors.append("duplicate case IDs")
    if git_revision(mr_bench) != data["software"]["mr_bench_revision"]:
        errors.append("mr_bench revision differs")
    for case in cases().values():
        for path_index in data["path_indices"]:
            issues = input_errors(case["id"], path_index)
            if issues:
                errors.append(f"{case['id']} path{path_index}: {issues}")
    return errors
