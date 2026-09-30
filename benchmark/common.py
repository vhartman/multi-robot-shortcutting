from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MANIFEST_PATH = ROOT / "manifest.json"
_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "mr-shortcutting-numba"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mr-shortcutting-matplotlib"))


def read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def manifest() -> dict[str, Any]:
    return read_json(MANIFEST_PATH)


def cases(kind: str | None = None) -> list[dict[str, Any]]:
    selected = manifest()["cases"]
    if kind is not None:
        selected = [case for case in selected if case["kind"] == kind]
    return selected


def case_by_id(case_id: str) -> dict[str, Any]:
    for case in cases():
        if case["id"] == case_id:
            return case
    raise KeyError(f"unknown benchmark case: {case_id}")


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_fingerprint(env: Any) -> str:
    def stable(value: Any, depth: int = 0) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        if isinstance(value, (list, tuple)):
            return [stable(item, depth + 1) for item in value]
        if isinstance(value, dict):
            return {
                str(key): stable(item, depth + 1)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if hasattr(value, "state") and callable(value.state):
            return stable(np.asarray(value.state()), depth + 1)
        if depth < 3 and hasattr(value, "__dict__"):
            return {
                "type": type(value).__qualname__,
                "fields": stable(vars(value), depth + 1),
            }
        return {"type": type(value).__qualname__}

    scene = None
    if hasattr(env, "C"):
        scene = {
            "joint_state": np.asarray(env.C.getJointState()),
            "frame_state": np.asarray(env.C.getFrameState()),
        }
    payload = stable({
        "class": type(env).__qualname__,
        "robots": getattr(env, "robots", None),
        "robot_dims": getattr(env, "robot_dims", None),
        "limits": getattr(env, "limits", None),
        "start": getattr(env, "start_pos", None),
        "tasks": getattr(env, "tasks", None),
        "scene": scene,
    })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_mrbench_environment(case: dict[str, Any]):
    from multi_robot_multi_goal_planning.problems import get_env_by_name

    seed = int(case["environment_seed"])
    np.random.seed(seed)
    random.seed(seed)
    env = get_env_by_name(case["environment"])
    settings = manifest()["cost"]
    env.cost_reduction = settings["reduction"]
    env.cost_metric = settings["per_agent"]
    return env


def restore_mrbench_path(env, records: list[dict[str, Any]]):
    from multi_robot_multi_goal_planning.problems.planning_env import State

    if not records:
        raise ValueError("saved path contains no states")

    current_mode = env.get_start_mode()
    previous_q = env.get_start_pos()
    restored = []

    for index, record in enumerate(records):
        q = env.get_start_pos().from_flat(np.asarray(record["q"], dtype=float))
        task_ids = [int(value) for value in record["mode"]]
        if list(current_mode.task_ids) != task_ids:
            next_modes = env.get_next_modes(previous_q, current_mode)
            matches = [mode for mode in next_modes if list(mode.task_ids) == task_ids]
            if not matches:
                available = [list(mode.task_ids) for mode in next_modes]
                raise ValueError(
                    f"state {index}: cannot restore transition {current_mode.task_ids} -> "
                    f"{task_ids}; candidates are {available}"
                )
            # Unassigned environments may enumerate the same symbolic successor once
            # per equivalent assignment.  Their duplicate Modes compare equal and
            # share the same scene graph; choosing the first is deterministic.
            if any(candidate != matches[0] for candidate in matches[1:]):
                raise ValueError(
                    f"state {index}: ambiguous non-equivalent modes for {task_ids}"
                )
            current_mode = matches[0]
        restored.append(State(q, current_mode))
        previous_q = q

    return restored


def mode_runs(path) -> list[list[int]]:
    out: list[list[int]] = []
    for state in path:
        signature = [int(value) for value in state.mode.task_ids]
        if not out or signature != out[-1]:
            out.append(signature)
    return out
