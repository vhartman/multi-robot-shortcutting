"""Run the additive mobile-wall continuation with saved-input RNG replay."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run as base  # noqa: E402

BASE_DEFINITION = base.definition
BASE_CHILD_ENV = base.child_env
ROOT = HERE / "online-wall-replay"
ROOT.mkdir(parents=True, exist_ok=True)
base.PROTOCOL_PATH = HERE / "wall_replay_protocol.json"
base.HERE = ROOT


def output_path(track, cell):
    return (ROOT / "results" / cell["case_id"] /
            f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json")


def log_path(track, cell):
    return (ROOT / "logs" / cell["case_id"] /
            f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.log")


def definition(track):
    value = BASE_DEFINITION(track)
    value["schema"] = "mr-shortcut-scaling-wall-replay-manifest-v1"
    value["source_hashes"][str(Path(__file__).relative_to(base.REPO))] = base.sha256(Path(__file__))
    return value


def child_env(mr_bench):
    env = BASE_CHILD_ENV(mr_bench)
    env["MR_SHORTCUT_NUMPY_REPLAY_DRAWS"] = str(
        base.protocol()["online"]["numpy_draws_after_seed"]
    )
    return env


base.output_path = output_path
base.log_path = log_path
base.definition = definition
base.child_env = child_env


if __name__ == "__main__":
    if "--track" not in sys.argv:
        sys.argv.extend(["--track", "online"])
    raise SystemExit(base.main())
