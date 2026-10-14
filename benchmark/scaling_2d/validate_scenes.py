"""Validate nested geometry, collision-free goals, and reproducible construction."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "expanded"))
os.environ["MR_SHORTCUT_PROTOCOL"] = str(HERE / "protocol.json")
os.environ["MR_SHORTCUT_OUTPUT_ROOT"] = str(HERE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr-bench-repo", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.mr_bench_repo / "src"))
    from common import environment_fingerprint
    from expanded_lib import build_mrbench_environment, cases, write_json

    snapshots = {}
    sequences = {}
    reference = None
    for case in reversed(list(cases().values())):
        env = build_mrbench_environment(case)
        count = len(env.robots)
        assert all(dimension == 2 for dimension in env.robot_dims.values())
        assert env.is_collision_free(env.start_pos, env.start_mode)
        snapshot = env.scene_metadata
        if reference is None:
            reference = snapshot
        assert snapshot["obstacles"] == reference["obstacles"]
        assert snapshot["starts"] == reference["starts"][:count]
        assert snapshot["goals"] == reference["goals"][:count]
        sequence = [env.tasks[index].name for index in env.sequence[:-1]]
        sequences[count] = sequence
        full_sequence = sequences[max(sequences)]
        assert sequence == [name for name in full_sequence if name in sequence]
        initial = env.C.getJointState().copy()
        for robot, goals in zip(env.robots, snapshot["goals"]):
            for goal in goals:
                env.C.setJointState(initial)
                env.C.setJointState(np.array(goal), env.robot_joints[robot])
                assert env.C.getCollisionFree(), (case["id"], robot, goal)
        env.C.setJointState(initial)
        first_fingerprint = environment_fingerprint(env)
        restored_env = build_mrbench_environment(case)
        assert first_fingerprint == environment_fingerprint(restored_env)
        snapshots[case["id"]] = snapshot
        print(f"[ok] {count} robots: nested geometry, task order, free goals, deterministic fingerprint", flush=True)
    write_json(HERE / "analysis" / "scene-layouts.json", snapshots)


if __name__ == "__main__":
    main()
