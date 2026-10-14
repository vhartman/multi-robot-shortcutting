"""Robot-local gain-ordering ablation for the standard tree policy."""
from __future__ import annotations

import numpy as np

from mr_shortcutting.api import Shortcutter, register_shortcutter
from mr_shortcutting.shortcutters import (
    _TreePrefixProposer,
    _candidate,
    _constant_runs,
)
from mr_shortcutting.api import Proposal


def _robot_length(path, robot: int) -> float:
    return float(sum(
        np.linalg.norm(
            np.asarray(after.q[robot], dtype=float)
            - np.asarray(before.q[robot], dtype=float)
        )
        for before, after in zip(path, path[1:])
    ))


def robot_local_run_gain(env, path, robot: int, first: int, last: int) -> float:
    """Predicted reduction of only the selected robot's Euclidean length."""
    segment = path[first : last + 1]
    candidate = _candidate(env, path, Proposal(first, last, (robot,)))
    return _robot_length(segment, robot) - _robot_length(candidate, robot)


class RobotLocalGainTreeShortcutter(Shortcutter):
    name = "tree_robot_local_gain"

    def make_proposer(self, env, path, rng, interpolation_resolution):
        runs = [
            _constant_runs(
                path,
                lambda state, robot=robot: int(state.mode.task_ids[robot]),
            )
            for robot in range(len(env.robots))
        ]
        return _TreePrefixProposer(
            env,
            path,
            rng,
            runs,
            singleton_gain_evaluator=robot_local_run_gain,
        )


register_shortcutter(RobotLocalGainTreeShortcutter(), replace=True)
