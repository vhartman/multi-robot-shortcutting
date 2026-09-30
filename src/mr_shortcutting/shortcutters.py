from __future__ import annotations

import heapq
import math
import time
from collections import deque
from itertools import combinations
from typing import Any

import numpy as np

from multi_robot_multi_goal_planning.problems.planning_env import State
from multi_robot_multi_goal_planning.problems.core.configuration import config_dist
from multi_robot_multi_goal_planning.problems.util import path_cost

from .api import (
    Proposal,
    Shortcutter,
    available_shortcutters,
    get_shortcutter,
    register_shortcutter,
)

POLICIES = (
    "full_configuration",
    "single_robot_random",
    "robot_subset_random",
    "robot_subset_mode",
    "single_robot_mode",
    "tree_prefix",
    "gain_only",
    "full_tree_gain_round_robin",
    "tree_group_gain",
    "wafr_slide_halton",
    "wafr_single_robot",
    "huang_round_robin",
    "huang_round_robin_no_prioritized",
    "huang_thompson",
)


def _constant_runs(path, signature) -> list[tuple[int, int]]:
    if not path:
        return []
    runs = []
    first = 0
    first_signature = signature(path[0])
    for index in range(1, len(path) + 1):
        current = signature(path[index]) if index < len(path) else None
        if index == len(path) or current != first_signature:
            if index - 1 - first >= 2:
                runs.append((first, index - 1))
            first = index
            first_signature = current
    return runs


def _ordered_pair(rng: np.random.Generator, first: int, last: int):
    i = int(rng.integers(first, last + 1))
    j = int(rng.integers(first, last + 1))
    return (i, j) if i <= j else (j, i)


def _make_proposer(
    policy: str,
    env,
    path,
    rng: np.random.Generator,
    interpolation_resolution: float = 0.5,
):
    robot_count = len(env.robots)
    all_robots = tuple(range(robot_count))
    robot_runs = [
        _constant_runs(path, lambda state, robot=robot: int(state.mode.task_ids[robot]))
        for robot in range(robot_count)
    ]
    eligible_runs = [
        (robot, first, last)
        for robot, runs in enumerate(robot_runs)
        for first, last in runs
    ]
    subset_runs = {}

    if policy in ("wafr_slide_halton", "wafr_single_robot"):
        if policy == "wafr_slide_halton":
            run_specs = [
                (first, last, all_robots)
                for first, last in _constant_runs(path, lambda state: state.mode)
            ]
            adaptation = (
                "independent constant-full-mode runs, round-robin; "
                "full-configuration arclength endpoints projected to nearest "
                "stored waypoint"
            )
        else:
            run_specs = [
                (first, last, (robot,))
                for robot, first, last in eligible_runs
            ]
            adaptation = (
                "independent constant-task single-robot runs, round-robin; "
                "single-robot arclength endpoints projected to nearest stored "
                "waypoint"
            )
        return _WafrSlideHaltonProposer(
            env, path, rng, run_specs, adaptation=adaptation
        )

    if policy in (
        "huang_round_robin",
        "huang_round_robin_no_prioritized",
        "huang_thompson",
    ):
        full_runs = _constant_runs(path, lambda state: state.mode)
        return _HuangMultiStrategyProposer(
            env,
            path,
            rng,
            robot_runs,
            full_runs,
            interpolation_resolution,
            selector="round_robin"
            if policy != "huang_thompson"
            else "thompson",
            strategies=("composite", "path")
            if policy == "huang_round_robin_no_prioritized"
            else None,
        )

    if policy == "full_tree_gain_round_robin":
        return _FullTreeGainRoundRobinProposer(
            env,
            path,
            rng,
            robot_runs,
        )

    if policy in ("tree_prefix", "gain_only", "tree_group_gain"):
        return _TreePrefixProposer(
            env,
            path,
            rng,
            robot_runs,
            group_gain_fallback=policy == "tree_group_gain",
            skip_tree_prefix=policy == "gain_only",
        )

    if policy == "single_robot_mode" and not eligible_runs:
        raise ValueError("path has no robot/task run spanning at least two edges")

    def propose() -> Proposal | None:
        if policy == "full_configuration":
            i, j = _ordered_pair(rng, 0, len(path) - 1)
            robots = all_robots
        elif policy == "single_robot_random":
            i, j = _ordered_pair(rng, 0, len(path) - 1)
            robots = (int(rng.integers(0, robot_count)),)
        elif policy == "robot_subset_random":
            i, j = _ordered_pair(rng, 0, len(path) - 1)
            size = int(rng.integers(1, robot_count + 1))
            robots = tuple(
                sorted(
                    int(value)
                    for value in rng.choice(robot_count, size=size, replace=False)
                )
            )
        elif policy == "robot_subset_mode":
            size = int(rng.integers(1, robot_count + 1))
            robots = tuple(
                sorted(
                    int(value)
                    for value in rng.choice(robot_count, size=size, replace=False)
                )
            )
            if robots not in subset_runs:
                subset_runs[robots] = _constant_runs(
                    path,
                    lambda state, robots=robots: tuple(
                        int(state.mode.task_ids[robot]) for robot in robots
                    ),
                )
            runs = subset_runs[robots]
            if not runs:
                return None
            first, last = runs[int(rng.integers(0, len(runs)))]
            i, j = _ordered_pair(rng, first, last)
        elif policy == "single_robot_mode":
            robot, first, last = eligible_runs[int(rng.integers(0, len(eligible_runs)))]
            i, j = _ordered_pair(rng, first, last)
            robots = (robot,)
        else:
            raise ValueError(f"unknown shortcutter: {policy}")

        if j - i < 2:
            return None

        if policy == "full_configuration":
            if path[i].mode != path[j].mode:
                return None
        else:
            for robot in robots:
                if path[i].mode.task_ids[robot] != path[j].mode.task_ids[robot]:
                    return None
        return Proposal(
            i,
            j,
            robots,
            source="subset_mode" if policy == "robot_subset_mode" else "random",
        )

    return propose

class _BuiltInShortcutter(Shortcutter):
    """Expose an existing built-in policy through the public API."""

    def __init__(self, name: str):
        self.name = name

    def make_proposer(
        self, env, path, rng, interpolation_resolution: float
    ):
        return _make_proposer(
            self.name, env, path, rng, interpolation_resolution
        )


for _policy_name in POLICIES:
    register_shortcutter(_BuiltInShortcutter(_policy_name))


def _registered_proposer(
    policy: str, env, path, rng, interpolation_resolution: float = 0.5
):
    return get_shortcutter(policy).make_proposer(
        env, path, rng, interpolation_resolution
    )



def _candidate(env, path, proposal: Proposal):
    i, j, robots = proposal.first, proposal.last, proposal.robots
    q0 = path[i].q
    shortcut_end = proposal.shortcut_end if proposal.shortcut_end is not None else j
    q1 = path[shortcut_end].q
    starts = {robot: np.asarray(q0[robot], dtype=float).copy() for robot in robots}
    increments = {
        robot: np.asarray(q1[robot], dtype=float) - starts[robot]
        for robot in robots
    }

    motion_steps = proposal.motion_steps
    if motion_steps is None:
        motion_steps = shortcut_end - i
    motion_steps = max(1, int(motion_steps))
    shift = max(0, shortcut_end - i - motion_steps)

    candidate = []
    for offset in range(j - i + 1):
        flat = np.asarray(path[i + offset].q.state(), dtype=float).copy()
        flat_offset = 0
        for robot, name in enumerate(env.robots):
            dimension = int(env.robot_dims[name])
            if robot in robots:
                if proposal.shortcut_type == "prioritized" and offset > motion_steps:
                    shifted_index = min(j, i + offset + shift)
                    flat[flat_offset : flat_offset + dimension] = np.asarray(
                        path[shifted_index].q[robot], dtype=float
                    )
                else:
                    interpolation_offset = min(offset, motion_steps)
                    if (
                        proposal.shortcut_type == "path"
                        and interpolation_offset != 0
                        and i + offset != shortcut_end
                        and path[i + offset].mode != path[i + offset - 1].mode
                    ):
                        interpolation_offset = max(0, interpolation_offset - 1)
                    alpha = interpolation_offset / motion_steps
                    flat[flat_offset : flat_offset + dimension] = (
                        starts[robot] + increments[robot] * alpha
                    )
            flat_offset += dimension
        candidate.append(State(q0.from_flat(flat), path[i + offset].mode))
    return candidate


def _group_run_gain(env, path, robots: tuple[int, ...], first: int, last: int) -> float:
    proposal = Proposal(first, last, robots)
    straightened = _candidate(env, path, proposal)
    old_cost = float(path_cost(path[first : last + 1], env.batch_config_cost))
    new_cost = float(path_cost(straightened, env.batch_config_cost))
    return old_cost - new_cost


def _full_run_gain(env, path, robot: int, first: int, last: int) -> float:
    return _group_run_gain(env, path, (robot,), first, last)


class _TreePrefixProposer:
    """Broadest-first collision-splitting prefix with a configurable fallback."""

    def __init__(
        self,
        env,
        path,
        rng,
        runs,
        group_gain_fallback=False,
        group_tree_prefix=False,
        full_singleton_tree_prefix=False,
        skip_tree_prefix=False,
        tree_only=False,
        zero_accept_subset=False,
        random_run_fallback=False,
        random_subset_fallback=False,
        split_tree_failures=True,
        max_tree_depth=None,
        stall_to_mode=False,
        warm_mode_seconds=0.0,
    ):
        self.env = env
        self.path = path
        self.rng = rng
        self.runs = runs
        self.group_gain_fallback = bool(group_gain_fallback)
        self.group_tree_prefix = bool(group_tree_prefix)
        self.full_singleton_tree_prefix = bool(full_singleton_tree_prefix)
        self.skip_tree_prefix = bool(skip_tree_prefix)
        self.tree_only = bool(tree_only)
        self.zero_accept_subset = bool(zero_accept_subset)
        self.random_run_fallback = bool(random_run_fallback)
        self.random_subset_fallback = bool(random_subset_fallback)
        self.split_tree_failures = bool(split_tree_failures)
        self.max_tree_depth = max_tree_depth
        self.stall_to_mode = bool(stall_to_mode)
        self.warm_mode_seconds = float(warm_mode_seconds)
        self.warm_finished = self.warm_mode_seconds <= 0.0
        self.robot_count = len(env.robots)
        self.eligible_runs = [
            (robot, first, last)
            for robot, robot_runs in enumerate(runs)
            for first, last in robot_runs
        ]
        self.used = [[0] * len(robot_runs) for robot_runs in runs]
        self.heap = []
        self.serial = 0
        self.initial_tree_intervals = 0
        if self.skip_tree_prefix:
            pass
        elif self.group_tree_prefix:
            for size in range(1, self.robot_count + 1):
                for robots in combinations(range(self.robot_count), size):
                    group_runs = _constant_runs(
                        self.path,
                        lambda state, robots=robots: tuple(
                            int(state.mode.task_ids[robot]) for robot in robots
                        ),
                    )
                    for first, last in group_runs:
                        self._push(tuple(robots), first, last)
                        self.initial_tree_intervals += 1
        elif self.full_singleton_tree_prefix:
            all_robots = tuple(range(self.robot_count))
            if self.robot_count > 1:
                for first, last in _constant_runs(self.path, lambda state: state.mode):
                    self._push(all_robots, first, last)
                    self.initial_tree_intervals += 1
            for robot, robot_runs in enumerate(runs):
                for first, last in robot_runs:
                    self._push((robot,), first, last)
                    self.initial_tree_intervals += 1
        else:
            for robot, robot_runs in enumerate(runs):
                for first, last in robot_runs:
                    self._push((robot,), first, last)
                    self.initial_tree_intervals += 1
        if not self.eligible_runs:
            raise ValueError("path has no robot/task run spanning at least two edges")
        if not self.heap and not self.skip_tree_prefix:
            raise ValueError("tree prefix has no eligible intervals")

        self.gains = None
        self.proposed = None
        self.accepted = None
        self.gain_floor = None
        self.group_gain_arms = None
        self.fallback = None
        self.stall_window_started = None
        self.stall_window_cost = None
        self.stats = {
            "tree_proposals": 0,
            "tree_accepted": 0,
            "tree_collisions": 0,
            "tree_cost_rejects": 0,
            "tree_failure_splits": 0,
            "split_tree_failures": self.split_tree_failures,
            "max_tree_depth": self.max_tree_depth,
            "group_tree_prefix": self.group_tree_prefix,
            "full_singleton_tree_prefix": self.full_singleton_tree_prefix,
            "tree_prefix_skipped": self.skip_tree_prefix,
            "tree_only": self.tree_only,
            "tree_initial_intervals": self.initial_tree_intervals,
            "tree_proposals_by_size": {},
            "tree_accepted_by_size": {},
            "gain_proposals": 0,
            "group_gain_arms": 0,
            "group_gain_proposals": 0,
            "group_gain_accepted": 0,
            "group_gain_proposals_by_size": {},
            "group_gain_accepted_by_size": {},
            "subset_proposals": 0,
            "mode_proposals": 0,
            "random_run_proposals": 0,
            "warm_mode_proposals": 0,
            "warm_mode_seconds": self.warm_mode_seconds,
            "fallback": None,
            "stall_switched": False,
            "stall_switch_time": None,
            "stall_switch_cost": None,
            "prefix_exhausted": False,
        }
        if self.skip_tree_prefix:
            self._initialize_gain_fallback()

    def _push(self, robots, first, last, depth=0):
        if last - first < 2:
            return
        heapq.heappush(
            self.heap,
            (
                -(last - first),
                float(self.rng.random()),
                self.serial,
                tuple(robots),
                first,
                last,
                int(depth),
            ),
        )
        self.serial += 1

    def _initialize_gain_fallback(self):
        self.fallback = "gain"
        self.gains = [
            [
                max(0.0, _full_run_gain(self.env, self.path, robot, first, last))
                for first, last in robot_runs
            ]
            for robot, robot_runs in enumerate(self.runs)
        ]
        all_gains = [gain for robot_gains in self.gains for gain in robot_gains]
        mean_gain = sum(all_gains) / len(all_gains)
        self.gain_floor = 0.05 * mean_gain if mean_gain > 0 else 1.0
        self.proposed = [[0] * len(robot_runs) for robot_runs in self.runs]
        self.accepted = [[0] * len(robot_runs) for robot_runs in self.runs]
        self.stats["prefix_exhausted"] = True
        self.stats["fallback"] = self.fallback

    def _initialize_group_gain_fallback(self):
        self.fallback = "group_gain"
        self.group_gain_arms = []
        for size in range(1, self.robot_count + 1):
            for robots in combinations(range(self.robot_count), size):
                group_runs = _constant_runs(
                    self.path,
                    lambda state, robots=robots: tuple(
                        int(state.mode.task_ids[robot]) for robot in robots
                    ),
                )
                for first, last in group_runs:
                    self.group_gain_arms.append({
                        "robots": tuple(robots),
                        "first": first,
                        "last": last,
                        "gain": max(
                            0.0,
                            _group_run_gain(
                                self.env,
                                self.path,
                                tuple(robots),
                                first,
                                last,
                            ),
                        ),
                        "used": 0,
                        "proposed": 0,
                        "accepted": 0,
                    })
        if not self.group_gain_arms:
            raise ValueError("path has no group/task run spanning at least two edges")
        mean_gain = sum(
            arm["gain"] for arm in self.group_gain_arms
        ) / len(self.group_gain_arms)
        self.gain_floor = 0.05 * mean_gain if mean_gain > 0 else 1.0
        self.stats["group_gain_arms"] = len(self.group_gain_arms)
        self.stats["prefix_exhausted"] = True
        self.stats["fallback"] = self.fallback


    def consider_time(self, elapsed, current_cost):
        if not self.warm_finished:
            if elapsed < self.warm_mode_seconds:
                return
            self.warm_finished = True
        if not self.stall_to_mode or self.fallback != "gain":
            return
        if self.stall_window_started is None:
            self.stall_window_started = float(elapsed)
            self.stall_window_cost = float(current_cost)
            return
        if elapsed - self.stall_window_started < 10.0:
            return
        relative_gain = (
            self.stall_window_cost - current_cost
        ) / max(self.stall_window_cost, 1e-12)
        if relative_gain < 0.01:
            self.fallback = "mode"
            self.stats["fallback"] = "gain_to_mode"
            self.stats["stall_switched"] = True
            self.stats["stall_switch_time"] = float(elapsed)
            self.stats["stall_switch_cost"] = float(current_cost)
            return
        self.stall_window_started = float(elapsed)
        self.stall_window_cost = float(current_cost)

    def _initialize_fallback(self):
        self.stats["prefix_exhausted"] = True
        if self.group_gain_fallback:
            self._initialize_group_gain_fallback()
        elif self.random_subset_fallback:
            self.fallback = "subset"
            self.stats["fallback"] = self.fallback
        elif self.random_run_fallback:
            self.fallback = "random_run"
            self.stats["fallback"] = self.fallback
        elif self.zero_accept_subset and self.stats["tree_accepted"] == 0:
            self.fallback = "subset"
            self.stats["fallback"] = self.fallback
        else:
            self._initialize_gain_fallback()

    def __call__(self):
        if not self.warm_finished:
            proposal = self._mode_proposal()
            if proposal is not None:
                self.stats["warm_mode_proposals"] += 1
            return proposal

        if self.heap:
            _, _, _, robots, first, last, depth = heapq.heappop(self.heap)
            self.stats["tree_proposals"] += 1
            size = str(len(robots))
            counts = self.stats["tree_proposals_by_size"]
            counts[size] = counts.get(size, 0) + 1
            return Proposal(first, last, robots, source="tree", tree_depth=depth)

        if self.tree_only:
            self.stats["prefix_exhausted"] = True
            return None

        if self.fallback is None:
            self._initialize_fallback()

        if self.fallback == "subset":
            i, j = _ordered_pair(self.rng, 0, len(self.path) - 1)
            if j - i < 2:
                return None
            size = int(self.rng.integers(1, self.robot_count + 1))
            robots = tuple(
                sorted(
                    int(value)
                    for value in self.rng.choice(
                        self.robot_count, size=size, replace=False
                    )
                )
            )
            for robot in robots:
                if self.path[i].mode.task_ids[robot] != self.path[j].mode.task_ids[robot]:
                    return None
            self.stats["subset_proposals"] += 1
            return Proposal(i, j, robots, source="subset")

        if self.fallback == "mode":
            proposal = self._mode_proposal()
            if proposal is not None:
                self.stats["mode_proposals"] += 1
            return proposal

        if self.fallback == "random_run":
            proposal = self._random_run_proposal()
            if proposal is not None:
                self.stats["random_run_proposals"] += 1
            return proposal

        if self.fallback == "group_gain":
            arm_index = max(
                range(len(self.group_gain_arms)),
                key=lambda index: (
                    (
                        self.group_gain_arms[index]["gain"]
                        * (1.0 + self.group_gain_arms[index]["accepted"])
                        / (2.0 + self.group_gain_arms[index]["proposed"])
                        + self.gain_floor
                    )
                    / (self.group_gain_arms[index]["used"] + 1),
                    -index,
                ),
            )
            arm = self.group_gain_arms[arm_index]
            arm["used"] += 1
            i, j = _ordered_pair(self.rng, arm["first"], arm["last"])
            if j - i < 2:
                return None
            arm["proposed"] += 1
            size = str(len(arm["robots"]))
            counts = self.stats["group_gain_proposals_by_size"]
            counts[size] = counts.get(size, 0) + 1
            self.stats["group_gain_proposals"] += 1
            return Proposal(
                i,
                j,
                arm["robots"],
                source="group_gain",
                arm=(arm_index,),
            )


        robot, run = max(
            (
                (robot, run)
                for robot, robot_runs in enumerate(self.runs)
                for run in range(len(robot_runs))
            ),
            key=lambda arm: (
                self.gains[arm[0]][arm[1]]
                * (1.0 + self.accepted[arm[0]][arm[1]])
                / (2.0 + self.proposed[arm[0]][arm[1]])
                + self.gain_floor
            )
            / (self.used[arm[0]][arm[1]] + 1),
        )
        self.used[robot][run] += 1
        first, last = self.runs[robot][run]
        i, j = _ordered_pair(self.rng, first, last)
        if j - i < 2:
            return None
        self.proposed[robot][run] += 1
        self.stats["gain_proposals"] += 1
        return Proposal(i, j, (robot,), source="gain", arm=(robot, run))

    def _random_run_proposal(self):
        eligible_robots = [
            robot
            for robot, robot_runs in enumerate(self.runs)
            if robot_runs
        ]
        robot = eligible_robots[int(self.rng.integers(0, len(eligible_robots)))]
        robot_runs = self.runs[robot]
        first, last = robot_runs[
            int(self.rng.integers(0, len(robot_runs)))
        ]
        i, j = _ordered_pair(self.rng, first, last)
        if j - i < 2:
            return None
        return Proposal(i, j, (robot,), source="random_run")

    def _mode_proposal(self):
        robot, first, last = self.eligible_runs[
            int(self.rng.integers(0, len(self.eligible_runs)))
        ]
        i, j = _ordered_pair(self.rng, first, last)
        if j - i < 2:
            return None
        return Proposal(i, j, (robot,), source="mode")

    def feedback(self, outcome, proposal, fail_index=None, **details):
        if proposal.source == "tree":
            if outcome == "accepted":
                self.stats["tree_accepted"] += 1
                size = str(len(proposal.robots))
                counts = self.stats["tree_accepted_by_size"]
                counts[size] = counts.get(size, 0) + 1
            elif outcome == "collision":
                self.stats["tree_collisions"] += 1
                if (
                    self.split_tree_failures
                    and (
                        self.max_tree_depth is None
                        or proposal.tree_depth < self.max_tree_depth
                    )
                    and fail_index is not None
                    and proposal.first < fail_index < proposal.last
                ):
                    self._push(
                        proposal.robots, proposal.first, fail_index,
                        proposal.tree_depth + 1,
                    )
                    self._push(
                        proposal.robots, fail_index, proposal.last,
                        proposal.tree_depth + 1,
                    )
                    self.stats["tree_failure_splits"] += 1
            elif outcome == "cost":
                self.stats["tree_cost_rejects"] += 1
            return

        if proposal.source == "group_gain":
            if outcome == "accepted":
                arm = self.group_gain_arms[proposal.arm[0]]
                arm["accepted"] += 1
                size = str(len(arm["robots"]))
                counts = self.stats["group_gain_accepted_by_size"]
                counts[size] = counts.get(size, 0) + 1
                self.stats["group_gain_accepted"] += 1
                for other in self.group_gain_arms:
                    if (
                        other["first"] < proposal.last
                        and proposal.first < other["last"]
                    ):
                        other["gain"] = max(
                            0.0,
                            _group_run_gain(
                                self.env,
                                self.path,
                                other["robots"],
                                other["first"],
                                other["last"],
                            ),
                        )
            return


        if proposal.source != "gain" or outcome != "accepted":
            return
        robot, run = proposal.arm
        self.accepted[robot][run] += 1
        for other_robot, robot_runs in enumerate(self.runs):
            for other_run, (first, last) in enumerate(robot_runs):
                if first < proposal.last and proposal.first < last:
                    self.gains[other_robot][other_run] = max(
                        0.0,
                        _full_run_gain(
                            self.env, self.path, other_robot, first, last
                        ),
                    )


class _FullTreeGainRoundRobinProposer:
    """Round robin over full-configuration, tree, and gain proposals.

    The collision-splitting tree is a finite prefix. Once it is exhausted,
    scheduling continues over full-configuration and gain proposals instead
    of silently turning the tree arm into a second gain arm.
    """

    def __init__(self, env, path, rng, robot_runs):
        self.path = path
        self.rng = rng
        self.all_robots = tuple(range(len(env.robots)))
        self.full_runs = _constant_runs(path, lambda state: state.mode)
        self.tree = _TreePrefixProposer(
            env,
            path,
            rng,
            robot_runs,
            tree_only=True,
        )
        self.gain = _TreePrefixProposer(
            env,
            path,
            rng,
            robot_runs,
            skip_tree_prefix=True,
        )
        self.arms = ["full", "tree", "gain"]
        self.index = 0
        self.selected = None
        self.proposals = {arm: 0 for arm in self.arms}
        self.accepted = {arm: 0 for arm in self.arms}

    def __call__(self):
        while self.arms:
            self.index %= len(self.arms)
            arm = self.arms[self.index]
            self.index = (self.index + 1) % len(self.arms)

            if arm == "full":
                if not self.full_runs:
                    removed = self.arms.index("full")
                    self.arms.pop(removed)
                    if removed < self.index:
                        self.index -= 1
                    continue
                first, last = self.full_runs[
                    int(self.rng.integers(0, len(self.full_runs)))
                ]
                i, j = _ordered_pair(self.rng, first, last)
                proposal = (
                    Proposal(i, j, self.all_robots, source="round_robin_full")
                    if j - i >= 2
                    else None
                )
            elif arm == "tree":
                proposal = self.tree()
                if proposal is None:
                    removed = self.arms.index("tree")
                    self.arms.pop(removed)
                    if removed < self.index:
                        self.index -= 1
                    continue
            else:
                proposal = self.gain()

            self.selected = arm
            self.proposals[arm] += 1
            return proposal
        return None

    def feedback(self, outcome, proposal, **details):
        arm = self.selected
        if arm is None:
            return
        if outcome == "accepted":
            self.accepted[arm] += 1
        if arm == "tree":
            self.tree.feedback(outcome, proposal, **details)
        elif arm == "gain":
            self.gain.feedback(outcome, proposal, **details)

    @property
    def stats(self):
        return {
            "round_robin_arms": ["full", "tree", "gain"],
            "round_robin_active_arms": list(self.arms),
            "round_robin_proposals": dict(self.proposals),
            "round_robin_accepted": dict(self.accepted),
            "round_robin_tree_stats": dict(self.tree.stats),
            "round_robin_gain_stats": dict(self.gain.stats),
        }


class _WafrSlideHaltonProposer:
    """Paper-faithful Slide+Halton scheduling over independent safe runs.

    The WAFR paper assumes one continuous path.  Our paths have discrete mode
    boundaries, so each supplied mode-safe run is treated as an independent
    path and visited round-robin.  Within each run the published 247 sliding
    intervals, 15-miss switch, and scrambled 2-D Halton sequence are used
    unchanged.
    """

    def __init__(self, env, path, rng, run_specs, adaptation):
        if not run_specs:
            raise ValueError("path has no eligible run spanning at least two edges")
        from scipy.stats import qmc

        sliding = []
        for level in range(7):
            length = 2.0 ** (-level)
            step = length * 0.5
            lower = 0.0
            upper = length
            while upper < 1.0:
                sliding.append((lower, upper))
                lower += step
                upper += step
            sliding.append((lower, 1.0))
        if len(sliding) != 247:
            raise RuntimeError(f"expected 247 WAFR sliding intervals, got {len(sliding)}")

        self.env = env
        self.path = path
        self.sliding = tuple(sliding)
        self.next_run = 0
        self.runs = []
        for first, last, robots in run_specs:
            halton_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            self.runs.append({
                "first": first,
                "last": last,
                "robots": tuple(robots),
                "sliding_index": 0,
                "recent_improvements": deque(maxlen=15),
                "switched": False,
                "halton": qmc.Halton(d=2, scramble=True, seed=halton_seed),
                "cumulative": None,
            })
        self.stats = {
            "wafr_adaptation": adaptation,
            "wafr_runs": len(self.runs),
            "wafr_sliding_intervals_per_run": len(self.sliding),
            "wafr_sliding_proposals": 0,
            "wafr_halton_proposals": 0,
            "wafr_early_switches": 0,
            "wafr_completed_prefixes": 0,
        }

    def _sliding_miss(self, state):
        state["recent_improvements"].append(False)
        if (
            not state["switched"]
            and len(state["recent_improvements"]) == 15
            and not any(state["recent_improvements"])
        ):
            state["switched"] = True
            self.stats["wafr_early_switches"] += 1

    def _fraction_index(self, state, fraction):
        cumulative = state["cumulative"]
        if cumulative is None:
            cumulative = [0.0]
            for index in range(state["first"], state["last"]):
                if len(state["robots"]) == len(self.env.robots):
                    edge_cost = float(
                        path_cost(
                            self.path[index : index + 2],
                            self.env.batch_config_cost,
                        )
                    )
                else:
                    # A constant scalar would not change the arclength
                    # fractions. Euclidean distance is the benchmark's
                    # per-robot geometric edge cost before reduction.
                    robot = state["robots"][0]
                    edge_cost = float(np.linalg.norm(
                        np.asarray(self.path[index + 1].q[robot], dtype=float)
                        - np.asarray(self.path[index].q[robot], dtype=float)
                    ))
                cumulative.append(cumulative[-1] + max(0.0, edge_cost))
            state["cumulative"] = cumulative
        if cumulative[-1] <= 0.0:
            return state["first"]
        target = float(fraction) * cumulative[-1]
        offset = int(np.searchsorted(cumulative, target, side="left"))
        if 0 < offset < len(cumulative):
            before = cumulative[offset - 1]
            after = cumulative[offset]
            if target - before <= after - target:
                offset -= 1
        return state["first"] + min(offset, len(cumulative) - 1)

    def __call__(self):
        run_index = self.next_run
        self.next_run = (self.next_run + 1) % len(self.runs)
        state = self.runs[run_index]
        source = "wafr_halton"

        if not state["switched"] and state["sliding_index"] < len(self.sliding):
            lower, upper = self.sliding[state["sliding_index"]]
            state["sliding_index"] += 1
            source = "wafr_sliding"
            self.stats["wafr_sliding_proposals"] += 1
            if state["sliding_index"] == len(self.sliding):
                state["switched"] = True
                self.stats["wafr_completed_prefixes"] += 1
        else:
            lower, upper = sorted(state["halton"].random(1)[0])
            self.stats["wafr_halton_proposals"] += 1

        first = self._fraction_index(state, lower)
        last = self._fraction_index(state, upper)
        if last - first < 2:
            if source == "wafr_sliding":
                self._sliding_miss(state)
            return None
        return Proposal(
            first,
            last,
            state["robots"],
            source=source,
            arm=(run_index,),
        )

    def feedback(self, outcome, proposal, fail_index=None, **details):
        if proposal.source != "wafr_sliding":
            return
        state = self.runs[proposal.arm[0]]
        improved = outcome == "accepted"
        if improved:
            state["cumulative"] = None
        state["recent_improvements"].append(improved)
        if (
            not state["switched"]
            and len(state["recent_improvements"]) == 15
            and not any(state["recent_improvements"])
        ):
            state["switched"] = True
            self.stats["wafr_early_switches"] += 1


class _HuangMultiStrategyProposer:
    """Huang et al. composite/prioritized/path selector on a fixed mode grid."""

    STRATEGIES = ("composite", "prioritized", "path")

    def __init__(
        self,
        env,
        path,
        rng,
        robot_runs,
        full_runs,
        interpolation_resolution,
        selector,
        strategies=None,
    ):
        if not full_runs:
            raise ValueError("path has no full-mode run spanning at least two edges")
        self.env = env
        self.path = path
        self.rng = rng
        self.robot_runs = [
            (robot, first, last)
            for robot, runs in enumerate(robot_runs)
            for first, last in runs
        ]
        if not self.robot_runs:
            raise ValueError("path has no robot/task run spanning at least two edges")
        self.full_runs = list(full_runs)
        self.all_robots = tuple(range(len(env.robots)))
        self.interpolation_resolution = float(interpolation_resolution)
        self.selector = selector
        self.strategies = tuple(strategies or self.STRATEGIES)
        unknown = set(self.strategies) - set(self.STRATEGIES)
        if unknown or not self.strategies:
            raise ValueError(f"invalid Huang strategies: {self.strategies}")
        self.round_robin_index = 0
        self.alphas = np.asarray(
            [10.0 if name == "composite" else 1.0 for name in self.strategies],
            dtype=float,
        )
        self.betas = np.ones(len(self.strategies), dtype=float)
        self.stats = {
            "huang_selector": selector,
            "huang_strategies": list(self.strategies),
            "huang_adaptation": (
                "published three strategies and selector on fixed mode-compatible "
                "grid; existing benchmark path cost replaces makespan"
            ),
            "huang_selected": {name: 0 for name in self.strategies},
            "huang_accepted": {name: 0 for name in self.strategies},
            "huang_no_need": {name: 0 for name in self.strategies},
            "huang_alpha": self.alphas.tolist(),
            "huang_beta": self.betas.tolist(),
            "huang_sigma": 0.01,
            "huang_gamma_alpha": 100.0,
            "huang_gamma_beta": 0.1,
            "huang_normalization": 1000.0,
        }

    def _select_strategy(self):
        if self.selector == "round_robin":
            index = self.round_robin_index % len(self.strategies)
            self.round_robin_index += 1
            return index
        samples = self.rng.beta(self.alphas, self.betas)
        return int(np.argmax(samples))

    def _sample_pair(self, first, last):
        while True:
            lower, upper = _ordered_pair(self.rng, first, last)
            if upper - lower >= 2:
                return lower, upper

    def __call__(self):
        strategy = self._select_strategy()
        name = self.strategies[strategy]
        self.stats["huang_selected"][name] += 1

        if name == "composite":
            run_index = int(self.rng.integers(0, len(self.full_runs)))
            run_first, run_last = self.full_runs[run_index]
            first, shortcut_end = self._sample_pair(run_first, run_last)
            motion_steps = _minimum_motion_steps(
                self.env,
                self.path,
                first,
                shortcut_end,
                self.all_robots,
                self.interpolation_resolution,
            )
            return Proposal(
                first,
                shortcut_end,
                self.all_robots,
                source="huang",
                arm=(strategy,),
                shortcut_type="composite",
                shortcut_end=shortcut_end,
                motion_steps=motion_steps,
                force_cost_reject=motion_steps >= shortcut_end - first,
            )

        robot, run_first, run_last = self.robot_runs[
            int(self.rng.integers(0, len(self.robot_runs)))
        ]
        first, shortcut_end = self._sample_pair(run_first, run_last)
        if name == "path":
            return Proposal(
                first,
                shortcut_end,
                (robot,),
                source="huang",
                arm=(strategy,),
                shortcut_type="path",
                shortcut_end=shortcut_end,
            )

        motion_steps = _minimum_motion_steps(
            self.env,
            self.path,
            first,
            shortcut_end,
            (robot,),
            self.interpolation_resolution,
        )
        return Proposal(
            first,
            run_last,
            (robot,),
            source="huang",
            arm=(strategy,),
            shortcut_type="prioritized",
            shortcut_end=shortcut_end,
            motion_steps=motion_steps,
            force_cost_reject=motion_steps >= shortcut_end - first,
        )

    def feedback(self, outcome, proposal, fail_index=None, **details):
        strategy = proposal.arm[0]
        name = self.strategies[strategy]
        if proposal.force_cost_reject:
            self.stats["huang_no_need"][name] += 1
        if outcome == "accepted":
            self.stats["huang_accepted"][name] += 1

        if self.selector == "thompson":
            if outcome == "accepted":
                elapsed = float(details.get("attempt_seconds") or 0.0)
                relative_gain = max(0.0, float(details.get("relative_gain") or 0.0))
                reward = relative_gain + max(0.0, 1.0 - elapsed / 0.01)
                self.alphas[strategy] += 100.0 * reward
            else:
                self.betas[strategy] += 0.1
            total = self.alphas[strategy] + self.betas[strategy]
            if total > 1000.0:
                self.alphas[strategy] = 1000.0 * self.alphas[strategy] / total
                self.betas[strategy] = 1000.0 * self.betas[strategy] / total

        self.stats["huang_alpha"] = self.alphas.tolist()
        self.stats["huang_beta"] = self.betas.tolist()


def _check_candidate(env, candidate, observe_failure):
    failing_q = [None]
    original_edge_check = None
    original_config_check = None
    if observe_failure:
        original_edge_check = env.is_edge_collision_free
        original_config_check = env.is_collision_free

        def edge_observer(q, *args, **kwargs):
            free = original_edge_check(q, *args, **kwargs)
            if not free:
                failing_q[0] = q
            return free

        def config_observer(q, *args, **kwargs):
            free = original_config_check(q, *args, **kwargs)
            if not free:
                failing_q[0] = q
            return free

        env.is_edge_collision_free = edge_observer
        env.is_collision_free = config_observer
    try:
        collision_free = env.is_path_collision_free(
            candidate,
            resolution=env.collision_resolution,
            tolerance=env.collision_tolerance,
            check_start_and_end=False,
            check_edges_in_order=True,
            binary_order=False,
        )
    finally:
        if observe_failure:
            env.is_edge_collision_free = original_edge_check
            env.is_collision_free = original_config_check

    fail_offset = None
    if not collision_free and failing_q[0] is not None:
        for offset, state in enumerate(candidate):
            if state.q is failing_q[0]:
                fail_offset = offset
                break
    return collision_free, fail_offset


def _interpolate_mode_safe(env, path, resolution: float, validate_input: bool = True):
    """Densify same-mode edges, falling back if extra samples expose a collision."""
    dense = []
    for index in range(len(path) - 1):
        start = path[index]
        end = path[index + 1]
        segments = 1
        if start.mode == end.mode:
            segments = max(1, int(config_dist(start.q, end.q, "max") / resolution))
        q0 = start.q.state()
        delta = (end.q.state() - q0) / segments
        for step in range(segments):
            dense.append(State(start.q.from_flat(q0 + delta * step), start.mode))
    dense.append(State(path[-1].q.from_flat(path[-1].q.state().copy()), path[-1].mode))

    if not validate_input:
        return dense, False

    certificate = dict(
        resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance),
        check_start_and_end=True,
        check_edges_in_order=True,
        binary_order=False,
    )
    if env.is_path_collision_free(dense, **certificate):
        return dense, False
    if not env.is_path_collision_free(path, **certificate):
        raise ValueError("input path failed strict collision validation")
    raw_copy = [State(state.q.from_flat(state.q.state().copy()), state.mode) for state in path]
    return raw_copy, True

def _empty_stats():
    return {
        "draws": 0,
        "compatible": 0,
        "cost_passing": 0,
        "collision_checks": 0,
        "accepted": 0,
        "collision_check_seconds_valid": 0.0,
        "collision_check_seconds_invalid": 0.0,
    }


def _new_lineage(path, robot_count):
    return {
        "owners": [[-1] * robot_count for _ in path],
        "events": [],
    }


def _record_lineage(lineage, proposal, local_gain, global_gain, check_seconds):
    if lineage is None:
        return
    event_id = len(lineage["events"])
    written_slots = max(0, proposal.last - proposal.first - 1) * len(proposal.robots)
    event = {
        "id": event_id,
        "first": proposal.first,
        "last": proposal.last,
        "robots": list(proposal.robots),
        "source": proposal.source,
        "local_gain": float(local_gain),
        "global_gain": float(global_gain),
        "check_seconds": float(check_seconds),
        "written_slots": written_slots,
    }
    lineage["events"].append(event)
    for index in range(proposal.first + 1, proposal.last):
        for robot in proposal.robots:
            lineage["owners"][index][robot] = event_id


def _finalize_lineage(lineage):
    survivors = [0] * len(lineage["events"])
    for row in lineage["owners"]:
        for owner in row:
            if owner >= 0:
                survivors[owner] += 1
    events = []
    for event, surviving_slots in zip(lineage["events"], survivors):
        record = dict(event)
        record["surviving_slots"] = surviving_slots
        record["surviving_fraction"] = (
            surviving_slots / event["written_slots"] if event["written_slots"] else 0.0
        )
        events.append(record)
    return events


def _attempt_once(env, path, propose, stats, current_cost, lineage=None):
    attempt_started = time.perf_counter()
    stats["draws"] += 1
    proposal = propose()
    if proposal is None:
        return current_cost, False

    stats["compatible"] += 1
    if proposal.force_cost_reject:
        if hasattr(propose, "feedback"):
            propose.feedback(
                "cost",
                proposal,
                attempt_seconds=time.perf_counter() - attempt_started,
                gain=0.0,
                relative_gain=0.0,
            )
        return current_cost, False

    candidate = _candidate(env, path, proposal)
    old_cost = float(
        path_cost(path[proposal.first : proposal.last + 1], env.batch_config_cost)
    )
    candidate_cost = float(path_cost(candidate, env.batch_config_cost))
    if candidate_cost >= old_cost:
        if hasattr(propose, "feedback"):
            propose.feedback(
                "cost",
                proposal,
                attempt_seconds=time.perf_counter() - attempt_started,
                gain=0.0,
                relative_gain=0.0,
            )
        return current_cost, False

    stats["cost_passing"] += 1
    stats["collision_checks"] += 1
    check_started = time.perf_counter()
    collision_free, fail_offset = _check_candidate(
        env,
        candidate,
        observe_failure=proposal.source == "tree",
    )
    check_seconds = time.perf_counter() - check_started
    if not collision_free:
        stats["collision_check_seconds_invalid"] += check_seconds
        if hasattr(propose, "feedback"):
            fail_index = (
                proposal.first + fail_offset if fail_offset is not None else None
            )
            propose.feedback(
                "collision",
                proposal,
                fail_index=fail_index,
                attempt_seconds=time.perf_counter() - attempt_started,
                check_seconds=check_seconds,
                gain=0.0,
                relative_gain=0.0,
            )
        return current_cost, False

    stats["collision_check_seconds_valid"] += check_seconds
    for offset, state in enumerate(candidate):
        path[proposal.first + offset].q = state.q
    stats["accepted"] += 1
    new_cost = float(path_cost(path, env.batch_config_cost))
    _record_lineage(
        lineage,
        proposal,
        old_cost - candidate_cost,
        current_cost - new_cost,
        check_seconds,
    )
    if hasattr(propose, "feedback"):
        propose.feedback(
            "accepted",
            proposal,
            attempt_seconds=time.perf_counter() - attempt_started,
            check_seconds=check_seconds,
            gain=max(0.0, current_cost - new_cost),
            relative_gain=max(0.0, current_cost - new_cost) / max(current_cost, 1e-12),
        )
    return new_cost, True


def run(
    env,
    input_path,
    policy: str,
    seconds: float,
    seed: int,
    interpolation_resolution: float = 0.5,
) -> tuple[list, dict[str, Any]]:
    if policy not in available_shortcutters():
        raise ValueError(
            f"unknown policy {policy!r}; choose from {available_shortcutters()}"
        )

    path, interpolation_fallback = _interpolate_mode_safe(
        env, input_path, interpolation_resolution
    )
    initial_cost = float(path_cost(path, env.batch_config_cost))
    current_cost = initial_cost
    rng = np.random.default_rng(seed)
    propose = _registered_proposer(
        policy, env, path, rng, interpolation_resolution
    )
    stats = _empty_stats()
    lineage = _new_lineage(path, len(env.robots))
    trace = [{"time": 0.0, "cost": initial_cost, "collision_checks": 0}]
    started = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= seconds:
            break
        if hasattr(propose, "consider_time"):
            propose.consider_time(elapsed, current_cost)
        current_cost, accepted = _attempt_once(
            env, path, propose, stats, current_cost, lineage=lineage
        )
        if accepted:
            trace.append({
                "time": time.perf_counter() - started,
                "cost": current_cost,
                "collision_checks": stats["collision_checks"],
            })

    elapsed = time.perf_counter() - started
    if trace[-1]["time"] < elapsed:
        trace.append({
            "time": elapsed,
            "cost": current_cost,
            "collision_checks": stats["collision_checks"],
        })
    if hasattr(propose, "stats"):
        stats.update(propose.stats)
    stats["accepted_lineage"] = _finalize_lineage(lineage)

    return path, {
        "policy": policy,
        "seed": seed,
        "budget_seconds": seconds,
        "wall_seconds": elapsed,
        "interpolation_resolution": interpolation_resolution,
        "interpolation_fallback_to_raw": interpolation_fallback,
        "initial_states": len(input_path),
        "working_states": len(path),
        "initial_cost": initial_cost,
        "final_cost": current_cost,
        "trace": trace,
        "stats": stats,
    }

def run_checks(
    env,
    input_path,
    policy: str,
    max_collision_checks: int,
    seed: int,
    interpolation_resolution: float = 0.5,
    seconds: float | None = None,
    max_attempts: int | None = None,
) -> tuple[list, dict[str, Any]]:
    """Run one planner-integrated shortcut pass with a check budget.

    The upstream planner requests 100 shortcut checks for each incumbent.  This
    adapter preserves that contract while allowing the benchmark policies to be
    inserted into the same loop.  ``seconds`` is an additional shared-deadline
    guard; it never grants work beyond the collision-check budget.
    """
    if policy not in available_shortcutters():
        raise ValueError(f"unsupported in-loop policy {policy!r}")
    if max_collision_checks < 0:
        raise ValueError("max_collision_checks must be nonnegative")

    started = time.perf_counter()
    path, interpolation_fallback = _interpolate_mode_safe(
        env, input_path, interpolation_resolution, validate_input=False
    )
    initial_cost = float(path_cost(path, env.batch_config_cost))
    current_cost = initial_cost
    rng = np.random.default_rng(seed)
    propose = _registered_proposer(
        policy, env, path, rng, interpolation_resolution
    )
    stats = _empty_stats()
    lineage = _new_lineage(path, len(env.robots))
    trace = [{"time": 0.0, "cost": initial_cost, "collision_checks": 0}]
    attempt_limit = (
        max(2500, 5 * max_collision_checks)
        if max_attempts is None
        else int(max_attempts)
    )

    while (
        stats["collision_checks"] < max_collision_checks
        and stats["draws"] < attempt_limit
    ):
        elapsed = time.perf_counter() - started
        if seconds is not None and elapsed >= seconds:
            break
        if hasattr(propose, "consider_time"):
            propose.consider_time(elapsed, current_cost)
        current_cost, accepted = _attempt_once(
            env, path, propose, stats, current_cost, lineage=lineage
        )
        if accepted:
            trace.append({
                "time": time.perf_counter() - started,
                "cost": current_cost,
                "collision_checks": stats["collision_checks"],
            })

    elapsed = time.perf_counter() - started
    if trace[-1]["time"] < elapsed:
        trace.append({
            "time": elapsed,
            "cost": current_cost,
            "collision_checks": stats["collision_checks"],
        })
    if hasattr(propose, "stats"):
        stats.update(propose.stats)
    stats["accepted_lineage"] = _finalize_lineage(lineage)
    return path, {
        "policy": policy,
        "seed": seed,
        "max_collision_checks": max_collision_checks,
        "max_attempts": attempt_limit,
        "seconds_guard": seconds,
        "wall_seconds": elapsed,
        "interpolation_resolution": interpolation_resolution,
        "interpolation_fallback_to_raw": interpolation_fallback,
        "initial_states": len(input_path),
        "working_states": len(path),
        "initial_cost": initial_cost,
        "final_cost": current_cost,
        "trace": trace,
        "stats": stats,
    }
