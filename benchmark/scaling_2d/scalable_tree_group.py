"""Scalable interleaved tree/compatible-group shortcutting policy."""
from __future__ import annotations

import numpy as np

from mr_shortcutting.api import Proposal, Shortcutter, register_shortcutter
from mr_shortcutting.shortcutters import (
    _TreePrefixProposer,
    _constant_runs,
    _ordered_pair,
)


class _VectorizedRandomGroup:
    """Sample exact mode-compatible groups with no subset enumeration/cache."""

    def __init__(self, path, rng):
        self.path = path
        self.rng = rng
        self.robot_count = len(path[0].mode.task_ids)
        self.task_ids = np.asarray(
            [state.mode.task_ids for state in path], dtype=int
        )
        self.proposals = 0
        self.accepted = 0
        self.run_evaluations = 0
        self.proposals_by_size = {}
        self.accepted_by_size = {}

    def _runs(self, robots):
        self.run_evaluations += 1
        signatures = self.task_ids[:, robots]
        changed = np.any(signatures[1:] != signatures[:-1], axis=1)
        boundaries = [0]
        boundaries.extend(int(value) + 1 for value in np.flatnonzero(changed))
        boundaries.append(len(self.path))
        return [
            (first, stop - 1)
            for first, stop in zip(boundaries, boundaries[1:])
            if stop - first >= 3
        ]

    def __call__(self):
        size = int(self.rng.integers(1, self.robot_count + 1))
        robots = tuple(sorted(
            int(value)
            for value in self.rng.choice(
                self.robot_count, size=size, replace=False
            )
        ))
        runs = self._runs(robots)
        if not runs:
            return None
        first, last = runs[int(self.rng.integers(0, len(runs)))]
        i, j = _ordered_pair(self.rng, first, last)
        if j - i < 2:
            return None
        key = str(size)
        self.proposals += 1
        self.proposals_by_size[key] = self.proposals_by_size.get(key, 0) + 1
        return Proposal(i, j, robots, source="scalable_random_group")

    def feedback(self, outcome, proposal, **details):
        if outcome == "accepted":
            key = str(len(proposal.robots))
            self.accepted += 1
            self.accepted_by_size[key] = self.accepted_by_size.get(key, 0) + 1

    @property
    def stats(self):
        return {
            "strategy": "vectorized_random_compatible_group",
            "subset_cache_entries": 0,
            "state_complexity": "O(n * path states)",
            "proposals": self.proposals,
            "accepted": self.accepted,
            "run_evaluations": self.run_evaluations,
            "proposals_by_size": dict(self.proposals_by_size),
            "accepted_by_size": dict(self.accepted_by_size),
        }


class _TreeVectorGroupRoundRobin:
    """Alternate a finite singleton tree and scalable compatible groups."""

    def __init__(self, env, path, rng, runs):
        # Ignore numerical-noise rewrites. Besides doing no useful work, these
        # can move a clearance-boundary edge across the collision tolerance.
        self.minimum_cost_improvement = 1e-6
        self.tree = _TreePrefixProposer(
            env, path, rng, runs, tree_only=True
        )
        self.group = _VectorizedRandomGroup(path, rng)
        self.arms = ["tree", "group"]
        self.index = 0
        self.selected = None
        self.proposals = {"tree": 0, "group": 0}
        self.accepted = {"tree": 0, "group": 0}

    def __call__(self):
        while self.arms:
            self.index %= len(self.arms)
            arm = self.arms[self.index]
            self.index = (self.index + 1) % len(self.arms)
            proposal = self.tree() if arm == "tree" else self.group()
            if proposal is None and arm == "tree":
                removed = self.arms.index("tree")
                self.arms.pop(removed)
                if removed < self.index:
                    self.index -= 1
                continue
            self.selected = arm
            self.proposals[arm] += 1
            return proposal
        return None

    def feedback(self, outcome, proposal, **details):
        if self.selected is None:
            return
        if outcome == "accepted":
            self.accepted[self.selected] += 1
        target = self.tree if self.selected == "tree" else self.group
        target.feedback(outcome, proposal, **details)

    @property
    def stats(self):
        return {
            "policy": "scalable_tree_group",
            "schedule": "tree_group_round_robin_until_tree_exhausted",
            "minimum_cost_improvement": self.minimum_cost_improvement,
            "active_arms": list(self.arms),
            "proposals": dict(self.proposals),
            "accepted": dict(self.accepted),
            "tree_stats": dict(self.tree.stats),
            "group_stats": dict(self.group.stats),
        }


class _TreeThenVectorGroup:
    """Exhaust the finite singleton tree, then sample compatible groups."""

    def __init__(self, env, path, rng, runs):
        self.minimum_cost_improvement = 1e-6
        self.tree = _TreePrefixProposer(
            env, path, rng, runs, tree_only=True
        )
        self.group = _VectorizedRandomGroup(path, rng)
        self.phase = "tree"
        self.selected = None
        self.proposals = {"tree": 0, "group": 0}
        self.accepted = {"tree": 0, "group": 0}

    def __call__(self):
        if self.phase == "tree":
            proposal = self.tree()
            if proposal is not None:
                self.selected = "tree"
                self.proposals["tree"] += 1
                return proposal
            self.phase = "group"
        proposal = self.group()
        self.selected = "group"
        self.proposals["group"] += 1
        return proposal

    def feedback(self, outcome, proposal, **details):
        if self.selected is None:
            return
        if outcome == "accepted":
            self.accepted[self.selected] += 1
        target = self.tree if self.selected == "tree" else self.group
        target.feedback(outcome, proposal, **details)

    @property
    def stats(self):
        return {
            "policy": "tree_then_compatible_group",
            "schedule": "exhaust_singleton_tree_then_compatible_random_groups",
            "minimum_cost_improvement": self.minimum_cost_improvement,
            "phase": self.phase,
            "proposals": dict(self.proposals),
            "accepted": dict(self.accepted),
            "tree_stats": dict(self.tree.stats),
            "group_stats": dict(self.group.stats),
        }


class ScalableTreeGroupShortcutter(Shortcutter):
    name = "scalable_tree_group"

    def make_proposer(self, env, path, rng, interpolation_resolution):
        runs = [
            _constant_runs(
                path,
                lambda state, robot=robot: int(state.mode.task_ids[robot]),
            )
            for robot in range(len(env.robots))
        ]
        return _TreeVectorGroupRoundRobin(env, path, rng, runs)


class TreeThenCompatibleGroupShortcutter(Shortcutter):
    name = "tree_then_compatible_group"

    def make_proposer(self, env, path, rng, interpolation_resolution):
        runs = [
            _constant_runs(
                path,
                lambda state, robot=robot: int(state.mode.task_ids[robot]),
            )
            for robot in range(len(env.robots))
        ]
        return _TreeThenVectorGroup(env, path, rng, runs)


register_shortcutter(ScalableTreeGroupShortcutter(), replace=True)
register_shortcutter(TreeThenCompatibleGroupShortcutter(), replace=True)
