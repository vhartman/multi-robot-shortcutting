"""Opt-in sampling variants for generating benchmark input paths."""
from __future__ import annotations

import numpy as np

from multi_robot_multi_goal_planning.planners.collision_free_sampler import (
    GibbsSampler,
    SamplingResult,
)


class RandomScanGibbsSampler(GibbsSampler):
    """Update one random unpinned robot, conditional on the other robots.

    Like upstream Gibbs, initialize from the planner's tree seed. Unlike its
    full sweep, modify just one robot per call. Bounded rejection retains the
    old coordinate on exhaustion; always validate the complete returned state.
    Pinning may invalidate the seed, in which case this call can return None.
    This is a tree-seeded local sampler, not a claim of uniform joint sampling.
    """

    def _sample_result(self, mode, pinned):
        seed = self.seed_fn(mode) if self.seed_fn else self.env.get_start_pos()
        q = seed.from_flat(seed.state().copy())
        self._apply_pinned(q, pinned)
        available = [i for i, robot in enumerate(self.env.robots) if robot not in pinned]
        checks = 0
        if available:
            index = np.random.choice(available)
            old = q[index].copy()
            limits = self.env.limits[:, self.env.robot_idx[self.env.robots[index]]]
            for _ in range(self.config.max_per_robot_attempts):
                q[index] = np.random.uniform(limits[0], limits[1])
                checks += 1
                if self.env.is_collision_free(q, mode):
                    return SamplingResult(q, checks, "gibbs_random_scan")
            q[index] = old
        checks += 1
        valid = self.env.is_collision_free(q, mode)
        return SamplingResult(q if valid else None, checks, "gibbs_random_scan")
