from __future__ import annotations

import random

import numpy as np


def coupled_box_stacking_env(
    *,
    num_robots: int = 6,
    num_boxes: int = 8,
    environment_seed: int = 10,
):
    """Construct box stacking with all UR10s sharing one table and one stack.

    The upstream environment supports at most six robots. For five or six
    robots it uses the four corner positions plus bases on the left and right
    sides of the table, with every arm facing toward the shared workspace.
    The default seed assigns at least one box to every robot.
    """
    if not 1 <= num_robots <= 6:
        raise ValueError("coupled box stacking supports one through six robots")
    if not 1 <= num_boxes <= 9:
        raise ValueError("coupled box stacking supports one through nine boxes")

    np.random.seed(environment_seed)
    random.seed(environment_seed)

    from multi_robot_multi_goal_planning.problems.rai.rai_envs import (
        rai_ur10_arm_box_stack_env,
    )

    return rai_ur10_arm_box_stack_env(
        num_robots=num_robots,
        num_boxes=num_boxes,
        robot_type="ur10",
    )
