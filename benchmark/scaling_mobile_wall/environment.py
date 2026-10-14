"""WAFR mobile-wall scaling environment adapter."""
from __future__ import annotations


def mobile_wall_scaling_env(num_robots: int, wall_x: int, wall_z: int):
    """Build the small-block sequential wall used by the WAFR scaling study."""
    from multi_robot_multi_goal_planning.problems.rai.rai_envs import (
        rai_mobile_manip_wall,
    )

    env = rai_mobile_manip_wall(
        num_robots=int(num_robots),
        wall_x=int(wall_x),
        wall_z=int(wall_z),
    )
    env.scene_metadata = {
        "study": "wafr_mobile_wall_scaling",
        "num_robots": int(num_robots),
        "wall_x": int(wall_x),
        "wall_z": int(wall_z),
        "num_blocks": int(wall_x) * int(wall_z),
    }
    return env
