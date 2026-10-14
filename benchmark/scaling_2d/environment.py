"""Nested disk-robot instances built from the benchmark's random-2D scene."""
from __future__ import annotations

import numpy as np
import robotic as ry

from multi_robot_multi_goal_planning.problems.rai.rai_config import make_random_two_dim
from multi_robot_multi_goal_planning.problems.rai_base_env import rai_env
from multi_robot_multi_goal_planning.problems.planning_env import (
    BaseModeLogic,
    ManipulationType,
    SequenceMixin,
    Task,
)
from multi_robot_multi_goal_planning.problems.core.goals import SingleGoal


class DiskScalingEnvironment(SequenceMixin, rai_env):
    def __init__(self, num_robots=4, num_goals=4, num_obstacles=10, max_robots=12, robot_radius=0.12):
        if not 1 <= num_robots <= max_robots or num_goals < 1:
            raise ValueError("invalid robot or goal count")
        self.C, _ = make_random_two_dim(
            num_agents=max_robots,
            num_goals=0,
            num_obstacles=num_obstacles,
            agents_can_rotate=False,
        )
        starts = self.C.getJointState().reshape(max_robots, 2).copy()
        shape_size = self.C.getFrame("a0").getSize().copy()
        shape_size[-1] = robot_radius
        for index in range(max_robots):
            self.C.delFrame(f"a{index}")
        for index in range(max_robots):
            self.C.addFrame(f"disk_{index:02d}").setParent(
                self.C.getFrame(f"pre_agent_{index}_frame")
            ).setShape(ry.ST.cylinder, size=shape_size).setContact(1).setJoint(
                ry.JT.transXY, limits=[-2, 2, -2, 2]
            ).setJointState(starts[index])
        keyframes = []
        candidate_scene = ry.Config()
        candidate_scene.addConfigurationCopy(self.C)
        for index in range(max_robots):
            previous = starts[index]
            for goal_index in range(num_goals):
                for attempt in range(10000):
                    candidate = np.random.uniform(-2.0, 2.0, size=2)
                    if np.linalg.norm(candidate - previous) < 0.8:
                        continue
                    candidate_scene.setJointState(starts.flatten())
                    candidate_scene.setJointState(candidate, [f"disk_{index:02d}"])
                    if candidate_scene.getCollisionFree():
                        keyframes.append(candidate)
                        previous = candidate
                        break
                else:
                    raise RuntimeError(f"cannot sample goal {goal_index} for disk {index}")
        self.robots = [f"disk_{index:02d}" for index in range(num_robots)]
        self.scene_metadata = {
            "arena_width": 4.0,
            "arena_height": 4.0,
            "starts": starts[:num_robots].tolist(),
            "goals": np.asarray(keyframes).reshape(max_robots, num_goals, 2)[:num_robots].tolist(),
            "robot_shape_size": shape_size.tolist(),
            "obstacles": [
                {
                    "position": self.C.getFrame(f"obs{index}").getPosition().tolist(),
                    "size": self.C.getFrame(f"obs{index}").getSize().tolist(),
                }
                for index in range(num_obstacles)
            ],
        }
        for index in range(num_robots, max_robots):
            self.C.delFrame(f"disk_{index:02d}")
            self.C.delFrame(f"pre_agent_{index}_frame")

        rai_env.__init__(self)
        self.tasks = [
            Task(
                f"goal_{robot}_{goal_index}",
                [robot],
                SingleGoal(keyframes[robot_index * num_goals + goal_index]),
            )
            for robot_index, robot in enumerate(self.robots)
            for goal_index in range(num_goals)
        ]
        self.sequence = []
        for goal_index in range(num_goals):
            for robot_index in np.random.permutation(max_robots):
                if robot_index < num_robots:
                    self.sequence.append(int(robot_index) * num_goals + goal_index)
        self.tasks.append(Task("terminal", self.robots, SingleGoal(self.C.getJointState())))
        self.sequence.append(len(self.tasks) - 1)
        BaseModeLogic.__init__(self)
        self.collision_tolerance = 0.0
        self.prev_mode = self.start_mode
        self.spec.manipulation = ManipulationType.STATIC
