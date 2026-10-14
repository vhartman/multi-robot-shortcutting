"""Render a certified scaling trajectory as a headless 3D MP4."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (BENCHMARK, BENCHMARK / "all_scaling"):
    sys.path.insert(0, str(directory))

import study_lib  # noqa: E402
from common import restore_mrbench_path  # noqa: E402
from multi_robot_multi_goal_planning.problems.planning_env import State  # noqa: E402
from multi_robot_multi_goal_planning.problems.rai.rai_config import (  # noqa: E402
    get_robot_joints,
)


ROBOT_COLORS = (
    "#3366AA", "#228833", "#CC6677", "#AA3377", "#EE7733",
    "#4477AA", "#66CCEE", "#999933", "#882255",
)


def read(path: Path):
    with path.open() as stream:
        return json.load(stream)


def records(result: dict, which: str):
    if which == "initial":
        return result["first_solution"]["states"]
    if which == "final":
        return result["selected_path"]
    raise ValueError(which)


def sampled_states(env, path, count: int):
    if count < 2:
        return [path[0]]
    distances = []
    transitions = []
    for first, second in zip(path[:-1], path[1:]):
        distance = max(
            float(np.linalg.norm(second.q[index] - first.q[index]))
            for index in range(len(env.robots))
        )
        distances.append(distance)
        transitions.append(first.mode != second.mode)
    positive = [value for value in distances if value > 1e-12]
    transition_weight = 0.25 * (float(np.median(positive)) if positive else 1.0)
    weights = np.asarray([
        max(distance, transition_weight) if transition else distance
        for distance, transition in zip(distances, transitions)
    ])
    cumulative = np.concatenate(([0.0], np.cumsum(weights)))
    targets = np.linspace(0.0, cumulative[-1], count)
    output = []
    for target in targets:
        edge = int(np.searchsorted(cumulative, target, side="right") - 1)
        edge = min(max(edge, 0), len(path) - 2)
        width = cumulative[edge + 1] - cumulative[edge]
        fraction = 0.0 if width <= 1e-15 else (target - cumulative[edge]) / width
        first, second = path[edge], path[edge + 1]
        if transitions[edge]:
            output.append(second if fraction >= 0.5 else first)
            continue
        flat = ((1.0 - fraction) * first.q.state()
                + fraction * second.q.state())
        output.append(State(first.q.from_flat(flat), first.mode))
    output[-1] = path[-1]
    return output


def set_state(env, state):
    env.set_to_mode(state.mode)
    for index, robot in enumerate(env.robots):
        env.C.setJointState(state.q[index], get_robot_joints(env.C, robot))


def visible(name: str) -> bool:
    lower = name.lower()
    if "goal" in lower:
        return False
    if "origin" in lower or "world" in lower:
        return False
    if lower.endswith("floor") or "gripper_center" in lower:
        return False
    return any(token in lower for token in (
        "table", "obj", "base", "link", "arm", "palm", "finger", "gripper",
        "coll", "cyl",
    ))


def frame_color(name: str, robots) -> tuple[str, float]:
    lower = name.lower()
    if "table" in lower:
        return "#B5B3AD", 0.93
    if "obj" in lower:
        return "#E5AD38", 1.0
    for index, robot in enumerate(robots):
        if robot.lower() in lower:
            return ROBOT_COLORS[index % len(ROBOT_COLORS)], 0.97
    match = re.search(r"(?:^|_)a(\d+)_", lower)
    if match:
        index = int(match.group(1))
        if any(robot.startswith("a0") for robot in robots):
            return ROBOT_COLORS[index % len(ROBOT_COLORS)], 0.97
        return ROBOT_COLORS[(index - 1) % len(ROBOT_COLORS)], 0.97
    return "#747B84", 0.92


def scene_mesh(env):
    faces = []
    colors = []
    points_for_bounds = []
    for frame in env.C.getFrames():
        if not visible(frame.name):
            continue
        points = np.asarray(frame.getMeshPoints())
        triangles = np.asarray(frame.getMeshTriangles(), dtype=int)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
                or triangles.ndim != 2 or triangles.shape[1] != 3
                or not len(triangles)):
            continue
        quaternion = np.asarray(frame.getQuaternion())
        rotation = Rotation.from_quat([
            quaternion[1], quaternion[2], quaternion[3], quaternion[0]
        ]).as_matrix()
        world = points @ rotation.T + np.asarray(frame.getPosition())
        color, alpha = frame_color(frame.name, env.robots)
        frame_faces = world[triangles]
        faces.extend(frame_faces)
        colors.extend([to_rgba(color, alpha)] * len(frame_faces))
        # Work surfaces are often deliberately much larger than the task.  Keep
        # drawing them, but let the moving geometry determine the camera crop.
        if "table" not in frame.name.lower():
            points_for_bounds.append(world)
    if not faces:
        raise RuntimeError("no visible meshes found")
    if not points_for_bounds:
        points_for_bounds = [np.vstack(faces)]
    return faces, colors, np.vstack(points_for_bounds)


def limits(env, path):
    clouds = []
    for state in (path[0], path[-1]):
        set_state(env, state)
        clouds.append(scene_mesh(env)[2])
    points = np.vstack(clouds)
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    widths = np.maximum(high - low, 1e-3)
    padding = np.array([0.05, 0.05, 0.08]) * widths
    return low - padding, high + padding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--which", choices=("initial", "final"), default="final")
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()

    case = study_lib.cases()[args.case]
    # Some scaling constructors print their generated task order.
    with contextlib.redirect_stdout(io.StringIO()):
        env = study_lib.build_environment(case)
    result = read(args.result)
    path = restore_mrbench_path(env, records(result, args.which))
    states = sampled_states(env, path, args.frames)
    low, high = limits(env, path)

    figure = plt.figure(figsize=(10.4, 7.2))
    axis = figure.add_subplot(111, projection="3d")
    set_state(env, states[0])
    faces, colors, _ = scene_mesh(env)
    collection = Poly3DCollection(
        faces, facecolors=colors, edgecolor="none", rasterized=True,
        zsort="average",
    )
    axis.add_collection3d(collection)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(max(0.0, low[2]), high[2])
    axis.set_box_aspect(high - low)
    axis.view_init(elev=55, azim=-48)
    axis.set_axis_off()
    axis.set_title(args.title, fontsize=18, pad=2)
    progress = axis.text2D(
        0.5, 0.015, "", transform=axis.transAxes, ha="center", fontsize=12,
        color="#444444",
    )
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.06, top=0.90)

    def update(index):
        set_state(env, states[index])
        faces, colors, _ = scene_mesh(env)
        collection.set_verts(faces)
        collection.set_facecolors(colors)
        progress.set_text(f"path progress  {100 * index / (len(states) - 1):.0f}%")
        return collection, progress

    update(0)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.preview, dpi=150, facecolor="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure, update, frames=len(states), interval=1000 / args.fps, blit=False,
    )
    movie.save(
        args.output,
        writer=animation.FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            bitrate=1800,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ),
        dpi=120,
    )
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
