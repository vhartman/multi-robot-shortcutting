from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from environment import coupled_box_stacking_env


ROBOT_COLORS = [
    "#2a6fbb",
    "#318c70",
    "#cb6d35",
    "#8256a8",
    "#be4c72",
    "#827238",
]


def _world_vertices(frame) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(frame.getMeshPoints())
    triangles = np.asarray(frame.getMeshTriangles(), dtype=int)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or points.size == 0
        or triangles.size == 0
    ):
        return None

    quaternion = np.asarray(frame.getQuaternion())
    rotation = Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    ).as_matrix()
    points = points @ rotation.T + np.asarray(frame.getPosition())
    return points, triangles


def _frame_style(name: str) -> tuple[str, float] | None:
    if name == "floor" or name.startswith("goal"):
        return None
    if name == "table":
        return "#b7b7b2", 0.92
    if name.startswith("obj"):
        return "#e3ad3e", 1.0
    for index in range(6):
        if name.startswith(f"a{index + 1}_"):
            return ROBOT_COLORS[index], 0.96
    return "#747b84", 0.9


def render(output: Path) -> None:
    env = coupled_box_stacking_env(
        num_robots=6,
        num_boxes=8,
        environment_seed=10,
    )

    env.C.computeCollisions()
    penetrations = [collision for collision in env.C.getCollisions() if collision[2] < 0]
    if penetrations:
        raise RuntimeError(f"initial scene contains collisions: {penetrations}")

    figure = plt.figure(figsize=(10.8, 8.2), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")

    faces = []
    facecolors = []
    for frame in env.C.getFrames():
        style = _frame_style(frame.name)
        if style is None:
            continue
        mesh = _world_vertices(frame)
        if mesh is None:
            continue
        points, triangles = mesh
        color, alpha = style
        frame_faces = points[triangles]
        faces.extend(frame_faces)
        facecolors.extend([to_rgba(color, alpha)] * len(frame_faces))

    collection = Poly3DCollection(
        faces,
        facecolors=facecolors,
        edgecolor="none",
        rasterized=True,
        zsort="average",
    )
    axis.add_collection3d(collection)

    for index in range(6):
        base = env.C.getFrame(f"a{index + 1}_ur_base").getPosition()
        axis.text(
            base[0],
            base[1],
            base[2] + 0.08,
            str(index + 1),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#202124",
        )

    axis.set_xlim(-1.65, 1.65)
    axis.set_ylim(-1.65, 1.65)
    axis.set_zlim(0.15, 1.5)
    axis.set_box_aspect((3.3, 3.3, 1.35))
    axis.view_init(elev=57, azim=-48)
    axis.set_axis_off()
    axis.set_title(
        "Coupled box stacking — six UR10s sharing one table",
        fontsize=18,
        pad=4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)

    print(f"robots={len(env.robots)} dof={len(env.C.getJointState())} tasks={len(env.tasks)}")
    print("initial_collisions=0")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "analysis"
        / "coupled-box-stacking-six-robots.png",
    )
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
