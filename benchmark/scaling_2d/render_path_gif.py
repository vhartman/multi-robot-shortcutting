"""Render an actual certified disk-scaling path as a loopable GIF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent


def load_path(path: Path):
    with path.open() as stream:
        record = json.load(stream)
    certificate = record.get("certificate") or {}
    if (
        certificate.get("strict_full_edge") is not True
        or certificate.get("terminal_mode") is not True
        or certificate.get("valid_plan") is False
    ):
        raise RuntimeError(f"path does not carry a strict certificate: {certificate}")
    scene = record.get("scene")
    if not scene:
        raise RuntimeError("path does not contain disk-scene metadata")
    robot_count = len(record["robot_dofs"])
    if record["robot_dofs"] != [2] * robot_count:
        raise RuntimeError("renderer expects translating 2-DoF disk robots")
    positions = np.asarray([state["q"] for state in record["states"]], dtype=float)
    positions = positions.reshape(len(positions), robot_count, 2)
    modes = np.asarray([state["mode"] for state in record["states"]], dtype=int)
    return record, scene, positions, modes


def constant_motion_samples(positions, modes, count):
    edge_lengths = np.max(np.linalg.norm(np.diff(positions, axis=0), axis=2), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    targets = np.linspace(0.0, cumulative[-1], count)
    sampled_positions = []
    sampled_modes = []
    for target in targets:
        first = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(positions) - 2)
        width = cumulative[first + 1] - cumulative[first]
        fraction = 0.0 if width <= 0 else (target - cumulative[first]) / width
        sampled_positions.append(
            positions[first] + fraction * (positions[first + 1] - positions[first])
        )
        sampled_modes.append(modes[first + 1] if fraction >= 1.0 - 1e-12 else modes[first])
    return np.asarray(sampled_positions), np.asarray(sampled_modes)


def completed_goals(mode, robot_count, goals_per_robot):
    terminal = robot_count * goals_per_robot
    return np.asarray([
        goals_per_robot if task == terminal else int(np.clip(task - robot * goals_per_robot, 0, goals_per_robot))
        for robot, task in enumerate(mode)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "inputs" / "rai-disk-scaling-r12-g4" / "path0.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "analysis" / "disk-scaling-r12-path0.gif",
    )
    parser.add_argument("--motion-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if args.motion_frames < 2 or args.fps < 1:
        raise ValueError("motion frames and fps must be positive")

    record, scene, positions, modes = load_path(args.input.resolve())
    robot_count = positions.shape[1]
    goals = np.asarray(scene["goals"], dtype=float)
    goals_per_robot = goals.shape[1]
    motion, sampled_modes = constant_motion_samples(
        positions, modes, args.motion_frames
    )
    hold = args.fps
    frame_indices = np.concatenate((
        np.zeros(hold, dtype=int),
        np.arange(args.motion_frames, dtype=int),
        np.full(hold, args.motion_frames - 1, dtype=int),
    ))

    colors = [plt.get_cmap("tab20")(index) for index in range(robot_count)]
    radius = float(scene["robot_shape_size"][-1])
    figure, axis = plt.subplots(figsize=(7.2, 7.2))
    axis.set_aspect("equal")
    axis.set_xlim(-2.12, 2.12)
    axis.set_ylim(-2.12, 2.12)
    axis.set_xticks([-2, -1, 0, 1, 2])
    axis.set_yticks([-2, -1, 0, 1, 2])
    axis.tick_params(labelsize=10)
    axis.grid(color="#d7d7d7", linewidth=0.7, alpha=0.6)
    axis.set_facecolor("#fafafa")
    shortcut = record.get("shortcut")
    if shortcut:
        reduction = 100.0 * (1.0 - float(shortcut["relative_cost"]))
        title = (
            f"After scalable tree + group · {robot_count} disk robots\n"
            f"cost {shortcut['original_cost']:.1f} → {shortcut['final_cost']:.1f} "
            f"(−{reduction:.1f}%)"
        )
    else:
        title = f"Certified first-solution path · {robot_count} disk robots"
    axis.set_title(title, fontsize=15, pad=12)
    for obstacle in scene["obstacles"]:
        x, y = obstacle["position"][:2]
        width, height = obstacle["size"][:2]
        axis.add_patch(Rectangle(
            (x - width / 2, y - height / 2), width, height,
            facecolor="#4b4b4b", edgecolor="#222222", linewidth=0.8,
            zorder=3,
        ))

    trails = []
    disks = []
    labels = []
    current_targets = []
    for robot in range(robot_count):
        color = colors[robot]
        axis.scatter(
            goals[robot, :, 0], goals[robot, :, 1],
            marker="x", s=24, linewidths=1.2, color=color, alpha=0.36,
            zorder=2,
        )
        start = np.asarray(scene["starts"][robot])
        axis.scatter(
            [start[0]], [start[1]], marker="o", s=42,
            facecolors="none", edgecolors=[color], linewidths=1.2,
            alpha=0.55, zorder=2,
        )
        trail, = axis.plot([], [], color=color, linewidth=1.25, alpha=0.34, zorder=1)
        trails.append(trail)
        disk = Circle((0, 0), radius, facecolor=color, edgecolor="white", linewidth=1.0, zorder=5)
        axis.add_patch(disk)
        disks.append(disk)
        labels.append(axis.text(
            0, 0, str(robot + 1), ha="center", va="center",
            color="white", fontsize=7, fontweight="bold", zorder=6,
        ))
        target = Circle(
            (0, 0), radius * 1.45, facecolor="none", edgecolor=color,
            linewidth=2.0, alpha=0.9, zorder=4,
        )
        axis.add_patch(target)
        current_targets.append(target)

    progress = axis.text(
        0.5, -0.085, "", transform=axis.transAxes, ha="center", va="center",
        fontsize=12,
    )
    axis.text(
        0.5, -0.13,
        "faint ×: goals   ·   ring: current goal   ·   outlined circle: start",
        transform=axis.transAxes, ha="center", va="center", fontsize=9,
        color="#555555",
    )

    def update(frame_number):
        index = int(frame_indices[frame_number])
        configuration = motion[index]
        completed = completed_goals(
            sampled_modes[index], robot_count, goals_per_robot
        )
        for robot in range(robot_count):
            x, y = configuration[robot]
            disks[robot].center = (x, y)
            labels[robot].set_position((x, y))
            trails[robot].set_data(
                motion[:index + 1, robot, 0], motion[:index + 1, robot, 1]
            )
            next_goal = int(completed[robot])
            if next_goal < goals_per_robot:
                current_targets[robot].center = tuple(goals[robot, next_goal])
                current_targets[robot].set_visible(True)
            else:
                current_targets[robot].set_visible(False)
        total_completed = int(completed.sum())
        progress.set_text(
            f"path progress {100 * index / (args.motion_frames - 1):.0f}%"
            f"   ·   goals reached {total_completed}/{robot_count * goals_per_robot}"
        )
        return [*trails, *disks, *labels, *current_targets, progress]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(frame_indices),
        interval=1000 / args.fps,
        blit=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(
        args.output.resolve(),
        writer=animation.PillowWriter(fps=args.fps),
        dpi=100,
    )
    plt.close(figure)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "source": str(args.input.resolve()),
        "source_case": record["case_id"],
        "source_path_index": record["path_index"],
        "robots": robot_count,
        "states": len(record["states"]),
        "modes": record["mode_count"],
        "frames": len(frame_indices),
        "fps": args.fps,
        "duration_seconds": len(frame_indices) / args.fps,
    }, indent=2))


if __name__ == "__main__":
    main()
