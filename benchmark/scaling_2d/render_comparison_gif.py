"""Render first-solution and shortcut paths on one physical-time axis."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from render_path_gif import completed_goals, load_path


HERE = Path(__file__).resolve().parent


def max_speed_timeline(positions, modes, vmax, times):
    """Sample piecewise-linear motion with max robot speed equal to vmax.

    Every nonzero source edge is assigned duration
    max_r ||q[i+1, r] - q[i, r]|| / vmax. Consequently, at least one
    robot travels at exactly vmax throughout each edge while all others are
    no faster. Zero-motion mode transitions have zero duration.
    """
    displacements = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    edge_distances = np.max(displacements, axis=1)
    edge_durations = edge_distances / float(vmax)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_durations)))
    duration = float(cumulative[-1])
    targets = np.minimum(np.asarray(times, dtype=float), duration)
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.clip(indices, 0, len(positions) - 2)
    widths = cumulative[indices + 1] - cumulative[indices]
    fractions = np.divide(
        targets - cumulative[indices],
        widths,
        out=np.zeros_like(targets),
        where=widths > 0,
    )
    sampled_positions = (
        positions[indices]
        + fractions[:, None, None]
        * (positions[indices + 1] - positions[indices])
    )
    sampled_modes = modes[indices].copy()
    at_end = fractions >= 1.0 - 1e-12
    sampled_modes[at_end] = modes[indices[at_end] + 1]
    sampled_positions[targets >= duration] = positions[-1]
    sampled_modes[targets >= duration] = modes[-1]

    nonzero = edge_durations > 0
    edge_max_speeds = np.divide(
        edge_distances[nonzero], edge_durations[nonzero]
    )
    if edge_max_speeds.size and not np.allclose(
        edge_max_speeds, vmax, rtol=0, atol=1e-12
    ):
        raise RuntimeError("physical-time parameterization failed vmax check")
    return sampled_positions, sampled_modes, duration, edge_max_speeds


def panel(axis, record, scene, robot_count, colors, title):
    goals = np.asarray(scene["goals"], dtype=float)
    radius = float(scene["robot_shape_size"][-1])
    axis.set_aspect("equal")
    axis.set_xlim(-2.12, 2.12)
    axis.set_ylim(-2.12, 2.12)
    axis.set_xticks([-2, -1, 0, 1, 2])
    axis.set_yticks([-2, -1, 0, 1, 2])
    axis.tick_params(
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    axis.grid(color="#d7d7d7", linewidth=0.7, alpha=0.6)
    axis.set_facecolor("#fafafa")
    axis.set_title(title, fontsize=19, pad=12)
    for obstacle in scene["obstacles"]:
        x, y = obstacle["position"][:2]
        width, height = obstacle["size"][:2]
        axis.add_patch(Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor="#4b4b4b",
            edgecolor="#222222",
            linewidth=0.8,
            zorder=3,
        ))

    artists = {"trails": [], "disks": [], "labels": [], "targets": []}
    for robot in range(robot_count):
        color = colors[robot]
        axis.scatter(
            goals[robot, :, 0],
            goals[robot, :, 1],
            marker="x",
            s=24,
            linewidths=1.2,
            color=color,
            alpha=0.36,
            zorder=2,
        )
        start = np.asarray(scene["starts"][robot])
        axis.scatter(
            [start[0]],
            [start[1]],
            marker="o",
            s=42,
            facecolors="none",
            edgecolors=[color],
            linewidths=1.2,
            alpha=0.55,
            zorder=2,
        )
        trail, = axis.plot(
            [], [], color=color, linewidth=1.25, alpha=0.34, zorder=1
        )
        artists["trails"].append(trail)
        disk = Circle(
            (0, 0),
            radius,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
        axis.add_patch(disk)
        artists["disks"].append(disk)
        artists["labels"].append(axis.text(
            0,
            0,
            "",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=6,
        ))
        target = Circle(
            (0, 0),
            radius * 1.45,
            facecolor="none",
            edgecolor=color,
            linewidth=2.0,
            alpha=0.9,
            zorder=4,
        )
        axis.add_patch(target)
        artists["targets"].append(target)
    artists["progress"] = axis.text(
        0.5,
        -0.085,
        "",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=15,
    )
    return artists, goals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--before",
        type=Path,
        default=HERE / "inputs" / "rai-disk-scaling-r12-g4" / "path0.json",
    )
    parser.add_argument(
        "--after",
        type=Path,
        default=(
            HERE
            / "analysis"
            / "disk-scaling-r12-path0-scalable-tree-group-seed0.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "analysis" / "disk-scaling-r12-before-after-vmax1.gif",
    )
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    if args.vmax <= 0 or args.fps < 1:
        raise ValueError("vmax and fps must be positive")

    before_record, before_scene, before_positions, before_modes = load_path(
        args.before.resolve()
    )
    after_record, after_scene, after_positions, after_modes = load_path(
        args.after.resolve()
    )
    robot_count = before_positions.shape[1]
    if after_positions.shape[1] != robot_count:
        raise RuntimeError("before and after paths have different robot counts")
    for key in ("starts", "goals", "obstacles", "robot_shape_size"):
        if before_scene[key] != after_scene[key]:
            raise RuntimeError(f"before and after scenes differ in {key}")

    def duration(positions):
        edges = np.linalg.norm(np.diff(positions, axis=0), axis=2)
        return float(np.max(edges, axis=1).sum() / args.vmax)

    before_duration = duration(before_positions)
    after_duration = duration(after_positions)
    total_duration = max(before_duration, after_duration)
    frame_count = int(np.ceil(total_duration * args.fps)) + 1
    times = np.arange(frame_count, dtype=float) / args.fps
    before_motion, before_sampled_modes, _, before_speeds = max_speed_timeline(
        before_positions, before_modes, args.vmax, times
    )
    after_motion, after_sampled_modes, _, after_speeds = max_speed_timeline(
        after_positions, after_modes, args.vmax, times
    )

    colors = [plt.get_cmap("tab20")(index) for index in range(robot_count)]
    before_title = "Initial path"
    after_title = "Final path"
    figure, axes = plt.subplots(1, 2, figsize=(14.4, 7.2))
    before_artists, goals = panel(
        axes[0], before_record, before_scene, robot_count, colors, before_title
    )
    after_artists, _ = panel(
        axes[1], after_record, after_scene, robot_count, colors, after_title
    )
    figure.subplots_adjust(
        left=0.055, right=0.985, bottom=0.10, top=0.91, wspace=0.14
    )
    goals_per_robot = goals.shape[1]

    def update_panel(artists, motion, sampled_modes, index, elapsed, path_duration):
        configuration = motion[index]
        completed = completed_goals(
            sampled_modes[index], robot_count, goals_per_robot
        )
        for robot in range(robot_count):
            x, y = configuration[robot]
            artists["disks"][robot].center = (x, y)
            artists["labels"][robot].set_position((x, y))
            artists["trails"][robot].set_data(
                motion[: index + 1, robot, 0], motion[: index + 1, robot, 1]
            )
            next_goal = int(completed[robot])
            if next_goal < goals_per_robot:
                artists["targets"][robot].center = tuple(goals[robot, next_goal])
                artists["targets"][robot].set_visible(True)
            else:
                artists["targets"][robot].set_visible(False)
        shown_time = min(elapsed, path_duration)
        artists["progress"].set_text(f"time {shown_time:.1f} s")
        return [
            *artists["trails"],
            *artists["disks"],
            *artists["labels"],
            *artists["targets"],
            artists["progress"],
        ]

    def update(index):
        elapsed = float(times[index])
        return [
            *update_panel(
                before_artists,
                before_motion,
                before_sampled_modes,
                index,
                elapsed,
                before_duration,
            ),
            *update_panel(
                after_artists,
                after_motion,
                after_sampled_modes,
                index,
                elapsed,
                after_duration,
            ),
        ]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / args.fps,
        blit=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # PillowWriter retains every full-resolution frame and becomes impractical
    # for a physical-time animation of this length. Stream once to a temporary
    # video, then use ffmpeg's two-pass GIF palette generation.
    destination = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="mr-shortcut-vmax-gif-") as temporary:
        temporary = Path(temporary)
        video = temporary / "comparison.mp4"
        palette = temporary / "palette.png"
        movie.save(
            video,
            writer=animation.FFMpegWriter(
                fps=args.fps,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
            ),
            dpi=100,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                "-vf", "palettegen=stats_mode=diff", str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                "-i", str(palette), "-lavfi",
                "paletteuse=dither=sierra2_4a", "-loop", "0", str(destination),
            ],
            check=True,
        )
    plt.close(figure)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "before": str(args.before.resolve()),
        "after": str(args.after.resolve()),
        "robots": robot_count,
        "vmax": args.vmax,
        "fps": args.fps,
        "frames": frame_count,
        "gif_duration_seconds": frame_count / args.fps,
        "before_physical_duration_seconds": before_duration,
        "after_physical_duration_seconds": after_duration,
        "before_edge_max_speed_range": [
            float(np.min(before_speeds)), float(np.max(before_speeds))
        ],
        "after_edge_max_speed_range": [
            float(np.min(after_speeds)), float(np.max(after_speeds))
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
