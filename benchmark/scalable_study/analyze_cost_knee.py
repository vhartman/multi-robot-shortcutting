"""Diagnose the knee in scalable tree/group shortcutting from saved lineages."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (BENCHMARK, HERE, BENCHMARK / "scaling_2d", REPO / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import scalable_tree_group  # noqa: E402,F401
from mr_shortcutting.api import Proposal  # noqa: E402
from mr_shortcutting.shortcutters import _candidate, _interpolate_mode_safe  # noqa: E402
from plot_style import method_color  # noqa: E402
from study_lib import cases, input_path, read_json  # noqa: E402


OUTPUT = HERE / "analysis" / "followups"
BASE_RESULTS = HERE / "offline" / "results"
GROUP_RESULTS = HERE / "offline" / "followups" / "offline" / "results"
METHODS = {
    "scalable_tree_group": "scalable tree + group",
    "random_compatible_group": "random compatible group",
}
DIAGNOSTIC_CASES = (
    "rai-coupled-box-stacking-r6-b8",
    "rai-disk-scaling-r12-g4",
)
FOUR_CASES = (
    "rai-spiral-tower",
    "assembly-car-mini-4r-openarm",
    "rai-coupled-box-stacking-r6-b8",
    "rai-disk-scaling-r12-g4",
)
TITLES = {
    "rai-spiral-tower": "Spiral tower",
    "assembly-car-mini-4r-openarm": "Car assembly",
    "rai-coupled-box-stacking-r6-b8": "Coupled stacking (6 robots)",
    "rai-disk-scaling-r12-g4": "Planar disks (12 robots)",
}


def result_path(case_id, method_id, path_index=0, seed=0):
    root = GROUP_RESULTS if method_id == "random_compatible_group" else BASE_RESULTS
    return root / case_id / f"{method_id}-path{path_index}-seed{seed}.json"


def step(trace, grid):
    times = np.asarray([float(point["time"]) for point in trace])
    costs = np.asarray([float(point["cost"]) for point in trace])
    indices = np.maximum(np.searchsorted(times, grid, side="right") - 1, 0)
    return costs[indices]


def aggregate(case_id, method_id, grid):
    path_curves = []
    for path_index in range(3):
        seeds = []
        for seed in range(3):
            result = read_json(result_path(case_id, method_id, path_index, seed))
            seeds.append(step(result["trace"], grid) / float(result["initial_cost"]))
        path_curves.append(np.median(seeds, axis=0))
    values = np.asarray(path_curves)
    return np.median(values, axis=0), np.min(values, axis=0), np.max(values, axis=0)


def last_tree_times(case_id):
    times = []
    for path_index in range(3):
        for seed in range(3):
            result = read_json(result_path(case_id, "scalable_tree_group", path_index, seed))
            events = result["stats"]["accepted_lineage"]
            trace = result["trace"]
            if len(trace) != len(events) + 2:
                raise RuntimeError("accepted lineage and cost trace are not aligned")
            tree_times = [
                float(trace[event["id"] + 1]["time"])
                for event in events if event["source"] == "tree"
            ]
            if tree_times:
                times.append(max(tree_times))
    return np.asarray(times)


def plot_source_transition():
    figure, axes = plt.subplots(1, 4, figsize=(21, 4.9), squeeze=False)
    for axis, case_id in zip(axes.flat, FOUR_CASES):
        seconds = float(cases()[case_id]["seconds"])
        grid = np.linspace(0, seconds, 301)
        for method_id, label in METHODS.items():
            median, low, high = aggregate(case_id, method_id, grid)
            color = method_color(method_id)
            axis.fill_between(grid, low, high, color=color, alpha=0.11, linewidth=0)
            axis.plot(grid, median, color=color, linewidth=3.2, label=label)
        transition = last_tree_times(case_id)
        center = float(np.median(transition))
        axis.axvspan(float(np.min(transition)), float(np.max(transition)), color="#666666", alpha=0.10)
        axis.axvline(center, color="#333333", linestyle="--", linewidth=2.0,
                    label="last accepted tree shortcut" if case_id == FOUR_CASES[0] else None)
        axis.set_title(TITLES[case_id], fontsize=17)
        axis.set_xlabel("time [s]", fontsize=17)
        axis.set_ylabel("relative cost", fontsize=17)
        axis.tick_params(labelsize=14)
        axis.grid(True, alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                  fontsize=16, bbox_to_anchor=(0.5, 0.995))
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    output = OUTPUT / "cost-knee-tree-transition.png"
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(output)


def robot_lengths(path):
    points = np.asarray([state.q.state() for state in path], dtype=float)
    slices = path[0].q._array_slice
    diff = points[1:] - points[:-1]
    return np.asarray([
        np.linalg.norm(diff[:, first:last], axis=1).sum()
        for first, last in slices
    ])


def geometry_only_environment(robot_dofs):
    from multi_robot_multi_goal_planning.problems.core.configuration import NpConfiguration

    slices = []
    offset = 0
    for dimension in robot_dofs:
        slices.append((offset, offset + int(dimension)))
        offset += int(dimension)
    prototype = NpConfiguration(np.zeros(offset), slices)
    robots = [f"robot{index}" for index in range(len(robot_dofs))]
    dimensions = {robot: int(dimension) for robot, dimension in zip(robots, robot_dofs)}
    return SimpleNamespace(
        robots=robots,
        robot_dims=dimensions,
        get_start_pos=lambda: prototype,
    )


def restore_symbolic_records(env, records):
    """Restore geometry and mode signatures without reevaluating transitions.

    Lineage replay only needs configuration slices and equality of mode task
    IDs; it performs no collision or task-constraint checks.
    """
    from multi_robot_multi_goal_planning.problems.planning_env import Mode, State

    prototype = env.get_start_pos()
    restored = []
    modes = {}
    for record in records:
        q = prototype.from_flat(np.asarray(record["q"], dtype=float))
        signature = tuple(int(value) for value in record["mode"])
        if signature not in modes:
            modes[signature] = Mode(list(signature), q)
        restored.append(State(q, modes[signature]))
    return restored


def replay(case_id, method_id):
    source = read_json(input_path(case_id, 0))
    result = read_json(result_path(case_id, method_id))
    env = geometry_only_environment(result["robot_dofs"])
    raw = restore_symbolic_records(env, source["states"])
    if result["interpolation_fallback_to_raw"]:
        path = raw
    else:
        path, _ = _interpolate_mode_safe(env, raw, 0.5, validate_input=False)
    if len(path) != int(result["working_states"]):
        raise RuntimeError(
            f"working path length differs for {case_id}: "
            f"{len(path)} != {result['working_states']}"
        )
    events = result["stats"]["accepted_lineage"]
    trace = result["trace"]
    if len(trace) != len(events) + 2:
        raise RuntimeError("accepted lineage and cost trace are not aligned")

    times = [0.0]
    current_lengths = robot_lengths(path)
    lengths = [current_lengths.copy()]
    for event in events:
        proposal = Proposal(
            int(event["first"]), int(event["last"]),
            tuple(int(robot) for robot in event["robots"]),
            source=event["source"],
        )
        before = robot_lengths(path[proposal.first : proposal.last + 1])
        candidate = _candidate(env, path, proposal)
        after = robot_lengths(candidate)
        selected = np.asarray(proposal.robots, dtype=int)
        current_lengths[selected] += after[selected] - before[selected]
        for offset, state in enumerate(candidate):
            path[proposal.first + offset].q = state.q
        times.append(float(trace[event["id"] + 1]["time"]))
        lengths.append(current_lengths.copy())
    if not np.allclose(current_lengths, robot_lengths(path), rtol=0, atol=1e-8):
        raise RuntimeError("incremental per-robot path lengths failed replay check")
    return np.asarray(times), np.asarray(lengths), result


def plot_per_robot_lengths():
    figure, axes = plt.subplots(2, 2, figsize=(15.8, 9.0), squeeze=False)
    color_map = plt.colormaps["turbo"]
    maximum_robots = max(
        int(cases()[case_id]["environment_kwargs"]["num_robots"])
        for case_id in DIAGNOSTIC_CASES
    )
    normalization = plt.Normalize(vmin=0, vmax=maximum_robots - 1)
    for row, case_id in enumerate(DIAGNOSTIC_CASES):
        for column, (method_id, method_label) in enumerate(METHODS.items()):
            axis = axes[row, column]
            times, lengths, result = replay(case_id, method_id)
            for robot in range(lengths.shape[1]):
                axis.plot(
                    times, lengths[:, robot],
                    color=color_map(normalization(robot)), linewidth=1.9,
                )
            exact = step(result["trace"], times)
            axis.plot(times, exact, color="#111111", linewidth=3.0,
                      label="complete cost")
            axis.set_title(f"{TITLES[case_id]} — {method_label}", fontsize=15)
            axis.set_xlabel("time [s]", fontsize=15)
            axis.set_ylabel("absolute path length / cost", fontsize=15)
            axis.tick_params(labelsize=12)
            axis.grid(True, alpha=0.22)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
                  frameon=False, fontsize=13, bbox_to_anchor=(0.46, 0.995))
    figure.subplots_adjust(top=0.91, left=0.07, right=0.90, bottom=0.08,
                           hspace=0.34, wspace=0.18)
    color_axis = figure.add_axes((0.925, 0.13, 0.014, 0.72))
    scalar = plt.cm.ScalarMappable(norm=normalization, cmap=color_map)
    colorbar = figure.colorbar(
        scalar, cax=color_axis, ticks=np.arange(maximum_robots),
    )
    colorbar.set_ticklabels([str(index + 1) for index in range(maximum_robots)])
    colorbar.set_label("robot", fontsize=13)
    colorbar.ax.tick_params(labelsize=10)
    output = OUTPUT / "cost-knee-per-robot-lengths.png"
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(output)


def broad_title(case_id):
    return case_id.removeprefix("rai-").removeprefix("assembly-").replace("-", " ").title()


def draw_absolute_panel(axis, case_id, times, lengths, result, color_map, normalization):
    for robot in range(lengths.shape[1]):
        axis.plot(
            times, lengths[:, robot], color=color_map(normalization(robot)),
            linewidth=2.0,
        )
    axis.plot(times, step(result["trace"], times), color="#111111", linewidth=3.0,
              label="complete cost")
    axis.set_title(broad_title(case_id), fontsize=15)
    axis.set_xlabel("time [s]", fontsize=14)
    axis.set_ylabel("absolute path length / cost", fontsize=14)
    axis.tick_params(labelsize=11)
    axis.grid(True, alpha=0.22)


def plot_broad_method(method_id, output_stem):
    available = []
    for case in cases().values():
        if case["family"] != "broad":
            continue
        path = result_path(case["id"], method_id)
        if path.is_file():
            available.append(case["id"])
    print(f"broad {method_id} lineage plots: {len(available)} available cases")

    replays = {
        case_id: replay(case_id, method_id)
        for case_id in available
    }
    maximum_robots = max(values[1].shape[1] for values in replays.values())
    color_map = plt.colormaps["turbo"]
    normalization = plt.Normalize(vmin=0, vmax=max(1, maximum_robots - 1))

    def render(selected, output, columns):
        rows = int(np.ceil(len(selected) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(5.4 * columns, 4.25 * rows), squeeze=False,
        )
        for axis, case_id in zip(axes.flat, selected):
            draw_absolute_panel(
                axis, case_id, *replays[case_id], color_map, normalization,
            )
        for axis in axes.flat[len(selected):]:
            axis.remove()
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", frameon=False,
                      fontsize=15, bbox_to_anchor=(0.48, 0.995))
        figure.subplots_adjust(
            top=0.955 if rows > 2 else 0.90, left=0.06, right=0.93,
            bottom=0.055, hspace=0.38, wspace=0.28,
        )
        color_axis = figure.add_axes((0.95, 0.12, 0.010, 0.76))
        scalar = plt.cm.ScalarMappable(norm=normalization, cmap=color_map)
        colorbar = figure.colorbar(
            scalar, cax=color_axis, ticks=np.arange(maximum_robots),
        )
        colorbar.set_ticklabels([str(index + 1) for index in range(maximum_robots)])
        colorbar.set_label("robot", fontsize=13)
        colorbar.ax.tick_params(labelsize=10)
        figure.savefig(output, dpi=190, bbox_inches="tight")
        plt.close(figure)
        print(output)

    render(available, OUTPUT / f"{output_stem}.png", columns=5)
    for page, start in enumerate(range(0, len(available), 10), 1):
        render(
            available[start : start + 10],
            OUTPUT / f"{output_stem}-page-{page}.png",
            columns=2,
        )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_source_transition()
    plot_per_robot_lengths()
    plot_broad_method("scalable_tree_group", "cost-knee-broad-scalable-tree")
    plot_broad_method("random_compatible_group", "cost-knee-broad-compatible-group")


if __name__ == "__main__":
    main()
