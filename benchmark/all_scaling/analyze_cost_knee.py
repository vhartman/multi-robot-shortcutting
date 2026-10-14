"""Replay saved shortcut lineages and plot per-robot path lengths."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (REPO / "src", BENCHMARK, HERE, BENCHMARK / "scaling_2d"):
    sys.path.insert(0, str(directory))

import scalable_tree_group  # noqa: E402,F401
from mr_shortcutting.api import Proposal  # noqa: E402
from mr_shortcutting.shortcutters import _candidate, _interpolate_mode_safe  # noqa: E402
from study_lib import cases, input_path, read_json  # noqa: E402


OUTPUT = HERE / "analysis"
RESULT_ROOTS = {
    "full_configuration": BENCHMARK / "scaling_2d" / "offline" / "results",
    "tree_gain": BENCHMARK / "scalable_study" / "offline" / "results",
}
CASE_IDS = ("rai-disk-scaling-r12-g4",)
METHODS = (
    ("full_configuration", "Full configuration"),
    ("tree_gain", "Tree + single-robot gain"),
)
SEEDS = (0, 1)


def result_path(case_id, method_id, seed):
    return RESULT_ROOTS[method_id] / case_id / f"{method_id}-path0-seed{seed}.json"


def robot_lengths(path):
    points = np.asarray([state.q.state() for state in path], dtype=float)
    slices = path[0].q._array_slice
    differences = points[1:] - points[:-1]
    return np.asarray(
        [
            np.linalg.norm(differences[:, first:last], axis=1).sum()
            for first, last in slices
        ]
    )


def geometry_only_environment(robot_dofs):
    from multi_robot_multi_goal_planning.problems.core.configuration import NpConfiguration

    slices = []
    offset = 0
    for dimension in robot_dofs:
        slices.append((offset, offset + int(dimension)))
        offset += int(dimension)
    prototype = NpConfiguration(np.zeros(offset), slices)
    robots = [f"robot{index}" for index in range(len(robot_dofs))]
    return SimpleNamespace(
        robots=robots,
        robot_dims={robot: int(dimension) for robot, dimension in zip(robots, robot_dofs)},
        get_start_pos=lambda: prototype,
    )


def restore_symbolic_records(env, records):
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


def replay(case_id, method_id, seed):
    source = read_json(input_path(case_id, 0))
    result = read_json(result_path(case_id, method_id, seed))
    env = geometry_only_environment(result["robot_dofs"])
    raw = restore_symbolic_records(env, source["states"])
    if result["interpolation_fallback_to_raw"]:
        path = raw
    else:
        path, _ = _interpolate_mode_safe(env, raw, 0.5, validate_input=False)
    if len(path) != int(result["working_states"]):
        raise RuntimeError(
            f"working path length differs for {case_id}/{method_id}/seed{seed}: "
            f"{len(path)} != {result['working_states']}"
        )

    events = result["stats"]["accepted_lineage"]
    trace = result["trace"]
    if len(trace) != len(events) + 2:
        raise RuntimeError("accepted lineage and cost trace are not aligned")

    times = [0.0]
    current = robot_lengths(path)
    lengths = [current.copy()]
    for event in events:
        proposal = Proposal(
            int(event["first"]),
            int(event["last"]),
            tuple(int(robot) for robot in event["robots"]),
            source=event["source"],
        )
        before = robot_lengths(path[proposal.first : proposal.last + 1])
        candidate = _candidate(env, path, proposal)
        after = robot_lengths(candidate)
        selected = np.asarray(proposal.robots, dtype=int)
        current[selected] += after[selected] - before[selected]
        for offset, state in enumerate(candidate):
            path[proposal.first + offset].q = state.q
        times.append(float(trace[event["id"] + 1]["time"]))
        lengths.append(current.copy())

    if not np.allclose(current, robot_lengths(path), rtol=0, atol=1e-8):
        raise RuntimeError("incremental per-robot lengths failed replay check")
    return np.asarray(times), np.asarray(lengths), result


def step_array(times, values, grid):
    indices = np.searchsorted(times, grid, side="right") - 1
    return np.asarray(values)[np.maximum(indices, 0)]


def objective_curve(result, grid):
    times = np.asarray([float(point["time"]) for point in result["trace"]])
    costs = np.asarray([float(point["cost"]) for point in result["trace"]])
    return step_array(times, costs, grid)


def grid_for(replays, log):
    budget = min(float(item[2]["budget_seconds"]) for item in replays)
    if not log:
        return np.linspace(0.0, budget, 501)
    first = min(float(times[1]) for times, _, _ in replays if len(times) > 1)
    return np.geomspace(first, budget, 501)


def draw_panel(axis, replays, title, colors, normalization, log):
    grid = grid_for(replays, log)
    robot_curves = np.asarray(
        [step_array(times, lengths, grid) for times, lengths, _ in replays]
    )
    objective = np.asarray([objective_curve(result, grid) for _, _, result in replays])
    center = np.median(robot_curves, axis=0)
    for robot in range(center.shape[1]):
        axis.step(
            grid,
            center[:, robot],
            where="post",
            color=colors(normalization(robot)),
            linewidth=3.0,
        )
    axis.step(
        grid,
        np.median(objective, axis=0),
        where="post",
        color="#111111",
        linewidth=4.2,
        label="complete cost",
    )
    if log:
        axis.set_xscale("log")
    axis.set_title(title, fontsize=25)
    axis.set_xlabel("time [s]", fontsize=25)
    axis.set_ylabel("absolute path length / cost", fontsize=25)
    axis.tick_params(labelsize=19)
    axis.grid(True, alpha=0.22)


def render(replay_map, output, seeds, log=False):
    maximum_robots = max(replay_map[key][0][1].shape[1] for key in replay_map)
    colors = plt.colormaps["turbo"]
    normalization = plt.Normalize(vmin=0, vmax=maximum_robots - 1)
    figure, axes = plt.subplots(
        len(CASE_IDS),
        len(METHODS),
        figsize=(16.5, 7.4),
        squeeze=False,
    )
    for row, case_id in enumerate(CASE_IDS):
        for column, (method_id, method_label) in enumerate(METHODS):
            selected = [replay_map[(case_id, method_id)][seed] for seed in seeds]
            draw_panel(
                axes[row, column],
                selected,
                method_label,
                colors,
                normalization,
                log,
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        fontsize=21,
        bbox_to_anchor=(0.47, 0.015),
    )
    figure.subplots_adjust(
        top=0.97,
        left=0.075,
        right=0.94,
        bottom=0.20,
        hspace=0.42,
        wspace=0.20,
    )
    color_axis = figure.add_axes((0.955, 0.24, 0.012, 0.64))
    scalar = plt.cm.ScalarMappable(norm=normalization, cmap=colors)
    colorbar = figure.colorbar(scalar, cax=color_axis, ticks=np.arange(maximum_robots))
    colorbar.set_ticklabels([str(index + 1) for index in range(maximum_robots)])
    colorbar.set_label("robot", fontsize=21)
    colorbar.ax.tick_params(labelsize=17)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(output)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    replay_map = {}
    for case_id in CASE_IDS:
        for method_id, _ in METHODS:
            replay_map[(case_id, method_id)] = {
                seed: replay(case_id, method_id, seed) for seed in SEEDS
            }
            print(f"replayed {case_id} {method_id}")
    render(
        replay_map,
        OUTPUT / "cost-knee-disk-r12-full-vs-single-gain.png",
        SEEDS,
    )
    render(
        replay_map,
        OUTPUT / "cost-knee-disk-r12-full-vs-single-gain-log.png",
        SEEDS,
        log=True,
    )
    for seed in SEEDS:
        render(
            replay_map,
            OUTPUT / f"cost-knee-disk-r12-full-vs-single-gain-seed{seed}.png",
            (seed,),
        )


if __name__ == "__main__":
    main()
