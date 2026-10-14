"""Plot the completed scalable-study follow-up experiments.

The offline plots combine the frozen three-method study with the additive
random-compatible-group ablation.  The online plots use the paired raw first
solution as time zero and normalize each planner seed by that same cost.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import analyze as base  # noqa: E402
from plot_style import method_color  # noqa: E402
from study_lib import cases, protocol, read_json  # noqa: E402


OUTPUT = HERE / "analysis" / "followups"
FOLLOWUPS = HERE / "offline" / "followups"
OFFLINE = FOLLOWUPS / "offline"
ONLINE = FOLLOWUPS / "online-v5"

METHODS = OrderedDict((
    ("tree_gain", "tree + gain"),
    ("tree_group_gain", "tree + group gain"),
    ("scalable_tree_group", "tree + compatible (interleaved)"),
    ("random_compatible_group", "compatible subset"),
))
ONLINE_METHODS = OrderedDict((
    ("upstream_tree", "standard tree"),
    ("scalable_tree_group", "scalable tree + group"),
))
CASE_TITLES = {
    "rai-random-2d": "Random 2D",
    "rai-box-rearrangement": "Box rearrangement",
    "rai-spiral-tower": "Spiral tower",
    "assembly-car-mini-4r-openarm": "Car assembly",
    "rai-coupled-box-stacking-r6-b8": "Coupled stacking (6 robots)",
    "rai-disk-scaling-r12-g4": "Planar disks (12 robots)",
}
FOUR_CASES = ("rai-spiral-tower", "assembly-car-mini-4r-openarm",
              "rai-coupled-box-stacking-r6-b8", "rai-disk-scaling-r12-g4")
BROAD_HIGHLIGHTS = ("rai-random-2d", "rai-box-rearrangement",
                    "rai-spiral-tower", "assembly-car-mini-4r-openarm")
FAMILIES = (
    "isolated_box_stacking",
    "disk_scaling",
    "coupled_box_stacking",
)


def title(case_id: str) -> str:
    return CASE_TITLES.get(
        case_id, case_id.removeprefix("rai-").replace("-", " ").title()
    )


def load_offline():
    records = base.load_results(allow_partial=True)
    manifest = read_json(OFFLINE / "manifest.json")
    followup_counts = {}
    missing = []
    for cell in manifest["cells"]:
        key = (
            cell["case_id"], cell["method_id"],
            int(cell["path_index"]), int(cell["shortcut_seed"]),
        )
        path = (
            OFFLINE / "results" / cell["case_id"]
            / f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json"
        )
        if not path.is_file():
            missing.append(path)
            continue
        result = read_json(path)
        if result.get("strict_collision_free") is not True:
            raise RuntimeError(f"uncertified follow-up result: {path}")
        records[key] = result
        followup_counts[cell["case_id"]] = followup_counts.get(cell["case_id"], 0) + 1
    if missing:
        raise RuntimeError(f"{len(missing)} offline follow-up cells are missing")
    print(f"offline follow-up loaded={len(manifest['cells'])}")
    return records, followup_counts


def load_online():
    manifest = read_json(ONLINE / "manifest.json")
    records = {}
    missing = []
    for cell in manifest["cells"]:
        path = (
            ONLINE / "results" / cell["case_id"]
            / f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json"
        )
        if not path.is_file():
            missing.append(path)
            continue
        result = read_json(path)
        if not result.get("certificate", {}).get("strict_full_edge"):
            raise RuntimeError(f"uncertified online result: {path}")
        key = (cell["case_id"], cell["method_id"], int(cell["path_index"]))
        records[key] = result
    if missing:
        raise RuntimeError(f"{len(missing)} online cells are missing")
    print(f"online loaded={len(records)}")
    return records


def offline_curve(records, case, method_id, grid):
    path_curves = []
    for path_index in protocol()["path_indices"]:
        seed_curves = []
        for seed in protocol()["shortcut_seeds"]:
            record = records.get((case["id"], method_id, path_index, seed))
            if record is None:
                continue
            seed_curves.append(
                base.step_value(record["trace"], "time", grid)
                / float(record["initial_cost"])
            )
        if seed_curves:
            path_curves.append(np.median(seed_curves, axis=0))
    if not path_curves:
        return None
    values = np.asarray(path_curves)
    return np.median(values, axis=0), np.min(values, axis=0), np.max(values, axis=0)


def online_curve(records, case_id, method_id, grid):
    curves = []
    for path_index in protocol()["path_indices"]:
        record = records[(case_id, method_id, path_index)]
        curves.append(
            base.step_value(record["comparison_curve"], "time", grid)
            / float(record["first_solution"]["cost"])
        )
    values = np.asarray(curves)
    return np.median(values, axis=0), np.min(values, axis=0), np.max(values, axis=0)


def first_positive_offline(records, case, method_ids):
    values = [
        float(point["time"])
        for key, record in records.items()
        if key[0] == case["id"] and key[1] in method_ids
        for point in record["trace"][1:]
        if float(point["time"]) > 0
    ]
    return min(values) if values else max(1e-4, float(case["seconds"]) / 10000)


def first_positive_online(records, case_id):
    values = [
        float(point["time"])
        for key, record in records.items()
        if key[0] == case_id
        for point in record["comparison_curve"][1:]
        if float(point["time"]) > 0
    ]
    return min(values) if values else 0.01


def style_axis(axis, plot_title):
    axis.set_title(plot_title, fontsize=25)
    axis.grid(True, alpha=0.22)
    axis.tick_params(labelsize=19)
    axis.set_xlabel("time [s]", fontsize=25)
    axis.set_ylabel("relative cost", fontsize=25)


def finish(figure, axes, output, legend_columns):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=legend_columns,
        frameon=False, fontsize=21, bbox_to_anchor=(0.5, 0.01),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(output)


def plot_offline_grid(records, selected, output, columns, log=False, highlight=False):
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(7.0 * columns, 5.4 * rows + 2.0) if highlight else (5.6 * columns, 5.0 * rows + 2.0),
        squeeze=False,
    )
    for axis, case in zip(axes.flat, selected):
        budget = float(case["seconds"])
        if log:
            start = first_positive_offline(records, case, METHODS)
            grid = np.geomspace(start, budget, 300)
        else:
            grid = np.linspace(0, budget, 301)
        for method_id, label in METHODS.items():
            curve = offline_curve(records, case, method_id, grid)
            if curve is None:
                continue
            median, low, high = curve
            color = method_color(method_id)
            axis.fill_between(grid, low, high, color=color, alpha=0.11, linewidth=0)
            axis.plot(grid, median, color=color, linewidth=4.0, label=label)
        if log:
            axis.set_xscale("log")
            axis.set_xlim(grid[0], budget)
        style_axis(axis, title(case["id"]))
    for axis in axes.flat[len(selected):]:
        axis.remove()
    if highlight:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2,
                      frameon=False, fontsize=21, bbox_to_anchor=(0.5, 0.01))
        figure.tight_layout(rect=(0, 0.12, 1, 1))
        figure.savefig(output, dpi=190, bbox_inches="tight")
        plt.close(figure)
        print(output)
    else:
        finish(figure, axes, output, len(METHODS))


def plot_online_grid(records, output, log=False):
    figure, axes = plt.subplots(1, 4, figsize=(21, 4.9), squeeze=False)
    horizon = 100.0
    for axis, case_id in zip(axes.flat, FOUR_CASES):
        if log:
            start = first_positive_online(records, case_id)
            grid = np.geomspace(start, horizon, 300)
        else:
            grid = np.linspace(0, horizon, 301)
        for method_id, label in ONLINE_METHODS.items():
            median, low, high = online_curve(records, case_id, method_id, grid)
            color = method_color(method_id)
            axis.fill_between(grid, low, high, color=color, alpha=0.11, linewidth=0)
            axis.plot(grid, median, color=color, linewidth=3.2, label=label)
        if log:
            axis.set_xscale("log")
            axis.set_xlim(grid[0], horizon)
        style_axis(axis, title(case_id))
    finish(figure, axes, output, len(ONLINE_METHODS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blog-broad-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.blog_broad_only:
        destination = args.output_dir or OUTPUT
        destination.mkdir(parents=True, exist_ok=True)
        records, counts = load_offline()
        resolved = cases()
        expected = len(protocol()["path_indices"]) * len(protocol()["shortcut_seeds"])
        broad = [case for case in resolved.values()
                 if case["family"] == "broad" and counts.get(case["id"]) == expected]
        for case in broad:
            for method in METHODS:
                for path_index in protocol()["path_indices"]:
                    for seed in protocol()["shortcut_seeds"]:
                        if (case["id"], method, path_index, seed) not in records:
                            raise RuntimeError(f"incomplete broad cell: {case['id']} {method} {path_index} {seed}")
        highlights = [resolved[case_id] for case_id in BROAD_HIGHLIGHTS]
        if not set(BROAD_HIGHLIGHTS) <= {case["id"] for case in broad}:
            raise RuntimeError("a highlighted case is incomplete")
        for log in (False, True):
            suffix = "-logx" if log else ""
            plot_offline_grid(records, broad, destination / f"offline-broad-comparison{suffix}.png", 4, log=log)
            plot_offline_grid(records, highlights, destination / f"offline-broad-highlights{suffix}.png", 2, log=log, highlight=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    offline, followup_counts = load_offline()
    online = load_online()
    resolved = cases()

    four = [resolved[case_id] for case_id in FOUR_CASES]
    plot_offline_grid(offline, four, OUTPUT / "offline-four-comparison.png", 4)
    plot_offline_grid(offline, four, OUTPUT / "offline-four-comparison-log.png", 4, log=True)
    plot_online_grid(online, OUTPUT / "online-four-comparison.png")
    plot_online_grid(online, OUTPUT / "online-four-comparison-log.png", log=True)

    expected = len(protocol()["path_indices"]) * len(protocol()["shortcut_seeds"])
    broad = [
        case for case in resolved.values()
        if case["family"] == "broad" and followup_counts.get(case["id"]) == expected
    ]
    plot_offline_grid(offline, broad, OUTPUT / "offline-broad-comparison.png", 4)
    plot_offline_grid(offline, broad, OUTPUT / "offline-broad-comparison-log.png", 4, log=True)
    for page, start in enumerate(range(0, len(broad), 10), 1):
        selected = broad[start:start + 10]
        plot_offline_grid(offline, selected, OUTPUT / f"offline-broad-page-{page}.png", 2)
        plot_offline_grid(offline, selected, OUTPUT / f"offline-broad-page-{page}-log.png", 2, log=True)

    for family in FAMILIES:
        selected = sorted(
            [case for case in resolved.values() if case["family"] == family],
            key=base.robot_count,
        )
        columns = 2 if family == "coupled_box_stacking" else 3
        plot_offline_grid(offline, selected, OUTPUT / f"offline-{family}.png", columns)
        plot_offline_grid(
            offline, selected, OUTPUT / f"offline-{family}-log.png", columns, log=True
        )


if __name__ == "__main__":
    main()
