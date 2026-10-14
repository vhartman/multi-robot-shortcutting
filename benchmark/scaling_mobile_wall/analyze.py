"""Plot mobile-wall curves and the joined isolated-arm scaling extension."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
os.environ["MR_SHORTCUT_PROTOCOL"] = str(HERE / "protocol.json")
os.environ["MR_SHORTCUT_OUTPUT_ROOT"] = str(HERE)
for directory in (BENCHMARK, BENCHMARK / "expanded"):
    sys.path.insert(0, str(directory))

from expanded_lib import cases, methods, protocol, read_json, result_path  # noqa: E402
from plot_style import method_color  # noqa: E402


ANALYSIS = HERE / "analysis"
BASE = BENCHMARK / "scalable_study"


def trace_curve(record, grid):
    times = np.asarray([float(point["time"]) for point in record["trace"]])
    costs = np.asarray([float(point["cost"]) for point in record["trace"]])
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.full_like(grid, float(record["initial_cost"]), dtype=float)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values / float(record["initial_cost"])


def records_for(case, method_id, roots):
    output = {}
    for path_index in case["path_indices"]:
        for seed in protocol()["shortcut_seeds"]:
            path = roots / case["id"] / f"{method_id}-path{path_index}-seed{seed}.json"
            if not path.is_file():
                continue
            value = read_json(path)
            if value.get("strict_collision_free") is not True:
                raise RuntimeError(f"uncertified result: {path}")
            output[(path_index, seed)] = value
    return output


def aggregate(case, method_id, root):
    records = records_for(case, method_id, root)
    if not records:
        return None
    grid = np.linspace(0.0, float(case["seconds"]), 301)
    paths = []
    for path_index in case["path_indices"]:
        curves = [
            trace_curve(records[(path_index, seed)], grid)
            for seed in protocol()["shortcut_seeds"]
            if (path_index, seed) in records
        ]
        if curves:
            paths.append(np.median(curves, axis=0))
    stacked = np.asarray(paths)
    return grid, np.median(stacked, axis=0), np.min(stacked, axis=0), np.max(stacked, axis=0), len(records)


def style(axis, title):
    axis.set_title(title, fontsize=16)
    axis.set_xlabel("time [s]", fontsize=13)
    axis.set_ylabel("relative cost", fontsize=13)
    axis.tick_params(labelsize=11)
    axis.grid(alpha=0.2)


def plot_grid(selected, output, roots, columns=3):
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 3.8 * rows), squeeze=False)
    first_axis = None
    for axis, case in zip(axes.flat, selected):
        for method_id, method in methods().items():
            curve = aggregate(case, method_id, roots)
            if curve is None:
                continue
            grid, center, low, high, _ = curve
            color = method_color(method_id)
            axis.step(grid, center, where="post", color=color, linewidth=3, label=method["label"])
            axis.fill_between(grid, low, high, step="post", color=color, alpha=0.12)
            first_axis = first_axis or axis
        kwargs = case["environment_kwargs"]
        style(axis, f"{kwargs['num_robots']} robot{'s' if kwargs['num_robots'] != 1 else ''}")
    for axis in axes.flat[len(selected):]:
        axis.remove()
    if first_axis is not None:
        handles, labels = first_axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=len(methods()), frameon=False, fontsize=14)
    else:
        plt.close(figure)
        return False
    figure.tight_layout(rect=(0, 0.14, 1, 1))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_map = cases()
    mobile = [case for case in case_map.values() if case["constructor"] == "rai_mobile_wall_scaling_env"]
    isolated_new = [case for case in case_map.values() if case["constructor"] == "rai_isolated_arm_box_stack_env"]

    missing = []
    expected = 0
    available = 0
    for case in case_map.values():
        for method_id in methods():
            for path_index in case["path_indices"]:
                for seed in protocol()["shortcut_seeds"]:
                    expected += 1
                    path = result_path(case["id"], method_id, path_index, seed)
                    if path.is_file():
                        available += 1
                    else:
                        missing.append(str(path))
    if missing and not args.allow_partial:
        raise RuntimeError(f"{len(missing)} cells missing; use --allow-partial for progress plots")

    shapes = sorted({
        (case["environment_kwargs"]["wall_x"], case["environment_kwargs"]["wall_z"])
        for case in mobile
    }, key=lambda shape: shape[0] * shape[1])
    for wall_x, wall_z in shapes:
        selected = sorted(
            [case for case in mobile if case["environment_kwargs"]["wall_x"] == wall_x and case["environment_kwargs"]["wall_z"] == wall_z],
            key=lambda case: case["environment_kwargs"]["num_robots"],
        )
        plot_grid(
            selected,
            output_dir / f"mobile-wall-{wall_x}x{wall_z}-curves.png",
            HERE / "offline" / "results",
        )

    # Join the already completed r1--r6 cells to the additive r7--r9 cells.
    base_protocol = read_json(BASE / "protocol.json")
    base_cases = []
    for source in base_protocol["case_sets"]:
        if source["family"] != "isolated_box_stacking":
            continue
        original = read_json(BENCHMARK.parent / source["source_protocol"])
        for case in original["cases"]:
            if case["track"] == source["source_track"]:
                joined = dict(case)
                joined["path_indices"] = list(base_protocol["path_indices"])
                base_cases.append(joined)
    isolated = sorted(base_cases + isolated_new, key=lambda case: case["environment_kwargs"]["num_robots"])
    base_case_ids = {case["id"] for case in base_cases}
    columns = 3
    rows = int(np.ceil(len(isolated) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 3.8 * rows), squeeze=False)
    first_axis = None
    for axis, case in zip(axes.flat, isolated):
        root = BASE / "offline" / "results" if case["id"] in base_case_ids else HERE / "offline" / "results"
        for method_id, method in methods().items():
            curve = aggregate(case, method_id, root)
            if curve is None:
                continue
            grid, center, low, high, _ = curve
            color = method_color(method_id)
            axis.step(grid, center, where="post", color=color, linewidth=3, label=method["label"])
            axis.fill_between(grid, low, high, step="post", color=color, alpha=0.12)
            first_axis = first_axis or axis
        count = case["environment_kwargs"]["num_robots"]
        style(axis, f"{count} robot{'s' if count != 1 else ''}")
    if first_axis is not None:
        handles, labels = first_axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=len(methods()), frameon=False, fontsize=14)
        figure.tight_layout(rect=(0, 0.14, 1, 1))
        figure.savefig(output_dir / "isolated-box-stacking-r1-r9-curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "expected_new_cells": expected,
        "available_new_cells": available,
        "missing_new_cells": len(missing),
        "mobile_wall_cases": len(mobile),
        "isolated_extension_cases": len(isolated_new),
        "aggregation": protocol()["aggregation"],
    }
    (output_dir / "availability.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
