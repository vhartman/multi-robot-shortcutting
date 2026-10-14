"""Compare the three compatible-group schedules used in the scaling draft.

This is additive: it reads the frozen all-scaling results for group-only and
interleaved tree/group, plus the separately certified tree-then-group sweep.
It does not modify either protocol or any raw result.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (BENCHMARK, HERE):
    sys.path.insert(0, str(directory))

import analyze as base  # noqa: E402
from plot_style import method_color  # noqa: E402
from study_lib import cases, protocol, read_json  # noqa: E402


OUTPUT = HERE / "analysis" / "tree-group-schedules"
NEW_ROOT = HERE / "offline" / "tree-then-compatible-group" / "results"
METHODS = OrderedDict(
    (
        (
            "robot_subset_mode",
            {"label": "compatible random groups"},
        ),
        (
            "tree_then_compatible_group",
            {"label": "tree → compatible groups"},
        ),
        (
            "scalable_tree_group",
            {"label": "tree + compatible groups (interleaved)"},
        ),
    )
)
FAMILY_ORDER = base.FAMILY_ORDER
FAMILY_TITLES = base.FAMILY_TITLES
THRESHOLDS = (0.95, 0.90, 0.80)


def new_result_path(case_id: str, path_index: int, seed: int) -> Path:
    return (
        NEW_ROOT
        / case_id
        / f"tree_then_compatible_group-path{path_index}-seed{seed}.json"
    )


def validate_new(record: dict, case: dict, path_index: int, seed: int, path: Path):
    expected = {
        "schema": "mr-shortcut-tree-then-group-scaling-v1",
        "case_id": case["id"],
        "family": case["family"],
        "method_id": "tree_then_compatible_group",
        "implementation_policy": "tree_then_compatible_group",
        "path_index": path_index,
        "shortcut_seed": seed,
    }
    errors = [key for key, value in expected.items() if record.get(key) != value]
    required_true = (
        "endpoint_start_preserved",
        "endpoint_end_preserved",
        "strict_collision_free",
        "terminal_mode_preserved",
    )
    errors.extend(key for key in required_true if record.get(key) is not True)
    if record.get("valid_plan") is False:
        errors.append("valid_plan")
    if record.get("input_mode_runs") != record.get("output_mode_runs"):
        errors.append("mode sequence")
    stats = record.get("stats", {})
    if stats.get("schedule") != "exhaust_singleton_tree_then_compatible_random_groups":
        errors.append("schedule")
    if stats.get("tree_stats", {}).get("prefix_exhausted") is not True:
        errors.append("tree prefix")
    if stats.get("group_stats", {}).get("strategy") != "vectorized_random_compatible_group":
        errors.append("group strategy")
    trace = record.get("trace", [])
    if not trace:
        errors.append("trace")
    elif any(float(a["time"]) > float(b["time"]) + 1e-12 for a, b in zip(trace, trace[1:])):
        errors.append("trace time order")
    elif any(float(a["cost"]) < float(b["cost"]) - 1e-9 for a, b in zip(trace, trace[1:])):
        errors.append("trace cost order")
    if errors:
        raise RuntimeError(f"invalid tree-then-group result {path}: {', '.join(errors)}")


def load_records():
    frozen = base.load_results()
    records = {
        key: value
        for key, value in frozen.items()
        if key[1] in ("robot_subset_mode", "scalable_tree_group")
    }
    for case in cases().values():
        for path_index in protocol()["path_indices"]:
            for seed in protocol()["shortcut_seeds"]:
                path = new_result_path(case["id"], path_index, seed)
                if not path.is_file():
                    raise RuntimeError(f"missing tree-then-group result: {path}")
                value = read_json(path)
                validate_new(value, case, path_index, seed, path)
                key = (case["id"], "tree_then_compatible_group", path_index, seed)
                records[key] = value
    expected = len(cases()) * len(METHODS) * len(protocol()["path_indices"]) * len(protocol()["shortcut_seeds"])
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} schedule cells, loaded {len(records)}")
    print(json.dumps({"loaded_cells": len(records), "cases": len(cases()), "methods": list(METHODS)}, indent=2))
    return records


def selected_records(records, case_id, method_id):
    return [value for key, value in records.items() if key[0] == case_id and key[1] == method_id]


def step_values(record, grid):
    times = np.asarray([float(point["time"]) for point in record["trace"]])
    costs = np.asarray([float(point["cost"]) for point in record["trace"]])
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.full(grid.shape, float(record["initial_cost"]), dtype=float)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values / float(record["initial_cost"])


def aggregate_curve(records, case, method_id, log=False):
    selected = selected_records(records, case["id"], method_id)
    budget = float(case["seconds"])
    if log:
        start = min(
            float(point["time"])
            for record in selected
            for point in record["trace"]
            if float(point["time"]) > 0
        )
        grid = np.geomspace(start, budget, 300)
    else:
        grid = np.linspace(0.0, budget, 301)
    curves = np.asarray([step_values(record, grid) for record in selected])
    return grid, np.median(curves, axis=0), np.min(curves, axis=0), np.max(curves, axis=0)


def style_axis(axis, title, log):
    axis.set_title(title, fontsize=17)
    axis.set_xlabel("time [s]", fontsize=15)
    axis.set_ylabel("relative cost", fontsize=15)
    axis.tick_params(labelsize=13)
    axis.grid(True, alpha=0.22)
    if log:
        axis.set_xscale("log")


def plot_curves(records, selected, output, columns, log=False, title=None):
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.3 * columns, 4.15 * rows), squeeze=False)
    first_axis = None
    for axis, case in zip(axes.flat, selected):
        for method_id, method in METHODS.items():
            x, center, low, high = aggregate_curve(records, case, method_id, log)
            color = method_color(method_id)
            axis.fill_between(x, low, high, step="post", color=color, alpha=0.10, linewidth=0)
            axis.step(x, center, where="post", color=color, linewidth=3.0, label=method["label"])
        count = base.robot_count(case)
        style_axis(axis, f"{count} robot{'s' if count != 1 else ''}", log)
        first_axis = first_axis or axis
    for axis in axes.flat[len(selected):]:
        axis.remove()
    if title:
        figure.suptitle(title, fontsize=20, y=0.995)
    handles, labels = first_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=14)
    figure.tight_layout(rect=(0, 0.13 if rows > 1 else 0.25, 1, 0.96 if title else 0.99))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def scalar_metrics(records, case, method_id):
    aucs, finals = [], []
    for record in selected_records(records, case["id"], method_id):
        budget = float(record["budget_seconds"])
        grid = np.linspace(0.0, budget, 301)
        curve = step_values(record, grid)
        aucs.append(float(np.trapz(curve, grid) / budget))
        finals.append(float(record["final_cost"]) / float(record["initial_cost"]))
    return np.asarray(aucs), np.asarray(finals)


def plot_summary(records, output):
    figure, axes = plt.subplots(2, 4, figsize=(22, 9.2), squeeze=False)
    for column, family in enumerate(FAMILY_ORDER):
        family_cases = [case for case in cases().values() if case["family"] == family]
        grouped = defaultdict(list)
        for case in family_cases:
            grouped[base.robot_count(case)].append(case)
        xs = np.asarray(sorted(grouped))
        for method_id, method in METHODS.items():
            centers = [[], []]
            lows = [[], []]
            highs = [[], []]
            for count in xs:
                per_case = [scalar_metrics(records, case, method_id) for case in grouped[int(count)]]
                for row in (0, 1):
                    values = np.asarray([np.median(metrics[row]) for metrics in per_case])
                    centers[row].append(float(np.median(values)))
                    lows[row].append(float(np.min(values)))
                    highs[row].append(float(np.max(values)))
            for row in (0, 1):
                axis = axes[row, column]
                color = method_color(method_id)
                axis.plot(xs, centers[row], marker="o", markersize=6, color=color, linewidth=2.8, label=method["label"])
                if family == "mobile_wall_scaling":
                    axis.fill_between(xs, lows[row], highs[row], color=color, alpha=0.08)
        axes[0, column].set_title(FAMILY_TITLES[family], fontsize=17)
        axes[1, column].set_xlabel("number of robots", fontsize=15)
        for row in (0, 1):
            axes[row, column].set_xticks(xs)
            axes[row, column].tick_params(labelsize=12)
            axes[row, column].grid(True, alpha=0.22)
    axes[0, 0].set_ylabel("normalized area under curve", fontsize=15)
    axes[1, 0].set_ylabel("final relative cost", fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=14)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def first_threshold_time(record, threshold):
    initial = float(record["initial_cost"])
    for point in record["trace"]:
        if float(point["cost"]) / initial <= threshold:
            return float(point["time"])
    return None


def write_metrics(records):
    metric_fields = (
        "case_id", "family", "num_robots", "wall_x", "wall_z", "wall_blocks",
        "method_id", "normalized_auc_median", "normalized_auc_min", "normalized_auc_max",
        "final_relative_cost_median", "final_relative_cost_min", "final_relative_cost_max",
    )
    with (OUTPUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        for case in cases().values():
            shape = base.wall_shape(case) if case["family"] == "mobile_wall_scaling" else (None, None)
            for method_id in METHODS:
                aucs, finals = scalar_metrics(records, case, method_id)
                writer.writerow({
                    "case_id": case["id"], "family": case["family"],
                    "num_robots": base.robot_count(case), "wall_x": shape[0], "wall_z": shape[1],
                    "wall_blocks": shape[0] * shape[1] if shape[0] else None, "method_id": method_id,
                    "normalized_auc_median": np.median(aucs), "normalized_auc_min": np.min(aucs),
                    "normalized_auc_max": np.max(aucs), "final_relative_cost_median": np.median(finals),
                    "final_relative_cost_min": np.min(finals), "final_relative_cost_max": np.max(finals),
                })

    threshold_fields = (
        "case_id", "family", "num_robots", "method_id", "target_relative_cost",
        "reached", "replicates", "time_median_successes", "time_min", "time_max",
    )
    with (OUTPUT / "time-to-thresholds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=threshold_fields)
        writer.writeheader()
        for case in cases().values():
            for method_id in METHODS:
                selected = selected_records(records, case["id"], method_id)
                for threshold in THRESHOLDS:
                    values = [first_threshold_time(record, threshold) for record in selected]
                    successes = [value for value in values if value is not None]
                    writer.writerow({
                        "case_id": case["id"], "family": case["family"],
                        "num_robots": base.robot_count(case), "method_id": method_id,
                        "target_relative_cost": threshold, "reached": len(successes),
                        "replicates": len(values),
                        "time_median_successes": np.median(successes) if successes else None,
                        "time_min": np.min(successes) if successes else None,
                        "time_max": np.max(successes) if successes else None,
                    })


def main():
    records = load_records()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for family in FAMILY_ORDER[:3]:
        selected = sorted(
            [case for case in cases().values() if case["family"] == family],
            key=base.robot_count,
        )
        columns = 2 if family == "coupled_box_stacking" else 3
        for log, suffix in ((False, ""), (True, "-log")):
            plot_curves(records, selected, OUTPUT / f"{family}-curves{suffix}.png", columns, log)

    mobile = [case for case in cases().values() if case["family"] == "mobile_wall_scaling"]
    shapes = sorted({base.wall_shape(case) for case in mobile}, key=lambda shape: shape[0] * shape[1])
    for x, z in shapes:
        selected = sorted(
            [case for case in mobile if base.wall_shape(case) == (x, z)],
            key=base.robot_count,
        )
        for log, suffix in ((False, ""), (True, "-log")):
            plot_curves(
                records,
                selected,
                OUTPUT / f"mobile_wall_{x}x{z}-curves{suffix}.png",
                4,
                log,
                f"Mobile-wall stacking: {x}×{z} wall ({x * z} blocks)",
            )
    plot_summary(records, OUTPUT / "scaling-summary.png")
    write_metrics(records)


if __name__ == "__main__":
    main()
