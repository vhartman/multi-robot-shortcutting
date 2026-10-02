"""Derived analyses for the expanded offline benchmark."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from expanded_lib import (  # noqa: E402
    cases,
    input_path,
    methods,
    methods_for,
    protocol,
    read_json,
    result_path,
)
from plot_style import method_color  # noqa: E402


MARKERS = {
    "single_robot_mode": "o",
    "tree_gain": "s",
    "tree_group_gain": "^",
    "full_tree_gain_round_robin": "D",
    "full_configuration": "P",
    "robot_subset_mode": "X",
}


def step_values(times, costs, grid):
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.ones_like(grid)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values


def performance(result, seconds):
    trace = result["trace"]
    times = np.asarray([point["time"] for point in trace], dtype=float)
    costs = np.asarray([point["cost"] for point in trace], dtype=float)
    costs /= float(result["initial_cost"])
    grid = np.linspace(0.0, float(seconds), 301)
    curve = step_values(times, costs, grid)
    return {
        "final_normalized_cost": float(curve[-1]),
        "normalized_auc": float(np.trapz(curve, grid) / float(seconds)),
    }


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float("nan")


def event_metrics(result):
    stats = result["stats"]
    events = stats.get("accepted_lineage") or []
    initial = float(result["initial_cost"])
    states = max(1, int(result["working_states"]) - 1)
    if events:
        spans = [
            100.0 * (int(event["last"]) - int(event["first"])) / states
            for event in events
        ]
        groups = [len(event["robots"]) for event in events]
        gains = [
            100.0 * max(0.0, float(event["local_gain"])) / initial
            for event in events
        ]
        retained = [
            100.0 * float(event["surviving_fraction"])
            for event in events
        ]
        median_span = float(np.median(spans))
        median_group = float(np.median(groups))
        median_gain = float(np.median(gains))
        mean_retained = float(np.mean(retained))
    else:
        median_span = float("nan")
        median_group = float("nan")
        median_gain = float("nan")
        mean_retained = float("nan")

    checks = int(stats["collision_checks"])
    accepted = int(stats["accepted"])
    improvement = (
        float(result["initial_cost"]) - float(result["final_cost"])
    ) / float(result["initial_cost"])
    return {
        "compatibility_percent": 100.0 * safe_ratio(
            stats["compatible"], stats["draws"]
        ),
        "acceptance_per_check_percent": 100.0 * safe_ratio(accepted, checks),
        "improvement_points_per_1000_checks": (
            100.0 * 1000.0 * safe_ratio(improvement, checks)
        ),
        "checks_per_second": safe_ratio(checks, result["wall_seconds"]),
        "collision_check_time_percent": 100.0 * safe_ratio(
            float(stats["collision_check_seconds_valid"])
            + float(stats["collision_check_seconds_invalid"]),
            result["wall_seconds"],
        ),
        "median_span_percent": median_span,
        "median_group_size": median_group,
        "median_local_gain_percent": median_gain,
        "mean_retained_percent": mean_retained,
    }


def available_result(case, method_id, path_index, seed):
    path = result_path(case["id"], method_id, path_index, seed)
    return read_json(path) if path.is_file() else None


def nested_summary(case, method_id, metric, run_rows_by_key):
    path_centers = []
    all_values = []
    paths_used = 0
    for path_index in case["path_indices"]:
        values = []
        for seed in protocol()["shortcut_seeds"]:
            row = run_rows_by_key.get((case["id"], method_id, path_index, seed))
            if row is not None and np.isfinite(row[metric]):
                values.append(float(row[metric]))
        if values:
            path_centers.append(float(np.median(values)))
            all_values.extend(values)
            paths_used += 1
    if not path_centers:
        return {
            "center": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "paths_used": 0,
            "runs_used": 0,
        }
    uncertainty = path_centers if len(path_centers) > 1 else all_values
    return {
        "center": float(np.median(path_centers)),
        "low": float(np.min(uncertainty)),
        "high": float(np.max(uncertainty)),
        "paths_used": paths_used,
        "runs_used": len(all_values),
    }


def dof_bin(value):
    if value <= 9:
        return "6–9"
    if value <= 20:
        return "12–20"
    return "24–28"


def mode_bin(value):
    if value <= 9:
        return "3–9"
    if value <= 19:
        return "10–19"
    if value <= 39:
        return "20–39"
    return "40+"


def build_rows():
    run_rows = []
    missing = []
    for case in cases().values():
        source = read_json(input_path(case["id"], case["path_indices"][0]))
        mode_runs = int(source.get("mode_count") or 0)
        for method_id in methods_for(case):
            for path_index in case["path_indices"]:
                for seed in protocol()["shortcut_seeds"]:
                    result = available_result(
                        case, method_id, path_index, seed
                    )
                    if result is None:
                        missing.append({
                            "case_id": case["id"],
                            "method_id": method_id,
                            "path_index": path_index,
                            "shortcut_seed": seed,
                        })
                        continue
                    row = {
                        "case_id": case["id"],
                        "track": case["track"],
                        "method_id": method_id,
                        "path_index": path_index,
                        "shortcut_seed": seed,
                        "robot_count": int(result["robot_count"]),
                        "total_dof": int(result["total_dof"]),
                        "mode_runs": mode_runs,
                        "seconds": float(case["seconds"]),
                    }
                    row.update(performance(result, case["seconds"]))
                    row.update(event_metrics(result))
                    run_rows.append(row)

    by_key = {
        (
            row["case_id"],
            row["method_id"],
            row["path_index"],
            row["shortcut_seed"],
        ): row
        for row in run_rows
    }
    metric_names = [
        "final_normalized_cost",
        "normalized_auc",
        "compatibility_percent",
        "acceptance_per_check_percent",
        "improvement_points_per_1000_checks",
        "checks_per_second",
        "collision_check_time_percent",
        "median_span_percent",
        "median_group_size",
        "median_local_gain_percent",
        "mean_retained_percent",
    ]
    scenario_rows = []
    for case in cases().values():
        exemplar = next(
            row for row in run_rows if row["case_id"] == case["id"]
        )
        for method_id in methods_for(case):
            row = {
                "case_id": case["id"],
                "track": case["track"],
                "method_id": method_id,
                "robot_count": exemplar["robot_count"],
                "total_dof": exemplar["total_dof"],
                "mode_runs": exemplar["mode_runs"],
                "dof_bin": dof_bin(exemplar["total_dof"]),
                "mode_bin": mode_bin(exemplar["mode_runs"]),
            }
            for metric in metric_names:
                summary = nested_summary(case, method_id, metric, by_key)
                row[metric] = summary["center"]
                row[f"{metric}_low"] = summary["low"]
                row[f"{metric}_high"] = summary["high"]
            row["runs_expected"] = (
                len(case["path_indices"]) * len(protocol()["shortcut_seeds"])
            )
            row["runs_used"] = sum(
                1
                for path_index in case["path_indices"]
                for seed in protocol()["shortcut_seeds"]
                if (case["id"], method_id, path_index, seed) in by_key
            )
            scenario_rows.append(row)

    tree_auc = {
        row["case_id"]: row["normalized_auc"]
        for row in scenario_rows
        if row["method_id"] == "tree_gain"
    }
    for row in scenario_rows:
        row["auc_delta_from_tree"] = (
            row["normalized_auc"] - tree_auc[row["case_id"]]
        )
    return run_rows, scenario_rows, missing


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_label(method_id):
    return methods()[method_id]["label"]


def style_axis(axis, title, ylabel):
    axis.set_title(title, fontsize=17)
    axis.set_ylabel(ylabel, fontsize=15)
    axis.tick_params(labelsize=12)
    axis.grid(alpha=0.22)


def legend_below(figure, axes, columns):
    handles = {}
    for axis in np.asarray(axes).flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=min(columns, len(handles)),
        fontsize=13,
        frameon=False,
    )


def errorbar_series(axis, rows, metric, method_ids, x_key, x_values):
    for method_id in method_ids:
        selected = {
            row[x_key]: row
            for row in rows
            if row["method_id"] == method_id
        }
        centers = np.asarray(
            [selected[value][metric] for value in x_values], dtype=float
        )
        lows = np.asarray(
            [selected[value][f"{metric}_low"] for value in x_values],
            dtype=float,
        )
        highs = np.asarray(
            [selected[value][f"{metric}_high"] for value in x_values],
            dtype=float,
        )
        axis.errorbar(
            x_values,
            centers,
            yerr=np.vstack((centers - lows, highs - centers)),
            color=method_color(method_id),
            marker=MARKERS[method_id],
            linewidth=2.5,
            markersize=7,
            capsize=3,
            label=method_label(method_id),
        )


def plot_scaling_performance(rows, output):
    selected = [row for row in rows if row["track"] == "scaling"]
    method_ids = list(methods_for(cases()["rai-isolated-box-stacking-r1-b4"]))
    robots = list(range(1, 7))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    errorbar_series(
        axes[0], selected, "final_normalized_cost",
        method_ids, "robot_count", robots,
    )
    errorbar_series(
        axes[1], selected, "normalized_auc",
        method_ids, "robot_count", robots,
    )
    style_axis(axes[0], "Final cost", "normalized cost")
    style_axis(axes[1], "Cost over time", "normalized AUC")
    for axis in axes:
        axis.set_xlabel("number of robots", fontsize=15)
        axis.set_xticks(robots)
    legend_below(figure, axes, 3)
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    path = output / "scaling-performance.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def grouped_values(rows, method_id, group_key, group_values, metric):
    centers, lows, highs = [], [], []
    for group in group_values:
        values = np.asarray([
            row[metric]
            for row in rows
            if row["method_id"] == method_id and row[group_key] == group
            and np.isfinite(row[metric])
        ], dtype=float)
        centers.append(float(np.median(values)))
        lows.append(float(np.percentile(values, 25)))
        highs.append(float(np.percentile(values, 75)))
    return np.asarray(centers), np.asarray(lows), np.asarray(highs)


def categorical_panel(
    axis,
    rows,
    metric,
    method_ids,
    group_key,
    groups,
    title,
    ylabel,
):
    positions = np.arange(len(groups), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(method_ids))
    for offset, method_id in zip(offsets, method_ids):
        center, low, high = grouped_values(
            rows, method_id, group_key, groups, metric
        )
        axis.errorbar(
            positions + offset,
            center,
            yerr=np.vstack((center - low, high - center)),
            color=method_color(method_id),
            marker=MARKERS[method_id],
            linewidth=2.0,
            markersize=6,
            capsize=3,
            label=method_label(method_id),
        )
    axis.axhline(
        0.0,
        color="#777777",
        linewidth=1.0,
        alpha=0.5,
        zorder=0,
    ) if metric == "auc_delta_from_tree" else None
    axis.set_xticks(positions, groups)
    style_axis(axis, title, ylabel)


def plot_complexity(rows, output):
    selected = [row for row in rows if row["track"] == "broad"]
    method_ids = list(methods_for(cases()["rai-simple"]))
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    categorical_panel(
        axes[0, 0], selected, "normalized_auc", method_ids,
        "dof_bin", ["6–9", "12–20", "24–28"],
        "Grouped by total DoF", "normalized AUC",
    )
    categorical_panel(
        axes[0, 1], selected, "auc_delta_from_tree", method_ids,
        "dof_bin", ["6–9", "12–20", "24–28"],
        "Difference from standard tree", "AUC difference",
    )
    categorical_panel(
        axes[1, 0], selected, "normalized_auc", method_ids,
        "mode_bin", ["3–9", "10–19", "20–39", "40+"],
        "Grouped by mode segments", "normalized AUC",
    )
    categorical_panel(
        axes[1, 1], selected, "auc_delta_from_tree", method_ids,
        "mode_bin", ["3–9", "10–19", "20–39", "40+"],
        "Difference from standard tree", "AUC difference",
    )
    for axis in axes[1]:
        axis.set_xlabel("complexity bin", fontsize=15)
    legend_below(figure, axes, 4)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    path = output / "broad-complexity-groups.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_efficiency(rows, output, group_key, groups, name, x_label):
    selected = [row for row in rows if row["track"] == "broad"]
    method_ids = list(methods_for(cases()["rai-simple"]))
    specifications = [
        ("compatibility_percent", "Compatible proposals", "compatible [%]"),
        (
            "acceptance_per_check_percent",
            "Accepted collision checks",
            "accepted/checks [%]",
        ),
        (
            "improvement_points_per_1000_checks",
            "Cost reduction per checking work",
            "normalized-cost points / 1000 checks",
        ),
        ("checks_per_second", "Collision-check throughput", "checks / s"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for axis, (metric, title, ylabel) in zip(axes.flat, specifications):
        categorical_panel(
            axis, selected, metric, method_ids, group_key, groups,
            title, ylabel,
        )
        axis.set_xlabel(x_label, fontsize=15)
    legend_below(figure, axes, 4)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    path = output / name
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_scaling_structure(rows, output):
    selected = [row for row in rows if row["track"] == "scaling"]
    method_ids = list(methods_for(cases()["rai-isolated-box-stacking-r1-b4"]))
    robots = list(range(1, 7))
    specifications = [
        (
            "median_span_percent",
            "Accepted shortcut span",
            "median span [% path]",
        ),
        (
            "median_group_size",
            "Accepted robot-group size",
            "median robots / shortcut",
        ),
        (
            "median_local_gain_percent",
            "Accepted shortcut gain",
            "median local gain [% initial cost]",
        ),
        (
            "mean_retained_percent",
            "Eventually retained shortcut content",
            "mean surviving slots [%]",
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    for axis, (metric, title, ylabel) in zip(axes.flat, specifications):
        errorbar_series(
            axis, selected, metric, method_ids, "robot_count", robots
        )
        style_axis(axis, title, ylabel)
        axis.set_xlabel("number of robots", fontsize=15)
        axis.set_xticks(robots)
    legend_below(figure, axes, 3)
    figure.tight_layout(rect=(0, 0.14, 1, 1))
    path = output / "scaling-shortcut-structure.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def main():
    output = HERE / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    run_rows, scenario_rows, missing = build_rows()
    write_csv(output / "run-metrics.csv", run_rows)
    write_csv(output / "scenario-metrics.csv", scenario_rows)
    (output / "derived-summary.json").write_text(json.dumps({
        "schema": "mr-shortcut-expanded-derived-analysis-v1",
        "aggregation": protocol()["aggregation"],
        "missing_cells": missing,
        "run_metrics": run_rows,
        "scenario_metrics": scenario_rows,
    }, indent=2) + "\n")

    paths = [
        plot_scaling_performance(scenario_rows, output),
        plot_complexity(scenario_rows, output),
        plot_efficiency(
            scenario_rows,
            output,
            "dof_bin",
            ["6–9", "12–20", "24–28"],
            "broad-efficiency-by-dof.png",
            "total DoF bin",
        ),
        plot_efficiency(
            scenario_rows,
            output,
            "mode_bin",
            ["3–9", "10–19", "20–39", "40+"],
            "broad-efficiency-by-mode-count.png",
            "mode-segment bin",
        ),
        plot_scaling_structure(scenario_rows, output),
    ]
    print(json.dumps({
        "valid_runs": len(run_rows),
        "missing_cells": len(missing),
        "figures": [str(path) for path in paths],
        "run_metrics": str(output / "run-metrics.csv"),
        "scenario_metrics": str(output / "scenario-metrics.csv"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
