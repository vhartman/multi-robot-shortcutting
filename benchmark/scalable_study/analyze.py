"""Analyze the broad and three scaling families in the scalable study."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from plot_style import method_color  # noqa: E402
from study_lib import cases, methods, protocol, read_json, result_path  # noqa: E402


ANALYSIS = HERE / "analysis"
METHOD_ORDER = tuple(methods())
FAMILY_ORDER = (
    "isolated_box_stacking",
    "disk_scaling",
    "coupled_box_stacking",
)
FAMILY_TITLES = {
    "isolated_box_stacking": "Isolated box stacking",
    "disk_scaling": "Planar disks with obstacles",
    "coupled_box_stacking": "Coupled box stacking",
}


def expected_results():
    for case in cases().values():
        for method_id in METHOD_ORDER:
            for path_index in protocol()["path_indices"]:
                for seed in protocol()["shortcut_seeds"]:
                    yield case, method_id, path_index, seed, result_path(
                        case["id"], method_id, path_index, seed
                    )


def load_results(allow_partial: bool):
    output = {}
    missing = []
    for case, method_id, path_index, seed, path in expected_results():
        if not path.exists():
            missing.append(path)
            continue
        value = read_json(path)
        if value.get("strict_collision_free") is not True:
            raise RuntimeError(f"uncertified result: {path}")
        output[(case["id"], method_id, path_index, seed)] = value
    if missing and not allow_partial:
        raise RuntimeError(
            f"{len(missing)} result cells are missing; pass --allow-partial for progress plots"
        )
    print(f"loaded={len(output)} missing={len(missing)}")
    return output


def step_value(trace, key, x):
    coordinates = np.asarray([float(point[key]) for point in trace])
    values = np.asarray([float(point["cost"]) for point in trace])
    index = np.searchsorted(coordinates, x, side="right") - 1
    return values[np.maximum(index, 0)]


def aggregated_curve(records, case, method_id, key="time", log=False):
    budget = float(case["seconds"])
    available = [
        record for record_key, record in records.items()
        if record_key[0] == case["id"] and record_key[1] == method_id
    ]
    if not available:
        return None
    if log:
        first = min(
            float(point[key])
            for record in available
            for point in record["trace"][1:]
            if float(point[key]) > 0
        )
        grid = np.geomspace(first, budget, 260)
    else:
        grid = np.linspace(0.0, budget, 301)
    path_curves = []
    for path_index in protocol()["path_indices"]:
        seed_curves = []
        for seed in protocol()["shortcut_seeds"]:
            record = records.get((case["id"], method_id, path_index, seed))
            if record is None:
                continue
            seed_curves.append(
                step_value(record["trace"], key, grid) / float(record["initial_cost"])
            )
        if seed_curves:
            path_curves.append(np.median(seed_curves, axis=0))
    if not path_curves:
        return None
    stacked = np.asarray(path_curves)
    return grid, np.median(stacked, axis=0), np.min(stacked, axis=0), np.max(stacked, axis=0)


def _style_axis(axis, title):
    axis.set_title(title, fontsize=15)
    axis.grid(True, alpha=0.22)
    axis.tick_params(labelsize=11)
    axis.set_xlabel("time [s]", fontsize=12)
    axis.set_ylabel("relative cost", fontsize=12)


def plot_case_grid(records, selected_cases, output, columns=3, log=False):
    rows = math.ceil(len(selected_cases) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 3.8 * rows), squeeze=False)
    for axis, case in zip(axes.flat, selected_cases):
        for method_id, method in methods().items():
            curve = aggregated_curve(records, case, method_id, log=log)
            if curve is None:
                continue
            x, median, low, high = curve
            color = method_color(method_id)
            axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
            axis.plot(x, median, color=color, linewidth=2.7, label=method["label"])
        if log:
            axis.set_xscale("log")
        _style_axis(axis, case["id"].replace("rai-", "").replace("-", " "))
    for axis in axes.flat[len(selected_cases):]:
        axis.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(METHOD_ORDER), frameon=False, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def robot_count(case):
    if case.get("constructor") == "rai_disk_scaling":
        return int(case["environment_kwargs"]["num_robots"])
    if case.get("constructor") in (
        "rai_isolated_arm_box_stack_env",
        "coupled_box_stacking_env",
    ):
        return int(case["environment_kwargs"]["num_robots"])
    raise ValueError(case["id"])


def threshold_value(record, improvement, key):
    target = float(record["initial_cost"]) * (1.0 - improvement)
    for point in record["trace"]:
        if float(point["cost"]) <= target:
            return float(point[key])
    return math.nan


def aggregate_scalar(records, case, method_id, function):
    paths = []
    for path_index in protocol()["path_indices"]:
        values = [
            function(records[(case["id"], method_id, path_index, seed)])
            for seed in protocol()["shortcut_seeds"]
            if (case["id"], method_id, path_index, seed) in records
        ]
        if len(values) != len(protocol()["shortcut_seeds"]):
            return math.nan, math.nan, math.nan
        if not all(np.isfinite(value) for value in values):
            return math.nan, math.nan, math.nan
        paths.append(float(np.median(values)))
    if len(paths) != len(protocol()["path_indices"]):
        return math.nan, math.nan, math.nan
    return float(np.median(paths)), float(np.min(paths)), float(np.max(paths))


def plot_thresholds(records, key, output):
    thresholds = protocol()["thresholds_relative_improvement"]
    figure, axes = plt.subplots(
        len(thresholds), len(FAMILY_ORDER),
        figsize=(15.5, 11.5), squeeze=False, sharex="col",
    )
    for row, threshold in enumerate(thresholds):
        for column, family in enumerate(FAMILY_ORDER):
            axis = axes[row, column]
            family_cases = sorted(
                [case for case in cases().values() if case["family"] == family],
                key=robot_count,
            )
            xs = np.asarray([robot_count(case) for case in family_cases])
            for method_id, method in methods().items():
                stats = [
                    aggregate_scalar(
                        records,
                        case,
                        method_id,
                        lambda record, threshold=threshold: threshold_value(record, threshold, key),
                    )
                    for case in family_cases
                ]
                median = np.asarray([item[0] for item in stats])
                low = np.asarray([item[1] for item in stats])
                high = np.asarray([item[2] for item in stats])
                if np.any(np.isfinite(median)):
                    axis.plot(
                        xs, median, marker="o", linewidth=2.5,
                        color=method_color(method_id), label=method["label"],
                    )
                    axis.fill_between(
                        xs, low, high,
                        color=method_color(method_id), alpha=0.12,
                    )
            if row == 0:
                axis.set_title(FAMILY_TITLES[family], fontsize=15)
            if column == 0:
                unit = "time [s]" if key == "time" else "collision checks"
                axis.set_ylabel(f"{int(100 * threshold)}% improvement\n{unit}", fontsize=12)
            if row == len(thresholds) - 1:
                axis.set_xlabel("number of robots", fontsize=12)
            axis.set_xticks(xs)
            axis.grid(True, alpha=0.22)
            axis.tick_params(labelsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(METHOD_ORDER), frameon=False, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_run_metrics(records, output):
    fields = [
        "case_id", "family", "robot_count", "total_dof", "method_id",
        "path_index", "shortcut_seed", "initial_cost", "final_cost",
        "normalized_final_cost", "normalized_auc",
        "time_to_5pct", "time_to_10pct", "time_to_20pct",
        "checks_to_5pct", "checks_to_10pct", "checks_to_20pct",
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(records):
            record = records[key]
            budget = float(record["budget_seconds"])
            grid = np.linspace(0, budget, 301)
            normalized = step_value(record["trace"], "time", grid) / float(record["initial_cost"])
            row = {
                "case_id": record["case_id"],
                "family": record["family"],
                "robot_count": record["robot_count"],
                "total_dof": record["total_dof"],
                "method_id": record["method_id"],
                "path_index": record["path_index"],
                "shortcut_seed": record["shortcut_seed"],
                "initial_cost": record["initial_cost"],
                "final_cost": record["final_cost"],
                "normalized_final_cost": float(record["final_cost"]) / float(record["initial_cost"]),
                "normalized_auc": float(np.trapz(normalized, grid) / budget),
            }
            for threshold in protocol()["thresholds_relative_improvement"]:
                suffix = f"{int(100 * threshold)}pct"
                row[f"time_to_{suffix}"] = threshold_value(record, threshold, "time")
                row[f"checks_to_{suffix}"] = threshold_value(record, threshold, "collision_checks")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    records = load_results(args.allow_partial)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    broad = [case for case in cases().values() if case["family"] == "broad"]
    page_paths = []
    log_paths = []
    for page, start in enumerate(range(0, len(broad), 10), start=1):
        selected = broad[start:start + 10]
        page_path = ANALYSIS / f"broad-curves-{page}.png"
        log_path = ANALYSIS / f"broad-curves-{page}-log.png"
        plot_case_grid(records, selected, page_path, columns=2, log=False)
        plot_case_grid(records, selected, log_path, columns=2, log=True)
        page_paths.append(page_path)
        log_paths.append(log_path)
    plot_case_grid(
        records,
        broad,
        ANALYSIS / "broad-curves-combined.png",
        columns=5,
        log=False,
    )
    plot_case_grid(
        records,
        broad,
        ANALYSIS / "broad-curves-combined-log.png",
        columns=5,
        log=True,
    )

    for family in FAMILY_ORDER:
        selected = sorted(
            [case for case in cases().values() if case["family"] == family],
            key=robot_count,
        )
        columns = 2 if family == "coupled_box_stacking" else 3
        plot_case_grid(records, selected, ANALYSIS / f"{family}-curves.png", columns=columns)
        plot_case_grid(records, selected, ANALYSIS / f"{family}-curves-log.png", columns=columns, log=True)

    plot_thresholds(records, "time", ANALYSIS / "scaling-time-to-improvement.png")
    plot_thresholds(records, "collision_checks", ANALYSIS / "scaling-checks-to-improvement.png")
    write_run_metrics(records, ANALYSIS / "run-metrics.csv")


if __name__ == "__main__":
    main()
