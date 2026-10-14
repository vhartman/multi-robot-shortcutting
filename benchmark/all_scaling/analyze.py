"""Plot the completed all-scaling comparison from its frozen manifest."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
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

from plot_style import method_color  # noqa: E402
from study_lib import cases, methods, protocol, read_json  # noqa: E402


ANALYSIS = HERE / "analysis"
MANIFEST = HERE / "execution-manifest.json"
METHOD_ORDER = tuple(methods())
FAMILY_ORDER = (
    "isolated_box_stacking",
    "disk_scaling",
    "coupled_box_stacking",
    "mobile_wall_scaling",
)
FAMILY_TITLES = {
    "isolated_box_stacking": "Isolated box stacking",
    "disk_scaling": "Planar disks with obstacles",
    "coupled_box_stacking": "Coupled box stacking",
    "mobile_wall_scaling": "Mobile-wall stacking",
}


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def robot_count(case):
    return int(case["environment_kwargs"]["num_robots"])


def wall_shape(case):
    kwargs = case["environment_kwargs"]
    return int(kwargs["wall_x"]), int(kwargs["wall_z"])


def wall_blocks(case):
    x, z = wall_shape(case)
    return x * z


def local_result(record):
    return (
        HERE
        / "offline"
        / "results"
        / record["case_id"]
        / (
            f"{record['method_id']}-path{record['path_index']}"
            f"-seed{record['shortcut_seed']}.json"
        )
    )


def load_results():
    manifest = read_json(MANIFEST)
    records = {}
    errors = []
    for frozen in manifest["cells"]:
        path = REPO / frozen["reused_result"] if frozen["reused_result"] else local_result(frozen)
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        value = read_json(path)
        key = (
            frozen["case_id"],
            frozen["method_id"],
            int(frozen["path_index"]),
            int(frozen["shortcut_seed"]),
        )
        expected = {
            "case_id": frozen["case_id"],
            "method_id": frozen["method_id"],
            "path_index": int(frozen["path_index"]),
            "shortcut_seed": int(frozen["shortcut_seed"]),
        }
        mismatches = [name for name, wanted in expected.items() if value.get(name) != wanted]
        if value.get("strict_collision_free") is not True:
            mismatches.append("strict_collision_free")
        if mismatches:
            errors.append(f"invalid {path}: {', '.join(mismatches)}")
        records[key] = value
    if errors:
        raise RuntimeError("\n".join(errors[:20]))
    if len(records) != len(manifest["cells"]):
        raise RuntimeError("manifest contains duplicate cells")
    print(
        json.dumps(
            {
                "manifest_sha256": digest(manifest),
                "loaded_cells": len(records),
                "cases": len({key[0] for key in records}),
            },
            indent=2,
        )
    )
    return records


def step_values(record, grid):
    times = np.asarray([float(point["time"]) for point in record["trace"]])
    costs = np.asarray([float(point["cost"]) for point in record["trace"]])
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.full(grid.shape, float(record["initial_cost"]), dtype=float)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values / float(record["initial_cost"])


def case_records(records, case_id, method_id):
    return [
        record
        for key, record in records.items()
        if key[0] == case_id and key[1] == method_id
    ]


def aggregate_curve(records, case, method_id, log=False):
    selected = case_records(records, case["id"], method_id)
    if not selected:
        return None
    budget = float(case["seconds"])
    if log:
        positives = [
            float(point["time"])
            for record in selected
            for point in record["trace"]
            if float(point["time"]) > 0
        ]
        start = min(positives)
        grid = np.geomspace(start, budget, 300)
    else:
        grid = np.linspace(0.0, budget, 301)
    curves = np.asarray([step_values(record, grid) for record in selected])
    return grid, np.median(curves, axis=0), np.min(curves, axis=0), np.max(curves, axis=0)


def style_curve_axis(axis, title, log):
    axis.set_title(title, fontsize=17)
    axis.set_xlabel("time [s]", fontsize=15)
    axis.set_ylabel("relative cost", fontsize=15)
    axis.tick_params(labelsize=13)
    axis.grid(True, alpha=0.22)
    if log:
        axis.set_xscale("log")


def plot_curve_grid(records, selected, output, columns=3, log=False, title=None):
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.3 * columns, 4.15 * rows),
        squeeze=False,
    )
    first_axis = None
    for axis, case in zip(axes.flat, selected):
        for method_id, method in methods().items():
            curve = aggregate_curve(records, case, method_id, log=log)
            if curve is None:
                continue
            x, center, low, high = curve
            color = method_color(method_id)
            axis.fill_between(x, low, high, step="post", color=color, alpha=0.10, linewidth=0)
            axis.step(
                x,
                center,
                where="post",
                color=color,
                linewidth=3.0,
                label=method["label"],
            )
            first_axis = first_axis or axis
        count = robot_count(case)
        style_curve_axis(axis, f"{count} robot{'s' if count != 1 else ''}", log)
    for axis in axes.flat[len(selected):]:
        axis.remove()
    if title:
        figure.suptitle(title, fontsize=20, y=0.995)
    handles, labels = first_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    bottom = 0.13 if rows > 1 else 0.26
    top = 0.96 if title else 0.99
    figure.tight_layout(rect=(0, bottom, 1, top))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def scalar_metrics(records, case, method_id):
    selected = case_records(records, case["id"], method_id)
    aucs, finals = [], []
    for record in selected:
        budget = float(record["budget_seconds"])
        grid = np.linspace(0.0, budget, 301)
        curve = step_values(record, grid)
        aucs.append(float(np.trapz(curve, grid) / budget))
        finals.append(float(record["final_cost"]) / float(record["initial_cost"]))
    return (
        float(np.median(aucs)),
        float(np.min(aucs)),
        float(np.max(aucs)),
        float(np.median(finals)),
        float(np.min(finals)),
        float(np.max(finals)),
    )


def plot_robot_scaling_summary(records, output):
    figure, axes = plt.subplots(2, 4, figsize=(22, 9.2), squeeze=False)
    for column, family in enumerate(FAMILY_ORDER):
        family_cases = [case for case in cases().values() if case["family"] == family]
        grouped = defaultdict(list)
        for case in family_cases:
            grouped[robot_count(case)].append(case)
        counts = sorted(grouped)
        positions = np.arange(len(counts), dtype=float)
        selected_methods = methods()
        bar_width = 0.84 / len(selected_methods)
        for method_index, (method_id, method) in enumerate(selected_methods.items()):
            auc_center, auc_low, auc_high = [], [], []
            final_center, final_low, final_high = [], [], []
            for count in counts:
                values = [scalar_metrics(records, case, method_id) for case in grouped[int(count)]]
                # Mobile-wall bars summarize the seven task counts. Other families
                # have one independent case per robot count.
                auc_values = np.asarray([value[0] for value in values])
                final_values = np.asarray([value[3] for value in values])
                auc_center.append(float(np.median(auc_values)))
                auc_low.append(float(np.min(auc_values)))
                auc_high.append(float(np.max(auc_values)))
                final_center.append(float(np.median(final_values)))
                final_low.append(float(np.min(final_values)))
                final_high.append(float(np.max(final_values)))
            color = method_color(method_id)
            for row, center, low, high in (
                (0, auc_center, auc_low, auc_high),
                (1, final_center, final_low, final_high),
            ):
                axis = axes[row, column]
                center = np.asarray(center)
                low = np.asarray(low)
                high = np.asarray(high)
                offset = (method_index - (len(selected_methods) - 1) / 2) * bar_width
                bar_centers = positions + offset
                axis.bar(
                    bar_centers,
                    center,
                    width=bar_width * 0.92,
                    color=color,
                    label=method["label"],
                )
                axis.errorbar(
                    bar_centers,
                    center,
                    yerr=np.vstack((center - low, high - center)),
                    fmt="none",
                    ecolor="#333333",
                    elinewidth=1.0,
                    capsize=1.8,
                    capthick=1.0,
                )
        axes[0, column].set_title(FAMILY_TITLES[family], fontsize=17)
        axes[1, column].set_xlabel("number of robots", fontsize=15)
        for row in (0, 1):
            axes[row, column].set_xticks(positions, labels=counts)
            axes[row, column].tick_params(labelsize=12)
            axes[row, column].grid(axis="y", alpha=0.22)
            axes[row, column].set_axisbelow(True)
            axes[row, column].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("normalized area under curve", fontsize=15)
    axes[1, 0].set_ylabel("final relative cost", fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=14)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_mobile_heatmaps(records, metric_index, output, label):
    mobile = [case for case in cases().values() if case["family"] == "mobile_wall_scaling"]
    robots = sorted({robot_count(case) for case in mobile})
    shapes = sorted({wall_shape(case) for case in mobile}, key=lambda shape: shape[0] * shape[1])
    values = {}
    for method_id in METHOD_ORDER:
        matrix = np.full((len(shapes), len(robots)), np.nan)
        for row, shape in enumerate(shapes):
            for column, count in enumerate(robots):
                case = next(
                    case
                    for case in mobile
                    if wall_shape(case) == shape and robot_count(case) == count
                )
                matrix[row, column] = scalar_metrics(records, case, method_id)[metric_index]
        values[method_id] = matrix
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in values.values()])
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    figure, axes = plt.subplots(
        1,
        len(METHOD_ORDER),
        figsize=(26, 5.4),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, method_id in zip(axes.flat, METHOD_ORDER):
        image = axis.imshow(values[method_id], origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        axis.set_title(methods()[method_id]["label"], fontsize=16)
        axis.set_xticks(range(len(robots)), labels=robots)
        axis.set_yticks(
            range(len(shapes)),
            labels=[f"{x}×{z} ({x * z})" for x, z in shapes],
        )
        axis.set_xlabel("number of robots", fontsize=14)
        if method_id == METHOD_ORDER[0]:
            axis.set_ylabel("wall shape (blocks)", fontsize=14)
        axis.tick_params(labelsize=12)
    colorbar = figure.colorbar(image, ax=list(axes.flat), shrink=0.92, pad=0.015)
    colorbar.set_label(label, fontsize=14)
    colorbar.ax.tick_params(labelsize=12)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_metrics(records, output):
    fields = [
        "case_id",
        "family",
        "num_robots",
        "wall_x",
        "wall_z",
        "wall_blocks",
        "method_id",
        "normalized_auc_median",
        "normalized_auc_min",
        "normalized_auc_max",
        "final_relative_cost_median",
        "final_relative_cost_min",
        "final_relative_cost_max",
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases().values():
            shape = wall_shape(case) if case["family"] == "mobile_wall_scaling" else (None, None)
            for method_id in METHOD_ORDER:
                auc, auc_low, auc_high, final, final_low, final_high = scalar_metrics(
                    records, case, method_id
                )
                writer.writerow(
                    {
                        "case_id": case["id"],
                        "family": case["family"],
                        "num_robots": robot_count(case),
                        "wall_x": shape[0],
                        "wall_z": shape[1],
                        "wall_blocks": shape[0] * shape[1] if shape[0] else None,
                        "method_id": method_id,
                        "normalized_auc_median": auc,
                        "normalized_auc_min": auc_low,
                        "normalized_auc_max": auc_high,
                        "final_relative_cost_median": final,
                        "final_relative_cost_min": final_low,
                        "final_relative_cost_max": final_high,
                    }
                )


def main():
    records = load_results()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    for family in FAMILY_ORDER[:3]:
        selected = sorted(
            [case for case in cases().values() if case["family"] == family],
            key=robot_count,
        )
        columns = 2 if family == "coupled_box_stacking" else 3
        plot_curve_grid(
            records,
            selected,
            ANALYSIS / f"{family}-curves.png",
            columns=columns,
        )
        plot_curve_grid(
            records,
            selected,
            ANALYSIS / f"{family}-curves-log.png",
            columns=columns,
            log=True,
        )

    mobile = [case for case in cases().values() if case["family"] == "mobile_wall_scaling"]
    shapes = sorted({wall_shape(case) for case in mobile}, key=lambda shape: shape[0] * shape[1])
    for x, z in shapes:
        selected = sorted(
            [case for case in mobile if wall_shape(case) == (x, z)],
            key=robot_count,
        )
        for log, suffix in ((False, ""), (True, "-log")):
            plot_curve_grid(
                records,
                selected,
                ANALYSIS / f"mobile_wall_{x}x{z}-curves{suffix}.png",
                columns=4,
                log=log,
                title=f"Mobile-wall stacking: {x}×{z} wall ({x * z} blocks)",
            )

    plot_robot_scaling_summary(records, ANALYSIS / "all-scaling-summary.png")
    plot_mobile_heatmaps(
        records,
        0,
        ANALYSIS / "mobile-wall-auc-heatmaps.png",
        "normalized area under curve",
    )
    plot_mobile_heatmaps(
        records,
        3,
        ANALYSIS / "mobile-wall-final-cost-heatmaps.png",
        "final relative cost",
    )
    write_metrics(records, ANALYSIS / "metrics.csv")


if __name__ == "__main__":
    main()
