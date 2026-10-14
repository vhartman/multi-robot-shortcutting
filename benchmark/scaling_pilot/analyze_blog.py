"""Produce the compact panels used by the scaling follow-up post."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import (
    FAMILY_LABELS,
    FAMILIES,
    LABELS,
    METHODS,
    curve,
    final_normalized,
    offline_results,
    path_median_curves,
    online_results,
    read,
    robot_count,
    scaling,
    step_values,
)
from plot_style import method_color


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "analysis" / "blog"
INTRO_METHODS = (
    "full_configuration", "tree_gain", "tree_group_gain", "robot_subset_mode",
)


def runs(records, case_id, method):
    return [value for cell, _, value in records
            if cell["case_id"] == case_id and cell["method_id"] == method]


def plot_cases(records, track, cases, methods, title, destination, log=False,
               panel_titles=None, columns=None):
    columns = columns or len(cases)
    rows = math.ceil(len(cases) / columns)
    width = 5.4 * columns
    height = 6.3 * rows + 2.0
    fig, axes = plt.subplots(rows, columns, figsize=(width, height),
                             squeeze=False, sharey=True)
    if panel_titles is None:
        panel_titles = [f"{robot_count(case_id)} robots" for case_id in cases]
    for axis, case_id, panel_title in zip(axes.flat, cases, panel_titles):
        values_for_case = [value for method in methods
                           for value in runs(records, case_id, method)]
        end = max(float(value["budget_seconds"] if track == "offline"
                        else value["refinement_seconds"])
                  for value in values_for_case)
        if log:
            positive = [float(point["time"]) for value in values_for_case
                        for point in (value["comparison_curve"] if track == "online"
                                      else value["trace"])
                        if float(point["time"]) > 0]
            start = max(min(positive), 1e-4)
            grid = np.geomspace(start, end, 301)
        else:
            grid = np.linspace(0, end, 301)
        for method in methods:
            matrix = path_median_curves(runs(records, case_id, method), track, grid)
            axis.plot(grid, np.median(matrix, axis=0),
                      color=method_color(method), linewidth=4.0,
                      label=LABELS[method])
            axis.fill_between(grid, np.min(matrix, axis=0), np.max(matrix, axis=0),
                              color=method_color(method), alpha=0.11, linewidth=0)
        if log:
            axis.set_xscale("log")
            axis.set_xlim(grid[0], grid[-1])
        axis.set_title(panel_title, fontsize=25)
        axis.set_xlabel("time [s]", fontsize=29)
        axis.tick_params(labelsize=21)
        axis.grid(alpha=0.22)
        axis.set_ylim(bottom=0)
    for row in range(rows):
        axes[row, 0].set_ylabel("normalized cost", fontsize=29)
    for axis in axes.flat[len(cases):]:
        axis.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=min(len(methods), 3), frameon=False, fontsize=20)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_scaling_summary(records, track, destination):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13.5), sharey=True)
    case_map = scaling.cases()
    protocol_cases = read(HERE / "protocol.json")["case_ids"]
    available_cases = {cell["case_id"] for cell, _, _ in records}
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for axis, family in zip(axes.flat, FAMILIES):
        cases = sorted(
            [case_id for case_id in protocol_cases
             if case_id in available_cases
             and case_map[case_id]["family"] == family],
            key=robot_count,
        )
        positions = np.arange(len(cases), dtype=float)
        counts = [robot_count(case_id) for case_id in cases]
        values = np.asarray([
            [np.median(final_normalized(records, track, case_id, method))
             for method in METHODS]
            for case_id in cases
        ])
        for position, row_values in zip(positions, values):
            axis.vlines(
                position, np.min(row_values), np.max(row_values),
                color="#d0d0d0", linewidth=2, zorder=0,
            )
        for method_index, method in enumerate(METHODS):
            axis.scatter(
                positions + offsets[method_index], values[:, method_index],
                s=105, color=method_color(method),
                edgecolor="white", linewidth=0.8,
                label=LABELS[method], zorder=3,
            )
        axis.set_title(FAMILY_LABELS[family], fontsize=25)
        axis.set_xlabel("robots", fontsize=29)
        axis.set_xticks(positions, labels=counts)
        axis.tick_params(labelsize=21)
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
    for axis in axes[:, 0]:
        axis.set_ylabel("final normalized cost", fontsize=29)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False, fontsize=20)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_family_curves(records, family, destination):
    case_map = scaling.cases()
    protocol_cases = read(HERE / "protocol.json")["case_ids"]
    cases = sorted(
        [case_id for case_id in protocol_cases
         if case_map[case_id]["family"] == family],
        key=robot_count,
    )
    columns = min(3, len(cases))
    rows = math.ceil(len(cases) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(18, 6.3 * rows + 2.0), squeeze=False,
        sharey=True,
    )
    for axis, case_id in zip(axes.flat, cases):
        values_for_case = [value for method in METHODS
                           for value in runs(records, case_id, method)]
        end = max(float(value["budget_seconds"]) for value in values_for_case)
        grid = np.linspace(0, end, 301)
        for method in METHODS:
            matrix = path_median_curves(runs(records, case_id, method), "offline", grid)
            axis.plot(grid, np.median(matrix, axis=0),
                      color=method_color(method), linewidth=4.0,
                      label=LABELS[method])
            axis.fill_between(grid, np.min(matrix, axis=0), np.max(matrix, axis=0),
                              color=method_color(method), alpha=0.11, linewidth=0)
        axis.set_title(f"{robot_count(case_id)} robots", fontsize=25)
        axis.set_xlabel("time [s]", fontsize=27)
        axis.tick_params(labelsize=20)
        axis.grid(alpha=0.22)
        axis.set_ylim(bottom=0)
    for axis in axes.flat[len(cases):]:
        axis.remove()
    for row in range(rows):
        axes[row, 0].set_ylabel("normalized cost", fontsize=27)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=False, fontsize=20)
    fig.tight_layout(rect=(0, 0.12 if rows > 1 else 0.20, 1, 1))
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    offline = offline_results()
    plot_cases(
        offline, "offline",
        ("rai-disk-scaling-r4-g4", "rai-disk-scaling-r8-g4",
         "rai-disk-scaling-r12-g4"),
        INTRO_METHODS, "2D disk scaling — offline",
        OUTPUT / "intro-disk-scaling.png",
    )
    plot_cases(
        offline, "offline",
        ("rai-coupled-box-stacking-r4-b8",
         "rai-coupled-box-stacking-r6-b8"),
        INTRO_METHODS, "Coupled box stacking — offline",
        OUTPUT / "intro-coupled-scaling.png",
    )
    plot_scaling_summary(offline, "offline", OUTPUT / "offline-scaling-summary.png")
    for family, filename in (
        ("disk_scaling", "offline-disk-scaling-curves.png"),
        ("isolated_box_stacking", "offline-isolated-box-stacking-curves.png"),
        ("coupled_box_stacking", "offline-coupled-box-stacking-curves.png"),
        ("mobile_wall_scaling", "offline-mobile-wall-scaling-curves.png"),
    ):
        plot_family_curves(offline, family, OUTPUT / filename)
    if args.offline_only:
        for path in sorted(OUTPUT.glob("*.png")):
            if path.name.startswith(("offline-", "intro-")):
                print(path)
        return
    online = online_results(require_complete=False)
    lower = (
        "rai-disk-scaling-r4-g4", "rai-isolated-box-stacking-r4-b4",
        "rai-coupled-box-stacking-r4-b8", "rai-mobile-wall-scaling-r4-x5-z5",
    )
    highest = (
        "rai-disk-scaling-r12-g4", "rai-isolated-box-stacking-r9-b4",
        "rai-coupled-box-stacking-r6-b8", "rai-mobile-wall-scaling-r7-x5-z5",
    )
    online_cases = lower + highest
    online_titles = (
        "2D disks · 4 robots", "Isolated stacking · 4 robots",
        "Coupled stacking · 4 robots", "Mobile wall · 4 robots",
        "2D disks · 12 robots", "Isolated stacking · 9 robots",
        "Coupled stacking · 6 robots", "Mobile wall · 7 robots",
    )
    plot_cases(
        online, "online", online_cases, METHODS,
        "Online scaling — moderate and largest settings",
        OUTPUT / "online-largest.png",
        panel_titles=online_titles, columns=4,
    )
    plot_cases(
        online, "online", online_cases, METHODS,
        "Online scaling — moderate and largest settings",
        OUTPUT / "online-largest-logx.png", log=True,
        panel_titles=online_titles, columns=4,
    )
    for path in sorted(OUTPUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
