"""Audit and plot the largest-setting sum-cost pilot."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
sys.path.insert(0, str(BENCHMARK))
from plot_style import method_color  # noqa: E402
from run_sum_cost_study import definition, output_path, result_errors  # noqa: E402
from analyze import LABELS  # noqa: E402


ROOT = HERE / "sum-cost"
RESULTS = ROOT / "results"
OUTPUT = ROOT / "analysis"
CASES = (
    ("rai-disk-scaling-r12-g4", "2D disks · 12 robots"),
    ("rai-isolated-box-stacking-r9-b4", "Isolated stacking · 9 robots"),
    ("rai-coupled-box-stacking-r6-b8", "Coupled stacking · 6 robots"),
    ("rai-mobile-wall-scaling-r7-x5-z5", "Mobile wall · 7 robots"),
)


def read(path):
    with path.open() as stream:
        return json.load(stream)


PROTOCOL = read(HERE / "protocol.json")
METHODS = tuple(item["id"] for item in PROTOCOL["methods"])


def load():
    results = {}
    errors = []
    for cell in definition()["cells"]:
        path = output_path(cell)
        issues = result_errors(path, cell)
        if issues:
            errors.append(f"{path}: {issues}")
            continue
        results.setdefault((cell["case_id"], cell["method_id"]), []).append(read(path))
    if errors:
        raise RuntimeError("sum-cost audit failed:\n" + "\n".join(errors))
    return results


def curve(result):
    times = [0.0]
    costs = [float(result["initial_cost"])]
    for point in result["trace"]:
        if float(point["time"]) <= float(result["budget_seconds"]):
            times.append(float(point["time"]))
            costs.append(float(point["cost"]))
    times.append(float(result["budget_seconds"]))
    costs.append(costs[-1])
    return np.asarray(times), np.asarray(costs) / costs[0]


def aggregate(values):
    curves = [curve(value) for value in values]
    times = np.unique(np.concatenate([item[0] for item in curves]))
    paths = {}
    for value, (x, y) in zip(values, curves):
        sampled = y[np.maximum(0, np.searchsorted(x, times, side="right") - 1)]
        paths.setdefault(value["path_index"], []).append(sampled)
    medians = np.asarray([np.median(paths[index], axis=0) for index in sorted(paths)])
    return times, np.median(medians, axis=0), medians.min(axis=0), medians.max(axis=0)


def plot(results, log_time=False):
    figure, axes = plt.subplots(2, 2, figsize=(16, 13.5), sharey=True)
    for axis, (case_id, title) in zip(axes.flat, CASES):
        first_positive = []
        for method in METHODS:
            times, costs, low, high = aggregate(results[(case_id, method)])
            axis.step(times, costs, where="post", linewidth=4.0,
                      color=method_color(method), label=LABELS[method])
            axis.fill_between(times, low, high, step="post", color=method_color(method), alpha=0.09)
            positive = times[times > 0]
            if len(positive):
                first_positive.append(float(positive[0]))
        if log_time:
            axis.set_xscale("log")
            axis.set_xlim(left=min(first_positive), right=float(
                results[(case_id, METHODS[0])][0]["budget_seconds"]
            ))
        else:
            axis.set_xlim(0, float(results[(case_id, METHODS[0])][0]["budget_seconds"]))
        axis.set_title(title, fontsize=25)
        axis.set_xlabel("time [s]", fontsize=29)
        axis.tick_params(labelsize=21)
        axis.grid(alpha=0.22)
        axis.set_ylim(bottom=0)
    for axis in axes[:, 0]:
        axis.set_ylabel("normalized sum cost", fontsize=29)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                  ncol=3, frameon=False, fontsize=20)
    figure.tight_layout(rect=(0, 0.14, 1, 1))
    suffix = "-logx" if log_time else ""
    destination = OUTPUT / f"offline-sum-cost-largest{suffix}.png"
    figure.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return destination


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = load()
    rows = []
    for (case_id, method), values in results.items():
        for value in values:
            _, normalized = curve(value)
            rows.append({
                "case_id": case_id,
                "method_id": method,
                "path_index": value["path_index"],
                "shortcut_seed": value["shortcut_seed"],
                "initial_cost": value["initial_cost"],
                "final_cost": value["final_cost"],
                "cost_at_budget": normalized[-1] * value["initial_cost"],
                "normalized_cost_at_budget": normalized[-1],
                "accepted": value["stats"]["accepted"],
                "collision_checks": value["stats"]["collision_checks"],
            })
    with (OUTPUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    audit = {"status": "pass", "cells": len(rows), "expected": len(definition()["cells"]),
             "aggregation": "median across seeds within path, then median across paths",
             "band": "range of path medians"}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    paths = [plot(results), plot(results, log_time=True)]
    print(json.dumps({"audit": audit, "plots": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
