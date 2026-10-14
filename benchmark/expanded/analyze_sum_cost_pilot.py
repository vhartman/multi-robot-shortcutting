"""Plot the one-path/one-seed sum-cost pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "analysis" / "sum-cost-scalable-pilot"
CASES = (
    ("rai-triple-waypoints", "triple waypoints"),
    ("rai-cube-four", "cube four"),
    ("assembly-cross-4r-openarm", "Cross, 4 robots"),
    ("assembly-car-mini-4r-openarm", "Car Mini, 4 robots"),
)
METHODS = (
    ("tree_prefix", "tree + gain (standard)", "#e66101"),
    ("scalable_tree_group", "scalable tree/group", "#332288"),
)


def load(case_id, method):
    path = ROOT / case_id / f"{method}-path0-seed0.json"
    with path.open() as stream:
        result = json.load(stream)
    if result["cost"] != {"per_agent": "euclidean", "reduction": "sum"}:
        raise RuntimeError(f"wrong cost metadata: {path}")
    return result


def normalized_trace(result):
    times = np.asarray([point["time"] for point in result["trace"]], dtype=float)
    costs = np.asarray([point["cost"] for point in result["trace"]], dtype=float)
    return times, costs / float(result["initial_cost"])


def auc(result):
    times, costs = normalized_trace(result)
    budget = float(result["budget_seconds"])
    grid = np.linspace(0.0, budget, 601)
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.ones_like(grid)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return float(np.trapz(values, grid) / budget)


def plot(results, *, log_time):
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 10.0), squeeze=False)
    for axis, (case_id, title) in zip(axes.flat, CASES):
        first_positive = []
        for method, label, color in METHODS:
            times, costs = normalized_trace(results[(case_id, method)])
            axis.step(times, costs, where="post", linewidth=3.0, color=color, label=label)
            positive = times[times > 0]
            if len(positive):
                first_positive.append(float(positive[0]))
        if log_time:
            axis.set_xscale("log")
            if first_positive:
                axis.set_xlim(left=min(first_positive))
        axis.set_title(title, fontsize=18)
        axis.set_xlabel("time [s]", fontsize=16)
        axis.set_ylabel("sum cost / initial sum cost", fontsize=16)
        axis.tick_params(labelsize=13)
        axis.grid(alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=16)
    figure.tight_layout(rect=(0, 0.07, 1, 0.98))
    suffix = "log" if log_time else "linear"
    destination = ROOT / f"sum-cost-standard-vs-scalable-{suffix}.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def main() -> None:
    results = {
        (case_id, method): load(case_id, method)
        for case_id, _ in CASES
        for method, _, _ in METHODS
    }
    rows = []
    for case_id, _ in CASES:
        for method, _, _ in METHODS:
            result = results[(case_id, method)]
            rows.append({
                "case_id": case_id,
                "method": method,
                "auc": auc(result),
                "final_normalized_cost": (
                    float(result["final_cost"]) / float(result["initial_cost"])
                ),
                "accepted": result["stats"]["accepted"]
                if isinstance(result["stats"]["accepted"], int)
                else sum(result["stats"]["accepted"].values()),
                "collision_checks": result["stats"]["collision_checks"],
            })
    with (ROOT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    paths = [plot(results, log_time=False), plot(results, log_time=True)]
    print(json.dumps({"rows": rows, "plots": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
