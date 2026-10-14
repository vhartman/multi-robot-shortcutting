"""Compare global max-objective and robot-local singleton gain ordering."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / "offline" / "results"
SCALABLE = HERE / "analysis" / "scalable-tree-group-final"
ABLATION = HERE / "analysis" / "max-gain-order-ablation"
CASES = (
    ("rai-triple-waypoints", "triple waypoints"),
    ("rai-cube-four", "cube four"),
    ("assembly-cross-4r-openarm", "Cross, 4 robots"),
    ("assembly-car-mini-4r-openarm", "Car Mini, 4 robots"),
)
METHODS = (
    ("tree_gain", "global max-cost gain (standard)", "#e66101"),
    ("tree_robot_local_gain", "robot-local gain", "#4c4c4c"),
    ("scalable_tree_group", "scalable tree/group", "#332288"),
)


def result_path(case_id: str, method: str) -> Path:
    if method == "tree_gain":
        root = OFFICIAL
    elif method == "scalable_tree_group":
        root = SCALABLE
    else:
        root = ABLATION
    return root / case_id / f"{method}-path0-seed0.json"


def load(case_id: str, method: str) -> dict:
    path = result_path(case_id, method)
    with path.open() as stream:
        result = json.load(stream)
    if result.get("strict_collision_free") is not True:
        raise RuntimeError(f"uncertified result: {path}")
    if result.get("valid_plan") is False:
        raise RuntimeError(f"invalid result: {path}")
    return result


def trace(result: dict) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray([point["time"] for point in result["trace"]], dtype=float)
    costs = np.asarray([point["cost"] for point in result["trace"]], dtype=float)
    return times, costs / float(result["initial_cost"])


def normalized_auc(result: dict) -> float:
    times, costs = trace(result)
    budget = float(result["budget_seconds"])
    grid = np.linspace(0.0, budget, 601)
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.ones_like(grid)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return float(np.trapz(values, grid) / budget)


def plot(results: dict, *, log_time: bool) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 10.0), squeeze=False)
    for axis, (case_id, title) in zip(axes.flat, CASES):
        first_positive = []
        for method, label, color in METHODS:
            times, costs = trace(results[(case_id, method)])
            axis.step(times, costs, where="post", linewidth=3.0, color=color, label=label)
            positive = times[times > 0]
            if len(positive):
                first_positive.append(float(positive[0]))
        if log_time:
            axis.set_xscale("log")
            axis.set_xlim(left=min(first_positive))
        axis.set_title(title, fontsize=18)
        axis.set_xlabel("time [s]", fontsize=16)
        axis.set_ylabel("cost / initial cost", fontsize=16)
        axis.tick_params(labelsize=13)
        axis.grid(alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
        fontsize=15,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.13,
        top=0.94,
        wspace=0.25,
        hspace=0.34,
    )
    suffix = "log" if log_time else "linear"
    destination = ABLATION / f"max-gain-order-ablation-{suffix}.png"
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
                "auc": normalized_auc(result),
                "final_normalized_cost": (
                    float(result["final_cost"]) / float(result["initial_cost"])
                ),
                "collision_checks": result["stats"]["collision_checks"],
            })
    ABLATION.mkdir(parents=True, exist_ok=True)
    with (ABLATION / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (ABLATION / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    destinations = [plot(results, log_time=False), plot(results, log_time=True)]
    print(json.dumps({
        "rows": rows,
        "plots": [str(path) for path in destinations],
    }, indent=2))


if __name__ == "__main__":
    main()
