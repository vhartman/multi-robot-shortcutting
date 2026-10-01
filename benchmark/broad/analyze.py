"""Summarize the broad one-path, three-seed offline first pass."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from broad_lib import HERE, cases, methods, protocol, protocol_sha256, read_json, result_path
from plot_style import method_color


def step_values(times, costs, grid):
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.ones_like(grid)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values


def aggregate(case_id, method_id):
    seconds = float(cases()[case_id]["seconds"])
    grid = np.linspace(0.0, seconds, 301)
    curves = []
    for seed in protocol()["shortcut_seeds"]:
        result = read_json(result_path(case_id, method_id, seed))
        trace = result["trace"]
        times = np.asarray([point["time"] for point in trace], dtype=float)
        costs = np.asarray([point["cost"] for point in trace], dtype=float)
        costs /= float(result["initial_cost"])
        curves.append(step_values(times, costs, grid))
    stacked = np.stack(curves)
    median = np.median(stacked, axis=0)
    return {
        "grid": grid,
        "median": median,
        "low": np.min(stacked, axis=0),
        "high": np.max(stacked, axis=0),
        "final_median": float(median[-1]),
        "final_low": float(np.min(stacked[:, -1])),
        "final_high": float(np.max(stacked[:, -1])),
        "normalized_auc": float(np.trapz(median, grid) / seconds),
    }


def main() -> int:
    output = HERE / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "mr-shortcut-broad-summary-v1",
        "broad_protocol_sha256": protocol_sha256(),
        "aggregation": protocol()["aggregation"],
        "cases": {},
    }
    rows = []
    aggregates = {}
    for case_id in cases():
        summary["cases"][case_id] = {}
        for method_id in methods():
            value = aggregate(case_id, method_id)
            aggregates[(case_id, method_id)] = value
            compact = {key: value[key] for key in (
                "final_median", "final_low", "final_high", "normalized_auc"
            )}
            summary["cases"][case_id][method_id] = compact
            rows.append({"case_id": case_id, "method_id": method_id, **compact})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    case_ids = list(cases())
    figure, axes = plt.subplots(4, 3, figsize=(16, 18), sharey=True)
    for axis, case_id in zip(axes.flat, case_ids):
        for method_id, method in methods().items():
            value = aggregates[(case_id, method_id)]
            axis.plot(
                value["grid"], value["median"],
                color=method_color(method_id), linewidth=3.0, label=method["label"],
            )
            axis.fill_between(
                value["grid"], value["low"], value["high"],
                color=method_color(method_id), alpha=0.14,
            )
        axis.set_title(case_id.removeprefix("rai-").replace("-", " "), fontsize=15)
        axis.set_xlabel("time [s]", fontsize=13)
        axis.set_ylabel("normalized cost", fontsize=13)
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.22)
    for axis in axes.flat[len(case_ids):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=14, frameon=False)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    plot = output / "broad-offline-first-pass.png"
    figure.savefig(plot, dpi=180)
    plt.close(figure)
    print(json.dumps({"summary": str(output / "summary.json"), "plot": str(plot)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

