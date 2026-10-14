"""Plot certified disk-scaling curves, summary metrics, and scene layouts."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
os.environ["MR_SHORTCUT_PROTOCOL"] = str(HERE / "protocol.json")
os.environ["MR_SHORTCUT_OUTPUT_ROOT"] = str(HERE)
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "expanded"))

from expanded_lib import cases, input_path, methods, protocol, read_json
from plot_style import method_color
from run import artifact_errors, offline_cells


def trace_curve(result, grid):
    times = np.array([point["time"] for point in result["trace"]])
    costs = np.array([point["cost"] for point in result["trace"]]) / result["initial_cost"]
    indices = np.searchsorted(times, grid, side="right") - 1
    curve = np.ones_like(grid)
    available = indices >= 0
    curve[available] = costs[indices[available]]
    return curve


def exact_auc(result, duration):
    points = {0.0: 1.0}
    points.update({
        float(point["time"]): float(point["cost"]) / result["initial_cost"]
        for point in result["trace"] if 0 <= point["time"] <= duration
    })
    times = np.array(sorted(points))
    widths = np.diff(np.append(times, duration))
    return float(np.dot(widths, [points[value] for value in times]) / duration)


def summarize(values):
    center = np.median(values, axis=0)
    return center, np.min(values, axis=0), np.max(values, axis=0)


def legend(figure, axis):
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=13)


def main():
    output = HERE / "analysis"
    output.mkdir(exist_ok=True)
    case_map = cases()
    results = {}
    unavailable = []
    for cell in offline_cells(list(case_map.values())):
        errors = artifact_errors(cell)
        if errors:
            unavailable.append({"case": cell.case_id, "method": cell.method_id,
                                "path": cell.path_index, "seed": cell.shortcut_seed,
                                "errors": errors})
        else:
            results[(cell.case_id, cell.method_id, cell.path_index, cell.shortcut_seed)] = read_json(cell.output)

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    rows = []
    for axis, case in zip(axes.flat, case_map.values()):
        grid = np.linspace(0, case["seconds"], 601)
        count = case["environment_kwargs"]["num_robots"]
        for method_id, method in methods().items():
            path_curves, path_finals, path_aucs = [], [], []
            seed_curves, seed_finals, seed_aucs = [], [], []
            run_count = 0
            for path_index in case["path_indices"]:
                selected = [
                    results[(case["id"], method_id, path_index, seed)]
                    for seed in protocol()["shortcut_seeds"]
                    if (case["id"], method_id, path_index, seed) in results
                ]
                if not selected:
                    continue
                curves = [trace_curve(result, grid) for result in selected]
                finals = [curve[-1] for curve in curves]
                aucs = [exact_auc(result, case["seconds"]) for result in selected]
                seed_curves.extend(curves)
                seed_finals.extend(finals)
                seed_aucs.extend(aucs)
                path_curves.append(np.median(curves, axis=0))
                path_finals.append(float(np.median(finals)))
                path_aucs.append(float(np.median(aucs)))
                run_count += len(selected)
            if not path_curves:
                continue
            center, _, _ = summarize(path_curves)
            _, low, high = summarize(path_curves if len(path_curves) > 1 else seed_curves)
            axis.step(grid, center, where="post", color=method_color(method_id), linewidth=3, label=method["label"])
            axis.fill_between(grid, low, high, step="post", color=method_color(method_id), alpha=0.12)
            row = {"robots": count, "dof": 2 * count, "method": method_id,
                   "paths_used": len(path_curves), "runs_used": run_count,
                   "runs_expected": len(case["path_indices"]) * len(protocol()["shortcut_seeds"])}
            for metric, path_values, seed_values in [
                ("final_cost", path_finals, seed_finals),
                ("auc", path_aucs, seed_aucs),
            ]:
                row[metric] = float(np.median(path_values))
                limits = path_values if len(path_values) > 1 else seed_values
                row[metric + "_low"] = float(min(limits))
                row[metric + "_high"] = float(max(limits))
            rows.append(row)
        axis.set_title(f"{count} robot{'s' if count != 1 else ''} · {2 * count} DoF", fontsize=19)
        axis.set_xlabel("time [s]", fontsize=16)
        axis.set_ylabel("normalized cost", fontsize=16)
        axis.tick_params(labelsize=13)
        axis.grid(alpha=0.2)
    legend(figure, next((axis for axis in axes.flat if axis.lines), axes.flat[0]))
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output / "disk-scaling-curves.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for axis, metric, title in zip(axes, ["final_cost", "auc"], ["Final normalized cost", "Normalized area under cost curve"]):
        for method_id, method in methods().items():
            selected = [row for row in rows if row["method"] == method_id]
            if not selected:
                continue
            centers = np.array([row[metric] for row in selected])
            lows = np.array([row[metric + "_low"] for row in selected])
            highs = np.array([row[metric + "_high"] for row in selected])
            axis.errorbar([row["robots"] for row in selected], centers,
                          yerr=[centers - lows, highs - centers], marker="o", capsize=3,
                          color=method_color(method_id), linewidth=3, label=method["label"])
        axis.set_title(title, fontsize=18)
        axis.set_xlabel("number of robots", fontsize=16)
        axis.set_xticks([case["environment_kwargs"]["num_robots"] for case in case_map.values()])
        axis.tick_params(labelsize=13)
        axis.grid(alpha=0.2)
    legend(figure, axes[0])
    figure.tight_layout(rect=(0, 0.22, 1, 1))
    figure.savefig(output / "disk-scaling-performance.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(14, 10))
    layout_file = output / "scene-layouts.json"
    layouts = read_json(layout_file) if layout_file.is_file() else {}
    for axis, case in zip(axes.flat, case_map.values()):
        source = input_path(case["id"], 0)
        scene = read_json(source)["scene"] if source.is_file() else layouts.get(case["id"])
        if scene:
            for obstacle in scene["obstacles"]:
                center, size = obstacle["position"], obstacle["size"]
                axis.add_patch(Rectangle((center[0] - size[0] / 2, center[1] - size[1] / 2), size[0], size[1], color="#444444"))
            for index, (start, goals) in enumerate(zip(scene["starts"], scene["goals"])):
                color = plt.get_cmap("tab20")(index)
                axis.add_patch(Circle(start, scene["robot_shape_size"][-1], color=color, alpha=0.8))
                axis.scatter(np.array(goals)[:, 0], np.array(goals)[:, 1], color=color, marker="x", s=35)
                axis.text(start[0], start[1], str(index + 1), ha="center", va="center", fontsize=9)
        axis.set_xlim(-2, 2)
        axis.set_ylim(-2, 2)
        axis.set_aspect("equal")
        count = case["environment_kwargs"]["num_robots"]
        axis.set_title(f"{count} robot{'s' if count != 1 else ''}", fontsize=17)
    figure.tight_layout()
    figure.savefig(output / "disk-scaling-scenes.png", dpi=180)
    plt.close(figure)
    if rows:
        with (output / "scenario-metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"valid_runs": len(results), "unavailable": unavailable,
               "aggregation": protocol()["aggregation"],
               "auc": "exact time-normalized integral of each accepted-cost step trace; nested scalar medians",
               "metrics": rows}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"valid_runs": len(results), "unavailable": len(unavailable),
                      "plots": str(output)}, indent=2))


if __name__ == "__main__":
    main()
