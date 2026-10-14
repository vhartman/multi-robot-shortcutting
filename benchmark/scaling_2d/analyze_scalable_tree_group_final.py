"""Aggregate the 3-path x 3-seed scalable-tree/group scaling study."""
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
RANDOM = HERE / "analysis" / "group-gain-trials-v2"
NEW = HERE / "analysis" / "scalable-tree-group-final"
OUTPUT = HERE / "analysis"
IMPLEMENTATION = HERE / "scalable_tree_group.py"
ROBOTS = (1, 2, 4, 6, 8, 12)
METHODS = (
    "tree_gain",
    "full_tree_gain_round_robin",
    "tree_random_group_mode",
    "scalable_tree_group",
)
LABELS = {
    "tree_gain": "tree → gain",
    "full_tree_gain_round_robin": "full/tree/gain round robin",
    "tree_random_group_mode": "tree → random compatible group",
    "scalable_tree_group": "scalable tree/group",
}
COLORS = {
    "tree_gain": "#e66101",
    "full_tree_gain_round_robin": "#008695",
    "tree_random_group_mode": "#7b3294",
    "scalable_tree_group": "#332288",
}


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def result_path(robots: int, method: str, path_index: int, seed: int) -> Path:
    case_id = f"rai-disk-scaling-r{robots}-g4"
    filename = f"{method}-path{path_index}-seed{seed}.json"
    if method == "tree_random_group_mode":
        root = RANDOM
    elif method == "scalable_tree_group":
        root = NEW
    else:
        root = OFFICIAL
    return root / case_id / filename


def step(result: dict, grid: np.ndarray) -> np.ndarray:
    times = np.asarray([point["time"] for point in result["trace"]], dtype=float)
    costs = np.asarray([point["cost"] for point in result["trace"]], dtype=float)
    costs /= float(result["initial_cost"])
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.ones_like(grid)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values


def aggregate(robots: int, method: str, grid: np.ndarray) -> dict:
    expected_hash = sha256(IMPLEMENTATION)
    path_curves = []
    path_aucs = []
    for path_index in range(3):
        curves = []
        aucs = []
        for seed in range(3):
            source = result_path(robots, method, path_index, seed)
            result = read(source)
            if method == "scalable_tree_group":
                if result.get("trial_implementation_sha256") != expected_hash:
                    raise RuntimeError(f"stale implementation hash: {source}")
                for field in (
                    "strict_collision_free",
                    "valid_plan",
                    "mode_runs_preserved",
                ):
                    if result.get(field) is not True:
                        raise RuntimeError(f"failed {field}: {source}")
            curve = step(result, grid)
            curves.append(curve)
            aucs.append(float(np.trapz(curve, grid) / grid[-1]))
        path_curves.append(np.median(np.stack(curves), axis=0))
        path_aucs.append(float(np.median(aucs)))
    stacked = np.stack(path_curves)
    median = np.median(stacked, axis=0)
    return {
        "curve": median,
        "low": np.min(stacked, axis=0),
        "high": np.max(stacked, axis=0),
        "auc": float(np.median(path_aucs)),
        "auc_low": float(min(path_aucs)),
        "auc_high": float(max(path_aucs)),
        "final": float(median[-1]),
        "final_low": float(np.min(stacked[:, -1])),
        "final_high": float(np.max(stacked[:, -1])),
    }


def main() -> None:
    grid = np.linspace(0.0, 15.0, 601)
    summaries = {}
    rows = []
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    for axis, robots in zip(axes.flat, ROBOTS):
        summaries[robots] = {}
        for method in METHODS:
            summary = aggregate(robots, method, grid)
            summaries[robots][method] = summary
            rows.append({
                "robots": robots,
                "dof": 2 * robots,
                "method": method,
                "auc": summary["auc"],
                "auc_low": summary["auc_low"],
                "auc_high": summary["auc_high"],
                "final": summary["final"],
                "final_low": summary["final_low"],
                "final_high": summary["final_high"],
            })
            axis.step(
                grid,
                summary["curve"],
                where="post",
                linewidth=2.8,
                color=COLORS[method],
                label=LABELS[method],
            )
            axis.fill_between(
                grid,
                summary["low"],
                summary["high"],
                step="post",
                color=COLORS[method],
                alpha=0.09,
            )
        axis.set_title(f"{robots} robots · {2 * robots} DoF", fontsize=16)
        axis.set_xlabel("time [s]", fontsize=13)
        axis.set_ylabel("cost / initial cost", fontsize=13)
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    curves_path = OUTPUT / "scalable-tree-group-final-curves.png"
    figure.savefig(curves_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    for axis, metric, title in (
        (axes[0], "auc", "Normalized area under curve"),
        (axes[1], "final", "Final normalized cost"),
    ):
        for method in METHODS:
            center = np.asarray([summaries[n][method][metric] for n in ROBOTS])
            low = np.asarray([summaries[n][method][f"{metric}_low"] for n in ROBOTS])
            high = np.asarray([summaries[n][method][f"{metric}_high"] for n in ROBOTS])
            axis.errorbar(
                ROBOTS,
                center,
                yerr=[center - low, high - center],
                marker="o",
                linewidth=2.8,
                capsize=3,
                color=COLORS[method],
                label=LABELS[method],
            )
        axis.set_title(title, fontsize=16)
        axis.set_xlabel("number of robots", fontsize=13)
        axis.set_xticks(ROBOTS)
        axis.tick_params(labelsize=11)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    summary_path = OUTPUT / "scalable-tree-group-final-scaling.png"
    figure.savefig(summary_path, dpi=180)
    plt.close(figure)

    csv_path = OUTPUT / "scalable-tree-group-final.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "rows": len(rows),
        "csv": str(csv_path),
        "plots": [str(curves_path), str(summary_path)],
        "implementation_sha256": sha256(IMPLEMENTATION),
    }, indent=2))


if __name__ == "__main__":
    main()
