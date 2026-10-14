"""Analyze the final scalable tree/group study against frozen baselines."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from expanded_lib import cases, methods_for, protocol, read_json


HERE = Path(__file__).resolve().parent
OFFICIAL = HERE / "offline" / "results"
NEW = HERE / "analysis" / "scalable-tree-group-final"
OUTPUT = HERE / "analysis"
IMPLEMENTATION = HERE.parent / "scaling_2d" / "scalable_tree_group.py"
NEW_METHOD = "scalable_tree_group"
LABELS = {
    "single_robot_mode": "random robot + compatible run",
    "tree_gain": "tree + gain",
    "tree_group_gain": "tree + group gain",
    "full_tree_gain_round_robin": "full/tree/gain round robin",
    "full_configuration": "full configuration",
    "robot_subset_mode": "random subset + compatible run",
    NEW_METHOD: "scalable tree/group",
}
COLORS = {
    "single_robot_mode": "#4477AA",
    "tree_gain": "#EE6677",
    "tree_group_gain": "#228833",
    "full_tree_gain_round_robin": "#CCBB44",
    "full_configuration": "#66CCEE",
    "robot_subset_mode": "#AA3377",
    NEW_METHOD: "#332288",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_path(case_id: str, method: str, path_index: int, seed: int) -> Path:
    root = NEW if method == NEW_METHOD else OFFICIAL
    return root / case_id / f"{method}-path{path_index}-seed{seed}.json"


def curve(result: dict, seconds: float, grid: np.ndarray) -> np.ndarray:
    times = np.asarray([point["time"] for point in result["trace"]], dtype=float)
    costs = np.asarray([point["cost"] for point in result["trace"]], dtype=float)
    costs /= float(result["initial_cost"])
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.ones_like(grid)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values


def aggregate(case: dict, method: str) -> dict:
    seconds = float(case["seconds"])
    grid = np.linspace(0.0, seconds, 301)
    path_curves = []
    expected_hash = sha256(IMPLEMENTATION)
    for path_index in case["path_indices"]:
        seed_curves = []
        for seed in protocol()["shortcut_seeds"]:
            source = result_path(case["id"], method, path_index, seed)
            if not source.is_file():
                if method == NEW_METHOD:
                    raise RuntimeError(f"missing final result: {source}")
                continue
            result = read_json(source)
            if method == NEW_METHOD:
                if result.get("trial_implementation_sha256") != expected_hash:
                    raise RuntimeError(f"stale implementation hash: {source}")
                for field in (
                    "strict_collision_free",
                    "endpoint_start_preserved",
                    "endpoint_end_preserved",
                    "terminal_mode_preserved",
                ):
                    if result.get(field) is not True:
                        raise RuntimeError(f"failed {field}: {source}")
                if result.get("valid_plan") is False:
                    raise RuntimeError(f"failed valid_plan: {source}")
                if result.get("input_mode_runs") != result.get("output_mode_runs"):
                    raise RuntimeError(f"failed mode-run preservation: {source}")
            seed_curves.append(curve(result, seconds, grid))
        if not seed_curves:
            raise RuntimeError(
                f"no results for {case['id']} {method} path{path_index}"
            )
        path_curves.append(np.median(np.stack(seed_curves), axis=0))
    paths = np.stack(path_curves)
    median = np.median(paths, axis=0)
    return {
        "grid": grid,
        "curve": median,
        "low": np.min(paths, axis=0),
        "high": np.max(paths, axis=0),
        "auc": float(np.trapz(median, grid) / seconds),
        "final": float(median[-1]),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def broad_analysis(broad: list[dict]) -> tuple[list[dict], list[dict], Path]:
    method_ids = list(methods_for(broad[0])) + [NEW_METHOD]
    rows = []
    values = {method: {"auc": [], "final": []} for method in method_ids}
    ranks = {method: [] for method in method_ids}
    wins = {method: 0 for method in method_ids}
    for case in broad:
        summaries = {method: aggregate(case, method) for method in method_ids}
        for method, summary in summaries.items():
            rows.append({
                "case_id": case["id"],
                "method": method,
                "auc": summary["auc"],
                "final": summary["final"],
            })
            values[method]["auc"].append(summary["auc"])
            values[method]["final"].append(summary["final"])
        ordered = sorted(method_ids, key=lambda method: summaries[method]["auc"])
        for rank, method in enumerate(ordered, start=1):
            ranks[method].append(rank)
        wins[ordered[0]] += 1
    summary_rows = [{
        "method": method,
        "median_auc": float(np.median(values[method]["auc"])),
        "median_final": float(np.median(values[method]["final"])),
        "mean_auc_rank": float(np.mean(ranks[method])),
        "auc_wins": wins[method],
    } for method in method_ids]

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for axis, metric, title in (
        (axes[0], "auc", "Normalized area under curve"),
        (axes[1], "final", "Final normalized cost"),
    ):
        data = [values[method][metric] for method in method_ids]
        artists = axis.boxplot(data, patch_artist=True, showfliers=False)
        for patch, method in zip(artists["boxes"], method_ids):
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.7)
        axis.set_xticks(range(1, len(method_ids) + 1))
        axis.set_xticklabels([LABELS[m] for m in method_ids], rotation=22, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    destination = OUTPUT / "scalable-tree-group-broad-summary.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return rows, summary_rows, destination


def scaling_analysis(scaling: list[dict]) -> tuple[list[dict], Path]:
    method_ids = list(methods_for(scaling[0])) + [NEW_METHOD]
    rows = []
    summaries = {}
    for case in scaling:
        robots = int(case["environment_kwargs"]["num_robots"])
        for method in method_ids:
            summary = aggregate(case, method)
            summaries[(robots, method)] = summary
            rows.append({
                "robots": robots,
                "method": method,
                "auc": summary["auc"],
                "final": summary["final"],
            })
    robot_counts = sorted({row["robots"] for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for axis, metric, title in (
        (axes[0], "auc", "Normalized area under curve"),
        (axes[1], "final", "Final normalized cost"),
    ):
        for method in method_ids:
            axis.plot(
                robot_counts,
                [summaries[(robots, method)][metric] for robots in robot_counts],
                marker="o",
                linewidth=2.6,
                color=COLORS[method],
                label=LABELS[method],
            )
        axis.set_title(title)
        axis.set_xlabel("number of robots")
        axis.set_xticks(robot_counts)
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.18, 1, 1))
    destination = OUTPUT / "scalable-tree-group-isolated-box-scaling.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return rows, destination


def main() -> None:
    selected = list(cases().values())
    broad = [case for case in selected if case["track"] == "broad"]
    scaling = [case for case in selected if case["track"] == "scaling"]
    broad_rows, summary_rows, broad_plot = broad_analysis(broad)
    scaling_rows, scaling_plot = scaling_analysis(scaling)
    write_csv(OUTPUT / "scalable-tree-group-broad.csv", broad_rows)
    write_csv(OUTPUT / "scalable-tree-group-broad-summary.csv", summary_rows)
    write_csv(OUTPUT / "scalable-tree-group-isolated-box-scaling.csv", scaling_rows)
    print(json.dumps({
        "validated_new_cells": len(broad) * 3 + len(scaling) * 9,
        "implementation_sha256": sha256(IMPLEMENTATION),
        "broad_summary": summary_rows,
        "plots": [str(broad_plot), str(scaling_plot)],
    }, indent=2))


if __name__ == "__main__":
    main()
