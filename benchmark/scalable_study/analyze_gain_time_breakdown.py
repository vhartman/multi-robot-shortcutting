"""Summarize timing already recorded by the 3x3 scaling experiments.

The historical result schema times collision checks in aggregate but does not
time proposal construction/scoring separately.  Consequently ``residual`` is
wall time minus measured collision checking; it must not be presented as a
more precise profiler category.
"""
from __future__ import annotations

import csv
import gc
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "offline" / "results"
OUTPUT = HERE / "analysis"
REPEAT_SUMMARY = HERE / "offline" / "group-gain-timing-repeat" / "summary.json"
METHODS = ("tree_gain", "tree_group_gain")
METHOD_LABELS = {
    "tree_gain": "tree + gain",
    "tree_group_gain": "tree + group gain",
}
FAMILIES = {
    "disk": ("rai-disk-scaling-r*-g4", "Planar disks", 15.0),
    "isolated": (
        "rai-isolated-box-stacking-r*-b4",
        "Isolated box stacking",
        30.0,
    ),
}


def robot_count(path: Path) -> int:
    return int(path.name.split("-r")[-1].split("-")[0])


def median(values):
    return float(statistics.median(values))


def load_rows():
    rows = []
    for family, (pattern, _, budget) in FAMILIES.items():
        directories = sorted(RESULTS.glob(pattern), key=robot_count)
        for directory in directories:
            robots = robot_count(directory)
            for method in METHODS:
                samples = []
                for path in sorted(directory.glob(f"{method}-path*.json")):
                    with path.open() as stream:
                        record = json.load(stream)
                    stats = record["stats"]
                    valid = float(stats["collision_check_seconds_valid"])
                    invalid = float(stats["collision_check_seconds_invalid"])
                    wall = float(record["wall_seconds"])
                    samples.append(
                        {
                            "wall": wall,
                            "valid": valid,
                            "invalid": invalid,
                            "residual": max(0.0, wall - valid - invalid),
                            "draws": int(stats["draws"]),
                            "collision_checks": int(stats["collision_checks"]),
                            "tree_proposals": int(stats["tree_proposals"]),
                            "gain_proposals": int(stats["gain_proposals"]),
                            "group_gain_arms": int(stats["group_gain_arms"]),
                            "group_gain_proposals": int(stats["group_gain_proposals"]),
                        }
                    )
                    del record
                    gc.collect()
                if not samples:
                    continue
                row = {
                    "family": family,
                    "robots": robots,
                    "method_id": method,
                    "replicates": len(samples),
                    "nominal_budget_seconds": budget,
                }
                for key in samples[0]:
                    values = [sample[key] for sample in samples]
                    row[f"{key}_median"] = median(values)
                    row[f"{key}_min"] = min(values)
                    row[f"{key}_max"] = max(values)
                row["collision_fraction_median"] = median(
                    (sample["valid"] + sample["invalid"]) / sample["wall"]
                    for sample in samples
                )
                row["residual_fraction_median"] = median(
                    sample["residual"] / sample["wall"] for sample in samples
                )
                rows.append(row)
    return rows


def write_csv(rows):
    destination = OUTPUT / "gain-time-breakdown.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def plot(rows):
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), squeeze=False)
    colors = ("#4c78a8", "#b9b9b9")
    component_labels = ("collision checking", "everything else")
    for row_index, (family, (_, family_label, budget)) in enumerate(FAMILIES.items()):
        for column, method in enumerate(METHODS):
            axis = axes[row_index, column]
            selected = sorted(
                [row for row in rows if row["family"] == family and row["method_id"] == method],
                key=lambda row: row["robots"],
            )
            x = np.arange(len(selected))
            bottom = np.zeros(len(selected))
            components = (
                np.asarray(
                    [row["valid_median"] + row["invalid_median"] for row in selected]
                ),
                np.asarray([row["residual_median"] for row in selected]),
            )
            for values, label, color in zip(components, component_labels, colors):
                axis.bar(x, values, bottom=bottom, color=color, label=label)
                bottom += values
            totals = np.asarray([row["wall_median"] for row in selected])
            low = totals - np.asarray([row["wall_min"] for row in selected])
            high = np.asarray([row["wall_max"] for row in selected]) - totals
            axis.errorbar(x, totals, yerr=np.vstack((low, high)), fmt="none", ecolor="#222", capsize=3)
            axis.axhline(budget, color="#222", linestyle="--", linewidth=1.4, label="nominal budget")
            axis.set_xticks(x, [int(row["robots"]) for row in selected])
            axis.set_xlabel("number of robots", fontsize=13)
            axis.set_ylabel("wall time [s]", fontsize=13)
            axis.set_title(f"{family_label}: {METHOD_LABELS[method]}", fontsize=15)
            axis.tick_params(labelsize=11)
            axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=12)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    destination = OUTPUT / "gain-time-breakdown.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_group_gain_repeat():
    if not REPEAT_SUMMARY.is_file():
        return None
    records = json.loads(REPEAT_SUMMARY.read_text())
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 4.9), squeeze=False)
    families = (
        ("disk_scaling", "Planar disks", 15.0),
        ("isolated_box_stacking", "Isolated box stacking", 30.0),
    )
    for axis, (family, title, budget) in zip(axes.flat, families):
        selected = sorted(
            [record for record in records if record["family"] == family],
            key=lambda record: record["robots"],
        )
        x = np.arange(len(selected))
        collision = np.asarray([record["collision_check_seconds"] for record in selected])
        other = np.asarray([record["everything_else_seconds"] for record in selected])
        axis.bar(x, collision, color="#4c78a8", label="collision checking")
        axis.bar(x, other, bottom=collision, color="#b9b9b9", label="everything else")
        axis.axhline(budget, color="#222", linestyle="--", linewidth=1.4, label="nominal budget")
        axis.set_xticks(x, [record["robots"] for record in selected])
        axis.set_xlabel("number of robots", fontsize=13)
        axis.set_ylabel("wall time [s]", fontsize=13)
        axis.set_title(f"{title}: tree + group gain", fontsize=15)
        axis.tick_params(labelsize=11)
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=12)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    destination = OUTPUT / "group-gain-time-breakdown-repeat.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    csv_path = OUTPUT / "group-gain-time-breakdown-repeat.csv"
    fields = list(records[0])
    fields.extend(
        key for record in records for key in record if key not in fields
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return destination


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    csv_path = write_csv(rows)
    plot_path = plot(rows)
    repeat_path = plot_group_gain_repeat()
    print(json.dumps({
        "rows": len(rows), "csv": str(csv_path), "plot": str(plot_path),
        "group_gain_repeat_plot": str(repeat_path) if repeat_path else None,
    }, indent=2))


if __name__ == "__main__":
    main()
