"""Quantify how much of the online refinement window reaches shortcutting."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BENCHMARK))

import analyze as base  # noqa: E402
from plot_style import method_color  # noqa: E402

OUTPUT = HERE / "analysis"


def accepted(stats):
    value = stats.get("accepted", 0)
    return sum(value.values()) if isinstance(value, dict) else int(value or 0)


def call_checks(call):
    stats = call.get("stats", {})
    if "collision_checks" in stats:
        return int(stats["collision_checks"])
    trace = call.get("trace", [])
    return int(trace[-1]["collision_checks"]) if trace else 0


def online_row(cell, value):
    calls = value["shortcut_calls"]
    first = float(value["first_solution"]["cost"])
    horizon = float(value["refinement_seconds"])
    return {
        "track": "online",
        "family": cell["family"],
        "case_id": cell["case_id"],
        "robots": base.robot_count(cell["case_id"]),
        "method_id": cell["method_id"],
        "shortcut_seed": cell["shortcut_seed"],
        "shortcut_calls": len(calls),
        "collision_checks": sum(call_checks(call) for call in calls),
        "accepted": sum(accepted(call.get("stats", {})) for call in calls),
        "shortcut_wall_seconds": float(value["shortcut_wall_seconds"]),
        "refinement_seconds": horizon,
        "shortcut_time_fraction": float(value["shortcut_wall_seconds"]) / horizon,
        "tracked_incumbents": int(value["incumbent_certification"]["tracked"]),
        "normalized_final_cost": float(value["comparison_final_cost"]) / first,
        "relative_improvement": 1 - float(value["comparison_final_cost"]) / first,
        "deadline_overrun_seconds": float(value["deadline_overrun_seconds"]),
    }


def offline_checks(cell, value):
    return {
        "family": cell["family"], "case_id": cell["case_id"],
        "robots": base.robot_count(cell["case_id"]), "method_id": cell["method_id"],
        "shortcut_seed": cell["shortcut_seed"],
        "collision_checks": int(value["trace"][-1]["collision_checks"]),
    }


def median(rows, key):
    return float(np.median([float(row[key]) for row in rows]))


def main():
    online = [online_row(cell, value) for cell, _, value in base.online_results()]
    offline = [offline_checks(cell, value) for cell, _, value in base.offline_results()]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "online-budget-diagnostics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(online[0]))
        writer.writeheader()
        writer.writerows(online)

    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    for column, family in enumerate(base.FAMILIES):
        cases = sorted({row["case_id"] for row in online if row["family"] == family},
                       key=base.robot_count)
        xs = np.asarray([base.robot_count(case) for case in cases])
        for method in base.METHODS:
            method_rows = [row for row in online
                           if row["family"] == family and row["method_id"] == method]
            calls, shares = [], []
            for case in cases:
                selected = [row for row in method_rows if row["case_id"] == case]
                calls.append(median(selected, "shortcut_calls"))
                shares.append(100 * median(selected, "shortcut_time_fraction"))
            axes[0, column].plot(xs, calls, marker="o", linewidth=2.2,
                                 color=method_color(method), label=base.LABELS[method])
            axes[1, column].plot(xs, shares, marker="o", linewidth=2.2,
                                 color=method_color(method))

        online_checks, offline_checks_values = [], []
        for case in cases:
            online_checks.append(median([row for row in online if row["case_id"] == case],
                                        "collision_checks"))
            offline_checks_values.append(median([row for row in offline if row["case_id"] == case],
                                                "collision_checks"))
        axes[2, column].plot(xs, online_checks, color="#222222", marker="o",
                             linewidth=2.7, label="online")
        axes[2, column].plot(xs, offline_checks_values, color="#777777", marker="o",
                             linewidth=2.7, linestyle="--", label="offline")

        for row in range(3):
            axis = axes[row, column]
            axis.set_title(base.FAMILY_LABELS[family], fontsize=15)
            axis.set_xlabel("robots", fontsize=13)
            axis.set_xticks(xs)
            axis.tick_params(labelsize=11)
            axis.grid(alpha=0.22)
            axis.set_yscale("log")
        if column == 0:
            axes[0, column].set_ylabel("shortcut calls", fontsize=13)
            axes[1, column].set_ylabel("shortcut time [%]", fontsize=13)
            axes[2, column].set_ylabel("collision checks", fontsize=13)

    method_handles, method_labels = axes[0, 0].get_legend_handles_labels()
    track_handles, track_labels = axes[2, 0].get_legend_handles_labels()
    first_legend = fig.legend(method_handles, method_labels, loc="upper center",
                              bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False, fontsize=12)
    fig.add_artist(first_legend)
    fig.legend(track_handles, track_labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.915), ncol=2, frameon=False, fontsize=12)
    fig.suptitle("What reaches shortcutting in the online planner?", fontsize=19, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    figure_path = OUTPUT / "online-budget-diagnostics.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {}
    for family in base.FAMILIES:
        summary[family] = {}
        cases = sorted({row["case_id"] for row in online if row["family"] == family},
                       key=base.robot_count)
        for case in cases:
            selected = [row for row in online if row["case_id"] == case]
            matching_offline = [row for row in offline if row["case_id"] == case]
            online_checks_value = median(selected, "collision_checks")
            summary[family][str(base.robot_count(case))] = {
                "median_shortcut_calls": median(selected, "shortcut_calls"),
                "median_online_collision_checks": online_checks_value,
                "median_shortcut_wall_seconds": median(selected, "shortcut_wall_seconds"),
                "median_shortcut_time_percent": 100 * median(selected, "shortcut_time_fraction"),
                "median_relative_improvement_percent": 100 * median(selected, "relative_improvement"),
                "median_offline_collision_checks": median(matching_offline, "collision_checks"),
                "offline_to_online_check_ratio": median(matching_offline, "collision_checks") / max(1, online_checks_value),
            }
    summary_path = OUTPUT / "online-budget-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"figure": str(figure_path), "csv": str(csv_path),
                      "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
