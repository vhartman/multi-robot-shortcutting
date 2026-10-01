"""Aggregate the final matrix path-first and generate the four blog panels."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
BENCHMARK_ROOT = HERE.parent
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from protocol_lib import (  # noqa: E402
    methods,
    offline_result_path,
    online_result_path,
    protocol,
    protocol_cases,
    protocol_sha256,
    read_json,
    write_json,
)
from plot_style import method_color, method_linestyle  # noqa: E402

TITLES = {
    "rai-random-2d": "Random 2D",
    "rai-box-rearrangement": "Box rearrangement",
    "rai-spiral-tower": "Spiral tower",
    "assembly-car-mini-4r-openarm": "Car assembly",
}


def result_path(
    phase: str, case_id: str, method_id: str, path_index: int,
    shortcut_seed: int | None,
) -> Path:
    if phase == "offline":
        assert shortcut_seed is not None
        return offline_result_path(case_id, method_id, path_index, shortcut_seed)
    return online_result_path(case_id, method_id, path_index, shortcut_seed)


def trace(phase: str, result: dict) -> tuple[np.ndarray, np.ndarray]:
    points = result["trace"] if phase == "offline" else result["full_curve"]
    if not points:
        raise ValueError("empty trace")
    times = np.asarray([point["time"] for point in points], dtype=float)
    if phase == "online":
        times = times - times[0]
        initial = float(points[0]["cost"])
    else:
        initial = float(result["initial_cost"])
    costs = np.asarray([point["cost"] for point in points], dtype=float) / initial
    return times, costs


def step_values(
    times: np.ndarray, costs: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.ones_like(grid)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values


def horizon(phase: str, case_id: str) -> float:
    if phase == "offline":
        return float(protocol_cases()[case_id]["offline_seconds"])
    return float(protocol()["online"]["analysis_horizon_seconds"])


def method_ids(phase: str) -> tuple[str, ...]:
    selected = tuple(methods())
    return selected + (("planner_only",) if phase == "online" else ())


def load_path_curves(
    phase: str,
    case_id: str,
    method_id: str,
    grid: np.ndarray,
    allow_incomplete: bool,
) -> tuple[list[np.ndarray], int]:
    data = protocol()["design"]
    path_curves = []
    run_count = 0
    for path_index in data["path_indices"]:
        seeds = [None] if method_id == "planner_only" else data["shortcut_seeds"]
        seed_curves = []
        for seed in seeds:
            path = result_path(phase, case_id, method_id, path_index, seed)
            if not path.exists():
                if allow_incomplete:
                    continue
                raise FileNotFoundError(path)
            times, costs = trace(phase, read_json(path))
            seed_curves.append(step_values(times, costs, grid))
            run_count += 1
        if seed_curves:
            path_curves.append(np.median(np.stack(seed_curves), axis=0))
        elif not allow_incomplete:
            raise RuntimeError(
                f"no runs for {phase}/{case_id}/{method_id}/path{path_index}"
            )
    return path_curves, run_count

def first_positive_time(
    phase: str, case_id: str, allow_incomplete: bool
) -> float:
    positives = []
    data = protocol()["design"]
    for method_id in method_ids(phase):
        seeds = [None] if method_id == "planner_only" else data["shortcut_seeds"]
        for path_index in data["path_indices"]:
            for seed in seeds:
                path = result_path(phase, case_id, method_id, path_index, seed)
                if not path.exists():
                    if allow_incomplete:
                        continue
                    raise FileNotFoundError(path)
                times, _ = trace(phase, read_json(path))
                positives.extend(float(value) for value in times if value > 0.0)
    if not positives:
        raise RuntimeError(f"no positive timestamps for {phase}/{case_id}")
    return min(positives)


def aggregate(
    phase: str,
    case_id: str,
    method_id: str,
    grid: np.ndarray,
    allow_incomplete: bool,
) -> dict | None:
    curves, run_count = load_path_curves(
        phase, case_id, method_id, grid, allow_incomplete
    )
    if not curves:
        return None
    stacked = np.stack(curves)
    median = np.median(stacked, axis=0)
    return {
        "median": median,
        "low": np.min(stacked, axis=0),
        "high": np.max(stacked, axis=0),
        "path_count": len(curves),
        "run_count": run_count,
        "final_median": float(median[-1]),
        "final_low": float(np.min(stacked[:, -1])),
        "final_high": float(np.max(stacked[:, -1])),
        "normalized_auc": float(np.trapz(median, grid) / grid[-1]),
    }

def plot_blog(
    phase: str, log_x: bool, allow_incomplete: bool
) -> Path:
    cases = protocol()["blog_cases"]
    figure, axes = plt.subplots(
        1, len(cases), figsize=(19.0, 8.6), sharex=False, sharey=True
    )
    handles = {}
    labels = {key: value["label"] for key, value in methods().items()}
    labels["planner_only"] = "planner only (no shortcutting)"
    for axis, case_id in zip(axes, cases):
        limit = horizon(phase, case_id)
        first = first_positive_time(phase, case_id, allow_incomplete) if log_x else 0.0
        grid = (
            np.geomspace(first, limit, 900)
            if log_x else np.linspace(0.0, limit, 900)
        )
        for method_id in method_ids(phase):
            stats = aggregate(
                phase, case_id, method_id, grid, allow_incomplete
            )
            if stats is None:
                continue
            line = axis.plot(
                grid,
                stats["median"],
                color=method_color(method_id),
                linestyle=method_linestyle(method_id),
                linewidth=4.0 if method_id == "planner_only" else 3.6,
            )[0]
            axis.fill_between(
                grid, stats["low"], stats["high"],
                color=method_color(method_id), alpha=0.07, linewidth=0,
            )
            handles[method_id] = line
        if log_x:
            axis.set_xscale("log")
            axis.set_xlim(first, limit)
        else:
            axis.set_xlim(0.0, limit)
        axis.set_title(TITLES.get(case_id, case_id), fontsize=25)
        axis.set_xlabel("time [s]", fontsize=29)
        axis.tick_params(labelsize=22)
        axis.grid(alpha=0.22, which="both" if log_x else "major")
    axes[0].set_ylabel(
        "cost / initial cost" if phase == "offline" else "cost / first solution cost",
        fontsize=29,
    )
    order = method_ids(phase)
    figure.legend(
        [handles[key] for key in order if key in handles],
        [labels[key] for key in order if key in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=20,
    )
    figure.tight_layout(rect=(0, 0.22 if phase == "online" else 0.18, 1, 1), w_pad=0.8)
    suffix = "logx" if log_x else "linear"
    output = HERE / "analysis" / "plots" / f"final-{phase}-blog-four-{suffix}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def summarize(allow_incomplete: bool) -> dict:
    summary = {
        "schema": "mr-shortcut-final-summary-v1",
        "protocol_sha256": protocol_sha256(),
        "aggregation": protocol()["design"]["aggregation"],
        "phases": {},
    }
    rows = []
    for phase in ("offline", "online"):
        phase_data = {}
        for case_id, case in protocol_cases().items():
            if phase == "online" and not case["online"]:
                continue
            limit = horizon(phase, case_id)
            grid = np.linspace(0.0, limit, 1001)
            case_data = {}
            for method_id in method_ids(phase):
                stats = aggregate(
                    phase, case_id, method_id, grid, allow_incomplete
                )
                if stats is None:
                    continue
                record = {
                    key: value for key, value in stats.items()
                    if key not in ("median", "low", "high")
                }
                case_data[method_id] = record
                rows.append({"phase": phase, "case_id": case_id, "method_id": method_id, **record})
            phase_data[case_id] = case_data
        summary["phases"][phase] = phase_data
    write_json(HERE / "analysis" / "summary.json", summary)
    csv_path = HERE / "analysis" / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["phase", "case_id", "method_id"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    summarize(args.allow_incomplete)
    outputs = []
    for phase in ("offline", "online"):
        for log_x in (False, True):
            outputs.append(str(plot_blog(phase, log_x, args.allow_incomplete)))
    print(json.dumps({"plots": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
