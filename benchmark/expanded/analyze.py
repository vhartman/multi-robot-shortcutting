"""Plot normalized cost-versus-time curves for the expanded offline benchmark."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from expanded_lib import (  # noqa: E402
    cases,
    methods_for,
    protocol,
    read_json,
    result_path,
)
from plot_style import method_color  # noqa: E402


def step_values(times, costs, grid):
    indices = np.searchsorted(times, grid, side="right") - 1
    values = np.ones_like(grid)
    valid = indices >= 0
    values[valid] = costs[indices[valid]]
    return values


def run_curve(case_id, method_id, path_index, seed, grid):
    result = read_json(result_path(case_id, method_id, path_index, seed))
    trace = result["trace"]
    times = np.asarray([point["time"] for point in trace], dtype=float)
    costs = np.asarray([point["cost"] for point in trace], dtype=float)
    costs /= float(result["initial_cost"])
    return step_values(times, costs, grid)


def aggregate(case, method_id):
    seconds = float(case["seconds"])
    grid = np.linspace(0.0, seconds, 301)
    path_medians = []
    seed_curves_by_path = []
    for path_index in case["path_indices"]:
        seed_curves = [
            run_curve(case["id"], method_id, path_index, seed, grid)
            for seed in protocol()["shortcut_seeds"]
            if result_path(case["id"], method_id, path_index, seed).is_file()
        ]
        if not seed_curves:
            continue
        stacked = np.stack(seed_curves)
        seed_curves_by_path.append(stacked)
        path_medians.append(np.median(stacked, axis=0))
    path_stack = np.stack(path_medians)
    median = np.median(path_stack, axis=0)
    if len(path_medians) == 1:
        uncertainty = seed_curves_by_path[0]
    else:
        uncertainty = path_stack
    return {
        "grid": grid,
        "median": median,
        "low": np.min(uncertainty, axis=0),
        "high": np.max(uncertainty, axis=0),
    }


def title(case):
    if case["track"] == "scaling":
        count = int(case["environment_kwargs"]["num_robots"])
        return f"{count} robot" + ("" if count == 1 else "s")
    text = case["id"]
    if text.startswith("rai-"):
        text = text[4:]
    if text.startswith("assembly-"):
        text = text[9:]
    return text.replace("-", " ")


def check_complete(selected):
    missing = []
    for case in selected:
        for method_id in methods_for(case):
            for path_index in case["path_indices"]:
                for seed in protocol()["shortcut_seeds"]:
                    path = result_path(case["id"], method_id, path_index, seed)
                    if not path.is_file():
                        missing.append(str(path))
    return missing


def plot_cases(selected, output, name, rows, columns):
    aggregates = {
        (case["id"], method_id): aggregate(case, method_id)
        for case in selected
        for method_id in methods_for(case)
    }
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.4 * columns, 4.4 * rows),
        squeeze=False,
    )
    for axis, case in zip(axes.flat, selected):
        for method_id, method in methods_for(case).items():
            value = aggregates[(case["id"], method_id)]
            color = method_color(method_id)
            axis.plot(
                value["grid"],
                value["median"],
                color=color,
                linewidth=3.6,
                label=method["label"],
            )
            axis.fill_between(
                value["grid"],
                value["low"],
                value["high"],
                color=color,
                alpha=0.13,
            )
        axis.set_title(title(case), fontsize=19)
        axis.set_xlabel("time [s]", fontsize=17)
        axis.set_ylabel("normalized cost", fontsize=17)
        axis.tick_params(labelsize=14)
        axis.grid(alpha=0.22)
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)

    handles = {}
    for axis in axes.flat[:len(selected)]:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    figure.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=min(len(handles), columns),
        fontsize=16,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    destination = output / name
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("broad", "scaling"))
    parser.add_argument("--case", action="append", choices=tuple(cases()))
    args = parser.parse_args()

    selected = list(cases().values())
    if args.track:
        selected = [case for case in selected if case["track"] == args.track]
    if args.case:
        wanted = set(args.case)
        selected = [case for case in selected if case["id"] in wanted]
    missing = check_complete(selected)

    output = HERE / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    broad = [case for case in selected if case["track"] == "broad"]
    for page, offset in enumerate(range(0, len(broad), 10), start=1):
        chunk = broad[offset:offset + 10]
        paths.append(plot_cases(
            chunk,
            output,
            f"expanded-broad-curves-{page}.png",
            rows=2,
            columns=5,
        ))
    if broad:
        paths.append(plot_cases(
            broad,
            output,
            "expanded-broad-curves-combined.png",
            rows=6,
            columns=5,
        ))
    scaling = [case for case in selected if case["track"] == "scaling"]
    if scaling:
        columns = 3
        paths.append(plot_cases(
            scaling,
            output,
            "isolated-box-stacking-scaling-curves.png",
            rows=math.ceil(len(scaling) / columns),
            columns=columns,
        ))

    if missing:
        print(f"warning: plotted available data with {len(missing)} missing cells")
        for path in missing:
            print(f"missing: {path}")
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
