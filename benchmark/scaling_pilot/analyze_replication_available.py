"""Plot audited replication data, preserving rejected cells in coverage metadata.

This is deliberately separate from the publication analyzers: incomplete
method/scenario groups are omitted, labelled, and listed in the coverage report.
By explicit choice, --successful-only instead retains every series using its
available successful runs. --blog-assets-dir installs those figures; accepted
benchmark results are never modified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import run as pilot
import run_sum_cost_study as summed
from analyze import FAMILY_LABELS, FAMILIES, LABELS, METHODS, path_median_curves, robot_count
from analyze_sum_cost import CASES as SUM_CASES
from plot_style import method_color

OUTPUT = pilot.HERE / "analysis" / "replication-preview"


def identity(cell):
    return cell["case_id"], cell["path_index"], cell["shortcut_seed"], cell["method_id"]


def collect(*, successful_only=False):
    failure_path = pilot.HERE / "analysis" / "continuation-failures.json"
    # Fully certified fresh runs need no rejection ledger. Missing results
    # without execution evidence still fail the coverage checks below.
    failures = pilot.read_json(failure_path) if failure_path.exists() else []
    rejected = {(entry["track"], identity(entry["cell"])): entry for entry in failures}
    groups = {"max": defaultdict(list), "sum": defaultdict(list)}
    expected = {"max": defaultdict(set), "sum": defaultdict(set)}
    rows, errors = [], []
    for track in ("max", "sum"):
        manifest = pilot.definition("offline") if track == "max" else summed.definition()
        frozen = (pilot.HERE / "offline" / "manifest.json" if track == "max"
                  else summed.ROOT / "replication-manifest.json")
        if pilot.read_json(frozen) != manifest:
            raise RuntimeError(f"{track}: frozen manifest changed")
        records = (pilot.status("offline", manifest, pilot.digest(manifest)) if track == "max"
                   else [(cell, "missing" if not summed.output_path(cell).exists() else
                          "invalid" if summed.result_errors(summed.output_path(cell), cell) else "local",
                          summed.output_path(cell)) for cell in manifest["cells"]])
        for cell, state, path in records:
            key = cell["case_id"], cell["method_id"]
            expected[track][key].add((cell["path_index"], cell["shortcut_seed"]))
            row = {"track": track, "cell": cell, "status": state, "path": str(path)}
            if state in ("local", "reused"):
                groups[track][key].append(pilot.read_json(path))
                row["status"] = "certified"
            elif (track, identity(cell)) in rejected:
                failure = rejected[(track, identity(cell))]
                if failure["cell"] != cell:
                    raise RuntimeError("rejected cell differs from frozen definition")
                diagnostic = pilot.read_json(Path(failure["diagnostic"]))
                arguments = diagnostic["arguments"]
                for flag, value in (("--case", cell["case_id"]), ("--method", cell["method_id"]),
                                    ("--path-index", str(cell["path_index"])),
                                    ("--shortcut-seed", str(cell["shortcut_seed"]))):
                    if arguments[arguments.index(flag) + 1] != value:
                        raise RuntimeError("diagnostic identity mismatch")
                if not diagnostic.get("diagnostic_only") or "certificate failed" not in diagnostic["exception"]:
                    raise RuntimeError("missing failed execution evidence")
                if not diagnostic["result"].get("trace") or not Path(failure["log"]).is_file():
                    raise RuntimeError("missing failed execution trace/log")
                row.update(status="rejected", diagnostic=failure["diagnostic"], log=failure["log"])
            else:
                errors.append(row)
            rows.append(row)
    omitted = []
    for track in groups:
        for key, pairs in expected[track].items():
            values = groups[track][key]
            actual = {(int(v["path_index"]), int(v["shortcut_seed"])) for v in values}
            if actual != pairs or len(values) != len(pairs):
                omitted.append({"track": track, "case_id": key[0], "method_id": key[1],
                                "certified": len(values), "expected": len(pairs)})
                if not successful_only:
                    groups[track][key] = []
    sample_counts = [
        {"track": track, "case_id": key[0], "method_id": key[1],
         "runs": len(values),
         "seeds_by_path": dict(Counter(int(v["path_index"]) for v in values))}
        for track, track_groups in groups.items() for key, values in track_groups.items()
    ]
    report = {
        "all_cells_accounted_for": not errors,
        "all_cells_certified": not errors and not omitted,
        "counts": {track: dict(Counter(row["status"] for row in rows if row["track"] == track))
                   for track in groups},
        "incomplete_series": omitted,
        "omitted_series": [] if successful_only else omitted,
        "sample_counts": sample_counts,
        "unaccounted_cells": errors, "cells": rows,
        "aggregation": "median of available successful seeds per initial path, then median of available path medians",
        "band": "range of the three path medians, not a confidence interval",
        "plot_policy": ("retain every method/scenario group; aggregate successful runs only"
                        if successful_only else
                        "omit entire incomplete method/scenario group; never average fewer than six runs"),
    }
    pilot.write_json(OUTPUT / "coverage.json", report)
    if errors:
        raise RuntimeError("unaccounted cells; inspect coverage.json")
    return groups, report


def legend(fig):
    handles = [Line2D([0], [0], color=method_color(m), linewidth=4, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=3, frameon=False, fontsize=20)


def render(groups, track, cases, filename, *, log=False, columns=3):
    rows = math.ceil(len(cases) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 6.3 * rows + 2.3),
                             squeeze=False, sharey=True)
    for axis, (case_id, title) in zip(axes.flat, cases):
        available = [v for method in METHODS for v in groups[(case_id, method)]]
        end = float(available[0]["budget_seconds"])
        positive = [float(p["time"]) for v in available for p in v["trace"] if float(p["time"]) > 0]
        grid = np.geomspace(max(min(positive), 1e-4), end, 301) if log else np.linspace(0, end, 301)
        missing = []
        for method in METHODS:
            values = groups[(case_id, method)]
            if not values:
                missing.append(LABELS[method])
                continue
            matrix = path_median_curves(values, "offline", grid)
            axis.plot(grid, np.median(matrix, axis=0), color=method_color(method), linewidth=4)
            axis.fill_between(grid, matrix.min(axis=0), matrix.max(axis=0),
                              color=method_color(method), alpha=0.11, linewidth=0)
        if log:
            axis.set_xscale("log")
        axis.set_xlim(grid[0], grid[-1])
        axis.set_ylim(bottom=0)
        axis.set_title(title, fontsize=25)
        axis.set_xlabel("time [s]", fontsize=27)
        axis.tick_params(labelsize=20)
        axis.grid(alpha=0.22)
        if missing:
            axis.text(0.98, 0.97, "Incomplete series omitted:\n" + "\n".join(missing),
                      transform=axis.transAxes, ha="right", va="top", fontsize=16,
                      bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"})
    for row in range(rows):
        axes[row, 0].set_ylabel("normalized sum cost" if track == "sum" else "normalized cost", fontsize=27)
    for axis in axes.flat[len(cases):]:
        axis.remove()
    legend(fig)
    fig.text(0.5, 0.004, "Each displayed curve: 3 paths × 2 seeds; shading: range of path medians.",
             ha="center", fontsize=16)
    fig.tight_layout(rect=(0, 0.13 if rows > 1 else 0.23, 1, 1))
    destination = OUTPUT / filename
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return destination


def summary(groups):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13.5), sharey=True)
    case_map = pilot.cases()
    protocol_cases = pilot.read_json(pilot.PROTOCOL_PATH)["case_ids"]
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    for axis, family in zip(axes.flat, FAMILIES):
        cases = sorted([c for c in protocol_cases if case_map[c]["family"] == family], key=robot_count)
        missing_counts = []
        for position, case_id in enumerate(cases):
            points = []
            for i, method in enumerate(METHODS):
                values = groups[(case_id, method)]
                if not values:
                    missing_counts.append(robot_count(case_id))
                    continue
                grid = np.asarray([float(values[0]["budget_seconds"])])
                value = float(np.median(path_median_curves(values, "offline", grid)))
                points.append(value)
                axis.scatter(position + offsets[i], value, s=105, color=method_color(method),
                             edgecolor="white", linewidth=0.8, zorder=3)
            axis.vlines(position, min(points), max(points), color="#d0d0d0", linewidth=2, zorder=0)
        axis.set_xticks(range(len(cases)), labels=[robot_count(c) for c in cases])
        axis.set_title(FAMILY_LABELS[family], fontsize=25)
        axis.set_xlabel("robots", fontsize=29)
        axis.tick_params(labelsize=21)
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(bottom=0)
        if missing_counts:
            axis.text(0.98, 0.97, "Incomplete series omitted at " + ", ".join(map(str, sorted(set(missing_counts)))) + " robots",
                      transform=axis.transAxes, ha="right", va="top", fontsize=15)
    for axis in axes[:, 0]:
        axis.set_ylabel("final normalized cost", fontsize=29)
    legend(fig)
    fig.text(0.5, 0.004, "Each displayed point: 3 paths × 2 seeds. Incomplete method/scenario groups are omitted.",
             ha="center", fontsize=15)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    destination = OUTPUT / "offline-scaling-summary.png"
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return destination


def successful_figures(groups, report):
    """Use the established blog layouts; keep reduced sample counts in metadata."""
    from analyze_blog import INTRO_METHODS, plot_cases, plot_family_curves, plot_scaling_summary
    import analyze_sum_cost as sum_plots
    records = [(row["cell"], Path(row["path"]), pilot.read_json(Path(row["path"])))
               for row in report["cells"] if row["track"] == "max" and row["status"] == "certified"]
    if any(not values for track in groups.values() for values in track.values()):
        raise RuntimeError("cannot draw a method/scenario group without successful results")
    paths = []
    for cases, name in (
        (("rai-disk-scaling-r4-g4", "rai-disk-scaling-r8-g4", "rai-disk-scaling-r12-g4"),
         "intro-disk-scaling.png"),
        (("rai-coupled-box-stacking-r4-b8", "rai-coupled-box-stacking-r6-b8"),
         "intro-coupled-scaling.png"),
    ):
        path = OUTPUT / name
        plot_cases(records, "offline", cases, INTRO_METHODS, "", path)
        paths.append(path)
    path = OUTPUT / "offline-scaling-summary.png"
    plot_scaling_summary(records, "offline", path)
    paths.append(path)
    for family, name in (
        ("disk_scaling", "offline-disk-scaling-curves.png"),
        ("isolated_box_stacking", "offline-isolated-box-stacking-curves.png"),
        ("coupled_box_stacking", "offline-coupled-box-stacking-curves.png"),
        ("mobile_wall_scaling", "offline-mobile-wall-scaling-curves.png"),
    ):
        path = OUTPUT / name
        plot_family_curves(records, family, path)
        paths.append(path)
    sum_plots.OUTPUT = OUTPUT
    paths.extend([sum_plots.plot(groups["sum"]), sum_plots.plot(groups["sum"], log_time=True)])
    return paths


def main():
    import argparse
    import shutil
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successful-only", action="store_true",
                        help="Keep incomplete series, aggregating their available successful runs.")
    parser.add_argument("--blog-assets-dir", type=Path,
                        help="Copy the successful-only figures to an existing blog assets directory.")
    args = parser.parse_args()
    if args.blog_assets_dir and not args.successful_only:
        parser.error("--blog-assets-dir requires --successful-only")
    if args.blog_assets_dir and not args.blog_assets_dir.is_dir():
        parser.error("blog assets directory must already exist")
    if args.successful_only:
        OUTPUT = pilot.HERE / "analysis" / "replication-successful"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, report = collect(successful_only=args.successful_only)
    if args.successful_only:
        paths = successful_figures(groups, report)
        if args.blog_assets_dir:
            for path in paths:
                shutil.copy2(path, args.blog_assets_dir / path.name)
        print(json.dumps({"counts": report["counts"], "incomplete_series": report["incomplete_series"],
                          "plots": [str(path) for path in paths],
                          "blog_assets_dir": str(args.blog_assets_dir) if args.blog_assets_dir else None}, indent=2))
        return
    paths = [summary(groups["max"])]
    case_map = pilot.cases()
    protocol_cases = pilot.read_json(pilot.PROTOCOL_PATH)["case_ids"]
    for family in FAMILIES:
        cases = [(c, f"{robot_count(c)} robots") for c in
                 sorted([c for c in protocol_cases if case_map[c]["family"] == family], key=robot_count)]
        for log in (False, True):
            suffix = "-logx" if log else ""
            paths.append(render(groups["max"], "max", cases, f"offline-{family}{suffix}.png", log=log))
    for log in (False, True):
        suffix = "-logx" if log else ""
        paths.append(render(groups["sum"], "sum", SUM_CASES, f"offline-sum-cost-largest{suffix}.png", log=log, columns=2))
    print(json.dumps({"counts": report["counts"], "omitted_series": report["omitted_series"],
                      "plots": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
