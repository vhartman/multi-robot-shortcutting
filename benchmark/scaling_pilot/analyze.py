"""Audit and plot the completed offline/online scaling pilot."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(BENCHMARK / "all_scaling"))

import study_lib as scaling  # noqa: E402
from plot_style import method_color  # noqa: E402

OUTPUT = HERE / "analysis"
METHODS = [
    "full_configuration", "tree_gain", "tree_group_gain", "robot_subset_mode",
    "scalable_tree_group", "tree_then_compatible_group",
]
LABELS = {
    "full_configuration": "full configuration",
    "tree_gain": "tree + gain",
    "tree_group_gain": "tree + group gain",
    "robot_subset_mode": "compatible subset",
    "scalable_tree_group": "tree + compatible (interleaved)",
    "tree_then_compatible_group": "tree + compatible (sequenced)",
}
FAMILIES = [
    "disk_scaling", "isolated_box_stacking", "coupled_box_stacking",
    "mobile_wall_scaling",
]
FAMILY_LABELS = {
    "disk_scaling": "2D disks",
    "isolated_box_stacking": "Isolated box stacking",
    "coupled_box_stacking": "Coupled box stacking",
    "mobile_wall_scaling": "Mobile wall 5×5",
}


def read(path: Path):
    with path.open() as stream:
        return json.load(stream)


def robot_count(case_id: str) -> int:
    match = re.search(r"-r(\d+)(?:-|$)", case_id)
    if not match:
        raise ValueError(f"cannot parse robot count from {case_id}")
    return int(match.group(1))


def offline_results():
    manifest = read(HERE / "offline" / "manifest.json")
    records = []
    for cell in manifest["cells"]:
        local = (HERE / "offline" / "results" / cell["case_id"] /
                 f"{cell['method_id']}-path{cell['path_index']}-seed{cell['shortcut_seed']}.json")
        path = local if local.is_file() else REPO / cell["reuse_path"]
        records.append((cell, path, read(path)))
    return records


def online_path(cell):
    case_id = cell["case_id"]
    count = robot_count(case_id)
    if cell["family"] != "mobile_wall_scaling" or count <= 2:
        root = HERE / "online" / "results"
    elif count <= 5:
        root = HERE / "online-wall-replay" / "results"
    else:
        root = HERE / "online-wall-generator-order" / "results"
    return root / case_id / f"{cell['method_id']}-path0-seed{cell['shortcut_seed']}.json"


def online_results(require_complete=True):
    protocol = read(HERE / "protocol.json")
    case_map = scaling.cases()
    records = []
    for case_id in protocol["case_ids"]:
        family = case_map[case_id]["family"]
        for seed in protocol["shortcut_seeds"]:
            for method in METHODS:
                cell = {"case_id": case_id, "family": family,
                        "method_id": method, "shortcut_seed": seed}
                path = online_path(cell)
                if not path.is_file() and not require_complete:
                    continue
                records.append((cell, path, read(path)))
    return records


def audit(offline, online):
    data = read(HERE / "protocol.json")
    expected_by_track = {
        track: len(data["case_ids"]) * len(data[track].get("path_indices", data["path_indices"]))
        * len(data["shortcut_seeds"]) * len(METHODS)
        for track in ("offline", "online")
    }
    errors = []
    for cell, path, value in offline:
        if value.get("case_id") != cell["case_id"] or value.get("method_id") != cell["method_id"]:
            errors.append(f"metadata: {path}")
        for key in ("strict_collision_free", "terminal_mode_preserved",
                    "endpoint_start_preserved", "endpoint_end_preserved"):
            if value.get(key) is not True:
                errors.append(f"{key}: {path}")
        if value.get("valid_plan") is False or not value.get("trace"):
            errors.append(f"offline certificate/trace: {path}")

    groups = defaultdict(list)
    for cell, path, value in online:
        if value.get("case_id") != cell["case_id"] or value.get("method_id") != cell["method_id"]:
            errors.append(f"metadata: {path}")
        certificate = value.get("certificate") or {}
        if certificate.get("strict_full_edge") is not True or certificate.get("terminal_mode") is not True:
            errors.append(f"online certificate: {path}")
        if certificate.get("valid_plan") is False or not value.get("comparison_curve"):
            errors.append(f"online plan/curve: {path}")
        groups[(cell["case_id"], cell["shortcut_seed"])].append((path, value))
    for key, values in groups.items():
        hashes = {value["first_solution"]["path_sha256"] for _, value in values}
        costs = {round(float(value["first_solution"]["cost"]), 12) for _, value in values}
        if len(values) != len(METHODS) or len(hashes) != 1 or len(costs) != 1:
            errors.append(f"first-path pairing: {key}")
    report = {
        "schema": "mr-shortcut-scaling-pilot-audit-v1",
        "offline_results": len(offline),
        "online_results": len(online),
        "online_pairing_groups": len(groups),
        "expected_by_track": expected_by_track,
        "errors": errors,
        "passed": (
            not errors
            and len(offline) == expected_by_track["offline"]
            and len(online) == expected_by_track["online"]
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def curve(value, track):
    points = value["trace"] if track == "offline" else value["comparison_curve"]
    initial = float(points[0]["cost"])
    return np.asarray([float(point["time"]) for point in points]), np.asarray(
        [float(point["cost"]) / initial for point in points]
    )


def step_values(times, values, grid):
    indices = np.searchsorted(times, grid, side="right") - 1
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]


def path_median_curves(values, track, grid):
    """One curve per initial path, after taking its seed-wise median."""
    if track == "online":
        return np.vstack([step_values(*curve(value, track), grid) for value in values])
    by_path = defaultdict(list)
    for value in values:
        times, costs = curve(value, track)
        by_path[int(value.get("path_index", 0))].append(step_values(times, costs, grid))
    return np.vstack([np.median(np.vstack(by_path[index]), axis=0)
                      for index in sorted(by_path)])


def plot_family(records, track, family):
    selected = [(cell, value) for cell, _, value in records if cell["family"] == family]
    cases = sorted({cell["case_id"] for cell, _ in selected}, key=robot_count)
    columns = min(3, len(cases))
    rows = math.ceil(len(cases) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.2 * rows), squeeze=False)
    for axis, case_id in zip(axes.flat, cases):
        for method in METHODS:
            runs = [value for cell, value in selected
                    if cell["case_id"] == case_id and cell["method_id"] == method]
            curves = [curve(value, track) for value in runs]
            end = (max(float(value["refinement_seconds"]) for value in runs)
                   if track == "online"
                   else max(float(value["budget_seconds"]) for value in runs))
            grid = np.linspace(0, end, 301)
            matrix = path_median_curves(runs, track, grid)
            axis.plot(grid, np.median(matrix, axis=0), color=method_color(method),
                      linewidth=2.7, label=LABELS[method])
            axis.fill_between(grid, np.min(matrix, axis=0), np.max(matrix, axis=0),
                              color=method_color(method), alpha=0.10, linewidth=0)
        axis.set_title(f"{robot_count(case_id)} robots", fontsize=15)
        axis.set_xlabel("time [s]", fontsize=14)
        axis.set_ylabel("normalized cost", fontsize=14)
        axis.tick_params(labelsize=12)
        axis.grid(alpha=0.22)
        axis.set_ylim(bottom=0)
    for axis in list(axes.flat)[len(cases):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"{FAMILY_LABELS[family]} — {track}", fontsize=18, y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.875))
    destination = OUTPUT / f"{track}-{family}-curves.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def final_normalized(records, track, case_id, method):
    values = defaultdict(list)
    for cell, _, value in records:
        if cell["case_id"] != case_id or cell["method_id"] != method:
            continue
        _, costs = curve(value, track)
        values[int(cell.get("path_index", 0))].append(float(costs[-1]))
    if track == "online":
        return np.asarray([value for group in values.values() for value in group])
    return np.asarray([np.median(values[index]) for index in sorted(values)])


def plot_summary(offline, online):
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharey="row")
    case_map = scaling.cases()
    for row, (track, records) in enumerate((("offline", offline), ("online", online))):
        for column, family in enumerate(FAMILIES):
            axis = axes[row, column]
            cases = sorted(
                [case_id for case_id in read(HERE / "protocol.json")["case_ids"]
                 if case_map[case_id]["family"] == family], key=robot_count,
            )
            positions = np.arange(len(cases), dtype=float)
            counts = [robot_count(case_id) for case_id in cases]
            bar_width = 0.84 / len(METHODS)
            for method_index, method in enumerate(METHODS):
                runs = [final_normalized(records, track, case_id, method) for case_id in cases]
                medians = np.asarray([np.median(item) for item in runs])
                lows = np.asarray([np.min(item) for item in runs])
                highs = np.asarray([np.max(item) for item in runs])
                offset = (method_index - (len(METHODS) - 1) / 2) * bar_width
                centers = positions + offset
                axis.bar(
                    centers, medians, width=bar_width * 0.92,
                    color=method_color(method), label=LABELS[method],
                )
                axis.errorbar(
                    centers, medians,
                    yerr=np.vstack((medians - lows, highs - medians)),
                    fmt="none", ecolor="#333333", elinewidth=0.9,
                    capsize=1.5, capthick=0.9,
                )
            axis.set_title(FAMILY_LABELS[family], fontsize=16)
            axis.set_xlabel("robots", fontsize=14)
            axis.tick_params(labelsize=12)
            axis.grid(axis="y", alpha=0.22)
            axis.set_axisbelow(True)
            axis.set_xticks(positions, labels=counts)
            axis.set_ylim(bottom=0)
            if column == 0:
                axis.set_ylabel(f"{track}\nfinal normalized cost", fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    destination = OUTPUT / "offline-online-final-scaling-summary.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def main():
    offline, online = offline_results(), online_results()
    report = audit(offline, online)
    figures = [plot_summary(offline, online)]
    for track, records in (("offline", offline), ("online", online)):
        figures.extend(plot_family(records, track, family) for family in FAMILIES)
    print(json.dumps({"audit": report, "figures": [str(path) for path in figures]}, indent=2))


if __name__ == "__main__":
    main()
