"""Audit both replicated offline studies without requiring new online runs."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import run as pilot
import run_sum_cost_study as summed


def audit():
    errors = []
    reports = {}
    for name in ("max", "sum"):
        if name == "max":
            manifest = pilot.definition("offline")
            frozen = pilot.HERE / "offline" / "manifest.json"
            records = pilot.status("offline", manifest, pilot.digest(manifest))
        else:
            manifest = summed.definition()
            frozen = summed.ROOT / "replication-manifest.json"
            records = []
            for cell in manifest["cells"]:
                path = summed.output_path(cell)
                issues = summed.result_errors(path, cell)
                state = "missing" if issues == ["missing"] else "invalid" if issues else "local"
                records.append((cell, state, path))
        if not frozen.exists() or pilot.read_json(frozen) != manifest:
            errors.append(f"{name}: frozen manifest does not match the current definition")
        identity = lambda cell: (cell["case_id"], cell["path_index"], cell["shortcut_seed"], cell["method_id"])
        if len({identity(cell) for cell in manifest["cells"]}) != len(manifest["cells"]):
            errors.append(f"{name}: duplicate cells")
        groups = defaultdict(list)
        for cell, state, path in records:
            if state not in ("local", "reused"):
                errors.append(f"{name}/{identity(cell)}: {state}")
                continue
            result = pilot.read_json(path)
            initial = float(result["initial_cost"])
            final = float(result["final_cost"])
            if not (math.isfinite(initial) and initial > 0 and math.isfinite(final) and final >= 0):
                errors.append(f"{name}/{identity(cell)}: invalid cost")
                continue
            if final > initial + 1e-7 * max(1, initial):
                errors.append(f"{name}/{identity(cell)}: increasing final cost")
            trace = result.get("trace", [])
            if not trace:
                errors.append(f"{name}/{identity(cell)}: missing trace")
            last_time, last_cost = -1.0, initial
            for point in trace:
                elapsed, cost = float(point["time"]), float(point["cost"])
                if not (math.isfinite(elapsed) and math.isfinite(cost) and elapsed >= last_time and cost >= 0):
                    errors.append(f"{name}/{identity(cell)}: invalid trace point")
                    break
                if cost > last_cost + 1e-7 * max(1, initial):
                    errors.append(f"{name}/{identity(cell)}: increasing trace")
                    break
                last_time, last_cost = elapsed, cost
            groups[(cell["case_id"], cell["path_index"])].append(initial)
        for key, costs in groups.items():
            if max(costs) - min(costs) > 1e-8 * max(1, max(costs)):
                errors.append(f"{name}/{key}: methods do not share the same initial cost")
        reports[name] = {"expected": len(manifest["cells"]),
                         "states": dict(Counter(state for _, state, _ in records)),
                         "input_path_groups": len(groups),
                         "manifest_sha256": pilot.digest(manifest)}
    report = {"passed": not errors, "studies": reports, "errors": errors,
              "aggregation": "median across two seeds within each path, then median across three paths",
              "scope": "offline scaling and sum cost only; online/broad unchanged"}
    pilot.write_json(pilot.HERE / "analysis" / "replication-audit.json", report)
    return report


if __name__ == "__main__":
    report = audit()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
