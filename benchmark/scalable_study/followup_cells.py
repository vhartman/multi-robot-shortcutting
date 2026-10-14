"""Additive ablations: reuse the frozen sampler and strict cell certification."""
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
for directory in (HERE.parent, HERE.parent / "scaling_2d", HERE.parent / "expanded",
                  HERE.parent.parent / "src"):
    sys.path.insert(0, str(directory))

import numpy as np
from mr_shortcutting.api import Shortcutter, register_shortcutter
from scalable_tree_group import _VectorizedRandomGroup
from study_lib import (assembly_definition, build_mrbench_environment, cases,
                       read_json, write_json)


class RandomCompatibleGroup(Shortcutter):
    name = "random_compatible_group"

    def make_proposer(self, env, path, rng, interpolation_resolution):
        proposer = _VectorizedRandomGroup(path, rng)
        proposer.minimum_cost_improvement = 1e-6
        return proposer


register_shortcutter(RandomCompatibleGroup(), replace=True)


def offline_cell():
    # Adapt only the method registry in the existing certified cell runner.
    # Do not change the frozen base protocol or implementation source files.
    import run_cell
    with patch.object(run_cell, "methods", return_value={
        "random_compatible_group": {"policy": "random_compatible_group"}
    }):
        return run_cell.main()


def check_online_pair(first, reference):
    """Compare two live online starts captured at the same planner hook."""
    actual, records = first["states"], reference["states"]
    if first["representation"] != reference["representation"]:
        raise RuntimeError("online first-path representations differ")
    if len(actual) != len(records):
        raise RuntimeError("paired online first paths have different state counts")
    for a, b in zip(actual, records):
        if len(a["q"]) != len(b["q"]):
            raise RuntimeError("paired online first-path dimensions differ")
        if not np.allclose(a["q"], b["q"], rtol=0, atol=1e-9):
            raise RuntimeError("paired online first-path configurations differ")
        if not np.array_equal(a["mode"], b["mode"]):
            raise RuntimeError("paired online first-path modes differ")
    expected = float(reference["cost"])
    if abs(float(first["cost"]) - expected) > 1e-9 * max(1, abs(expected)):
        raise RuntimeError("paired online first-solution costs differ")


def online_cell():
    from common import ROOT, environment_fingerprint
    from online import ShortcutPatch, _assembly_plan, _certificate, _mode_runs, _scalar
    from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
    from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
    from multi_robot_multi_goal_planning.problems.planning_env import State
    from multi_robot_multi_goal_planning.problems.util import path_cost

    def certify_candidate(candidate):
        snapshot = copy.deepcopy(candidate)
        # planner.plan() adds the same-configuration state in the new mode only
        # to its final returned path. Incumbents in info["paths"] are captured
        # before that normalization, so reproduce it before strict checking.
        normalized = []
        for index, state in enumerate(snapshot):
            normalized.append(state)
            if index + 1 < len(snapshot) and state.mode != snapshot[index + 1].mode:
                normalized.append(State(state.q, snapshot[index + 1].mode))
        candidate_certificate = _certificate(env, normalized)
        valid = (
            candidate_certificate["strict_full_edge"]
            and candidate_certificate["terminal_mode"]
            and candidate_certificate["valid_plan"] is not False
        )
        return normalized, candidate_certificate, valid

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    settings = read_json(HERE / "followups.json")["online"]
    if args.case not in settings["cases"] or args.method not in settings["methods"]:
        raise ValueError("case/method outside online pilot")
    case = cases()[args.case]
    source = read_json(args.input)
    reference = read_json(args.reference) if args.reference else None
    if args.method != "upstream_tree" and reference is None:
        raise RuntimeError("candidate policy requires its paired online baseline")
    if reference is not None:
        for key, expected in (("case_id", args.case),
                              ("method_id", "upstream_tree"),
                              ("planner_seed", source["planner_seed"]),
                              ("path_index", source["path_index"])):
            if reference.get(key) != expected:
                raise RuntimeError(f"online reference mismatch: {key}")
    if source["case_id"] != args.case:
        raise RuntimeError("input case mismatch")
    if os.environ.get("PYTHONHASHSEED") != str(source["pythonhashseed"]):
        raise RuntimeError("PYTHONHASHSEED differs from saved input")
    if case["kind"] == "mrbench":
        env = build_mrbench_environment(case)
    else:
        from assembly_loader import build
        env, _, _ = build(assembly_definition(case), ROOT)
        env.cost_reduction, env.cost_metric = "max", "euclidean"
    fingerprint = environment_fingerprint(env)
    if fingerprint != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from paired input")
    if reference is not None and fingerprint != reference["environment_fingerprint"]:
        raise RuntimeError("environment differs between paired online runs")
    np.random.seed(source["planner_seed"])
    random.seed(source["planner_seed"])
    cap = float(settings["first_solution_cap_seconds"])
    horizon = float(settings["refinement_seconds"])
    started = time.perf_counter()
    first = {}
    events = []
    planner_config = {}
    class ChargedShortcutPatch(ShortcutPatch):
        def __enter__(self):
            super().__enter__()
            integrated = self.integrated

            def measured(*args, **kwargs):
                call_started = time.perf_counter()
                result = integrated(*args, **kwargs)
                self.calls[-1]["total_call_wall_seconds"] = time.perf_counter() - call_started
                return result

            for module, _ in self.restores:
                if getattr(module, "robot_mode_shortcut", None) is integrated:
                    module.robot_mode_shortcut = measured
            self.integrated = measured
            return self

    shortcut = ChargedShortcutPatch(args.method, settings["shortcut_seed"], started,
                                    started + cap, settings["checks_per_call"])

    class Budget:
        def should_terminate(self, current_iterations=None, current_time=None):
            deadline = started + (first["time"] + horizon if first else cap)
            return time.perf_counter() >= deadline

    original_plan = BidirectionalRRTstar.plan

    def controlled_plan(planner, ptc, optimize=True, **kwargs):
        planner_config.update(asdict(planner.config))
        original_tracking = planner.update_results_tracking

        def tracking(cost, path):
            elapsed = time.perf_counter() - started
            if path and env.is_terminal_mode(path[-1].mode):
                if not first:
                    if elapsed > cap:
                        raise RuntimeError("first solution arrived after the configured cap")
                    # Deep-copy at the same pre-shortcut hook for both methods.
                    # Saved offline paths are NOT references: plan() inserts
                    # additional mode-boundary states before returning them.
                    records = json.loads(json.dumps([state.to_dict() for state in path]))
                    first.update(time=elapsed, cost=_scalar(cost), states=records,
                                 representation="internal_path_before_first_shortcut",
                                 path_sha256=hashlib.sha256(
                                     json.dumps(records, sort_keys=True).encode()).hexdigest())
                    if reference is not None:
                        if planner_config != reference["planner_config"]:
                            raise RuntimeError("planner settings differ between online runs")
                        check_online_pair(first, reference["first_solution"])
                    shortcut.deadline = started + elapsed + horizon
                events.append({"time": elapsed, "cost": _scalar(cost), "source": "planner"})
            return original_tracking(cost, path)

        planner.update_results_tracking = tracking
        return original_plan(planner, ptc=Budget(), optimize=optimize, **kwargs)

    with shortcut, patch.object(BidirectionalRRTstar, "plan", controlled_plan):
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            if case["kind"] == "assembly":
                path, info = _assembly_plan(env, cap + horizon,
                                           source["planner_seed"], True, True)
            else:
                config = BaseRRTConfig(**source["planner_config"]) if source.get("planner_config") else BaseRRTConfig()
                config.distance_metric = "max_euclidean"
                config.shortcutting = True
                config.with_mode_validation = False
                if not source.get("planner_config"):
                    config.stepsize = 1.0
                planner = BidirectionalRRTstar(env, config=config)
                if source.get("sampling_strategy", {}).get("name") == "gibbs_random_scan":
                    from input_sampling import RandomScanGibbsSampler
                    planner.collision_free_sampler = RandomScanGibbsSampler(
                        env, planner.collision_free_sampler.config, planner._get_seed_for_mode)
                path, info = planner.plan(ptc=Budget(), optimize=True)
    elapsed = time.perf_counter() - started
    if not first or not path:
        raise RuntimeError("no first solution within cap; do not pool as a successful cell")
    if not shortcut.calls:
        raise RuntimeError("planner never invoked its shortcut hook")
    for call in shortcut.calls:
        if call["input_mode_runs"] != call["output_mode_runs"]:
            raise RuntimeError("shortcut altered mode sequence")
        events.extend(dict(point, source="shortcut_unverified") for point in call["trace"])

    # Evaluate the anytime planner using only incumbents that pass the strict
    # certificate. The planner can retain a later invalid path after reinsertion;
    # that path stays in diagnostics but cannot improve the reported curve.
    info = dict(info or {})
    tracked_paths = list(info.get("paths", []))
    tracked_times = [float(value) for value in info.get("times", [])]
    tracked_costs = [_scalar(value) for value in info.get("costs", [])]
    if not (len(tracked_paths) == len(tracked_times) == len(tracked_costs)):
        raise RuntimeError("planner incumbent paths/costs/times are not aligned")
    certified_events = []
    rejected_events = []
    selected_path = None
    selected_certificate = None
    selected_cost = float("inf")
    horizon_end = first["time"] + horizon
    for index, (event_time, event_cost, candidate) in enumerate(
            zip(tracked_times, tracked_costs, tracked_paths)):
        snapshot, candidate_certificate, valid = certify_candidate(candidate)
        record = {
            "index": index,
            "time": event_time,
            "cost": event_cost,
            "certificate": candidate_certificate,
        }
        if not valid:
            rejected_events.append(record)
            continue
        certified_events.append(record)
        if first["time"] <= event_time <= horizon_end and event_cost < selected_cost:
            selected_path = snapshot
            selected_certificate = candidate_certificate
            selected_cost = event_cost
    if selected_path is None:
        raise RuntimeError("planner produced no strictly certified incumbent in the comparison horizon")
    measured_cost = _scalar(path_cost(selected_path, env.batch_config_cost))
    if abs(measured_cost - selected_cost) > 1e-8 * max(1, abs(selected_cost)):
        raise RuntimeError("selected incumbent cost does not match its recorded cost")

    certified_curve, best = [], float("inf")
    for point in sorted(certified_events, key=lambda item: (item["time"], item["cost"])):
        if point["cost"] < best - 1e-12:
            best = point["cost"]
            certified_curve.append({key: point[key] for key in ("time", "cost")})
    # Both methods share this exact raw first solution, so it is the fair cost
    # and time origin even when its internal (pre-return-normalization) form is
    # not itself the selected certified output. Every subsequent point and the
    # reported final cost come exclusively from strictly certified incumbents.
    curve = [{"time": first["time"], "cost": first["cost"],
              "status": "paired_raw_first_solution"}]
    curve.extend(point for point in certified_curve if point["time"] > first["time"])
    within = [dict(p, time=p["time"] - first["time"]) for p in curve
              if p["time"] <= horizon_end]
    if len(within) < 2:
        raise RuntimeError("no strictly certified incumbent followed the paired raw first solution")
    if abs(within[-1]["cost"] - selected_cost) > 1e-8 * max(1, abs(selected_cost)):
        raise RuntimeError("comparison curve does not end at the selected certified incumbent")
    write_json(args.output, {
        "schema": "mr-shortcut-scalable-online-pilot-v5",
        "case_id": args.case, "method_id": args.method,
        "planner_seed": source["planner_seed"], "path_index": source["path_index"],
        "shortcut_seed": settings["shortcut_seed"],
        "environment_fingerprint": fingerprint, "planner_config": planner_config,
        "sampling_strategy": source.get("sampling_strategy"),
        "first_solution": first,
        "first_path_pairing": {
            "role": "reference" if reference is None else "paired_candidate",
            "validated": None if reference is None else True,
            "reference_path_sha256": None if reference is None else reference["first_solution"]["path_sha256"],
        },
        "refinement_seconds": horizon, "total_wall_seconds": elapsed,
        "deadline_overrun_seconds": max(0, elapsed - first["time"] - horizon),
        "full_curve": curve, "comparison_curve": within,
        "comparison_final_cost": within[-1]["cost"],
        "selected_path_cost": measured_cost,
        "certificate": selected_certificate, "shortcut_calls": shortcut.calls,
        "incumbent_certification": {
            "tracked": len(tracked_paths),
            "strictly_valid": len(certified_events),
            "rejected": rejected_events,
            "selected_recorded_cost": selected_cost,
        },
        "shortcut_wall_seconds": sum(call["total_call_wall_seconds"] for call in shortcut.calls),
        "planner_info_times": tracked_times,
        "planner_info_costs": tracked_costs,
        "output_mode_runs": _mode_runs(selected_path),
    })
    return 0


if __name__ == "__main__":
    track = sys.argv.pop(1)
    raise SystemExit(offline_cell() if track == "offline" else online_cell())
