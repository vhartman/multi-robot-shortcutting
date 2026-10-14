"""Execute one certified offline or paired online scaling-pilot cell."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE.parent
REPO = BENCHMARK.parent
for directory in (REPO / "src", BENCHMARK, BENCHMARK / "all_scaling",
                  BENCHMARK / "scaling_2d", HERE):
    sys.path.insert(0, str(directory))

import study_lib as scaling  # noqa: E402
from common import environment_fingerprint, mode_runs, restore_mrbench_path  # noqa: E402


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol():
    return read_json(HERE / "protocol.json")


def method(method_id):
    return next(item for item in protocol()["methods"] if item["id"] == method_id)


def certify(env, candidate):
    from online import _certificate
    from multi_robot_multi_goal_planning.problems.planning_env import State

    snapshot = copy.deepcopy(candidate)
    normalized = []
    for index, state in enumerate(snapshot):
        normalized.append(state)
        if index + 1 < len(snapshot) and state.mode != snapshot[index + 1].mode:
            normalized.append(State(state.q, snapshot[index + 1].mode))
    certificate = _certificate(env, normalized)
    valid = (
        certificate["strict_full_edge"]
        and certificate["terminal_mode"]
        and certificate["valid_plan"] is not False
    )
    return normalized, certificate, valid


def offline(args):
    import scalable_tree_group  # noqa: F401
    from mr_shortcutting import run

    data = protocol()
    case = scaling.cases()[args.case]
    selected = method(args.method)
    source_path = scaling.input_path(args.case, args.path_index).resolve()
    source = read_json(source_path)
    seconds = float(data["offline"]["seconds_by_family"][case["family"]])
    if args.seconds != seconds:
        raise ValueError("offline budget differs from protocol")
    env = scaling.build_environment(case)
    if environment_fingerprint(env) != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from input")
    path = restore_mrbench_path(env, source["states"])
    from multi_robot_multi_goal_planning.problems.util import path_cost

    restored = float(path_cost(path, env.batch_config_cost))
    if abs(restored - float(source["initial_cost"])) > 1e-9 * max(1, abs(restored)):
        raise RuntimeError("restored input cost changed")
    started = time.perf_counter()
    output, result = run(
        env, path, selected["policy"], seconds, args.shortcut_seed,
        interpolation_resolution=float(data["offline"]["interpolation_resolution"]),
    )
    elapsed = time.perf_counter() - started
    start_ok = bool(np.allclose(output[0].q.state(), path[0].q.state(), rtol=0, atol=1e-8))
    end_ok = bool(np.allclose(output[-1].q.state(), path[-1].q.state(), rtol=0, atol=1e-8))
    strict = bool(env.is_path_collision_free(
        output, resolution=float(env.collision_resolution),
        tolerance=float(env.collision_tolerance), check_start_and_end=True,
        check_edges_in_order=True, binary_order=False,
    ))
    valid = bool(env.is_valid_plan(output)) if hasattr(env, "is_valid_plan") else None
    terminal = bool(env.is_terminal_mode(output[-1].mode))
    before, after = mode_runs(path), mode_runs(output)
    if not (start_ok and end_ok and strict and terminal and valid is not False and before == after):
        raise RuntimeError("offline output certificate failed")
    result.update({
        "schema": "mr-shortcut-scaling-pilot-offline-v1",
        "case_id": args.case, "family": case["family"],
        "method_id": args.method, "implementation_policy": selected["policy"],
        "path_index": args.path_index, "shortcut_seed": args.shortcut_seed,
        "budget_seconds": seconds, "total_wall_seconds": elapsed,
        "source_path": str(source_path), "source_sha256": sha256(source_path),
        "environment_fingerprint": source["environment_fingerprint"],
        "input_mode_runs": before, "output_mode_runs": after,
        "endpoint_start_preserved": start_ok, "endpoint_end_preserved": end_ok,
        "terminal_mode_preserved": terminal, "strict_collision_free": strict,
        "valid_plan": valid, "protocol_sha256": sha256(HERE / "protocol.json"),
    })
    write_json(args.output, result)


def compare_first(actual, reference):
    if actual["representation"] != reference["representation"]:
        raise RuntimeError("first-path representations differ")
    if len(actual["states"]) != len(reference["states"]):
        raise RuntimeError("paired first paths have different state counts")
    for left, right in zip(actual["states"], reference["states"]):
        if not np.allclose(left["q"], right["q"], rtol=0, atol=1e-9):
            raise RuntimeError("paired first-path configurations differ")
        if not np.array_equal(left["mode"], right["mode"]):
            raise RuntimeError("paired first-path modes differ")
    if abs(float(actual["cost"]) - float(reference["cost"])) > 1e-9 * max(1, abs(float(reference["cost"]))):
        raise RuntimeError("paired first-solution costs differ")


def online(args):
    data = protocol()
    settings = data["online"]
    case = scaling.cases()[args.case]
    selected = method(args.method)
    source_path = scaling.input_path(args.case, args.path_index).resolve()
    source = read_json(source_path)
    reference = read_json(args.reference) if args.reference else None
    reference_id = settings["reference_method"]
    if args.method != reference_id and reference is None:
        raise RuntimeError("candidate policy requires its paired reference")
    if reference is not None:
        expected = {
            "case_id": args.case, "method_id": reference_id,
            "planner_seed": source["planner_seed"], "path_index": args.path_index,
            "shortcut_seed": args.shortcut_seed,
        }
        if any(reference.get(key) != value for key, value in expected.items()):
            raise RuntimeError("online reference metadata mismatch")
    if os.environ.get("PYTHONHASHSEED") != str(data["pythonhashseed"]):
        raise RuntimeError("PYTHONHASHSEED differs from protocol")
    env = scaling.build_environment(case)
    fingerprint = environment_fingerprint(env)
    if fingerprint != source["environment_fingerprint"]:
        raise RuntimeError("environment fingerprint differs from input")
    np.random.seed(source["planner_seed"])
    random.seed(source["planner_seed"])
    rng_replay_draws = int(os.environ.get("MR_SHORTCUT_NUMPY_REPLAY_DRAWS", "0"))
    if rng_replay_draws < 0:
        raise ValueError("MR_SHORTCUT_NUMPY_REPLAY_DRAWS must be nonnegative")
    if rng_replay_draws:
        # The saved-input generator seeded NumPy immediately before importing
        # the planner.  At the frozen mr-bench revision those imports consume
        # five NumPy draws.  The wall continuation advances the already-loaded
        # process explicitly; exact equality is enforced between the online
        # reference and every paired candidate below.
        np.random.random(rng_replay_draws)
    # Preserve the input generator's import order: construct the environment,
    # seed the planner stream, and only then import/construct the planner.
    from multi_robot_multi_goal_planning.planners.planner_birrtstar import BidirectionalRRTstar
    from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTConfig
    from multi_robot_multi_goal_planning.problems.util import path_cost

    config = BaseRRTConfig(**source["planner_config"]) if source.get("planner_config") else BaseRRTConfig()
    config.distance_metric = "max_euclidean"
    config.shortcutting = True
    config.with_mode_validation = False
    if not source.get("planner_config"):
        config.stepsize = 1.0
    planner = BidirectionalRRTstar(env, config=config)
    # Match the behavior-preserving bindings used to create the certified
    # initial paths and retain their timing counters for later analysis.
    planner._sample_config_fn = planner._timed_sample_configuration
    planner._sample_transition_config_fn = planner._timed_sample_transition_configuration
    planner._check_edge_cf = planner._timed_edge_collision_free
    planner._check_config_cf = planner._timed_collision_free

    # Policy registration is irrelevant to planning.  Import it only after
    # planner construction and restore both global RNG streams defensively.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    import scalable_tree_group  # noqa: F401
    from online import ShortcutPatch, _scalar
    np.random.set_state(numpy_state)
    random.setstate(python_state)

    cap = float(settings["first_solution_cap_seconds"])
    horizon = float(settings["refinement_seconds"])
    started = time.perf_counter()
    first, events, planner_config = {}, [], {}

    class ChargedPatch(ShortcutPatch):
        def __enter__(self):
            super().__enter__()
            integrated = self.integrated

            def measured(*values, **kwargs):
                call_started = time.perf_counter()
                result = integrated(*values, **kwargs)
                self.calls[-1]["total_call_wall_seconds"] = time.perf_counter() - call_started
                return result

            for module, _ in self.restores:
                if getattr(module, "robot_mode_shortcut", None) is integrated:
                    module.robot_mode_shortcut = measured
            self.integrated = measured
            return self

    shortcut = ChargedPatch(
        selected["policy"], args.shortcut_seed, started, started + cap,
        int(settings["checks_per_call"]),
    )

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
                    records = json.loads(json.dumps([state.to_dict() for state in path]))
                    first.update(
                        time=elapsed, cost=_scalar(cost), states=records,
                        representation="internal_path_before_first_shortcut",
                        path_sha256=hashlib.sha256(
                            json.dumps(records, sort_keys=True).encode()
                        ).hexdigest(),
                    )
                    if reference is not None:
                        if planner_config != reference["planner_config"]:
                            raise RuntimeError("planner settings differ between paired runs")
                        compare_first(first, reference["first_solution"])
                    shortcut.deadline = started + elapsed + horizon
                events.append({"time": elapsed, "cost": _scalar(cost), "source": "planner"})
            return original_tracking(cost, path)

        planner.update_results_tracking = tracking
        return original_plan(planner, ptc=Budget(), optimize=optimize, **kwargs)

    if source.get("sampling_strategy", {}).get("name") == "gibbs_random_scan":
        from input_sampling import RandomScanGibbsSampler
        planner.collision_free_sampler = RandomScanGibbsSampler(
            env, planner.collision_free_sampler.config, planner._get_seed_for_mode
        )
    with shortcut, patch.object(BidirectionalRRTstar, "plan", controlled_plan):
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            returned_path, info = planner.plan(ptc=Budget(), optimize=True)
    elapsed = time.perf_counter() - started
    if not first or not returned_path:
        raise RuntimeError("no first solution within cap")
    if not shortcut.calls:
        raise RuntimeError("planner never invoked its shortcut hook")
    for call in shortcut.calls:
        if call["input_mode_runs"] != call["output_mode_runs"]:
            raise RuntimeError("shortcut altered mode sequence")
        events.extend(dict(point, source="shortcut_unverified") for point in call["trace"])

    info = dict(info or {})
    paths = list(info.get("paths", []))
    times = [float(value) for value in info.get("times", [])]
    costs = [_scalar(value) for value in info.get("costs", [])]
    if not (len(paths) == len(times) == len(costs)):
        raise RuntimeError("planner incumbent records are not aligned")
    valid_events, rejected = [], []
    selected_path, selected_certificate, selected_cost = None, None, float("inf")
    horizon_end = first["time"] + horizon
    for index, (event_time, event_cost, candidate) in enumerate(zip(times, costs, paths)):
        snapshot, certificate, valid = certify(env, candidate)
        record = {"index": index, "time": event_time, "cost": event_cost,
                  "certificate": certificate}
        if not valid:
            rejected.append(record)
            continue
        valid_events.append(record)
        if first["time"] <= event_time <= horizon_end and event_cost < selected_cost:
            selected_path, selected_certificate, selected_cost = snapshot, certificate, event_cost
    if selected_path is None:
        raise RuntimeError("no strictly certified incumbent in comparison horizon")
    measured = _scalar(path_cost(selected_path, env.batch_config_cost))
    if abs(measured - selected_cost) > 1e-8 * max(1, abs(selected_cost)):
        raise RuntimeError("certified incumbent cost mismatch")
    curve, best = [], float("inf")
    for point in sorted(valid_events, key=lambda item: (item["time"], item["cost"])):
        if point["cost"] < best - 1e-12:
            best = point["cost"]
            curve.append({key: point[key] for key in ("time", "cost")})
    comparison = [{"time": 0.0, "cost": first["cost"],
                   "status": "paired_raw_first_solution"}]
    comparison.extend(dict(point, time=point["time"] - first["time"])
                      for point in curve if first["time"] < point["time"] <= horizon_end)
    if len(comparison) < 2:
        raise RuntimeError("no certified incumbent followed the paired first solution")
    if abs(comparison[-1]["cost"] - selected_cost) > 1e-8 * max(1, abs(selected_cost)):
        raise RuntimeError("comparison curve does not end at selected incumbent")
    write_json(args.output, {
        "schema": "mr-shortcut-scaling-pilot-online-v1",
        "case_id": args.case, "family": case["family"],
        "method_id": args.method, "implementation_policy": selected["policy"],
        "planner_seed": source["planner_seed"], "path_index": args.path_index,
        "shortcut_seed": args.shortcut_seed, "source_sha256": sha256(source_path),
        "environment_fingerprint": fingerprint, "planner_config": planner_config,
        "sampling_strategy": source.get("sampling_strategy"),
        "planner_rng_replay": {
            "numpy_draws_after_seed": rng_replay_draws,
            "reference_cost_checked_against_saved_input": False,
            "exact_candidate_pairing_checked": reference is not None,
        },
        "first_solution": first,
        "first_path_pairing": {
            "role": "reference" if reference is None else "paired_candidate",
            "validated": None if reference is None else True,
            "reference_path_sha256": None if reference is None else reference["first_solution"]["path_sha256"],
        },
        "refinement_seconds": horizon, "total_wall_seconds": elapsed,
        "deadline_overrun_seconds": max(0, elapsed - first["time"] - horizon),
        "comparison_curve": comparison, "comparison_final_cost": selected_cost,
        "selected_path_cost": measured, "certificate": selected_certificate,
        "shortcut_calls": shortcut.calls,
        "shortcut_wall_seconds": sum(call["total_call_wall_seconds"] for call in shortcut.calls),
        "incumbent_certification": {"tracked": len(paths), "strictly_valid": len(valid_events),
                                     "rejected": rejected},
        "protocol_sha256": sha256(HERE / "protocol.json"),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("track", choices=("offline", "online"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--path-index", required=True, type=int)
    parser.add_argument("--shortcut-seed", required=True, type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = protocol()
    if args.case not in data["case_ids"]:
        raise ValueError("case outside protocol")
    if args.method not in {item["id"] for item in data["methods"]}:
        raise ValueError("method outside protocol")
    if args.path_index not in data["path_indices"] or args.shortcut_seed not in data["shortcut_seeds"]:
        raise ValueError("replicate outside protocol")
    (offline if args.track == "offline" else online)(args)


if __name__ == "__main__":
    main()
