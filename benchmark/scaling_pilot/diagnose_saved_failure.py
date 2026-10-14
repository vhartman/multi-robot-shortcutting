"""Replay a rejected path without planning, shortcutting, or tolerance changes.

Run only after the serial timed batch. Original mode entry configurations are
retained; a separate fresh restoration diagnoses serialization differences.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np

import run as pilot
from common import restore_mrbench_path


def worker_argument(data, name):
    arguments = data["arguments"]
    return arguments[arguments.index(name) + 1]


def certify(env, path):
    samples = []
    original = env.is_collision_free_np

    def observe(q, mode, *args, **kwargs):
        free = original(q, mode, *args, **kwargs)
        if not free and len(samples) < 8:
            samples.append((np.asarray(q).copy(), mode))
        return free

    env.is_collision_free_np = observe
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            strict = bool(env.is_path_collision_free(
                path, resolution=float(env.collision_resolution),
                tolerance=float(env.collision_tolerance),
                check_start_and_end=True, check_edges_in_order=True, binary_order=False,
            ))
            valid = bool(env.is_valid_plan(path))
    finally:
        env.is_collision_free_np = original
    bad_states = [index for index, state in enumerate(path)
                  if not env.is_collision_free(state.q, state.mode)]
    return {"strict": strict, "native": valid, "native_messages": stream.getvalue(),
            "bad_waypoints": bad_states,
            "rejected_edge_samples": [{"q": q.tolist(), "mode": list(mode.task_ids)} for q, mode in samples]}, samples


def probe(env, q, mode, *, fresh=False):
    if fresh:
        env.prev_mode = None
        env.set_to_mode(mode, use_cached=False, place_in_cache=False)
    records = []
    for _ in range(5):
        free = bool(env.is_collision_free(q, mode))
        binary = bool(env.C.getCollisionFree())
        penetration = float(env.C.getCollisionsTotalPenetration())
        pairs = [[str(pair[0]), str(pair[1]), float(pair[2])]
                 for pair in env.C.getCollisions() if float(pair[2]) < 0]
        records.append({"free": free, "binary_free": binary,
                        "penetration": penetration, "negative_pairs": pairs})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pilot.read_json(args.diagnostic)
    if data.get("diagnostic_only") is not True:
        raise ValueError("expected a captured failed-cell diagnostic")
    case_id = worker_argument(data, "--case")
    index = int(worker_argument(data, "--path-index"))
    case = pilot.cases()[case_id]
    source_path = pilot.input_path(case_id, index)
    source = pilot.read_json(source_path)
    env = pilot.scaling.build_environment(case)
    env.cost_reduction = "sum" if Path(data["worker"]).name == "run_sum_cost.py" else "max"
    original_path = restore_mrbench_path(env, source["states"])
    modes = {tuple(state.mode.task_ids): state.mode for state in original_path}
    from multi_robot_multi_goal_planning.problems.planning_env import State

    candidate = [State(original_path[0].q.from_flat(np.asarray(state["q"], dtype=float)),
                       modes[tuple(state["mode"])]) for state in data["states"]]
    print(f"[replay] {case_id}, {len(candidate)} waypoints", flush=True)
    certificate, samples = certify(env, candidate)
    report = {"diagnostic_only": True, "case_id": case_id,
              "captured_path": str(args.diagnostic), "source_sha256": pilot.sha256(source_path),
              "collision_tolerance": float(env.collision_tolerance),
              "collision_resolution": float(env.collision_resolution),
              "original_modes_certificate": certificate, "waypoint_probes": {}, "edge_probes": []}
    for point in certificate["bad_waypoints"][:8]:
        state = candidate[point]
        tests = {"cached": probe(env, state.q, state.mode),
                 "fresh_mode_geometry": probe(env, state.q, state.mode, fresh=True)}
        first, last = max(0, point - 2), min(len(candidate), point + 3)
        tests["local_fragment_certificate"] = certify(env, candidate[first:last])[0]
        tests["last_owner_per_robot"] = {}
        for event in data["result"]["stats"].get("accepted_lineage", []):
            if event["first"] < point < event["last"]:
                for robot in event["robots"]:
                    tests["last_owner_per_robot"][str(robot)] = event
        report["waypoint_probes"][str(point)] = tests
    for q, mode in samples:
        conf = original_path[0].q.from_flat(q)
        report["edge_probes"].append({"q": q.tolist(), "mode": list(mode.task_ids),
                                      "cached": probe(env, conf, mode),
                                      "fresh_mode_geometry": probe(env, conf, mode, fresh=True)})
    fresh_env = pilot.scaling.build_environment(case)
    restored = restore_mrbench_path(fresh_env, data["states"])
    report["freshly_restored_certificate"] = certify(fresh_env, restored)[0]
    report["mode_entry_max_deltas"] = {
        str(list(state.mode.task_ids)): float(np.max(np.abs(
            state.mode.entry_configuration.state() - modes[tuple(state.mode.task_ids)].entry_configuration.state())))
        for state in restored if state.mode.entry_configuration is not None
    }
    pilot.write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "original_modes": certificate,
                      "fresh_restore": report["freshly_restored_certificate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
