from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np


def _prepend_assembly_checkout() -> Path:
    configured = os.environ.get("ASSEMBLY_REPO")
    if not configured:
        raise RuntimeError(
            "assembly benchmarks require ASSEMBLY_REPO=/path/to/assembly"
        )
    checkout = Path(configured).expanduser().resolve()
    if not (checkout / "mp_export.py").is_file():
        raise RuntimeError(
            f"assembly checkout not found at {checkout}; set ASSEMBLY_REPO"
        )
    sys.path.insert(0, str(checkout))
    return checkout


def build(case: dict, root: Path):
    checkout = _prepend_assembly_checkout()

    import assembly as assembly_module
    import mp_export
    import replay
    from plan_io import read_plan
    from solver import RobotIK

    plan_file = root / case["plan"]
    path_file = root / case["export"]
    cfg, plan = read_plan(str(plan_file))
    plan = [tuple(action) for action in plan]
    C, assembly = replay.build_scene(cfg)
    robots = assembly_module.get_robots(C)
    gripper_kind = (
        assembly_module.wrench_kind_arg(C, robots)
        if cfg.get("wrench")
        else None
    )

    robot_ik = RobotIK(C, robots)
    robot_ik.ik_seeding = (
        True if cfg.get("ik_seeding") is None else bool(cfg.get("ik_seeding"))
    )
    robot_ik.pre_insertion_delta = float(cfg.get("pre_insertion") or 0.0)
    robot_ik.pre_pick_delta = float(cfg.get("pre_pick") or 0.0)
    if hasattr(robot_ik, "post_pick_delta"):
        robot_ik.post_pick_delta = float(cfg.get("post_pick") or 0.0)
    robot_ik.pre_place_delta = float(cfg.get("pre_place") or 0.0)
    robot_ik.pre_add_grasp_delta = float(cfg.get("pre_add_grasp") or 0.0)
    if hasattr(robot_ik, "post_release_delta"):
        robot_ik.post_release_delta = float(cfg.get("post_release") or 0.0)

    bundle = mp_export.load_export_bundle(
        str(path_file), plan, list(robot_ik.C.getJointNames())
    )
    settings = dict(bundle.get("settings") or {})
    for attribute, key in (
        ("pre_insertion_delta", "pre_insertion"),
        ("pre_pick_delta", "pre_pick"),
        ("post_pick_delta", "post_pick"),
        ("pre_place_delta", "pre_place"),
        ("pre_add_grasp_delta", "pre_add_grasp"),
        ("post_release_delta", "post_release"),
    ):
        if hasattr(robot_ik, attribute) and key in settings:
            setattr(robot_ik, attribute, float(settings[key]))

    kwargs = {
        "bracing": bool(settings.get("bracing", cfg.get("bracing"))),
        "depgraph": bool(settings.get("depgraph", False)),
        "collision_tolerance": float(
            settings.get("collision_tolerance", cfg.get("mp_collision_tolerance", 0.01))
        ),
        "pre_witnesses": bundle["pre_witnesses"],
        "robot_robot_margin": float(settings.get("robot_robot_margin", 0.0)),
        "robot_margin_ramp": float(settings.get("robot_margin_ramp", 0.0)),
        "robot_margin_mode": str(settings.get("robot_margin_mode", "linear")),
        "table_margin": float(settings.get("table_margin", 0.0)),
        "table_float": float(settings.get("table_float", 0.0)),
        "wrench_gate": bool(settings.get("wrench_gate", False)),
        "witness_sets": bundle.get("witness_sets"),
        "resolved_pre_witnesses": dict(bundle["pre_witnesses"]),
        "resolve_missing_pre": False,
        "task_records": bundle["tasks"],
        "saved_step_names": bundle["step_names"],
        "saved_sequence_names": bundle["sequence_names"],
        "physical_finger_collisions": bool(
            settings.get("physical_finger_collisions", False)
        ),
    }
    accepted = inspect.signature(mp_export.make_problem).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    env = mp_export.make_problem(
        C,
        assembly,
        plan,
        bundle["witnesses"],
        robots,
        robot_ik,
        gripper_kind,
        **kwargs,
    )

    saved = mp_export.load_path(str(path_file))
    chain = mp_export.mode_sequence(env)
    by_task_ids = {}
    for mode in chain:
        by_task_ids.setdefault(
            tuple(int(value) for value in np.atleast_1d(mode.task_ids)),
            mode,
        )

    from multi_robot_multi_goal_planning.problems.planning_env import State

    configuration_type = type(env.get_start_pos())
    slices = env.get_start_pos()._array_slice
    path = []
    for index, state in enumerate(saved):
        signature = tuple(
            int(value) for value in np.atleast_1d(state.mode.task_ids)
        )
        if signature not in by_task_ids:
            raise RuntimeError(
                f"saved state {index} uses mode {signature}, absent from rebuilt chain"
            )
        q = configuration_type(np.asarray(state.q, dtype=float), slices)
        path.append(State(q, by_task_ids[signature]))

    wanted = getattr(mp_export.load_path, "last_signature", None)
    actual = mp_export.task_signature(env)
    if wanted is not None and wanted != actual:
        raise RuntimeError(
            f"assembly task signature changed: saved {wanted}, rebuilt {actual}"
        )

    return env, path, {
        "assembly_checkout": str(checkout),
        "task_signature": actual,
        "saved_task_signature": wanted,
        "plan_file": str(plan_file),
        "path_file": str(path_file),
    }


def restore_records(env, records):
    import mp_export
    from multi_robot_multi_goal_planning.problems.planning_env import State

    chain = mp_export.mode_sequence(env)
    by_task_ids = {
        tuple(int(value) for value in np.atleast_1d(mode.task_ids)): mode
        for mode in chain
    }
    configuration_type = type(env.get_start_pos())
    slices = env.get_start_pos()._array_slice
    restored = []
    for index, record in enumerate(records):
        signature = tuple(int(value) for value in record["mode"])
        if signature not in by_task_ids:
            raise RuntimeError(
                f"saved state {index} uses mode {signature}, absent from rebuilt chain"
            )
        q = configuration_type(np.asarray(record["q"], dtype=float), slices)
        restored.append(State(q, by_task_ids[signature]))
    return restored
