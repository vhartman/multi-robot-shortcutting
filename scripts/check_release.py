"""Fast release checks. No planning, rendering, or timed benchmark is started."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_policies():
    from mr_shortcutting import available_shortcutters, get_shortcutter, load_shortcutter_plugins
    legacy_file = ROOT / "benchmark/scaling_2d/scalable_tree_group.py"
    legacy = load_file("frozen_scalable", legacy_file)
    load_shortcutter_plugins(["mr_shortcutting.scalable"])
    from mr_shortcutting import scalable

    # Keep the packaged implementation aligned with the frozen benchmark.
    old = {n.name: ast.dump(n) for n in ast.parse(legacy_file.read_text()).body
           if isinstance(n, ast.ClassDef)}
    new = {n.name: ast.dump(n) for n in ast.parse(Path(scalable.__file__).read_text()).body
           if isinstance(n, ast.ClassDef)}
    assert all(new[name] == tree for name, tree in old.items()), "scalable source drift"
    policies = ("scalable_tree_group", "tree_then_compatible_group", "random_compatible_group")
    assert set(policies) <= set(available_shortcutters())
    pairs = (
        (legacy.ScalableTreeGroupShortcutter(), scalable.ScalableTreeGroupShortcutter()),
        (legacy.TreeThenCompatibleGroupShortcutter(), scalable.TreeThenCompatibleGroupShortcutter()),
    )
    draws = 0
    for robots in (1, 2, 6, 12):
        env = SimpleNamespace(robots=list(range(robots)))
        path = [SimpleNamespace(mode=SimpleNamespace(task_ids=[
            index // (5 + 3 * robot) for robot in range(robots)
        ])) for index in range(64)]
        for seed in (0, 1, 2):
            streams = [(a.make_proposer(env, path, np.random.default_rng(seed), 0.5),
                        b.make_proposer(env, path, np.random.default_rng(seed), 0.5))
                       for a, b in pairs]
            original_group = legacy._VectorizedRandomGroup(path, np.random.default_rng(seed))
            original_group.minimum_cost_improvement = 1e-6
            streams.append((original_group, get_shortcutter("random_compatible_group").make_proposer(
                env, path, np.random.default_rng(seed), 0.5
            )))
            for old_stream, new_stream in streams:
                for draw in range(400):
                    a, b = old_stream(), new_stream()
                    assert a == b, (robots, seed, draw, a, b)
                    draws += 1
                    if b is not None:
                        assert 0 <= b.first < b.last < len(path) and b.last - b.first >= 2
                        assert b.robots and len(set(b.robots)) == len(b.robots)
                        for robot in b.robots:
                            assert len({s.mode.task_ids[robot] for s in path[b.first:b.last + 1]}) == 1
                        outcome = ("accepted", "collision", "cost")[draw % 3]
                        fail_index = (b.first + b.last) // 2
                        old_stream.feedback(outcome, a, fail_index=fail_index)
                        new_stream.feedback(outcome, b, fail_index=fail_index)
                    assert old_stream.stats == new_stream.stats
    return draws


def check_input_planning():
    sys.path.insert(0, str(ROOT / "benchmark/scaling_pilot"))
    import prepare_inputs
    import run as pilot
    with tempfile.TemporaryDirectory(prefix="mr-shortcut-release-") as directory:
        def target(case, index):
            return Path(directory) / case / f"path{index}.json"
        with patch.object(prepare_inputs, "input_path", target):
            pending = prepare_inputs.missing_inputs()
            assert len(pending) == 81
            assert len({(c, i) for c, i, *_ in pending}) == 81
            for case_id in pilot.protocol()["case_ids"]:
                assert len({seed for c, _, seed, *_ in pending if c == case_id}) == 3
            case = pilot.protocol()["case_ids"][0]
            first = target(case, 0)
            first.parent.mkdir(parents=True)
            first.write_text(json.dumps({"planner_seed": 99, "environment_seed": 2}))
            with patch.object(prepare_inputs.scaling, "input_errors", return_value=[]):
                remaining = prepare_inputs.missing_inputs([case])
                assert len(remaining) == 2 and all(index != 0 for _, index, *_ in remaining)
                assert all(seed != 99 for _, _, seed, *_ in remaining)
            with patch.object(prepare_inputs.scaling, "input_errors", return_value=["strict certificate"]):
                try:
                    prepare_inputs.missing_inputs([case])
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("invalid existing input accepted")
    return 81


def check_saved_source_hashes():
    checked = 0
    for relative in ("benchmark/scaling_pilot/offline/manifest.json",
                     "benchmark/scaling_pilot/sum-cost/replication-manifest.json"):
        path = ROOT / relative
        if not path.exists():
            continue  # Results/manifests are intentionally not shipped.
        for name, expected in json.loads(path.read_text())["source_hashes"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            assert actual == expected, f"frozen source changed: {name}"
            checked += 1
    return checked


if __name__ == "__main__":
    print(json.dumps({"equivalent_proposal_draws": check_policies(),
                      "fresh_scaling_inputs_planned": check_input_planning(),
                      "saved_source_hashes_checked": check_saved_source_hashes()}, indent=2))
