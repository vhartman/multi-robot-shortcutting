# Multi-robot shortcutting

Reusable shortcutting policies and the frozen benchmark protocol for the multi-robot shortcutting study. The repository follows a small `src/` package plus a separate benchmark runner: implementation code is importable, while protocols and generated data remain separate.

## Install

The final protocol requires Python 3.10 and the exact `mr_bench` revision recorded in [`protocol.json`](benchmark/final/protocol.json). Once that revision is published, a normal editable install is enough:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -e .
```

Until then, install a local checkout at revision `3142dacfbfc22d619f52d69d07164eacae21b26c` first, then install this package without resolving its remote dependency again:

```bash
.venv/bin/pip install -e /path/to/mr_bench
.venv/bin/pip install -e . --no-deps
```

Assembly cases also need an assembly checkout. No machine-specific checkout path is embedded in the code; pass both checkout locations to the runner or set `MR_BENCH_REPO` and `ASSEMBLY_REPO`.

## Use a shortcutter

```python
from mr_shortcutting import run, run_checks

path, report = run(env, path, "tree_group_gain", seconds=30, seed=0)
path, report = run_checks(
    env, path, "tree_group_gain", max_collision_checks=250, seed=0
)
```

`run` is the offline wall-clock interface. `run_checks` uses a collision-check budget and is the interface used inside the planner. Both return the updated path and a trace/report dictionary.

## Add a shortcutter

A new method only supplies proposals. The common engine keeps interpolation, cost gating, collision checks, budgets, accepted-path updates, traces, and certificates identical across methods.

```python
import numpy as np
from mr_shortcutting import Proposal, Shortcutter, register_shortcutter

class MyShortcutter(Shortcutter):
    name = "my_shortcutter"

    def make_proposer(self, env, path, rng, interpolation_resolution):
        def propose():
            robot = int(rng.integers(len(env.robots)))
            first, last = sorted(rng.integers(len(path), size=2))
            if last - first < 2:
                return None
            if path[first].mode.task_ids[robot] != path[last].mode.task_ids[robot]:
                return None
            return Proposal(int(first), int(last), (robot,))
        return propose

register_shortcutter(MyShortcutter())
```

Proposal sources may expose `feedback(...)`, `consider_time(...)`, or a `stats` mapping when a policy needs observations from the shared engine. External modules can be loaded explicitly with `load_shortcutter_plugins(["my_package.shortcutters"])`.

## Reproduce the final benchmark

Nothing executes without `execute --confirm`. Start with the matrix and preflight checks:

```bash
python benchmark/final/run.py plan
python benchmark/final/run.py preflight \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly
```

Then run each phase serially:

```bash
python benchmark/final/run.py execute --phase inputs --confirm \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/final/run.py execute --phase offline --confirm \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/final/run.py execute --phase online --confirm \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/final/validate.py \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/final/analyze.py
```

See [`benchmark/final/README.md`](benchmark/final/README.md) for the frozen design, aggregation rules, resume behavior, and expected matrix size. Generated inputs, results, logs, and plots are intentionally ignored by Git.

## Layout

- `src/mr_shortcutting/`: public API, registry, maintained policies, and shared execution engine.
- `benchmark/manifest.json`: the 15 frozen scenarios and deterministic environment seeds.
- `benchmark/fixtures/assembly/`: source plans/exports required to reconstruct the two assembly tasks.
- `benchmark/online.py`: planner-in-the-loop adapter.
- `benchmark/final/`: frozen protocol, serial runner, validation, lock, and analysis.
