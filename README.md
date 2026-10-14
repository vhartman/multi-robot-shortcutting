# Multi-robot shortcutting

Shortcut proposal policies, a shared validation/execution engine, and the
protocols behind [Multi-robot shortcutting](https://vhartmann.com/multi-robot-shortcutting/)
and [Scaling shortcutting to many robots](https://vhartmann.com/scalable-multi-robot-shortcutting/).

The package is independent of the benchmark runners: use an existing planning
environment and path, or implement a new proposal policy without duplicating
collision checking, interpolation, cost gating, or logging.

## Install

Use Python 3.10 (the experiments used 3.10.12). From this repository:

**Release prerequisite:** the pinned planner commit below is not currently
fetchable from the public remote. Until that exact source is published, use
the local-checkout installation below. The normal installation command will
not work on a clean machine yet; do not substitute another planner revision
when reproducing the published protocol.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e .
source .venv/bin/activate
```

The package pins the planner dependency to
`3142dacfbfc22d619f52d69d07164eacae21b26c`. If working with a local planner
checkout, install that revision with its dependencies first, then install this
package without resolving the remote dependency again:

```bash
.venv/bin/python -m pip install -e /path/to/mr_bench
.venv/bin/python -m pip install -e . --no-deps
```

Assembly scenarios additionally require the assembly checkout specified by the
study. Supply checkout locations through the runner's `--mr-bench-repo` and
`--assembly-repo` arguments. The first-post runner also accepts
`MR_BENCH_REPO` and `ASSEMBLY_REPO`. Serial runners use Linux power/suspend
monitoring and file locks; keep a laptop plugged in for timing experiments.

## Use a shortcutter

```python
from mr_shortcutting import load_shortcutter_plugins, run, run_checks

# Register the scalable policies from the second post.
load_shortcutter_plugins(["mr_shortcutting.scalable"])

path, report = run(env, path, "scalable_tree_group", seconds=30, seed=0)
path, report = run_checks(
    env, path, "scalable_tree_group", max_collision_checks=250, seed=0
)
```

`env` and `path` use the planner package's environment and `State` interfaces.
`run` is the offline wall-clock interface; `run_checks` uses a collision-check
budget for planner integration. Each call creates fresh policy state and returns
the updated path plus a trace/report dictionary. Query
`available_shortcutters()` after loading plugins for all registered policy IDs;
`POLICIES` lists only the original built-ins.

| Name in the scaling post | Policy ID |
|---|---|
| full configuration | `full_configuration` |
| tree + gain | `tree_prefix` (benchmark alias: `tree_gain`) |
| tree + group gain | `tree_group_gain` |
| compatible subset, scaling comparison | `robot_subset_mode` |
| compatible subset, broad vectorized ablation | `random_compatible_group` |
| tree + compatible (interleaved) | `scalable_tree_group` |
| tree + compatible (sequenced) | `tree_then_compatible_group` |

The last three IDs are registered by `mr_shortcutting.scalable`. The two
compatible-subset implementations have the same sampling idea but different
bookkeeping: the broad ablation uses the vectorized sampler and the scalable
methods' `1e-6` minimum cost improvement. They are not interchangeable run IDs.
“Tree + gain” here is the benchmark policy, not a promise about what a newer
upstream planner uses by default.

## Add a shortcutter

Implement the abstract `Shortcutter` factory, returning a callable proposal
source. Return `None` to skip a draw. A compatible interval must have no mode
change for any selected robot, including its interior states.

```python
from mr_shortcutting import Proposal, Shortcutter, register_shortcutter

class MyShortcutter(Shortcutter):
    name = "my_shortcutter"

    def make_proposer(self, env, path, rng, interpolation_resolution):
        def propose():
            robot = int(rng.integers(len(env.robots)))
            first, last = sorted(map(int, rng.integers(len(path), size=2)))
            if last - first < 2:
                return None
            task = path[first].mode.task_ids[robot]
            if any(s.mode.task_ids[robot] != task for s in path[first:last + 1]):
                return None
            return Proposal(first, last, (robot,))
        return propose

register_shortcutter(MyShortcutter())
```

The engine changes the bound path in place as it accepts shortcuts. A proposer
can inspect that updated path on subsequent calls. Optional hooks are
`feedback(outcome, proposal, **details)`, `consider_time(elapsed, current_cost)`,
and a `stats` mapping. External plugin modules register their policies on import
and can be loaded with `load_shortcutter_plugins(["my_package.shortcutters"])`.
Use the same shared engine and validation settings for comparisons.

## Reproduce the studies

Start at the [benchmark index](benchmark/README.md). The main entry points are:

- [Scaling, online, and alternative-cost comparisons](benchmark/scaling_pilot/README.md):
  27 scaling cases, six methods, 3 paths × 2 seeds offline; online remains 1 × 2.
- [Broad comparison](benchmark/scalable_study/README.md):
  27 retained broad cases, four methods, 3 paths × 3 seeds.
- [First-post four-environment comparison](benchmark/final/README.md):
  the historical 3 × 3 protocol.

For the current scaling study, inspection does not start experiments:

```bash
python benchmark/scaling_pilot/prepare_inputs.py plan --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_pilot/run.py plan --track offline --mr-bench-repo /path/to/mr_bench
```

Generated paths, results, logs, videos, and plots are not committed. A fresh
checkout therefore needs input generation before shortcutting, and result
generation before analysis. Regenerated paths are new samples, not a guarantee
of reproducing the exact published trajectories or wall-clock curves.
The study README describes certified-result reuse and the published
successful-run aggregation, including three rejected cells.

### Historical source locks

The first-post content lock matches benchmark revision `7aef53d`. Use a
separate checkout of that revision for an exact first-post protocol run;
later shared-engine and plot-style changes intentionally do not satisfy its
old lock. Do not re-lock old data to make a validation warning disappear.

The scaling runners retain their original implementation module at
`benchmark/scaling_2d/scalable_tree_group.py` because manifests hash its bytes.
The installed plugin is a behavior-equivalent copy, verified by
`python scripts/check_release.py`. Published workers still use the frozen
module. No existing result needs to be rewritten to use the installed package.

## Repository layout

- `src/mr_shortcutting/`: public extension API, shared engine, and policies.
- `benchmark/scaling_pilot/`: final second-post scaling/online/sum-cost protocol.
- `benchmark/scalable_study/`: broad study and compatible-group ablation.
- `benchmark/scaling_2d/`, `scaling_coupled/`, `scaling_mobile_wall/`: scenes,
  renderers, and historical source protocols.
- `benchmark/final/`, `broad/`, `expanded/`, `all_scaling/`: earlier protocols
  and source definitions retained for provenance and certified reuse.
- `benchmark/fixtures/assembly/`: source fixtures, not benchmark results.
- `benchmark/online_scheduling/`: unpublished planner exploration, explicitly
  separate from the published experiments.
- `scripts/check_release.py`: source/package smoke checks; no timed experiment.

See the benchmark index before launching an older trial script: several
historical protocols cover larger grids than the final published comparison.
