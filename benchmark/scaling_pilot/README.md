# Scaling comparison

The published offline comparison uses 27 scenarios, six methods, three
initial paths (indices 0--2), and two shortcut seeds (0--1): 972 cells.
Budgets remain 15 seconds for disks and 30 seconds for other families.
The online protocol retains path 0 and two seeds; it is not expanded.

The completed replication has 970/972 certified max-cost results and 143/144
certified sum-cost results. The post retains every series using successful runs:
three method/scenario groups have five instead of six samples, with all three
initial paths represented. The strict audit still reports those three failures.
The matrices and source locks are not rewritten to make them appear complete.

Completed cells are reused only when the input hash, method, seed, budget and
certificates match. Existing manifests are archived when execution freezes a
strictly additive extension. Missing input files block timed execution.
No timed runs or input generation are started by `plan` or `preflight`.

Use the configured Python environment and replace `/path/to/mr_bench` below.

```bash
python benchmark/scaling_pilot/prepare_inputs.py plan --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_pilot/run.py plan --track offline --mr-bench-repo /path/to/mr_bench
```

When ready, generate only missing paths, using distinct planner seeds and a
200-second cap per path. The generator stops on failure; a failed attempt
does not silently change seeds or overwrite existing inputs. Every accepted
path must pass strict certification and match the existing paths' environment
and cost. With no saved inputs, the first generated path establishes that
reference. Add repeated `--case CASE_ID` arguments to prepare a subset.

Fresh input generation uses `generate_input.py`, which supports all four
families (including mobile wall and isolated stacking above six robots), with
random-scan Gibbs for disks. It records its generator hash and planner settings.
Archived inputs are never overwritten; new paths are not claimed to reproduce
the exact historical first solutions.

```bash
python benchmark/scaling_pilot/prepare_inputs.py execute --confirm --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_pilot/run.py preflight --track offline --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_pilot/run.py execute --confirm --track offline --max-hours 8 --mr-bench-repo /path/to/mr_bench
```

Execution is serial, requires mains power, and stops on suspend or power loss.
The eight-hour limit is checked between cells; it is not a per-cell deadline.
Resume with the same command to run only missing cells.

Offline plots take the median across the two shortcut seeds for each input
path, then the median across the three paths. Bands show the range across path
medians, not confidence intervals. No new curves should be presented as a
completed 3-by-2 comparison until all expected cells are certified.

The separate general/broad comparison uses its existing 3-by-3 results. Its
blog figures can be regenerated without experiments:

```bash
python benchmark/scalable_study/analyze_followups.py --blog-broad-only --output-dir /path/to/blog/assets/scalable-multi-robot-shortcutting
```

## Sum-cost replication

`sum_cost_protocol.json` expands the existing largest-setting comparison to
three paths and two seeds: four cases, six methods, 144 cells. It preserves the
original 24 certified cells and uses the same paths and per-case budgets as the
max-cost study, but optimizes the sum of robot path lengths. The remaining 120
cells contain 52.5 minutes of nominal shortcutting, plus setup, certification,
and budget overshoot. Run this after, not alongside, the scaling comparison:

```bash
python benchmark/scaling_pilot/run_sum_cost_study.py plan --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_pilot/run_sum_cost_study.py execute --confirm --max-hours 2 --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_pilot/analyze_sum_cost.py
```

## Online comparison

The published online budget is 45 seconds after the paired first solution,
with 100 collision checks per shortcut call, one initial-path seed, and two
shortcut seeds. The first-solution cap is 240 seconds. These are separate runs,
not the offline paths injected into a planner.

```bash
python benchmark/scaling_pilot/run.py plan --track online --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_pilot/run.py preflight --track online --mr-bench-repo /path/to/mr_bench
```

Historical mobile-wall continuations are in `run_wall_replay.py` (3–5 robots)
and `run_wall_generator_order.py` (6–7 robots). They preserve the first-path
pairing fixes and have their own frozen protocols. The online analyzer uses
those results for the corresponding cases; do not substitute the rejected
earlier wall runs. These wrappers accept `plan`, `preflight`, and
`execute --confirm --mr-bench-repo /path/to/mr_bench`, like the main runner.

After all online cells are available, `python benchmark/scaling_pilot/analyze_blog.py`
regenerates the online and offline figures. Do not run the unpublished
`online_scheduling` experiments to reproduce these plots.

## Certification and plotting

If final validation rejects a cell, do not silently drop that run from a
six-run aggregate or relax the collision threshold. The continuation runner
keeps failed paths and logs separately from accepted results. A separate
preview renderer checks execution coverage and omits the entire incomplete
method/scenario series, with a visible annotation:

```bash
python benchmark/scaling_pilot/analyze_replication_available.py
```

These figures and `coverage.json` go to `analysis/replication-preview/`, not
the publication figure directory. `all_cells_accounted_for` means every cell
has either a certified result or recorded rejection; it does **not** mean
`all_cells_certified`. The strict publication audit remains unchanged.

To retain all curves using the available successful runs (the selected policy
for the blog), use the explicit successful-only option. The median is still
taken within each path, then across paths; a path with one successful shortcut
seed contributes that single value. Actual counts remain in `coverage.json`.
Outputs go to `analysis/replication-successful/`:

```bash
python benchmark/scaling_pilot/analyze_replication_available.py --successful-only --blog-assets-dir /path/to/blog/assets/scalable-multi-robot-shortcutting
```

Both runners share the execution lock and power/suspend supervision. The
sum-cost analyzer requires all 144 certified cells and uses the same nested
seed/path median and path-range bands as the offline scaling figures.

For a fully certified new run, audit the exact offline matrices and regenerate
only their figures (no new online runs or online plot changes). These strict
commands intentionally do not accept the three rejected published cells; use
the successful-only command above for the published aggregation:

```bash
python benchmark/scaling_pilot/audit_replication.py
python benchmark/scaling_pilot/analyze_blog.py --offline-only
python benchmark/scaling_pilot/analyze_sum_cost.py
```
