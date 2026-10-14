# Scalable shortcutting study

## Published broad comparison

The second post uses the 27 retained broad scenarios, three initial paths and
three shortcut seeds, with four methods: tree + gain, tree + group gain,
tree + compatible (interleaved), and compatible subset. The fourth method is
the additive vectorized `random_compatible_group` ablation in `followups.json`.
Regenerate these figures from saved data with:

```bash
python benchmark/scalable_study/analyze_followups.py --blog-broad-only
```

The final scaling, online, and sum-cost comparisons are now in
[`../scaling_pilot`](../scaling_pilot/README.md). Use that protocol, not the older
scaling matrix below, for the published scaling figures.

## Original study and retained source definitions

The original offline study compares three methods:

- `tree_gain`: the current standard tree-prefix policy;
- `tree_group_gain`: the strongest policy retained from the earlier shortcutting study;
- `scalable_tree_group`: the finite singleton tree interleaved with mode-compatible random robot groups.

The matrix contains three independently planned initial paths and three shortcut seeds for every method and case. Execution is strictly serial. The four families are:

- the 30-case broad benchmark;
- isolated box stacking with one through six UR10s and four boxes per robot;
- planar disks with obstacles at 2, 4, 6, 8, 10, and 12 robots;
- coupled box stacking with two through six UR10s, eight shared boxes, one table, and one shared stack.

Existing certified inputs are reused from `benchmark/expanded`, `benchmark/final`, and `benchmark/scaling_2d`. Missing broad paths and all coupled-box-stacking paths are generated locally. The coupled cases use environment seed 10.

## Inspect and prepare

```bash
python benchmark/scalable_study/run.py plan \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly

python benchmark/scalable_study/run.py generate-inputs --confirm \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly

python benchmark/scalable_study/run.py preflight \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly
```

Filters `--family`, `--case`, and `--path-index` can be repeated. Input generation stops at the first solution; the configured durations are caps. Imported paths must carry a strict full-edge certificate; legacy paths without one are regenerated locally.

## Run and analyze

```bash
python benchmark/scalable_study/run.py execute --confirm \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly

python benchmark/scalable_study/analyze.py
```

Execution is resumable. Existing valid cells are skipped, stale or invalid cells are refused, and `--force` explicitly replaces selected local artifacts. The full matrix has 1,269 shortcutting cells and 6.32 nominal shortcut-hours, excluding environment construction and input/output certification. Broad-case budgets are selected from the earlier three-method traces: cases already flat by one second receive two seconds, while active cases retain progressively longer 5, 10, 15, 20, or 30 second horizons. In particular, combinatorial group methods can overrun their nominal budget when one proposal itself is expensive; total wall time is retained separately.

Results retain the complete accepted cost/time/collision-check trace, proposer counters, collision-check timing, and accepted-shortcut lineage. Analysis produces broad scenario pages, full curves for all three scaling families, time and collision checks to 5%, 10%, and 20% relative cost improvements, and a reusable run-level CSV.

## Additive follow-up protocols

`followups.json` defines two additive experiments without modifying the frozen
three-method study or its results:

- Offline: remove the tree from `scalable_tree_group`, retaining its exact
  vectorized compatible-group sampler and `1e-6` improvement threshold. Reuse
  every certified scalable-method cell's path, shortcut seed, and time budget.
  This tests whether the tree adds value, independently of the older subset cache.
- Online pilot: spiral tower, car assembly, six-arm shared-table box stacking,
  and twelve disks. Three paired planner seeds (the three saved paths), one
  shortcut seed each, two policies: the actual upstream `run_choice="tree"`
  implementation and `scalable_tree_group`. This is 24 cells, not a 3x3 matrix.
  Each run gets a 200-second first-solution cap and 100 seconds after the first
  solution, with 100 collision checks per shortcut call. Reproduce the saved
  planner configuration, including random-scan Gibbs for the disk scene.

Online runs start planning from scratch; saved files supply seeds and planner
settings, not injected paths. The baseline's first unshortcutted internal path
is captured before the first shortcut. The candidate's first path is compared
against that online reference, configuration by configuration and mode by mode,
with matching cost and planner settings. It is not compared against the old
offline path, whose return representation adds mode-boundary states.
A mismatch fails the cell and stops the batch. Shortcut RNG is
isolated from planner RNG, and proposer state is recreated on each shortcut call.
The upstream implementation may finish a call beyond the deadline; raw overruns
are retained, but only events within the 100-second horizon count in comparison
curves. Planner construction, shortcutting setup, and reinsertion are charged.
Environment construction and final output certification are excluded.

```bash
python benchmark/scalable_study/run_followups.py plan --track offline \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/scalable_study/run_followups.py plan --track online \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly

# Only when ready and plugged in; these commands are intentionally separate.
python benchmark/scalable_study/run_followups.py execute --track offline --confirm \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/scalable_study/run_followups.py execute --track online --confirm \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
```

`plan` and `preflight` do not execute planners or shortcutters. `execute` freezes
a track-specific manifest of code/input/reference hashes under
`offline/followups/<track>/` (`online-v5` for the normalized, certified-incumbent online runner), skips matching certified results, and refuses stale
outputs. Tracks share a serial execution lock. Power loss or suspend detection
stops execution without promoting the interrupted candidate. This is not an
automatic overnight job: explicit execution is required. Do not run other
benchmarks concurrently. No blog figures or text are changed by these commands.

Use `execute --track online --confirm --smoke-pair` with the same repository
arguments to validate the first complete method pair at full budgets. Then
execute without `--smoke-pair` to resume the remaining cells, retaining that
pair. Failed v1 logs/manifests remain under `offline/followups/online/`.
