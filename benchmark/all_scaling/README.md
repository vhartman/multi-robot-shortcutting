# All scaling comparison

This is the frozen offline comparison across every retained scaling family:

- isolated box stacking with 1--9 robots and four boxes per robot;
- planar disks with obstacles at 2, 4, 6, 8, 10, and 12 robots;
- coupled shared-table box stacking with 2 through 6 robots;
- the WAFR mobile-wall 7-by-7 robot/task-count grid.

To keep the complete scaling sweep near six hours, every scenario uses its
strictly certified path 0 and shortcut seeds 0 and 1. The compared methods are
tree + gain, full/tree/gain round robin, scalable tree + group, random robot
subset + compatible run, and full-configuration shortcutting. Disk cases use
their existing 15-second budget; all other cases use 30 seconds.

The runner reuses a result only if its input hash, method/policy, seed, budget,
cost definition, and output certificate match the current cell. Its path and
hash are frozen into `execution-manifest.json`. Missing cells are written under
this directory. Execution is strictly serial and stops on mains-power loss,
suspend/clock jumps, timeouts, or execution errors. A completed shortcut run
whose output fails the strict final certificate is recorded as a failed
replicate and is not retried; the remaining matrix continues. This avoids
selecting a favorable wall-clock cutoff while retaining the historical 0.5
shortcut interpolation resolution.

The protocol explicitly records both compatible core implementation hashes.
The earlier hash differs from the current one only by the injectable gain
evaluator used by the new scalable policy and by an equivalent strict
cost-improvement comparison; neither changes the reused policies.

```bash
python benchmark/all_scaling/run.py plan \
  --mr-bench-repo /path/to/mr_bench
python benchmark/all_scaling/run.py preflight \
  --mr-bench-repo /path/to/mr_bench
python benchmark/all_scaling/run.py execute --confirm \
  --mr-bench-repo /path/to/mr_bench
```

Repeatable family, case, method, path-index, and shortcut-seed filters are
available for inspection and pilots. A filtered execution still uses the same
full-study manifest, so later unfiltered execution resumes without mixing
protocols.

## Compatible-group schedule ablation

The follow-up post compares three schedules on the same 69 scaling cases,
path 0, seeds 0 and 1, and the same 15/30-second budgets:

- compatible random groups only (`robot_subset_mode`, reused from the frozen
  all-scaling comparison);
- exhaust the finite singleton tree, then use compatible random groups
  (`tree_then_compatible_group`);
- alternate singleton-tree and compatible-group proposals until the tree is
  exhausted, then continue with groups (`scalable_tree_group`, reused from the
  frozen comparison).

The additive runner never rewrites the two reused methods. It resumes only
missing tree-then-group cells and validates input/implementation hashes, cost,
mode sequence, endpoints, and the strict output certificate before reuse.

```bash
python benchmark/all_scaling/run_tree_then_group.py plan
python benchmark/all_scaling/run_tree_then_group.py execute --confirm --max-hours 6
python benchmark/all_scaling/analyze_tree_group_schedules.py
```

The analyzer writes linear and logarithmic curves, a four-family scaling
summary, scalar metrics, and explicit time-to-threshold reach counts under
`benchmark/all_scaling/analysis/tree-group-schedules/`. Raw timed results stay
under the ignored `offline/` data directory.
