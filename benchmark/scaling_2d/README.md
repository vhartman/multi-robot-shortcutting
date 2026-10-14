# Disk-robot scaling with obstacles

This offline study uses 2, 4, 6, 8, 10, and 12 translating disk robots (2 DoF each, radius 0.12), ten rectangular obstacles in a fixed 4-by-4 arena, and four random goals per robot followed by a return home. Arena, obstacles, and home positions come from `make_random_two_dim` in the pinned motion-planning repository. Disk radius is reduced from the original 0.2 after an input-planning pilot. Robot joint names are made unambiguous because upstream substring matching confuses `a1` with `a10` and `a11`.

Every instance is a subset of one twelve-robot scene. Obstacles, robot sizes, starts, and each retained robot's goals remain identical as robots are added. Goal visits are interleaved in four rounds, with a reproducible random robot order per round. More robots therefore increase dimension, the number of mode transitions, and congestion together. This is a fixed-workspace study, not a constant-density study or a test of dimension alone. Goals are checked against the full home configuration; input planning and full-edge certification establish feasibility.

The planned protocol uses three initial paths and three shortcut seeds, with the same six methods as the box-stacking scaling study. Each shortcutter gets 15 seconds. The default 180-second input-generation budget is a cap; the 10-robot case uses a 900-second cap and random-scan Gibbs sampling. Generation stops at the first solution. Runs execute serially. Pilot with `--path-index 0 --shortcut-seed 0` before executing the full matrix. The full 324-cell shortcutting budget is 81 minutes, excluding construction and certification. A planning timeout does not establish that a scene is infeasible; missing input paths prevent the corresponding benchmark from running.

```bash
python benchmark/scaling_2d/validate_scenes.py --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_2d/run.py plan --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_2d/run.py generate-inputs --confirm --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_2d/run.py execute --confirm --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_2d/analyze.py
```

Run the scalable tree/group follow-up on all saved paths and seeds:

```bash
python benchmark/scaling_2d/run_scalable_tree_group.py \
  --robots 2 --robots 4 --robots 6 --robots 8 --robots 10 --robots 12 \
  --path-index 0 --path-index 1 --path-index 2 \
  --seed 0 --seed 1 --seed 2 \
  --output-root benchmark/scaling_2d/analysis/scalable-tree-group-final
python benchmark/scaling_2d/analyze_scalable_tree_group_final.py
```

Protocol and environment code are separate from the completed broad and arm-scaling study. Inputs, results, logs, and plots are ignored by Git. Saved results include time/cost/check traces, compatibility and acceptance counters, checking time, and accepted-shortcut lineage. Input and output paths must pass full-edge collision checks with zero penetration tolerance and preserve the task sequence and endpoints.

## Difficult input paths

Input generation accepts explicit runtime and planner-sampling overrides without changing the scene or invalidating previously saved inputs. New inputs record the actual runtime cap and the complete planner configuration. For example:

```bash
python benchmark/scaling_2d/run.py generate-inputs --confirm \
  --case rai-disk-scaling-r12-g4 --path-index 0 \
  --input-seconds 900 --input-stepsize 0.25 \
  --input-sampler gibbs_random_scan --input-home-bias 1 \
  --mr-bench-repo /path/to/mr_bench
```

Explicit Gibbs sampling uses the planner's existing Gibbs implementation for both exploration and transition sampling. Its default automatic selector can remain on joint rejection sampling; ordinary joint exploration attempts do not update the selector's statistics. Optional `--input-home-bias` affects transition samples, placing inactive robots at home with the specified probability. Input generation records sampling and collision-check time and uses a monotonic runtime deadline. The same saved input is supplied to every shortcutter. These inputs must be identified as using different planner settings when comparing scaling trends to inputs generated with the default sampler. Timed shortcutting runs should wait until the machine is on mains power.

The opt-in `gibbs_random_scan` variant in `benchmark/expanded/input_sampling.py` updates one uniformly selected unpinned robot per sample, starting from an existing tree configuration. Other robot coordinates remain unchanged. It uses the upstream Gibbs per-robot attempt limit and full-configuration collision checker, retains the previous coordinate on exhaustion, and rejects any invalid final state. This makes exploration more local than a full twelve-robot sweep; it does not imply uniform joint samples. With home bias 1, transitions use the checked goal-plus-home configurations, while exploration uses random-scan Gibbs. The saved input's `sampling_strategy` names and hashes this extension separately from the native planner configuration. No upstream planner source is modified, and existing input paths are preserved.
