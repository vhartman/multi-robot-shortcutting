# WAFR scaling extensions

This additive offline protocol brings the two larger WAFR scaling families into
the shortcutting benchmark without changing the frozen `scalable_study`
protocol or invalidating its completed results.

The mobile-wall matrix is the original 7-by-7 Cartesian grid:

- one through seven mobile manipulators;
- walls of 2x2, 3x2, 3x3, 4x3, 4x4, 5x4, and 5x5 blocks (4, 6, 9, 12, 16,
  20, and 25 blocks).

The independent-arm extension adds isolated box stacking with seven, eight,
and nine UR10s and four boxes per robot. Its one-through-six robot cells remain
in `benchmark/scalable_study`; the analysis joins the old and new curves rather
than rerunning them.

Every new case uses three independently planned first-solution paths and three
shortcut seeds. It compares the current standard (`tree_gain`), the previous
best (`tree_group_gain`), and `scalable_tree_group`. Each shortcutter gets 30
seconds. Input planning stops at the first solution; the 600-second mobile-wall
and 1,800-second isolated-arm values are caps. Runs are strictly serial.

Inspect the matrix without running anything:

```bash
python benchmark/scaling_mobile_wall/run.py plan \
  --mr-bench-repo /path/to/mr_bench
```

Generate and strictly certify first-solution paths, then preflight and execute:

```bash
python benchmark/scaling_mobile_wall/run.py generate-inputs --confirm \
  --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_mobile_wall/run.py preflight \
  --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_mobile_wall/run.py execute --confirm \
  --mr-bench-repo /path/to/mr_bench
python benchmark/scaling_mobile_wall/analyze.py --allow-partial
```

Use repeatable `--case`, `--method`, `--path-index`, and `--shortcut-seed`
filters for pilots or resuming selected cells. Generated inputs, results, logs,
audit records, and plots are ignored by Git. Result JSON retains accepted
cost/time/check traces, proposal and acceptance counts, collision-check timing,
and accepted-shortcut lineage for later analyses.
