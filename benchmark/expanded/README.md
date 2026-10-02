# Expanded offline benchmark

This follow-up is separate from the frozen four-environment final protocol and from the completed broad first pass. It contains:

- the original 30 heterogeneous scenarios, using their existing certified first-solution paths;
- isolated box stacking with one through six UR10 robots and exactly four boxes per robot;
- three shortcut seeds for every input;
- one initial path for each archived scenario and three independently planned initial paths for each scaling scenario;
- the four selected methods on every case, plus full-configuration and compatible random-subset sampling on the scaling cases.

Execution is strictly serial. Result JSON retains the full time/cost/check trace, aggregate attempt counters, collision-check time, and accepted-shortcut lineage. The analysis produces normalized cost-versus-time curves as well as final-cost, AUC, collision-check, acceptance, shortcut-length, and retention summaries.

## Prepare inputs

Adopt the original inputs without modifying the archive:

```bash
python benchmark/expanded/adopt_inputs.py \
  --source /path/to/mr-shortcutting-archive/benchmark/experiments/paper-comparison-full/inputs
python benchmark/expanded/adopt_inputs.py \
  --source /path/to/mr-shortcutting-archive/benchmark/experiments/paper-comparison-full/inputs \
  --write
```

Generate the 18 scaling inputs serially:

```bash
python benchmark/expanded/run.py generate-inputs --confirm \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
```

The 1,800-second input limit is a cap per input, not an intended runtime: planning stops at the first solution.

## Run and plot

```bash
python benchmark/expanded/run.py plan \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/expanded/run.py execute --confirm \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/expanded/analyze.py
python benchmark/expanded/analyze_metrics.py
```

The runner is resumable and refuses stale or invalid cells. Use `--case`, `--track`, `--method`, `--path-index`, or `--shortcut-seed` to run a subset. `analyze_metrics.py` writes reusable run-level and scenario-level CSV files alongside the figures. Generated inputs, logs, result JSON, audit records, and plots are ignored by Git.
