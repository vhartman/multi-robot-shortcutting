# Broad offline first pass

This protocol extends the final four-environment comparison to the eleven remaining manifest cases. It is deliberately a screening pass: one previously generated certified initial path per case, three shortcut seeds, and three simple methods (compatible random robot, tree + gain, and tree + group gain). The four final environments are not rerun here.

Generated inputs, results, logs, analysis, and the execution audit are ignored by Git.

```bash
python benchmark/broad/adopt_inputs.py --source /path/to/paper-comparison-full/inputs
python benchmark/broad/adopt_inputs.py --source /path/to/paper-comparison-full/inputs --write
python benchmark/broad/run.py plan \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/broad/run.py execute --confirm \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
python benchmark/broad/analyze.py
```

The runner is strictly serial and resumable. Existing valid cells are skipped. Invalid cells are never silently replaced; inspect the log and rerun the identical cell with `--force` only after fixing its cause.

