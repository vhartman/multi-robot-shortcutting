# Final four-environment experiment

This directory is the frozen four-environment comparison from the first post.
The later broad and scaling studies have separate protocols; see the
[benchmark index](../README.md). The lock is historical: do not regenerate it
just to accept later implementation changes. Use the first-post revision
specified in the root README when reproducing this exact protocol.

## Design

- Environments: Random 2D, box rearrangement, spiral tower, and car assembly.
- Three independently planned initial paths per environment.
- Three shortcut RNG seeds per method and path: nine runs per method/environment.
- Strictly serial execution.
- Offline budgets: 20 seconds for Random 2D and 30 seconds for the other cases.
- Online budget: the prior-work setup—100 collision checks per shortcut invocation and 30 analyzed seconds after the first solution. A 5-second replay margin is run but discarded.
- No uncharged final shortcut pass.
- Each run is normalized by its own initial/first-solution cost. We first take the pointwise median over the three shortcut seeds for each initial path, and then the median over the three path-level curves. Shading spans the minimum to maximum path-level median.

The eight methods listed in the blog are:

1. random robot + compatible run;
2. random robot subset + compatible run;
3. WAFR single robot;
4. Huang round robin without prioritization;
5. tree + gain (the current standard);
6. tree + group gain;
7. gain only;
8. full/tree/gain round robin.

The first four already have complete 3×3 prior-work data. The final run computes only the last four.

## Matrix

| artifact | complete comparison | adopted prior work | new run |
|---|---:|---:|---:|
| certified inputs | 12 | 12 | 0 |
| offline method cells | 288 | 144 | 144 |
| online method cells | 288 | 144 | 144 |
| planner-only controls | 12 | 12 | 0 |

The missing offline budget is 1.1 hours. Missing online refinement is 1.2 hours; first-solution replay and the five-second margins bring the nominal total to roughly 3.3 hours. Allow about four hours of serial wall time.

## Adopt the completed prior-work cells

The importer does not run a planner or shortcutter. It verifies schemas, revisions, seeds, fingerprints, initial costs, budgets, recorded certificates, online collision-check budgets, and the post-solution horizon before writing adopted copies into the ignored final-data directories. Validate first, then write:

```bash
python benchmark/final/adopt_prior_work.py \
  --source /path/to/prior-work-multipath
python benchmark/final/adopt_prior_work.py \
  --source /path/to/prior-work-multipath --write
```

The expected source directory contains `inputs/`, `shortcut-only/results/`, and `planner-loop/results/`. Adopted artifacts record their original schema and SHA-256 hash.

## Inspect without running

```bash
python benchmark/final/run.py plan
python benchmark/final/run.py preflight \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly
python benchmark/final/validate.py --allow-incomplete --no-write \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly
```

After adoption, `plan` must report 12 valid inputs, 144 valid and 144 missing offline cells, and 156 valid and 144 missing online cells.

## Execute later

Nothing runs without `execute --confirm`. The runner is strictly serial and skips every valid adopted artifact.

```bash
python benchmark/final/run.py execute --phase offline --pending-only --confirm \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly
python benchmark/final/run.py execute --phase online --pending-only --confirm \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly
python benchmark/final/validate.py \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly
python benchmark/final/analyze.py
```

Do not launch multiple runner instances. A failed cell is never replaced by a different seed or path; fix infrastructure and rerun the identical cell. `--force` is only for an intentional replacement of that same frozen cell.

## Reproducibility contract

The exact cost, RNG rules, aliases, source revisions, validation rules, and output names are in [`protocol.json`](protocol.json). [`protocol.lock.json`](protocol.lock.json) hashes the protocol, runners, maintained shortcutters, manifest, and assembly fixtures. Regenerate the lock only after an intentional protocol change:

```bash
python benchmark/final/lock_protocol.py --write
```

Required revisions are Python 3.10.12, `mr_bench` `3142dacfbfc22d619f52d69d07164eacae21b26c`, and assembly `10a7ff87aa6940b765ff3bcf88ee17a366d67534` for car assembly. Checkout paths are supplied through the CLI or `MR_BENCH_REPO` and `ASSEMBLY_REPO`; none are embedded in the runners.
