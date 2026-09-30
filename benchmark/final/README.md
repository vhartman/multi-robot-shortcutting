# Final experiment protocol

This directory is the single source of truth for the final experiments behind
the multi-robot shortcutting article. The protocol was frozen on 2026-09-30.
No final experiment has been run from this directory yet.

## Frozen design

- 15 offline cases and 14 online cases; mobile strut is offline-only.
- Seven methods, exactly matching the article's experiment section.
- Three independently planned initial paths per case.
- Three shortcut RNG seeds per method and path (the full 3 x 3 product).
- Strictly serial execution; concurrent cells are outside the protocol.
- The regularized makespan cost used by the planner:
  `max(robot Euclidean edge lengths) + 0.01 * sum(robot Euclidean edge lengths)`,
  summed over path edges.
- Offline case-specific wall-clock budgets of 10, 20, or 30 seconds.
- Online calls receive 250 collision checks. Each run provides 100 seconds
  after its first solution; the extra 15-second replay margin is discarded by
  analysis.
- Online shortcut RNG state is isolated from planner RNG state.
- Planner-only controls run once per initial path.

The complete specification, including case budgets, seeds, aliases, validation
rules, aggregation, revisions, and output paths, is in
[`protocol.json`](protocol.json). Do not infer settings from an older
experiment directory.

The matrix contains:

| phase | cells |
|---|---:|
| input generation | 45 |
| offline shortcutting | 945 |
| online methods | 882 |
| online planner-only controls | 42 |

## Reproducibility contract

Every input is an unshortcutted first solution produced with
`planner_seed = environment_seed + path_index` and `PYTHONHASHSEED=0`.
Every method sees the same certified input for a given case and path index.

Online runs replay that planner seed. A result is accepted only if its first
solution has the same environment fingerprint and cost as the saved input.
Every offline and online output must pass a deterministic full-edge collision
certificate, preserve the terminal mode, and pass the native plan validator
when one exists. Offline outputs additionally preserve both endpoints and the
complete mode sequence.

Failed cells are never replaced with another seed or path. Their log remains
the audit record. After an infrastructure fix, rerun the identical cell.

The content lock hashes the protocol, runners, implementation, manifest, and
dependency file. Execution refuses to start when a locked file changes. After
an intentional reviewed change, regenerate it with:

```bash
python benchmark/final/lock_protocol.py --write
```

This is a protocol revision, not routine setup.

## Required source revisions

- Python 3.10.12.
- `mr_bench` revision
  `3142dacfbfc22d619f52d69d07164eacae21b26c`.
- Cross assembly revision
  `db09fada24e47ab19148f75574df84703b14b068`.
- Car-mini assembly revision
  `10a7ff87aa6940b765ff3bcf88ee17a366d67534`.

Pass `--mr-bench-repo` and `--assembly-repo` to `run.py`, or set
`MR_BENCH_REPO` and `ASSEMBLY_REPO`. The assembly checkout must expose
worktrees at both required revisions. No local checkout path is assumed.

The required `mr_bench` commit is not yet available from the public remote.
Publish that exact commit before relying on the direct dependency URL in
`pyproject.toml`. A local checkout at that revision is sufficient in the
meantime, and the content lock keeps local final runs exact.

## Inspect without running

These commands do not invoke a planner or shortcutter:

```bash
python benchmark/final/run.py plan
python benchmark/final/run.py preflight \
  --mr-bench-repo /path/to/mr_bench \
  --assembly-repo /path/to/assembly
python benchmark/final/validate.py --allow-incomplete --no-write \
  --mr-bench-repo /path/to/mr_bench --assembly-repo /path/to/assembly
```

`run.py` has no implicit execution mode. Experiments start only through the
`execute --confirm` subcommand.

## Execute later

Use the same Python environment for every command and run the phases in order:

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

The runner is resumable: a valid existing cell is skipped. An invalid existing
artifact stops that cell rather than being silently trusted. Use `--force`
only for an intentional rerun of the same frozen cell. Filters such as
`--case`, `--method`, `--path-index`, and `--shortcut-seed` are
available for recovery and small checks.

Do not launch multiple runner instances. The protocol intentionally exposes no
jobs option.

## Aggregation

Costs are normalized independently by each initial path. At every plotted time,
the three shortcut seeds are first reduced to a median within one initial path.
Those three path-level curves are then reduced to a median. The shaded range is
the minimum to maximum of the three path medians. Thus, an initial path—not an
individual shortcut run—is the statistical unit.

Offline time starts when shortcutting starts. Online time is rebased to the
first solution and truncated at 100 seconds. The analysis writes a JSON/CSV
summary for all cases and linear/logarithmic four-case figures under
`analysis/`.
