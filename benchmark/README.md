# Benchmark index

Only source definitions, runners, and assembly fixtures belong in Git.
Generated inputs and results are local artifacts; they are not included with
the source repository.

## Published studies

| Study | Entry point | Paths × shortcut seeds |
|---|---|---|
| First post, four environments | [final](final/README.md), historical revision `7aef53d` | 3 × 3 |
| Second post, broad offline comparison | [scalable_study](scalable_study/README.md) | 3 × 3 |
| Second post, offline scaling | [scaling_pilot](scaling_pilot/README.md) | 3 × 2 |
| Second post, sum-of-path-lengths cost | [scaling_pilot](scaling_pilot/README.md#sum-cost-replication) | 3 × 2 |
| Second post, online scaling | [scaling_pilot](scaling_pilot/README.md#online-comparison) | 1 × 2 |

The offline scaling matrix contains six methods and 27 cases:

- Disks: 2, 4, 6, 8, 10, and 12 robots; ten obstacles, four goals per robot.
- Isolated stacking: 1–9 robots; four boxes per robot.
- Coupled stacking: 2–6 robots; eight shared boxes on one table.
- Mobile wall: 1–7 robots; a fixed 5×5 wall with 25 bricks.

Published curves use nested medians: seeds within each initial path, then
across paths. Shading is the range of path-level medians, not a confidence
interval. The final offline replication has 970/972 certified max-cost cells
and 143/144 certified sum-cost cells. All series are retained, with five
instead of six successful runs in three method/scenario groups. Rejected
cells remain rejected; `--successful-only` is an explicit plotting policy,
not a change to collision validation.

## Historical dependencies and exploration

`expanded`, `all_scaling`, and the individual scaling directories contain
earlier protocols and reusable environment definitions. The final studies
import these definitions and may refer to their certified results. Their
larger matrices and old labels are historical, not instructions to repeat
every experiment. Do not rename policy IDs or alter frozen JSON merely for
presentation: both are part of the provenance hashes.

The extra `expanded/*trial*`, gain-order/sum-cost pilot scripts, and
`online_scheduling` are retained exploratory source, not part of the published
matrix. The latter includes notes for later planner work and depends on
unpublished planner changes. Its results and profiling outputs are ignored.

## Before running

Install the package as described in the [root README](../README.md). Use a
separate, clean planner checkout at the study's pinned revision; do not point a
timed reproduction at a development branch with local planner changes.
Assembly tasks also need the pinned assembly checkout. Inspect `plan` and
`preflight` first. Execution is serial, with explicit confirmation; avoid
concurrent benchmarks and laptop sleep.

Fresh runs generate inputs locally. Saved inputs/results can be reused only
when their identity, hashes, budget, and certificates match. Merely having the
same seed or filename is not sufficient. Old output directories must not be
silently overwritten or re-certified during cleanup.

For exact first-post source locks, use revision `7aef53d`; the current shared
engine contains subsequent scaling changes. Final scaling workers and their
frozen manifests are kept unchanged when packaging the scalable policies.
