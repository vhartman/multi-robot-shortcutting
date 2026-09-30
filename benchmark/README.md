# Benchmark

The benchmark is intentionally split into source definitions and generated artifacts. Only definitions and the two assembly source fixtures belong in Git.

- [`manifest.json`](manifest.json) defines the 15 environments and deterministic environment seeds.
- [`fixtures/assembly`](fixtures/assembly/) contains the task plans and exports needed to reconstruct the two assembly environments.
- [`online.py`](online.py) inserts any registered shortcutter into the planner's normal shortcut-and-reinsert hook.
- [`final`](final/README.md) contains the frozen 3-initial-path × 3-shortcut-seed offline and online protocol.

Inputs, results, logs, plots, and analysis products are generated under `final/` and ignored by Git. Installation, the public shortcutter interface, and the runnable commands are documented in the repository [`README`](../README.md).
