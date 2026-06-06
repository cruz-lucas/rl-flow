# Roadmap

This roadmap is about making `rl-flow` a proper research framework, not only a workflow editor.

## Near-Term

- Keep this docs site current with generated CLI and component references.
- Add golden example checks for RiverSwim, Navix DQN, DQN + R-Max, sweep compilation, and learning-curve export.
- Add richer artifact pages in the UI for histories, metrics, checkpoints, replay datasets, and generated plots.
- Improve failure surfaces: expose last stderr line, status history, run command, and manifest links in the UI.
- Add explicit docs for expected result-reporting practice, including metric windows, seeds, and sweep grouping.

## Research-Grade

- Define benchmark suites with stable workflow IDs, environment versions, baseline configs, and accepted metric definitions.
- Add result cards that bundle config, metric summary, plots, provenance, seed count, and links to run directories.
- Add dataset and artifact registries with typed metadata, checksums, and provenance links.
- Add plugin templates for environments, agents, intrinsic rewards, replay buffers, loggers, and analysis modules.
- Capture full lockfile state, dirty diffs, hardware information, and dataset versions in run manifests.
- Add compatibility rules for component versions and workflow migrations.

## Framework-Grade

- Stabilize a public component API with semantic compatibility expectations.
- Decouple algorithm implementations from runner mechanics so new algorithms can share execution infrastructure cleanly.
- Add robust job orchestration with restart recovery, scheduler polling, cancellation semantics, and failure classification.
- Support additional backends through a backend interface, such as cloud batch systems and Kubernetes.
- Implement tracking integrations, including MLflow or a first-party experiment database.
- Add publishable report exports for sweeps, benchmarks, and individual result cards.

## Non-Goals for the Current Documentation Pass

- Do not implement the architecture improvements yet.
- Do not redesign the UI.
- Do not add new algorithms solely for documentation coverage.
- Do not replace the existing CLI/API workflows.
