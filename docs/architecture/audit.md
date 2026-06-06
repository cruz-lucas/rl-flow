# Architecture Audit

This audit is intentionally direct. The project has a strong shape for a research workflow compiler, but several pieces need hardening before it should be treated as a full research platform.

## Strengths

Schema-driven UI
: Component JSON schemas drive forms. This keeps the frontend generic and lets researchers add modules in Python.

Python-owned RL semantics
: Validation, compilation, runners, and algorithms live in Python, close to the research code.

Deterministic compilation artifacts
: The compiler writes workflow, resolved config, Gin, command, status, and manifest files before execution.

Entry-point component discovery
: Third-party packages can expose `rlflow.components` without changing the core registry.

Local/SLURM executor split
: Local smoke tests and cluster-scale sweeps share the same compiled run model.

Run manifests
: File hashes, dependency versions, git state, platform, backend, seed, and sweep metadata are captured.

Seed-grouped sweep summaries
: Sweeps treat seed replicates as repeated measurements for one non-seed hyperparameter assignment.

## Gaps

Limited job recovery
: Local job state is not robustly reconstructed after API restart.

Minimal artifact browsing
: Artifacts are listed, but not typed, previewed, versioned, or linked into result cards.

MLflow is not integrated
: The optional dependency exists, but tracking integration is still an architecture stub.

Weak benchmark and result registry
: There is no stable benchmark suite, baseline registry, or paper-style result card format.

No formal plugin compatibility policy
: Component IDs and versions exist, but compatibility rules and migrations are not enforced.

Algorithm/runtime coupling
: Built-in runners and algorithm implementations are still tightly coupled, especially in the DQN path.

Partial provenance
: The manifest captures selected dependency versions, but not full lockfile state, dirty diffs, dataset versions, or hardware details.

No multi-user model
: Authentication, authorization, ownership, and shared lab storage policies are not in place.

## Architectural Direction

The most important next step is not adding more algorithms. It is turning experiment artifacts into durable research records. That means typed artifacts, benchmark metadata, complete provenance, reproducible examples, plugin contracts, and a job orchestration layer that can survive restarts and cluster realities.
