# DRLMOTS implementation workspace

This workspace is for an incremental, test-first implementation of the DRLMOTS cloud–fog scheduling approach in the paper:

> Choppara, P. and Mangalampalli, S. “An efficient deep reinforcement learning based task scheduler in cloud-fog environment.” *Cluster Computing* 28:67 (2025). DOI: 10.1007/s10586-024-04712-z.

## Current status

Only the planning/specification documents exist in this starter pack. No implementation is claimed. The current task is to verify the local development environment before choosing versions/install commands.

## Documents

- `paper_requirements.md` — system extraction, equations, hyperparameters, paper gaps, and traceability matrix R001–R040.
- `implementation_plan.md` — phased execution roadmap with microtask IDs and verification gates.
- `assumptions.md` — decisions that must remain explicit and configurable.
- `progress.md` — verified state and next microtask.

## Core approach

1. Use SimPy to model arrivals, queues, service execution and task completion.
2. Implement a small deterministic environment first.
3. Derive and test task/VM state and action mapping from the source, recording unresolved interpretations.
4. Implement the DQN with replay memory, a target network and epsilon-greedy exploration as specified by Algorithm 1.
5. Verify metric calculations on hand-worked examples before any long training run.
6. Add evaluation and approximate baseline variants only with explicit labels if the paper does not describe them fully.
7. Save configs, random seeds, logs, metrics and figures for every experiment.

## Reproducibility caveat

The paper leaves some implementation-critical details undefined and contains apparent inconsistencies in reward semantics, fault tolerance, energy units, training repetitions, task counts and reported percentage improvements. These are itemized in `paper_requirements.md` and `assumptions.md`. The implementation will document these limitations and will not fabricate results or claim exact reproduction without evidence.

## Proposed runtime commands

Exact setup and run commands will be added only after the operating system, Python version and environment are verified. Do not run guessed install commands yet.
