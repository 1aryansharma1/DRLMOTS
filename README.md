# DRLMOTS implementation workspace

This workspace is for an incremental, test-first implementation of the DRLMOTS cloud–fog scheduling approach in the paper:

> Choppara, P. and Mangalampalli, S. “An efficient deep reinforcement learning based task scheduler in cloud-fog environment.” *Cluster Computing* 28:67 (2025). DOI: 10.1007/s10586-024-04712-z.

## Current status

Version 0.1 is a runnable deterministic scheduling baseline. It includes validated
Task/Host/VM models, fog/cloud topology, task lifecycle transitions, finite configuration
validation, a fixed synthetic workload factory, and a SimPy execution loop. The example
completes three tasks on two VMs with a 4 ms makespan and an 8/3 ms average completion
time. All 194 tests passed on Python 3.11.9 on 2026-10-13.

This version verifies software behavior. It does not implement DQN training or reproduce
the paper's experimental results. The source supplied during this iteration was the
different DRL-MLS Kubernetes paper (DOI 10.1016/j.future.2026.108459); its algorithm has
not been merged into the DRLMOTS model. Before making paper-fidelity claims, review the
original DRLMOTS paper against the existing extraction.

## Documents

- `paper_requirements.md` — system extraction, equations, hyperparameters, paper gaps, and traceability matrix R001–R040.
- `implementation_plan.md` — phased execution roadmap with microtask IDs and verification gates.
- `assumptions.md` — decisions that must remain explicit and configurable.
- `progress.md` — verified state and next microtask.
- [PROGRESS_REPORT.md](PROGRESS_REPORT.md) — dated delivery report, verification results, and remaining work.
- [HANDOFF.md](HANDOFF.md) — collaborator starting point, module map, and the next task's acceptance criteria.

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

## Setup and run on Windows

Run these commands from the repository root. Python 3.11 is verified; the package
supports Python >=3.11,<3.13. Keep installation inside the local virtual environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m drlmots --output outputs/metrics/smoke.json
```

Expected checks: no broken requirements; 194 passing tests. The last command prints
JSON and saves the exact topology, workload, input checksum, queue rules, per-task
start/finish times and metrics. The fixture is deterministic, so repeated runs produce
identical JSON. Generated results are ignored by Git.

If Python 3.11 is unavailable, inspect `py -0p` and install a supported Python version
before creating the environment. If the package cannot be imported, run the editable
installation command using the same `.venv` interpreter. Do not use system Python 3.13.

## Baseline semantics and limits

- This fixture uses explicitly byte-valued task lengths: 1000 bytes take 2 ms under
  the recorded Eq. (8) interpretation. The original table's unitless length of 700,000
  remains unresolved. Task data sizes are a separate input; no relationship is inferred.
- Every VM executes one task at a time. Its positive, explicit queue capacity counts
  waiting tasks only. `None` is unknown and cannot be treated as unlimited.
- Placement chooses the eligible VM with the fewest running plus waiting tasks,
  breaking ties by VM ID. A VM's waiting tasks execute first-come, first-served.
- Simultaneous arrivals preserve input order. Completions release capacity before
  arrivals at the same time. All queues full means an explicit failed task.
- Makespan is the last successful finish from simulation time zero, including arrival
  gaps. Average completion time includes waiting and service for successful tasks.
  Failed tasks are counted separately; an empty workload has no average.
- The example's small topology and queue limits are test fixtures. Paper experiment
  settings are separate in `configs/default.yaml`, including the 20-200 VM range.
- VM-to-host allocation, CPU/memory feasibility, network delay, priority/threshold
  placement, energy, reliability, reward and DQN are not yet implemented. Nonempty
  task resource requirements are rejected rather than ignored. Network delay is zero.
- The API copies input tasks and keeps caller-owned VM queues unchanged, allowing
  repeated runs on the same fixture.

## Versions

| Version | Completion target | Status |
|---|---|---|
| 0.1 | Deterministic topology, queue simulation, exact timelines and saved inputs/results | VERIFIED |
| 0.2 | Configurable seeded synthetic workloads and an explicit resource/placement model | Next |
| 0.3 | Documented state/action/reward definitions, DQN CPU smoke training and checkpoints | Planned |
| 1.0 | Source audit, matched experiments and honest evaluation against the selected paper | Planned; source gaps remain |

See `implementation_plan.md` for microtask mapping and `progress.md` for actual
verification evidence. No later version is complete merely because a smoke run passes.
