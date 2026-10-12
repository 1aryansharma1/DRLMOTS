# DRLMOTS collaborator handoff

**Last updated:** 2026-10-13
**Current checkpoint:** v0.2 P4.3/P4.6 seeded workload generation VERIFIED;
v0.1 foundation is commit `993344d`.
**Next task:** configuration-driven topology construction for v0.2 (configurable P5.1).
**Task ownership:** P4.3 completed by Codex for ashishjha20; the next task is unassigned. Record the owner and status
in `progress.md` when starting so another contributor can see what is in progress.

## Read this first

1. [README.md](README.md): setup, runnable example, versions and baseline semantics.
2. [PROGRESS_REPORT.md](PROGRESS_REPORT.md): delivered functionality and latest checks.
3. [progress.md](progress.md): authoritative microtask state and resume point.
4. [assumptions.md](assumptions.md): open scientific questions; A-017 defines v0.1.
5. [implementation_plan.md](implementation_plan.md): version gates and full roadmap.
6. [paper_requirements.md](paper_requirements.md): recorded DRLMOTS source extraction.

The target of the existing repository plan is DRLMOTS cloud/fog scheduling, DOI
10.1007/s10586-024-04712-z. The recently supplied PDF describes DRL-MLS Kubernetes
scheduling, DOI 10.1016/j.future.2026.108459. They are separate papers. v0.1 continues
the existing repository plan; an independent original-source audit is still pending.
Do not treat the attached paper's GPU/action/reward definitions as DRLMOTS requirements.

## Get a working checkout

For a new checkout, clone `https://github.com/1aryansharma1/DRLMOTS.git`, enter its
directory, and follow the setup commands in README. For an existing checkout, inspect
`git status` first, preserve your local changes, then fetch and fast-forward `main` when
appropriate. Do not reset someone else's changes.

From a configured repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m drlmots --output outputs/metrics/smoke.json
.\.venv\Scripts\python.exe -m drlmots --task-count 100 --seed 42 --output outputs/metrics/seeded-demo.json
```

Expected current checks: 246 tests pass; dependency check is clean; fixed example completes three
tasks with zero failures and 4 ms makespan. The seeded example completes 96 and rejects
4 because all queues are full, accounting for every input. The `.venv` is local, not in Git. Python
3.11.9 was used here; the project requires Python >=3.11,<3.13.

## Where the current behavior lives

| File / interface | Responsibility and constraints |
|---|---|
| `src/drlmots/entities.py` | Task/Host/VM, ResourceTier, TaskStatus; `Task.transition_to` enforces lifecycle rules |
| `src/drlmots/config.py` | YAML validation; nullable unresolved settings; optional paper experiment settings separate from small fixtures |
| `src/drlmots/topology.py` | AP/end/fog/cloud links and single host/VM node membership; no VM-to-physical-host assignment |
| `src/drlmots/workload.py` | `SyntheticWorkloadSettings`, `seeded_workload`, `fixed_workload`, explicit-unit `processing_time_ms`, and `workload_manifest` with SHA-256/settings/runtime provenance |
| `src/drlmots/environment.py` | `simulate(topology, tasks, task_length_unit="bytes")`; returns immutable outcomes and completion metrics |
| `src/drlmots/__main__.py` | Fixed topology; fixed or CLI-configured seeded workloads and JSON reports; topology/resources not configurable yet |
| `tests/unit/` | Entities, topology, configuration and workload boundaries |
| `tests/integration/test_environment.py` | Exact timelines, fallback, overflow, same-time events, repeatability and accounting |
| `tests/integration/test_seeded_cli.py` | Seeded simulator accounting, actual CLI output files and invalid-input handling |
| `configs/default.yaml` | Unresolved fields stay null; source-reported settings are recorded separately |

Preserve these tested semantics unless a documented task intentionally changes them:

- A VM runs one task and has a separately counted waiting queue with explicit capacity.
- Placement is least running-plus-waiting count; VM ID breaks ties. This is a baseline,
  not DQN or the paper's threshold policy. All configured tiers are eligible.
- Simultaneous arrivals preserve input order. Completions precede same-time arrivals.
- All eligible queues full produces a FAILED outcome with a reason, not a lost task.
- Task lengths must explicitly use bytes in this model; timestamps/durations use ms.
  Data size is a separate field. Do not interpret the paper's unitless 700,000 value.
- Inputs remain unchanged across runs. Resource requirements must be empty until
  feasibility is actually implemented; queue_capacity=None is unknown, not unlimited.
- Network delay is zero. No energy, reliability or reward result is available yet.

## Completed P4.3 checkpoint

`seeded_workload(SyntheticWorkloadSettings(...))` returns tasks. Pass the same settings
as `generator_settings` to `workload_manifest` to record random-workload provenance.
The settings explicitly contain task_count, seed, interarrival_ms_range,
length_bytes_range and data_size_kb_range. A-018 documents the uniform synthetic model;
the original paper distribution/dataset identity remain unresolved.

All previous P4.3 acceptance criteria are verified: local RNG without global mutation;
repeatable tasks/checksums; zero tasks; unique IDs; validated finite bounds and counts;
ordered arrivals; seed/settings/versions recorded; actual CLI execution and no output
file for invalid CLI inputs; existing baseline tests retained. The suite totals 246 tests.

## Start configuration-driven topology

The CLI currently constructs its topology directly in `smoke_report`; do not imply that
`configs/default.yaml` controls it. Add an explicit runnable configuration and builder
using the existing topology invariants. Keep the unresolved paper configuration separate
from a small software fixture; do not silently invent VM/host distributions.

Acceptance criteria:

1. Caller-configured AP/end/fog/cloud links and host/VM memberships construct a valid
   `Topology`; counts/IDs/capacities match the input exactly.
2. Missing counts or required capacities fail clearly; unknown is not unlimited.
   References, duplicate IDs and wrong-tier/orphan resources still fail.
3. No source-ambiguous unit, distribution or mapping is presented as a paper default.
4. The CLI can run a seeded workload using that topology and saves exact configuration,
   input settings/checksums and results. Fixed/seeded baseline examples still work.
5. A tiny configured topology yields hand-calculated timelines, and reruns agree.
   Report actual test results and update progress/owner/resume point.

After configurable topology, add an explicit resource/placement model. Keep each change
small enough to review and document behavior changes. v0.2 is not complete yet.
Do not begin DQN training until state/action/reward definitions, metrics and simulator
prerequisites are verified. Source ambiguity should become a recorded assumption or
open item, never an undocumented formula or a claim of reproduced results.
