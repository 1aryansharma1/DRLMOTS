# DRLMOTS collaborator handoff

**Last updated:** 2026-10-13
**Current checkpoint:** v0.1 deterministic baseline, implementation commit `993344d`.
**Next task:** P4.3 - seeded synthetic workload generation for v0.2.
**Task ownership:** not assigned in the repository yet. Record the owner and status
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
```

Expected baseline: 194 tests pass; dependency check is clean; example completes three
tasks with zero failures and 4 ms makespan. The `.venv` is local, not in Git. Python
3.11.9 was used here; the project requires Python >=3.11,<3.13.

## Where the current behavior lives

| File / interface | Responsibility and constraints |
|---|---|
| `src/drlmots/entities.py` | Task/Host/VM, ResourceTier, TaskStatus; `Task.transition_to` enforces lifecycle rules |
| `src/drlmots/config.py` | YAML validation; nullable unresolved settings; optional paper experiment settings separate from small fixtures |
| `src/drlmots/topology.py` | AP/end/fog/cloud links and single host/VM node membership; no VM-to-physical-host assignment |
| `src/drlmots/workload.py` | `fixed_workload`, explicit-unit `processing_time_ms`, and input `workload_manifest` with SHA-256 |
| `src/drlmots/environment.py` | `simulate(topology, tasks, task_length_unit="bytes")`; returns immutable outcomes and completion metrics |
| `src/drlmots/__main__.py` | Fixed smoke fixture and `python -m drlmots` JSON report; not a configurable experiment runner yet |
| `tests/unit/` | Entities, topology, configuration and workload boundaries |
| `tests/integration/test_environment.py` | Exact timelines, fallback, overflow, same-time events, repeatability and accounting |
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

## Start P4.3

Build on `workload.py` and `tests/unit/test_workload.py`. Start with a small configurable
synthetic generator, not a guessed Google Jobs import. The current paper extraction only
says arrivals occur at random intervals; it does not establish their distribution.

Before implementing, record the selected distribution and its units in `assumptions.md`
as an implementation choice. Use a local seeded random generator, with explicit task
count, arrival parameters and positive byte-length/data-size ranges. Keep all unresolved
paper-specific values visible rather than adding purported paper defaults.

Acceptance criteria for this task:

1. Same seed and settings produce identical tasks and identical manifest/checksum.
2. Task IDs are unique; requested count is exact, including an explicitly supported
   zero-task case. Negative/fractional/bool counts and invalid ranges fail clearly.
3. Arrival timestamps are finite, nonnegative and ordered; byte lengths and data sizes
   are finite/positive and stay within the requested bounds.
4. Seed, distribution name, units, ranges and generator settings appear in the manifest.
   Changing an effective setting changes the recorded provenance.
5. A small generated workload completes or explicitly fails every task through the
   existing simulator; repeated runs agree. Existing deterministic tests still pass.
6. Record the actual test command/results and task owner/status in `progress.md`.
   Update this handoff with the next task; do not mark the whole version complete.

After P4.3, continue configurable topology construction and the explicit resource/
placement model. Keep each change small enough to review and document behavior changes.
Do not begin DQN training until state/action/reward definitions, metrics and simulator
prerequisites are verified. Source ambiguity should become a recorded assumption or
open item, never an undocumented formula or a claim of reproduced results.
