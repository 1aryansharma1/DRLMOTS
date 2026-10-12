# DRLMOTS collaborator handoff

**Last updated:** 2026-10-13
**Current checkpoint:** v0.2 seeded workload/configured-topology checkpoints VERIFIED;
latest verified implementation is `e97a14d` on `main`; v0.1 foundation is `993344d`.
**Next task:** explicit resource feasibility and placement behavior for v0.2.
**Task ownership:** both checkpoints completed by Codex for ashishjha20; next task unassigned. Record the owner and status
in `progress.md` when starting so another contributor can see what is in progress.

**Full remaining-work list:** [NEXT_STEPS.md](NEXT_STEPS.md). Start with its resource
feasibility checklist, then follow the ordered backlog. No implementation task is active.

## Read this first

1. [README.md](README.md): setup, runnable example, versions and baseline semantics.
2. [PROGRESS_REPORT.md](PROGRESS_REPORT.md): delivered functionality and latest checks.
3. [progress.md](progress.md): authoritative microtask state and resume point.
4. [assumptions.md](assumptions.md): open scientific questions; A-017 defines v0.1.
5. [implementation_plan.md](implementation_plan.md): version gates and full roadmap.
6. [paper_requirements.md](paper_requirements.md): recorded DRLMOTS source extraction.
7. [NEXT_STEPS.md](NEXT_STEPS.md): all remaining work through DQN and final evaluation.

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
.\.venv\Scripts\python.exe -m drlmots --config configs/synthetic.yaml --output outputs/metrics/configured-demo.json
```

Expected current checks: 304 tests pass; dependency check is clean; fixed example completes three
tasks with zero failures and 4 ms makespan. The seeded example completes 96 and rejects
4 because all queues are full, accounting for every input. Configured fixture: six
tasks on three VMs, all completed in 4 ms. The `.venv` is local, not in Git. Python
3.11.9 was used here; the project requires Python >=3.11,<3.13.

## Where the current behavior lives

| File / interface | Responsibility and constraints |
|---|---|
| `src/drlmots/entities.py` | Task/Host/VM, ResourceTier, TaskStatus; `Task.transition_to` enforces lifecycle rules |
| `src/drlmots/config.py` | YAML validation; nullable unresolved settings; optional paper experiment settings separate from small fixtures |
| `src/drlmots/topology.py` | Validated AP/end/fog/cloud links, single host/VM node membership and strict `build_topology`; no VM-to-host assignment |
| `src/drlmots/run_config.py` | `RunConfig`, strict `load_run_config`/`parse_run_config`, effective settings/checksum; duplicate-safe YAML |
| `src/drlmots/workload.py` | `SyntheticWorkloadSettings`, `seeded_workload`, `fixed_workload`, explicit-unit `processing_time_ms`, and `workload_manifest` with SHA-256/settings/runtime provenance |
| `src/drlmots/environment.py` | `simulate(topology, tasks, task_length_unit="bytes")`; returns immutable outcomes and completion metrics |
| `src/drlmots/__main__.py` | Fixed/seeded examples and `configured_report`; `--config` selects explicit topology/workload; resources not modelled yet |
| `tests/unit/` | Entities, topology, configuration and workload boundaries |
| `tests/integration/test_environment.py` | Exact timelines, fallback, overflow, same-time events, repeatability and accounting |
| `tests/integration/test_seeded_cli.py` | Seeded simulator accounting, actual CLI output files and invalid-input handling |
| `tests/unit/test_run_config.py`, `tests/integration/test_configured_run.py` | Strict schema/topology/YAML, effective config replay, exact configured timelines and CLI behavior |
| `configs/default.yaml` | Unresolved fields stay null; source-reported settings are recorded separately |
| `configs/synthetic.yaml` | Explicit runnable three-VM/six-task software fixture, not paper defaults |

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
file for invalid CLI inputs; existing baseline tests retained. That checkpoint passed
246 tests; the current configured-topology suite totals 304.

## Completed configuration-driven topology

Use `--config configs/synthetic.yaml` or `load_run_config`/`configured_report` to run
explicit lists/links/capacities and an explicit seeded workload. All schema fields are
required, unknown fields rejected, counts derived from lists and queue limits positive.
`configs/default.yaml` is not used as a runnable configuration. CLI seed/count/range
flags cannot be combined with a run file. A-019 documents this checkpoint's scope.

The effective config/checksum is recorded as `run_configuration`; remove only `sha256`
and feed that mapping to `parse_run_config` to reproduce the run. Formatting/local
pathname are excluded. The configured hand calculation and changed-queue-limit tests
pass; the total suite is 304 tests. Full paper topology and VM-to-host allocation are open.

## Start explicit resource feasibility

Currently `simulate` rejects every nonempty task `required_resources` mapping. Host
storage/memory fields and VM bandwidth are recorded, but no CPU/memory allocation is
modelled. Decide a minimal explicit test resource schema before enabling these requests;
do not infer per-VM capacities, CPU speeds or physical host budgets from missing data.
Record units, capacity semantics and the no-eligible-resource policy as a new assumption.

Acceptance criteria:

1. Task demands and per-VM capacities have matching explicit resource names and units;
   all values are validated. Unknown capacities cannot satisfy a positive request.
2. Placement selects only feasible VMs, preserves the documented queue/ordering rules,
   and records a clear outcome when no feasible VM or queue slot exists.
3. Resource reservations/release follow task start/completion, with no oversubscription
   or resource leak. Do not claim a physical host budget without a declared VM mapping.
4. Requirements/capacities/policy appear in configured-run provenance; impossible,
   exact-boundary and competing-task cases have hand-checked tests.
5. Existing empty-resource examples remain unchanged; every task is accounted for.
   Update reports, task owner and resume point with actual verification results.

Keep each resource/placement change
small enough to review and document behavior changes. v0.2 is not complete yet.
Do not begin DQN training until state/action/reward definitions, metrics and simulator
prerequisites are verified. Source ambiguity should become a recorded assumption or
open item, never an undocumented formula or a claim of reproduced results.
