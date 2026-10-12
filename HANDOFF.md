# DRLMOTS collaborator handoff

**Last updated:** 2026-10-13
**Current checkpoint:** v0.2.0 synthetic resource-aware baseline VERIFIED;
latest verified implementation is `4f4307f`; v0.1 foundation is `993344d`.
**Next task:** v0.3 state/action contract, runtime observations and encoding (P6.1/P7.2).
**Task ownership:** v0.2 completed by Codex for ashishjha20; next task unassigned. Record the owner and status
in `progress.md` when starting so another contributor can see what is in progress.

**Full remaining-work list:** [NEXT_STEPS.md](NEXT_STEPS.md). Start with its state/action
checklist, then follow the ordered backlog. No implementation task is active.

## Read this first

1. [README.md](README.md): setup, runnable example, versions and baseline semantics.
2. [PROGRESS_REPORT.md](PROGRESS_REPORT.md): delivered functionality and latest checks.
3. [progress.md](progress.md): authoritative microtask state and resume point.
4. [assumptions.md](assumptions.md): open scientific questions; A-017/A-020 define baseline scopes.
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
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m drlmots --output outputs/metrics/smoke.json
.\.venv\Scripts\python.exe -m drlmots --task-count 100 --seed 42 --output outputs/metrics/seeded-demo.json
.\.venv\Scripts\python.exe -m drlmots --config configs/synthetic.yaml --output outputs/metrics/configured-demo.json
.\.venv\Scripts\python.exe -m drlmots --config configs/resources.yaml --output outputs/metrics/resources-demo.json
```

Expected current checks: 367 tests pass; dependency check is clean; fixed example completes three
tasks with zero failures and 4 ms makespan. The seeded example completes 96 and rejects
4 because all queues are full, accounting for every input. Configured fixture: six
tasks on three VMs, all completed in 4 ms. Resource fixture: four complete/two full-feasible-
queue failures, 4 ms makespan, eight resource events, all balances restored. The editable
install refreshes version metadata to 0.2.0 after pulling. `.venv` is local, not in Git. Python
3.11.9 was used here; the project requires Python >=3.11,<3.13.

## Where the current behavior lives

| File / interface | Responsibility and constraints |
|---|---|
| `src/drlmots/entities.py` | Task demands, optional VM resource capacities, Host/VM and lifecycle; amount validation/copying |
| `src/drlmots/resources.py` | Declared unit catalogue, `fits_resources`, one-running-task ResourceLedger and reserve/release events |
| `src/drlmots/config.py` | YAML validation; nullable unresolved settings; optional paper experiment settings separate from small fixtures |
| `src/drlmots/topology.py` | Validated AP/end/fog/cloud links, single host/VM node membership and strict `build_topology`; no VM-to-host assignment |
| `src/drlmots/run_config.py` | Strict schemas 1/2, effective config/checksum, unit/profile validation and replay; duplicate-safe YAML |
| `src/drlmots/workload.py` | `SyntheticWorkloadSettings`, `seeded_workload`, `fixed_workload`, explicit-unit `processing_time_ms`, and `workload_manifest` with SHA-256/settings/runtime provenance |
| `src/drlmots/environment.py` | `simulate(..., task_length_unit="bytes", resource_units=...)`; feasible placement, live local queues/running state, ledger, outcomes/events/final balances |
| `src/drlmots/__main__.py` | Fixed/seeded/resource examples; `--config`, units/policies, resource events/balances and JSON provenance |
| `tests/unit/` | Entities, topology, configuration and workload boundaries |
| `tests/integration/test_environment.py` | Exact timelines, fallback, overflow, same-time events, repeatability and accounting |
| `tests/integration/test_seeded_cli.py` | Seeded simulator accounting, actual CLI output files and invalid-input handling |
| `tests/unit/test_run_config.py`, `tests/integration/test_configured_run.py` | Strict schema/topology/YAML, effective config replay, exact configured timelines and CLI behavior |
| `tests/unit/test_resource_*.py`, `tests/integration/test_resource_*.py` | Capacity/unit boundaries, ledger invariants, heterogeneous execution, actual CLI and resource replay |
| `configs/default.yaml` | Unresolved fields stay null; source-reported settings are recorded separately |
| `configs/synthetic.yaml` | Explicit runnable three-VM/six-task software fixture, not paper defaults |
| `configs/resources.yaml` | Schema-2 capacities/units/demand profile; two feasible VMs and a third too small |

Preserve these tested semantics unless a documented task intentionally changes them:

- A VM runs one task and has a separately counted waiting queue with explicit capacity.
- Placement filters componentwise resource-feasible VMs with a waiting slot, then uses
  least running-plus-waiting count; VM ID breaks ties. This is a baseline,
  not DQN or the paper's threshold policy. All configured tiers are eligible.
- Simultaneous arrivals preserve input order. Completions precede same-time arrivals.
- No VM satisfying demands and full feasible queues produce distinct FAILED reasons.
- Task lengths must explicitly use bytes in this model; timestamps/durations use ms.
  Data size is a separate field. Do not interpret the paper's unitless 700,000 value.
- Inputs remain unchanged across runs. Every capacity/demand uses the same explicitly
  declared per-run unit catalogue. Missing/null capacity is unknown, not unlimited.
  Queue limits stay explicit. Only running tasks reserve; completion releases before
  queued starts and same-time arrivals. CPU admission units do not change service speed.
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
246 tests; the configured-topology checkpoint passed 304. The latest suite passes 367.

## Completed configuration-driven topology

Use `--config configs/synthetic.yaml` or `load_run_config`/`configured_report` to run
explicit lists/links/capacities and an explicit seeded workload. All schema fields are
required, unknown fields rejected, counts derived from lists and queue limits positive.
`configs/default.yaml` is not used as a runnable configuration. CLI seed/count/range
flags cannot be combined with a run file. A-019 documents this checkpoint's scope.

The effective config/checksum is recorded as `run_configuration`; remove only `sha256`
and feed that mapping to `parse_run_config` to reproduce the run. Formatting/local
pathname are excluded. The configured hand calculation and changed-queue-limit tests
passed with 304 tests at that checkpoint. The latest suite is 367; full paper topology
and VM-to-host allocation remain open.

## Completed v0.2 resource feasibility

Schema 2 adds run-level `resource_units`, each VM's `resource_capacities`, and workload
`required_resources`. Existing schema-1 examples remain compatible, including input
fingerprints. Generated tasks have a constant explicit profile; hand-built workloads
can vary task demands. A-020 records the componentwise admission, serial execution,
queue and resource accounting semantics; there is no physical host budget inference.

The 367-test suite verifies exact-fit/oversized/unknown capacity cases, distinct failures,
fallback, reserve/release ordering, no negative balance or leak, empty/profile provenance,
heterogeneous 100-task accounting and replay. This completes the version's synthetic
scope, not the paper's priority/threshold, physical allocation or full Phase 7 requirements.

## Start v0.3 state/action design

Write the P6.1/P7.2 contract and a new assumption before choosing features. Define one
decision-time observation for an arriving task after all same-time completions and
before placement. Queues/running flags and ledger balances live inside `simulate`;
input VM queues are intentionally unchanged, and final report balances are post-run.
Expose a real runtime snapshot rather than treating either as a live observation.

Acceptance criteria:

1. Feature list, order, units, dimensions and runtime snapshot boundary are documented.
   Keep missing paper fields visible; unknown capacity is distinguishable from zero.
2. VM action ordering is explicit/stable; action index maps to exactly one configured VM.
3. Legal actions match the resource/queue model: fitting total capacity allows queueing
   behind a running task even when currently available capacity is reserved; full waiting
   queues and infeasible VMs are masked. No valid action has an explicit outcome.
4. A pure encoder returns fixed-shape finite values for the configured run. Hand-built
   empty/busy/full/unknown snapshots and same-time event tests agree with the contract.
5. Before later learning, normalization uses documented training-only rules and avoids
   evaluation leakage. Save feature/action schema in provenance and keep baseline
   outcomes/resource balances unchanged. Update reports with actual verification.

Keep each change small enough to review and document behavior changes. v0.2 is complete
for the synthetic baseline scope; v0.3 is not started.
Do not begin DQN training until state/action/reward definitions, metrics and simulator
prerequisites are verified. Source ambiguity should become a recorded assumption or
open item, never an undocumented formula or a claim of reproduced results.
