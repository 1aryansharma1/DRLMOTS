# DRLMOTS progress report

**Report date:** 2026-10-13 (Asia/Kolkata)
**Branch:** main
**Implementation checkpoint:** `e97a14d` - v0.2 configured topology;
prior seeded checkpoint `7ebf8d6`.
**Working status:** v0.1 complete; v0.2 in progress. P4.3/P4.6 seeded workload checkpoint
and configuration-driven topology are verified. Resource/placement behavior is next.

**Pickup status:** no active implementation task; next owner unassigned. The complete
ordered remaining-work list is [NEXT_STEPS.md](NEXT_STEPS.md), including the immediate
resource-feasibility acceptance criteria and later source/DQN/evaluation gates.

## Latest checkpoint - configured topology

Runnable experiments now load explicit hosts, VMs, positive waiting-queue capacities,
fog/cloud memberships and AP/end links from a separate strict YAML schema. The
workload contains explicit units, generator and all A-018 settings. Unknown or missing
fields, duplicate YAML keys and invalid topology references/tier memberships fail clearly.
`configs/default.yaml` remains an unresolved planning config; `configs/synthetic.yaml`
is the runnable fixture. A-019 records this distinction and the remaining limitations.

```powershell
.\.venv\Scripts\python.exe -m drlmots --config configs/synthetic.yaml --output outputs/metrics/configured-demo.json
```

Actual result: three hosts/VMs, two fog nodes and one cloud datacenter; six simultaneous
tasks, **6 completed, 0 failed, 4 ms makespan, 3 ms average completion time**. Exact
timestamps match the two-wave hand calculation. The saved effective config (excluding
its SHA-256 field) reproduces the same report when parsed again.

Effective config SHA-256:
`a4ad989a835c5719717d8e0fb15f025efb2d97429603222e7485e42162cc3720`.
Workload SHA-256:
`2e9bf785db9c7ef9ba06048c962dec49fc04894117a103d9913f6134e8b4d62d`.
Checks on 2026-10-13: **304 tests passed**; no broken requirements. Tests cover actual
CLI output, settings replay, queue-limit changes and overflow accounting, strict YAML
validation, ambiguous override rejection and unchanged existing fixtures.

This is the configurable portion of P5.1. Full paper topology, physical VM-to-host
placement, resource feasibility and transfer delays remain open; v0.2 is not complete.

## Previous checkpoint - seeded synthetic workloads

P4.3 adds explicit task count, seed, uniform interarrival-ms bounds, inclusive integer
byte lengths and uniform data-size-kB ranges. The local RNG does not change global
random state. P4.6 now records all settings, distribution/unit names, RNG identity,
generator/Python versions, exact tasks and their SHA-256. Zero-task and simultaneous
arrival cases are supported; invalid settings and accumulated-time overflow fail clearly.

Actual checks on 2026-10-13: **246 tests passed**; `pip check` found no broken requirements.
The original fixed fixture still yields three completions and 4 ms makespan.

```powershell
.\.venv\Scripts\python.exe -m drlmots --task-count 100 --seed 42 --output outputs/metrics/seeded-demo.json
```

This run accounted for all 100 inputs: **96 completed, 4 explicitly failed** because
all waiting queues were full. Makespan was **155.1273394932697 ms**. Its workload
SHA-256 was `b22d2e1654e5f8f0f4a5a79a201966d97e8033dba19b89f9f98239848122b231`
with the recorded Python 3.11.9/generator-v1 settings. The 2-VM topology is still the
fixed test fixture. These numbers verify execution/accounting, not paper performance.

P4.4-P4.5 remain open; no experiment-count conflict or real dataset mapping is resolved.
A-018 labels the synthetic distribution as an implementation choice. HANDOFF now
describes the next resource-feasibility task; the seeded checkpoint is retained here.

## Delivered in v0.1

The repository now has a runnable deterministic software baseline rather than only
task/resource models. It supports an explicit cloud/fog topology, task lifecycle rules,
a fixed synthetic workload, per-VM waiting queues, serial service within a VM, concurrent
service across VMs, and recorded completion outcomes.

| Work | Current state | Evidence |
|---|---|---|
| P3.3 topology | Verified | Unique resource IDs, parent references, tier membership, and no orphan hosts/VMs |
| P3.4 lifecycle | Verified | Legal transitions tested; invalid transitions and terminal restarts rejected |
| P3.5 validation | Verified for current model | Finite values, integer counts, DQN setting consistency, and separate 20-200 VM experiment range |
| P4.1 fixed workload | Verified | Exact fields, copied mappings, duplicate-ID rejection |
| P4.2 service-time conversion | Verified for byte-valued fixtures | 1000 bytes -> 2 ms; units must be explicit |
| P4.6 manifest | Verified for fixed-list inputs | Exact inputs, units, generator identity, null seed, SHA-256 |
| P5.2-P5.6/P5.8 execution | Verified for baseline scope | Exact timelines, FCFS queues, alternate-VM fallback, explicit overflow failure, event ordering and task accounting |

These are scoped software checks. They do not establish that whole Phases 4-5, the
paper's scheduling algorithm, or its reported experimental improvements are reproduced.

## Verification

The v0.1 checks below were rerun on 2026-10-13 using this checkout's Python 3.11.9
virtual environment before its publication. The latest suite is 304 tests above:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 194 passed
.\.venv\Scripts\python.exe -m pip check
# No broken requirements found.
```

The deterministic smoke example has also been executed and saved:

```powershell
.\.venv\Scripts\python.exe -m drlmots --output outputs/metrics/smoke.json
```

| Smoke metric | Result |
|---|---:|
| Configured VMs | 2 |
| Input tasks | 3 |
| Completed | 3 |
| Failed | 0 |
| Makespan from time zero | 4 ms |
| Average completion time, successful tasks | 8/3 ms |

Tests include a 100-task fractional-time fixture, repeatability, no input mutation,
no overlapping service on the same VM, no lost/duplicated outcomes, and completion
before same-time arrival. Generated JSON and local `.venv` are intentionally ignored
by Git; the command recreates the example. No training results or checkpoints exist.

## Remaining work and source limitations

- P4.4-P4.5: experiment-size handling and real dataset provenance; P4.3 is now verified.
- Full source-audited P5.1 paper topology: the configurable builder is verified, while
  the bundled example remains a tiny synthetic fixture.
- P5.7: a documented network-delay model; the baseline explicitly uses zero delay.
- Resource feasibility and VM-to-host allocation: not implemented. Nonempty task
  resource requests are rejected rather than treated as supported.
- Paper priorities, thresholds, state/action encoding, reward, energy/fault metrics,
  DQN training, checkpoints, baselines and comparative evaluation: pending.
- The original DRLMOTS source needs an independent audit against `paper_requirements.md`.
  The 16-page DRL-MLS Kubernetes PDF supplied in this session is a different paper;
  its GPU/resource/reward definitions have not been merged into DRLMOTS.
- The existing source extraction has unresolved units, rewards, training parameters,
  dataset identity and conflicting experiment settings. See assumptions A-001 through A-017.

## Next delivery

P4.3, its seeded manifest and the configurable topology builder are complete. Continue
v0.2 with explicit resource/placement behavior and its own
tests. DQN work remains a later milestone after these behavior and source decisions.

See [HANDOFF.md](HANDOFF.md) for concrete starting steps and acceptance criteria.
Update [progress.md](progress.md) when a task is claimed or actually verified.
