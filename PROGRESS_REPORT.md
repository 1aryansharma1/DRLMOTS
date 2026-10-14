# DRLMOTS progress report

**Report date:** 2026-10-14 (Asia/Kolkata)
**Branch:** main
**Implementation checkpoint:** `a7a08bb`, v0.3.0a1 initial state/action step VERIFIED;
prior `2083faa` professor demo;
resource model `4f4307f`;
prior configured/seeded checkpoints `e97a14d` / `7ebf8d6`.
**Working status:** initial state/action step completed by Codex for **Aravindak3 /
Aravind**. Repository-local commit identity uses `195939576+Aravindak3@users.noreply.github.com`
and credential selection requests that account. v0.1/v0.2 and the prior professor
demo are verified. Training-only normalization is the next unassigned implementation
task; rewards, transitions and DQN training remain pending.

**Pickup status:** no active implementation task; next owner unassigned. The complete
ordered remaining-work list is [NEXT_STEPS.md](NEXT_STEPS.md), including the immediate
normalization acceptance criteria and later source/DQN/evaluation gates.

**Publication at handoff:** local commits on main; push failed because the Windows
sandbox could not run Git Credential Manager's authentication helper. Remote main
was still `62e0202`. Run `git push origin main` in a normal Aravindak3-authenticated
terminal and verify publication; author identity is already correct on the commits.

## Latest checkpoint - v0.3.0a1 initial state/action contract

New frozen observation structures detach arrival-time queues, running service,
capacities and ledger balances from runtime state. The observation boundary follows
all same-time completions and queued starts, before each arrival's placement.
Simultaneous arrivals retain input order and see earlier placements. Raw encoding
uses sorted VM/resource columns, explicit unknown flags and documented units; its
dimension is `3+R+V*(6+4R)`. A schema hash identifies the feature/action contract.

Stable VM indices and componentwise masks match total-capacity fit plus waiting-slot
availability, including busy-but-queueable VMs. All-false masks retain explicit
failure and a null action/VM. Opt-in `--capture-decisions` records observations,
vectors, masks and selected actions; ordinary reports remain compact. A-022 and
STATE_ACTION_CONTRACT.md document the provisional synthetic scope.

Final evidence: **419 tests passed**, including all 41 new tests, dependency imports
and figure generation; `pip check` is clean. The resource
fixture has six 47-element states, least-queue actions 0,1,0,1,null,null, four completions,
two rejections, 4 ms makespan and restored balances. Trace replay, both policies,
unknown-vs-zero, invalid snapshots/actions, immutable snapshots, event boundaries,
baseline preservation and actual CLI pass. The professor demo was regenerated and
the offline DOM-stub selections/exports check passed again.

The old Store Python 3.11 environment was inaccessible inside this session. It was
preserved and a separate ignored `.venv312` created using supported Python 3.12.14.
The fresh environment uses the same pinned dependencies. A corrupt PyTorch download
was rejected, and its replacement matched PyPI's published SHA-256; stale generated
package metadata was regenerated through setuptools. The runtime version changes workload provenance hashes, not the shared config checksum
or scheduling outcomes. No reward, normalizer, learned selector, step/transition API,
Markov sufficiency or paper-faithful state representation is claimed.

## Previous checkpoint - v0.2.1 professor demo

The user requested a practical professor demonstration before continuing learning.
The offline report now compares least queue and first fit on identical workloads and
resource constraints; it shows explicit failures, timings, VM occupancy, execution
timelines, outcomes and provenance. Selectors switch precomputed runs. No learned
policy, physical cloud deployment or paper reproduction claim is made.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m drlmots.demo --config configs/resources.yaml
```

| Preset | Least queue done/failed | First fit done/failed |
|---|---|---|
| Worked example (6 simultaneous tasks) | 4 / 2 | 4 / 2 |
| Seeded workload (100 tasks, seed 42) | 96 / 4 | 91 / 9 |
| Queue pressure (25 simultaneous tasks) | 4 / 21 | 4 / 21 |

Seeded successful makespan: 155.1273394932697 ms versus 154.5713394932697 ms;
average completion time: 3.5399653116546155 versus 3.757690860833562 ms. First fit
rejects more work, so a smaller makespan alone is not better performance. These are
one-seed synthetic observations. Input hashes match and all resources are restored
in each pair. A-021 documents policies, presets and successful-task metric semantics.

Generated artifacts: self-contained `outputs/demos/drlmots-demo.html`, full
`comparison.json`, 15 SVG and 15 PNG figures. They are ignored by Git and regenerated
from committed source. PROFESSOR_DEMO.md includes setup, a five-minute talk track,
metric explanations, likely questions and exact limitations. The optional dependency
lock extends the unchanged base lock; the wheel packages the HTML template.

Actual verification: **378 tests passed**, clean dependency check, valid SVG/tooltips,
safe embedded JSON for HTML-like identifiers, paired resource/accounting/replay,
hand-calculated placement timelines and actual generator CLI. Representative timeline,
comparison and occupancy PNGs were visually inspected. Node's offline DOM-stub check
passed all six selections and three JSON exports; it does not verify layout. The
browser tool disallowed local-file preview, so desktop/mobile layout needs manual
preview before presenting. No task is active; state/action design remains next.

## Previous checkpoint - v0.2.0 resource-aware baseline

Tasks now use explicit per-VM capacity eligibility. A shared unit catalogue describes
every named resource; capacities and demands are finite/positive. Missing capacity is
unknown, never inferred from host fields. Feasible VMs retain the least-queue/VM-ID rule;
no feasible VM and full feasible queues produce distinct recorded terminal failures.

Running tasks reserve their demands, queues reserve none, and completions release
before queued starts or same-time arrivals. JSON includes demands/capacities/units,
policies, reserve/release events and final balances. Schema 2 preserves all constraints
for replay. Schema 1 and generator-v1 workload/config fingerprints remain compatible.

```powershell
.\.venv\Scripts\python.exe -m drlmots --config configs/resources.yaml --output outputs/metrics/resources-demo.json
```

Actual result: **4 completed, 2 failed** (full feasible queues), **4 ms makespan**,
**3 ms average completion time**, **8 reservation/release events**, all per-VM balances
restored. The third VM is too small and receives no demanded task. Config SHA-256:
`76e284813d5c9166d3fef29c454ce08d28d10acaf549dbc9a0b9bc84ec2b520f`.

Checks: **367 tests passed**; `pip check` clean; package metadata refreshed to **0.2.0**.
Coverage includes exact-fit, impossible/missing capacities, unknown units, full feasible
queues, fallback, release ordering, empty workloads, heterogeneous 100-task accounting,
independent input maps and replay/provenance sensitivity. The existing configured
six-task result and seed-42 96/4 outcome/fingerprints remain unchanged.

This completes v0.2's synthetic software milestone. CPU units are admission units,
not a speed model; service time remains the byte-valued test formula. Physical host
allocation, paper-specific priority/tier thresholds, source audit, rewards and DQN remain
open. Assumption A-020 separates this approximation from original paper requirements.

## Previous checkpoint - configured topology

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
placement, resource feasibility and transfer delays were open at that checkpoint;
resource feasibility is now implemented in the explicit synthetic model above.

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
virtual environment before its publication. The latest suite is 419 tests above:

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
- Source-audited resource values and physical VM-to-host allocation remain unresolved.
  Explicit synthetic per-VM feasibility/reservation is implemented, with no inferred host budget.
- Paper priorities, thresholds, source-audited state/action encoding, reward, energy/fault metrics,
  DQN training/checkpoints and full paper comparators/evaluation: pending. Two simple
  synthetic policies and paired demo evaluation are now verified under A-021.
- The original DRLMOTS source needs an independent audit against `paper_requirements.md`.
  The 16-page DRL-MLS Kubernetes PDF supplied in this session is a different paper;
  its GPU/resource/reward definitions have not been merged into DRLMOTS.
- The existing source extraction has unresolved units, rewards, training parameters,
  dataset identity and conflicting experiment settings. See assumptions A-001 through A-022.

## Next delivery

v0.2's scoped baseline and the initial v0.3 state/action implementation are delivered.
Next is training-only normalization/schema persistence. Metric/reward/source decisions must pass
their gates before DQN work. Keep original paper-specific tasks open until verified.

See [HANDOFF.md](HANDOFF.md) for concrete starting steps and acceptance criteria.
Update [progress.md](progress.md) when a task is claimed or actually verified.
