# DRLMOTS progress report

## Latest verified delivery - 2026-10-25

Owner: Codex for Aravindak3 / Aravind. Multi-objective extension A-028 is VERIFIED
for its documented synthetic scope. Standard latency-only scheduling is retained.
Optional live reward contract 3 uses weighted latency, active/input energy and
analytical destination risk. Actual failure/backup/recovery simulation is still open.

**Verification:** 698 tests pass; final packaging test rerun passed; pip check clean.
Nine actual 20-episode models: 180 episodes / 18,000 training tasks / 17,062 optimizer
updates. Three independent model seeds, five fresh workload seeds 4100-4104 and three
load scenarios; 195 distinct evaluation simulations, represented in 315 paired rows.
All checkpoints round-tripped and evaluation left training fingerprints unchanged.

**Results:** normal-load fog-balanced DQN versus least queue: mean paired modeled
energy reduction 18.710%, successful mean latency reduction 4.163%; admitted-task
analytical failure risk is worse (11.547% relative increase). The fog-first heuristic
still performs better in normal/burst aggregate comparisons. Burst admission is only
12/100 across all policies; do not interpret low energy from rejection as efficiency.

**Reports:** outputs/efficiency/professor-fresh/efficiency-report.html and
 efficiency-report.zip (extract first), plus CSV/JSON/Markdown and 9 PNG/9 SVG plots.
All 86 local links, 96 ZIP entries/CRC, 45 paired trace groups and resource balances
verified; representative figures inspected and overlapping labels corrected. Normal
browser desktop/mobile layout remains a manual presentation check. Generated outputs
are ignored; share the ZIP or regenerate. Source/config/tests and the tracked results
summary provide the collaborator handoff.

Read [MULTI_OBJECTIVE.md](MULTI_OBJECTIVE.md) and
[MULTI_OBJECTIVE_RESULTS.md](MULTI_OBJECTIVE_RESULTS.md) first. Run:

~~~powershell
python -m drlmots.efficiency --config configs/rl_network.yaml --training-config configs/efficiency_training.yaml --objectives configs/objectives.yaml --output-dir outputs/efficiency/professor-fresh --model-seeds 42 43 44
~~~

Use --resume for matching existing output; choose a fresh folder for a different plan.
Next work: calibrated coefficients and real failure/backup behavior with its costs;
validation-only tuning and a new untouched test set; larger task/VM sweeps and paper
baseline/source fidelity. Independent initializations are implemented, but this fixed
budget study has no validation tuning or significance claim. No implementation task
is currently active.

## Earlier milestone context (2026-10-15)


**Report date:** 2026-10-15 (Asia/Kolkata)
**Branch:** main
**Implementation checkpoint:** v0.3.1 live DQN for both environments VERIFIED;
prior fog-first `2ad7e4b`;
prior Q-network `96a5407` and template fix `6f87c53`;
prior rewards `72255bf`;
normalization `29debfb`;
state/action foundation `a7a08bb`;
prior `2083faa` professor demo;
resource model `4f4307f`;
prior configured/seeded checkpoints `e97a14d` / `7ebf8d6`.
**Working status:** state/action, normalization, synthetic rewards and Q-network completed by Codex for **Aravindak3 /
Aravind**. Repository-local commit identity uses `195939576+Aravindak3@users.noreply.github.com`
and credential selection requests that account. v0.1/v0.2 and the prior professor
demo are verified. Contract-2 state/reward, live placement, replay, Adam/target updates,
full resume and held-out evaluation are complete. Standard and fog-preferred models
were actually trained. Next is an independent initialization/validation study.

**Pickup status:** no active implementation task; next owner unassigned. The complete
ordered remaining-work list is [NEXT_STEPS.md](NEXT_STEPS.md), including the immediate
multi-initialization/validation experiment and later source/model-fidelity gates.

**Git publication:** the prior state/action implementation/handoff was pushed by the
user and confirmed on main at `9604536`. For the latest checkpoint compare local HEAD
with GitHub's main reference. Authentication may require a normal user terminal when
the sandbox cannot run Git Credential Manager.

## Latest checkpoint - v0.3.1 end-to-end live CPU DQN

The Q-network now chooses actual VM placements during the event loop. Both the
original zero-network simulator and new upload/propagation model support immutable
contract-2 states with ordered queue readiness, legal masks and immediate causal
end-to-end rewards. Standard DQN chooses all legal VMs; fog-preferred DQN learns
inside an explicit completion preference. Reference rules and old contracts remain.

Bounded replay folds forced arrivals without fake actions; masked Bellman targets,
Smooth L1/Adam, finite clipped gradients and a separate frozen/synchronized target
network are implemented. Checkpoints contain weights, optimizer, replay, local RNG,
training-only scaler/provenance, configs and counters. Exact resume, immutable greedy
evaluation and seed/task overlap rejection are tested. Windows atomic-save lock retry
was exercised in the actual run; it resumed without restarting or refitting.

**677 tests pass**, including **59 new live-training tests**; pip check is clean.
Three actual models trained for 20 episodes each: standard-zero **1878** updates,
standard-network **1885**, fog-preferred-network **1920** (total **5683**). All changed
initial weights and reloaded full checkpoints. Held-out seeds 3000/3001/3002 were
evaluated without any update/refit; comparisons pair exact inputs within each model.

Mean completions per 100 tasks: zero-network least-queue 95.67, first-fit 93.00,
standard DQN 93.00; network least-queue 94.33, first-fit 93.00, fog-first rule 97.67,
standard DQN 94.33, fog-preferred DQN 96.00. These short runs do not beat every
heuristic and do not establish convergence or statistical robustness. Only one
initialization seed was used per model. Do not tune on the reported test seeds.

The static offline training report includes loss/return curves, paired results,
actual traces, resumable checkpoints and a ZIP with verified local links/integrity.
Five PNG/five SVG figures were generated; representative figures were inspected.
Browser layout remains a manual check. Full commands/results/contracts and the
next research gate are in LIVE_DQN.md/NEXT_STEPS.md. Generated outputs are ignored.

Collaborator documentation through main `5dffde0` was merged without discarding the
new scheduling work. Commit identity remains Aravind / Aravindak3.

## Previous checkpoint - v0.3.0a5 fog-first completion and actual upload timing

The requested policy compares eligible local fog and cloud predicted finishes and
prefers fog on equal or earlier finish, including busy fog with queue space. Schema-3
settings explicitly declare source end node, decimal kB/Mbps units, propagation and
independent per-VM uploads. Actual starts wait for both upload readiness and preceding
FCFS work; CPU/memory reserve at execution start. Queue slots include uploading tasks.
Every decision exports candidate timing estimates/reason and selected transfer data.

**618 tests pass in 48.53 seconds**, including 41 new network/policy tests; pip check
is clean. Busy fog, faster cloud, ties, variable input size, upload overlap/head-of-line
waiting, reachable fog membership, full queues, resources, event ordering, causality,
forecast/actual agreement, strict config/replay, CLI and demo generation are verified.
Both zero-network and network demos pass Node control/export checks (6/9 choices,
three exports each), as does raw-template startup. Network figures: 21 PNG/21 SVG;
representative charts inspected. Real browser layout remains a manual check.

Seed-42 network fixture: fog-first 65 completed/35 rejected, makespan
165.06162267920254 ms, successful mean 10.524262058797573 ms. Each baseline completes
62/rejects 38 on those same network settings. These are single synthetic observations,
not a paper result or proof of general superiority. Old zero-network 96/4 outcomes
remain unchanged under their own inputs; do not compare them across timing models.

Legacy RL state/reward/capture is guarded for network configs because uploading queues
need a new contract. Next version that contract before learning on the network model;
replay/targets, live agent control and training remain unfinished. See A-026 and
FOG_CLOUD_POLICY.md for units, assumptions and reproducible commands.

## Previous fix - opening the demo source template

The user opened `src/drlmots/templates/demo.html`, which contains the data placeholder,
and saw empty controls/charts. The source now shows a clear explanation, generation
command and relative link to `outputs/demos/drlmots-demo.html`; its dashboard is hidden
and its script exits safely without data. The generator marks the real report ready,
populates its embedded data and retains the existing controls/charts.

Verification: **577 tests pass**, including the new source-opening regression; both
source startup and all six generated selections/three exports pass the Node DOM-stub
checks. Regenerated the offline report and 15 PNG/15 SVG figures. Browser automation
was unavailable, so this is not a new browser-layout verification. Next implementation
task remains replay/target updates.

## Previous checkpoint - v0.3.0a4 CPU Q-network and legal selection

The configurable two-hidden-layer CPU float32 network derives input/action counts
from the schema. The resource fixture uses 47 -> 50/ReLU -> 50/ReLU -> 3 and 5,103
parameters. Explicit context binds full topology/schema, ordered actions, fitted
normalizer and synthetic reward identifiers. Equal shape alone is insufficient.
Forward rejects invalid dtype/device/bounds, incompatible context and nonfinite
parameters/outputs; gradients remain available for the next learning-update gate.

Local seeds reproduce initialization/exploration without changing caller CPU/Python
RNG. Greedy ties select the first legal index, exploration uses only legal indices,
and all-false masks return None. Strict checksummed JSON network snapshots validate
architecture, parameter names/shapes/numbers and context; same-runtime reload gives
exact Q-values. Training optimizer/replay/exploration state is not persisted yet.

Actual verification: **576 tests pass in 33.19 seconds**, including **66 new tests**;
dependency check clean. The CLI fitted 100 training observations and inferred on 100
held-out baseline observations: 92 legal selections/8 forced None selections, exact
checkpoint reload. The scaler and existing simulator/reward results are preserved.
Hand outputs/gradients, local seeds, illegal masks, context mismatches, malformed
snapshots, deterministic CLI and input-overwrite guards are verified.

The choices are untrained and were not executed in the simulator. The report makes
no learned-policy completion/return claim. A-025/Q_NETWORK.md record all architecture,
initialization and persistence assumptions. A-009/A-010 paper decisions remain open.
Next: bounded replay, masked Bellman targets, finite updates and target sync; then
interactive agent-controlled episodes and bounded CPU smoke training/evaluation.

## Previous checkpoint - v0.3.0a3 causal rewards and transition exports

Explicit `--reward-config configs/reward.yaml` enables decision capture and synthetic
placement scoring. An admitted task gets S/(S+projected wait+own service), using only
arrival-time queue/running work. Forced rejection gets -P. Fixture S=10 ms/P=1 are
documented choices; this does not implement paper Eq. (25), energy or reliability.
There is no completion/terminal bonus, and all-arrival return includes rejections.

One raw state/action/reward/next-state record is exported per arrival. Only the last
arrival is terminal. A separate pure learner adapter absorbs following forced costs
with explicit gamma per arrival, then bootstraps an actionable state or terminates.
Forced prefixes remain in episode metrics without invented actions. This is baseline
transition data and an adapter, not live agent control, replay or DQN training.

Actual verification: **510 tests pass**, including 50 new reward/transition tests;
dependency check clean. Six-task result remains 4 completed/2 rejected and 4 ms,
with return 23/21. The 100-task fixture remains 96/4, with 100 records, exactly one
terminal and return 67.73152495135943. Hand values, forecast/actual latency, causal
future extensions, normalizer preservation, empty/all-rejected runs, forced-chain
gamma/terminal cases, invalid configs/actions, replay and actual CLI are verified.
See A-024 and REWARD_CONTRACT.md for limits and the future learning boundary.

## Previous checkpoint - v0.3.0a2 training-only normalization

The frozen scaler fits only explicitly training-labelled raw traces. It scales
numeric columns by their training maxima, excluding unknown resource amounts;
binary flags pass through. Zero/unseen columns use scale 1 and held-out outliers
clip to 1 before division, keeping even subnormal/huge inputs finite. A-023 and
NORMALIZATION.md document these approximations and caller split responsibilities.

Separate fit/apply commands save/load JSON with maxima/counts/scales, full feature
schema, canonical topology hash, training configuration/workload/generator/software
provenance and checksum. Apply never fits or changes saved statistics. Schema,
units, capacities, queues, tiers and links must match; pure list permutations are
compatible. Held-out training-source/exact-task overlap and malformed artifacts
are rejected. Outputs retain the raw simulator report and add normalized vectors.

Actual verification: **460 tests pass**, including 41 normalization tests; `pip check`
is clean. Named seed-42 training fixture fits 100 observations; seed-314159 held-out
fixture transforms 100 47-feature observations into [0,1], retaining 92 completed/
8 rejected tasks. Training remains 96/4. Hand ratios, zero/constant/unknown columns,
binary-unit spelling, extreme values, immutable fit, repeated save/load, actual CLI,
equal-dimension incompatible contexts and input overwrite protection pass.
These are software checks; there is still no learned policy, reward or DQN training.

## Previous checkpoint - v0.3.0a1 initial state/action contract

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
preview before presenting. At that checkpoint, state/action design was next;
the current delivery and pickup task are recorded above.

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
virtual environment before its publication. The latest suite is 677 tests above:

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
by Git; the command recreates the example. No training results or training checkpoints
exist; current untrained network snapshots are described above.

## Remaining work and source limitations

- P4.4-P4.5: experiment-size handling and real dataset provenance; P4.3 is now verified.
- Full source-audited P5.1 paper topology: the configurable builder is verified, while
  the bundled example remains a tiny synthetic fixture.
- Paper P5.7: source-audit the implemented synthetic A-026 upload model and decide
  whether contention, downloads or other link details are justified.
- Source-audited resource values and physical VM-to-host allocation remain unresolved.
  Explicit synthetic per-VM feasibility/reservation is implemented, with no inferred host budget.
- Paper priorities, thresholds, source-audited state/action encoding, reward, energy/fault metrics,
  full paper comparators/evaluation: pending. Synthetic DQN training/checkpoints and
  the reference policies are now implemented. Two simple
  synthetic policies and paired demo evaluation are now verified under A-021.
- The original DRLMOTS source needs an independent audit against `paper_requirements.md`.
  The 16-page DRL-MLS Kubernetes PDF supplied in this session is a different paper;
  its GPU/resource/reward definitions have not been merged into DRLMOTS.
- The existing source extraction has unresolved units, rewards, training parameters,
  dataset identity and conflicting experiment settings. See assumptions A-001 through A-027.

## Next delivery

v0.2's scoped baseline and the initial v0.3 state/action implementation are delivered.
Training-only normalization, synthetic rewards/transitions and CPU Q-network/selection are verified;
Fog-first/network-aware and standard live DQN training are verified. Next is the
predeclared multi-initialization/validation study in NEXT_STEPS.md; original
paper-specific tasks remain open until independently verified.

See [HANDOFF.md](HANDOFF.md) for concrete starting steps and acceptance criteria.
Update [progress.md](progress.md) when a task is claimed or actually verified.
