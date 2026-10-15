# DRLMOTS implementation workspace

This workspace is for an incremental, test-first implementation of the DRLMOTS cloud–fog scheduling approach in the paper:

> Choppara, P. and Mangalampalli, S. “An efficient deep reinforcement learning based task scheduler in cloud-fog environment.” *Cluster Computing* 28:67 (2025). DOI: 10.1007/s10586-024-04712-z.

## Current status

Version **0.3.1** delivers actual end-to-end CPU DQN: live VM placement, replay,
masked targets, optimizer/target updates, resumable checkpoints and frozen held-out
evaluation. Both standard zero-network and network-aware environments are retained;
standard DQN and an explicit fog-preferred variant are available. All earlier rules,
normalization/reward APIs and baseline demos remain available.
It supports fixed/seeded
workloads, configured fog/cloud topology, explicit per-VM resource capacities and task
demands, feasible placement, queue overflow, resource reservation/release, and replayable
JSON reports and optional arrival-time observations, raw vectors and legal VM masks.
The current suite passes **677 tests** with the demo dependencies on Python 3.12.14
(2026-10-15); the dependency check is clean.
See [STATE_ACTION_CONTRACT.md](STATE_ACTION_CONTRACT.md) and the presentation guide
[PROFESSOR_DEMO.md](PROFESSOR_DEMO.md).

The original fixed example still completes three tasks with a 4 ms makespan. The
resource example accounts for six simultaneous tasks: four complete and two fail when
all feasible waiting queues are full; makespan is 4 ms and all capacities are restored.
v0.2 is complete for this explicitly documented software scope. The first v0.3
state/action, normalization, live control, replay and DQN training are implemented.
Three models were each trained for 20 episodes and evaluated on three held-out workload
seeds. This is a tested synthetic prototype, not proof of optimality or paper reproduction.
See [LIVE_DQN.md](LIVE_DQN.md) for the complete command, actual results, contracts and resume workflow.
See [Q_NETWORK.md](Q_NETWORK.md) for the preserved legacy inference/checkpoint API.
See [FOG_CLOUD_POLICY.md](FOG_CLOUD_POLICY.md) and `configs/fog_cloud.yaml` for the
network-aware rule. Legacy contract-1 capture stays guarded on network runs;
the new live contract 2 supports both environments. Schema-1/2 behavior is preserved.

This version implements synthetic DQN training and verifies software behavior; it does
not reproduce the paper's experimental results. The source supplied during this iteration was the
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
- [NEXT_STEPS.md](NEXT_STEPS.md) — complete remaining-work checklist, pickup order, open decisions, and handoff rules.
- [PROFESSOR_DEMO.md](PROFESSOR_DEMO.md) — offline report, five-minute walkthrough and honest answers to likely questions.
- [STATE_ACTION_CONTRACT.md](STATE_ACTION_CONTRACT.md) — observation timing, feature units/order, stable actions and trace replay.
- [NORMALIZATION.md](NORMALIZATION.md) — training-only fitting, held-out application, scaling rules and artifact compatibility.
- [REWARD_CONTRACT.md](REWARD_CONTRACT.md) — causal scoring, all-arrival returns, terminal boundaries and forced-chain adaptation.

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

Expected checks: no broken requirements; 677 passing tests with the demo extra
(674 pass/three figure tests skip without it). The last command prints
JSON and saves the exact topology, workload, input checksum, queue rules, per-task
start/finish times and metrics. The fixture is deterministic, so repeated runs produce
identical JSON. Generated results are ignored by Git.

If Python 3.11 is unavailable, inspect `py -0p` and install a supported Python version
before creating the environment. If the package cannot be imported, run the editable
installation command using the same `.venv` interpreter. Do not use system Python 3.13.
After pulling v0.2 into an existing editable environment, rerun the editable install
command to refresh package version metadata. The report should identify version 0.3.1.

In the agent session on 2026-10-14, the Microsoft Store Python 3.11 interpreter was
inaccessible. A separate ignored `.venv312` was created using the bundled Python
3.12.14; `.venv` was preserved. You can use any supported 3.11/3.12 installation.
For this checkout's verified 3.12 environment, replace `.venv` in commands with
`.venv312`. Workload hashes include Python version, so compare them within the same
runtime; the input config checksum and scheduling results are unaffected.
If tests see stale editable-package version metadata in `src/drlmots.egg-info`,
regenerate it using the same interpreter:
`python -c "from setuptools import setup; setup(script_args=['egg_info'])"`.

## Arrival-time states and legal actions (initial v0.3)

```powershell
.\.venv\Scripts\python.exe -m drlmots --config configs/resources.yaml --capture-decisions --output outputs/metrics/decisions.json
```

The opt-in `decision_trace` contains schema/feature units, raw finite vectors,
resource/queue masks and chosen action indices/VM IDs. Every arrival, including a
rejection, gets a detached immutable snapshot after same-time completions and queued
starts, before placement. VM/resource columns are sorted and stable for the run.
Unknown capacity has an explicit known flag; a busy VM may accept a waiting task
that fits total capacity even when its available resources are zero. The resource
example has 47 features, six decisions and actions `0,1,0,1,null,null` for least queue.

Ordinary reports remain compact; capture does not change outcomes, timings or input
hashes. See STATE_ACTION_CONTRACT.md for Python APIs and exact ordering. Vectors are
unscaled in the raw trace. The optional normalization command below adds scaled
vectors. Synthetic rewards and baseline transition records are opt-in below; there
is no learned selector, interactive step API or DQN yet.

## Training-only normalization (P6.3)

```powershell
.\.venv312\Scripts\python.exe -m drlmots.normalization fit --train-config configs/normalization_train.yaml --output outputs/metrics/normalizer.json
.\.venv312\Scripts\python.exe -m drlmots.normalization apply --config configs/normalization_evaluation.yaml --normalizer outputs/metrics/normalizer.json --output outputs/metrics/normalized-evaluation.json
```

The training fixture fits maxima from 100 observations (seed 42); held-out seed 314159
uses the saved artifact without refitting. Numeric values scale into [0,1], clipping
above the training range. Binary flags stay unchanged; unknown resource amounts stay
zero with their known flags. The saved JSON includes counts/maxima, training provenance,
feature schema, full topology fingerprint and checksum. Apply rejects incompatible
topologies/catalogues, malformed artifacts and declared training-task overlap.

The held-out report retains its raw decisions, masks, timings and resource evidence
and adds `normalized_state`. It still completes 92/100 and rejects 8; no learned-policy
performance is claimed. See NORMALIZATION.md for the scale-1 fallback, clipping,
subnormal values, library APIs and caller responsibilities for split declarations.

## Synthetic rewards and transitions (P8/P11 subsets)

```powershell
.\.venv312\Scripts\python.exe -m drlmots --config configs/resources.yaml --reward-config configs/reward.yaml --output outputs/metrics/reward-example.json
```

Reward uses only pre-placement work: `S / (S + projected_wait_ms + service_ms)` for
admission, `-P` for forced rejection. The explicit fixture uses S=10 ms and P=1;
these are software choices, not paper Eq. (25). Scoring implies decision capture and
preserves all placements/timings/resource balances. The six-task example still gives
4 completions/2 rejections and 4 ms makespan; return is 23/21 across all six arrivals.

The report includes one reward/raw transition per arrival. Only the final arrival is
terminal. Forced rejections have null actions; they never become fake VM choices.
`build_learning_transitions(decisions, gamma=...)` absorbs intervening forced costs
with the correct arrival discount power before the next legal-action bootstrap.
See REWARD_CONTRACT.md for prefix/terminal handling and the provisional reward's limits.
There is no interactive agent, replay buffer or DQN training in this checkpoint.

## Professor demo and paired comparisons

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m drlmots.demo --config configs/resources.yaml --output-dir outputs/demos
Start-Process .\outputs\demos\drlmots-demo.html
```

This self-contained HTML displays three precomputed paired scenarios, task timelines,
VM occupancy, outcomes and checksums; it works offline. The command also exports full
JSON and 15 SVG/15 PNG figures. The optional demo lock extends the unchanged base lock;
Matplotlib is only required for figure generation. Alternatively install `-e '.[dev,demo]'`
for compatible unpinned dependencies. Generated outputs stay ignored and are regenerated
by each collaborator. Read PROFESSOR_DEMO.md before presenting.

Both policies use resource/queue eligibility and FCFS VM queues. `least_queue_fcfs`
(default) chooses minimum running-plus-waiting count, then VM ID; `first_fit_fcfs`
chooses minimum eligible VM ID. These are synthetic software comparators, not trained
agents or the paper's CNN/LSTM/GGCN baselines. For the seed-42 100-task resource preset,
least queue completes 96 and rejects 4; first fit completes 91 and rejects 9. Timing
must be read alongside these counts, because successful tasks alone determine timings.
These single-seed observations do not establish general performance improvements.

The normal simulator CLI also accepts `--policy first_fit_fcfs`. For replay, call
`configured_report(parse_run_config(saved_config_without_sha), policy=report["policy"])`;
the policy is recorded separately from the shared input configuration checksum.

## Seeded synthetic workloads (v0.2 checkpoint)

An explicit task count and seed select random generation. Without synthetic options,
the CLI retains the original fixed three-task example. These seed/count CLI examples
use the fixed two-VM fixture with one waiting slot per VM. Use `--config` below to
select a different explicit topology.

```powershell
.\.venv\Scripts\python.exe -m drlmots --task-count 100 --seed 42 --output outputs/metrics/seeded-demo.json
.\.venv\Scripts\python.exe -m drlmots --task-count 10 --seed 7 --interarrival-ms 1 3 --length-bytes 500 2000 --data-size-kb 20 150
```

The synthetic model places the first arrival at 0 ms and samples subsequent intervals
uniformly. Lengths are integer-uniform bytes; data sizes are independently uniform kB.
These are explicit implementation choices in A-018, not claims about the paper's
distribution. Software-demo range defaults are 0-3 ms, 500-2000 bytes, and 20-150 kB.
All bounds/seed/counts are configurable and recorded. Zero tasks and zero intervals
are supported; invalid or non-finite inputs are rejected.

The manifest includes the seed, distributions, all generator settings, units, RNG
identity, generator/Python versions, exact tasks, and SHA-256. Identical settings within
the recorded runtime/generator version reproduce both tasks and results, without
changing global random state. The 100-task seed-42 example completes 96 tasks and
explicitly rejects 4 because every waiting queue is full; all 100 are accounted for.
It is a simulator check, not a scheduling-performance result against the paper.

## Configured topology and workload

```powershell
.\.venv\Scripts\python.exe -m drlmots --config configs/synthetic.yaml --output outputs/metrics/configured-demo.json
```

`configs/synthetic.yaml` is a runnable `synthetic_baseline` fixture with three hosts,
three VMs (two fog, one cloud), explicit AP/end/fog/cloud links, and one waiting slot
per VM. Six simultaneous 1000-byte tasks complete in two service waves: all six
complete, makespan is 4 ms, and average completion time is 3 ms.

Edit its explicit resource lists, memberships, queue capacities and workload settings
to run another fixture. Counts derive from list lengths; no host/VM distribution is
inferred. All fields and units are required. Unknown fields, missing capacities,
duplicate YAML keys, invalid references and wrong-tier resources are rejected.
`--config` cannot be combined with CLI seed/count/range flags; edit the run file instead.

The report saves the effective run configuration and SHA-256, derived topology counts,
workload provenance and outcomes. The saved `run_configuration`, excluding its
`sha256` field, can be loaded again through `parse_run_config` to reproduce the run.
File location and YAML formatting do not affect the effective configuration checksum.
`configs/default.yaml` retains unresolved paper settings and is not a runnable config.
Schema 1 preserves the demand-free baseline. Use schema 2 below for resource-aware
admission. Physical VM placement remains pending; DQN training is available separately.
Schema 3 adds an explicitly
configured network model; see FOG_CLOUD_POLICY.md.

## Resource-aware example (schema 2)

```powershell
.\.venv\Scripts\python.exe -m drlmots --config configs/resources.yaml --output outputs/metrics/resources-demo.json
```

Schema 2 additionally requires a run-level `resource_units` catalogue, explicit
`resource_capacities` on every VM, and workload `required_resources`. Generated tasks
share this demand profile; hand-built tasks can have different profiles. Each named
resource uses the same declared unit throughout the run, with no implicit conversions.
Capacities/demands are finite and positive; null or an absent capacity is unknown and
cannot satisfy a positive demand. No VM capacity is inferred from host metadata.

The example declares abstract CPU admission units and memory in GB. Every task needs
2 CPU units/4 GB; two VMs fit and a third is too small. Among resource-feasible VMs
with a waiting slot, the existing least-load/VM-ID rule applies. The fixture completes
four tasks, fails two because all feasible queues are full, and emits eight reserve/
release events. All VM balances return to their configured values.

Only running tasks reserve resources; queued tasks reserve none. Completion releases
resources before the next queued task starts and before same-time arrivals. A task with
no feasible VM gets `no VM can satisfy task resources`; full feasible queues have a
distinct failure reason. Demand-free legacy examples retain their outcomes/checksums.

Reports record demands/capacities/units, admission/queue policies, resource events,
final balances and exact replay configuration. Resource demand profiles use generator
version 2; the demand-free generator-v1 provenance remains compatible. CPU capacities
control admission, not task speed: service time still uses the explicit byte-valued
fixture formula. Each VM runs one task at a time; there is no physical host budget or
cross-VM resource sharing. See assumption A-020 for this model's limits.

## Baseline semantics and limits

- This fixture uses explicitly byte-valued task lengths: 1000 bytes take 2 ms under
  the recorded Eq. (8) interpretation. The original table's unitless length of 700,000
  remains unresolved. Task data sizes are a separate input; no relationship is inferred.
- Every VM executes one task at a time. Its positive, explicit queue capacity counts
  waiting tasks only. `None` is unknown and cannot be treated as unlimited.
- Placement chooses the resource-feasible VM with a waiting slot and the fewest running plus waiting tasks,
  breaking ties by VM ID. A VM's waiting tasks execute first-come, first-served.
- Simultaneous arrivals preserve input order. Completions release capacity before
  arrivals at the same time. All queues full means an explicit failed task.
- Makespan is the last successful finish from simulation time zero, including arrival
  gaps. Average completion time includes waiting and service for successful tasks.
  Failed tasks are counted separately; an empty workload has no average.
- The example's small topology and queue limits are test fixtures. Paper experiment
  settings are separate in `configs/default.yaml`, including the 20-200 VM range.
- Explicit resource feasibility/reservation is implemented for the synthetic per-VM
  model. VM-to-host allocation, paper-specific priorities/thresholds, energy, reliability,
  paper reward remain open. DQN training uses the documented synthetic reward.
  These schema-1/2 examples use zero network delay;
  the separate schema-3 fixture includes explicit uploads/propagation under A-026.
- The API copies input tasks and keeps caller-owned VM queues unchanged, allowing
  repeated runs on the same fixture.

## Versions

| Version | Completion target | Status |
|---|---|---|
| 0.1 | Deterministic topology, queue simulation, exact timelines and saved inputs/results | VERIFIED |
| 0.2 | Configurable seeded synthetic workloads and an explicit resource/placement model | VERIFIED; original resource checkpoint passed 367 tests |
| 0.2.1 | Offline professor report, paired simple policies, figures and presentation guide | VERIFIED software checks: 378 tests; browser layout requires manual preview |
| 0.3.0a1 | Initial synthetic observation/encoder/mask contract and optional traces | VERIFIED scoped checkpoint `a7a08bb`; 419 tests pass |
| 0.3.0a2 | Training-only state scaling, schema/topology-bound artifacts and held-out application | VERIFIED `29debfb`; 460 tests pass |
| 0.3.0a3 | Causal synthetic rewards, arrival transitions and forced-chain learner adapter | VERIFIED `72255bf`; 510 tests pass |
| 0.3.0a4 | CPU Q-network, legal selection and context-bound network snapshots | VERIFIED under A-025; 576 tests pass |
| 0.3.0a5 | Fog-first completion, explicit uploads/reachability and three-policy demo | VERIFIED under A-026; 618 tests pass |
| 0.3.1 | Live DQN across both environments, standard/fog-preferred variants, full resume and paired evaluation | VERIFIED under A-027; 677 tests; three 20-episode models; see LIVE_DQN.md |
| 0.3 | Documented synthetic state/action/reward, DQN CPU training and checkpoints | Delivered for the documented prototype scope; broader research/model fidelity remains open |
| 1.0 | Source audit, matched experiments and honest evaluation against the selected paper | Planned; source gaps remain |

See `implementation_plan.md` for microtask mapping and `progress.md` for actual
verification evidence. No later version is complete merely because a smoke run passes.
