# DRLMOTS implementation workspace

<!-- A037:START -->
## Latest: teacher-assisted DQN completed (2026-10-27)

Point 3 is implemented and actually run: 40 observation-only multi-objective teacher episodes, fixed 20-epoch masked imitation, then 300 ordinary RL episodes for each of three seeds, plus equally budgeted RL-only controls. 1,800 RL episodes / 117,608 updates; 342 frozen fresh-test simulations. Existing baselines/default and the 40/40/20 objective are unchanged.

Fault-panel teacher + RL: 232.936 ms makespan, 40.862 J, 89.19% completion; combined cost 0.640721. Versus matched RL-only, cost is 6.01% lower. The paired model-seed interval includes zero: a repeatable teacher benefit is not established. Best eligible among the main three: **Without DQN (native rule)**.

**Main finding:** imitation-only was the strongest evaluated neural policy on the fresh fault panel: 95.48% completion, 0.564300 cost and 45/45 service-feasible runs, close to the native rule (95.44%, 0.563988). Continuing RL raised its cost by 13.54%, reduced completion by 6.30 percentage points and failed service floors in 3/45 runs. The fine-tuned model stays opt-in; preserving imitation during RL is the next research question.

Read [TEACHER_DQN_RESULTS.md](TEACHER_DQN_RESULTS.md) for concise findings and [TEACHER_DQN.md](TEACHER_DQN.md) for the teacher equations, training protocol, limits and reproduction commands. [Portable report](reports/teacher-dqn-20261027/report.html) includes charts, imitation/RL controls, preserved checkpoints' fresh metrics and uncertainty. **1,064 tests passed; independent saved-artifact audit passed.** Earlier entries below are historical.
<!-- A037:END -->
This workspace is for an incremental, test-first implementation of the DRLMOTS cloud–fog scheduling approach in the paper:

> Choppara, P. and Mangalampalli, S. “An efficient deep reinforcement learning based task scheduler in cloud-fog environment.” *Cluster Computing* 28:67 (2025). DOI: 10.1007/s10586-024-04712-z.

## Predictive DQN and longer validation-selected training (2026-10-27)

Predictive training finished: 900 episodes / 54,000 task arrivals across three model seeds; 58,837 optimizer updates. All checkpoints were selected on validation before 216 fresh frozen test simulations.

The observed mean fault-panel makespan was 1.42% lower than the same-feature initialization, while energy changed by +0.20% and completion by -0.07 percentage points. All four paired model-seed intervals include zero, so a repeatable gain is not established. The new variant did not beat the original 16-episode DQN's mean combined cost on this panel. It remains opt-in; the preserved baseline remains the default. These results do not establish an improvement on all three objectives.

Read [PREDICTIVE_DQN_RESULTS.md](PREDICTIVE_DQN_RESULTS.md) for the simple three-method/three-metric results and [PREDICTIVE_DQN.md](PREDICTIVE_DQN.md) for the 325-input feature model, selection rule and commands. The current baseline remains available. **1,033 full-suite tests passed.**

~~~powershell
Start-Process ".\outputs\objective-model\predictive-training\study-20261027\report.html"
~~~

## Frozen enhanced DQN comparison: milestone 5 of 5 complete (2026-10-27)

**Short report:** [three methods, three metrics and one overall ranking](DQN_SIMPLE_REPORT.md).
Create the simple offline page from the saved results:

~~~powershell
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.simple_comparison_report --output-dir outputs/objective-model/simple-comparison/my-report
Start-Process ".\outputs\objective-model\simple-comparison\my-report\report.html"
~~~


Frozen untrained DQN, the actual final A-034 trained checkpoint and all four native
heuristics were evaluated on the same held-out workloads and failure traces:
**108 simulations, 18 matched scenarios, 6,480 task arrivals**. Neither DQN learned
or changed during evaluation. All five agreed milestones are complete.

With failures, trained DQN averaged **58.07/60 completions versus 55.87/60 untrained**,
with more energy (41.779 vs 41.450 J) and a later successful makespan (249.728 vs
245.153 ms). Combined return was almost unchanged. The native multi-objective rule
averaged 59.27/60, 41.502 J and 231.342 ms. Its information access differs from DQN:
this is an implementation comparison, not proof of algorithm superiority.

Read [ENHANCED_DQN_COMPARISON.md](ENHANCED_DQN_COMPARISON.md) for all results, uncertainty,
method limits, charts, the portable HTML/ZIP report and professor notes.
**997 full-suite tests passed**, including 27 new comparison checks.

~~~powershell
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.enhanced_comparison --training-dir outputs/objective-model/dqn-sanity/milestone4-20261027 --output-dir outputs/objective-model/frozen-comparison/my-comparison
Start-Process ".\outputs\objective-model\frozen-comparison\my-comparison\comparison-report.html"
~~~

Choose a new output directory. On another checkout, first regenerate the checkpoints
with [ENHANCED_DQN_TRAINING_SANITY.md](ENHANCED_DQN_TRAINING_SANITY.md). The final
16-episode checkpoint is fixed; this comparison does not tune or select it.
The existing enhanced_dqn inference command still initializes an untrained network
when --checkpoint is omitted. Earlier report commands keep their original meaning.

See [ENHANCED_DQN_INTEGRATION.md](ENHANCED_DQN_INTEGRATION.md) for the live connection,
[ENHANCED_DQN_CONTRACT.md](ENHANCED_DQN_CONTRACT.md) for the 253-input contract and
[ENHANCED_DQN_VERIFICATION.md](ENHANCED_DQN_VERIFICATION.md) for connected checks.

## Enhanced objective model and presentation reference (2026-10-26)

All ten objective-model improvements are available in a separate A-030 mode:
MI/MIPS runtime, explicit VM/host mapping, utilization-based host energy, sampled
exponential/Weibull outages, separate detection/repair/reboot/restart, shared upload
and output bandwidth, deadlines versus censoring, reliability metrics, paired failure
trials and constrained multi-objective reporting. Legacy baselines and trained DQN
contracts are preserved; the enhanced mode now has a small sanity-trained checkpoint,
with frozen comparative results recorded in ENHANCED_DQN_COMPARISON.md.

Start with [OBJECTIVE_MODEL_PRESENTATION.md](OBJECTIVE_MODEL_PRESENTATION.md): the
1-10 checklist, formulas, simple examples, file map and professor walkthrough.
[OBJECTIVE_MODEL_RESULTS.md](OBJECTIVE_MODEL_RESULTS.md) records actual results from
132 simulations. The new coefficients are synthetic, with measured-curve input support.

~~~powershell
.\.venv312\Scripts\python.exe -m drlmots.objective_report --config configs/rl_network.yaml --model-config configs/objective_model.yaml --output-dir outputs/objective-model/professor-final
Start-Process ".\outputs\objective-model\professor-final\objective-report.html"
~~~

The generated offline HTML includes charts, reliability/availability tables, a recovered
task example and CSV/JSON/ZIP downloads. Extract the ZIP before sharing/opening elsewhere.
Generated outputs are ignored; source, configuration, presentation guide and results
summary are tracked. Old report commands below retain their original meanings.

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
The current suite passes **1,033 tests** with the demo dependencies on Python 3.12.14
(2026-10-27); the dependency check is clean.
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

## Observed failures and restart recovery (2026-10-25)

A separate versioned mode now interrupts actual simulated tasks when fog/cloud nodes
fail, retries eligible tasks on another node, and accounts for partial/repeated uploads,
execution, idle power and recovery overhead over a common observation horizon.
Existing no-failure simulators and analytical-risk DQN results are preserved.
See [FAILURE_RECOVERY.md](FAILURE_RECOVERY.md) for the teaching approach/contracts,
and [RECOVERY_RESULTS.md](RECOVERY_RESULTS.md) for verified results and remaining work.

~~~powershell
.\.venv312\Scripts\python.exe -m drlmots.recovery_report --config configs/rl_network.yaml --objectives configs/objectives.yaml --recovery-config configs/failure_recovery.yaml --seeds 5200 5201 5202 5203 5204 --output-dir outputs/recovery/professor
Start-Process .\outputs\recovery\professor\recovery-report.html
~~~

The report compares no failures, failures without recovery, and identical failures
with recovery. It contains actual task attempts, energy/time ledgers, charts and a
ZIP/CSV/JSON/Markdown bundle. The old DQN state/reward lacks these fault/retry events. The later enhanced contract,
training and frozen comparison are documented in ENHANCED_DQN_COMPARISON.md.


## Multi-objective efficiency extension (2026-10-25)

The original latency-only modes are retained. Optional reward contract 3 combines
latency, active/input-transmission energy and **analytical failure risk** with explicit
weights and synthetic VM coefficients. This is not an implementation of backup/failover
or a source-faithful paper reward. See [MULTI_OBJECTIVE.md](MULTI_OBJECTIVE.md) for units,
methodology, checkpoint compatibility and report interpretation, and
[MULTI_OBJECTIVE_RESULTS.md](MULTI_OBJECTIVE_RESULTS.md) for actual results.

~~~powershell
.\.venv312\Scripts\python.exe -m drlmots.efficiency --config configs/rl_network.yaml --training-config configs/efficiency_training.yaml --objectives configs/objectives.yaml --output-dir outputs/efficiency/professor-fresh --model-seeds 42 43 44
Start-Process .\outputs\efficiency\professor-fresh\efficiency-report.html
~~~

Add --resume for matching existing output. The experiment compares seven policies,
three independent training initializations, five fresh workload seeds and three load
scenarios. HTML, CSV, JSON, written reports, plots and an offline ZIP are generated.
Generated artifacts are ignored; collaborators regenerate or receive the ZIP separately.


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

Expected checks: no broken requirements; 732 passing tests with the demo extra. Optional figure/report tests may skip without it. The last command prints
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
This section describes the preserved reward/transition API; live replay and DQN training are available through train.py.

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
