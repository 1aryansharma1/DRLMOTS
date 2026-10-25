# DRLMOTS Implementation Progress

## Latest verified delivery - enhanced DQN contract, milestone 1 of 5 (2026-10-26)

Owner: Codex for ashishjha20. **Milestone 1 is complete.** User scoped the next aim to:
(1) contract, (2) basic masked-DQN integration, (3) connected small-scenario tests,
(4) small training sanity run, (5) frozen untrained/trained/four-heuristic comparison.
Only milestone 1 is implemented in this delivery. Double DQN/PPO and larger studies
are outside this agreed aim. Earlier entries below describe historical deliveries.

Read [ENHANCED_DQN_CONTRACT.md](ENHANCED_DQN_CONTRACT.md) first. The executable module
src/drlmots/enhanced_rl_contract.py defines contract 4, immutable public context hashes,
253 fixed-scale observation features for this topology, six sorted VM actions,
initial/recovery masks, and cumulative actual-energy/time/outcome rewards. Seeds,
sampled future faults and exact future repair times are excluded. Gamma=1; finite
training horizon H terminates with zero bootstrap, while report unfinished stays
censored. Service shortfall penalties default to 10 each and do not guarantee hard
constraints; report service floors still gate recommendations.

**Verified:** 839 full-suite tests passed (57 new), dependency check clean and whitespace
check passed. New checks include strict/canonical context guards, unknown/stale/future
telemetry rules, ordered queues, total-resource fit, prior-node recovery exclusion,
masked actions, hand-calculated rewards, delayed costs, interval partition invariance,
cutoff semantics and reward parity with actual reports from all four existing rules.
No simulator/controller/DQN/replay/checkpoint behavior was changed. No new enhanced
DQN training or learned-policy performance result exists yet.

**Next: milestone 2, unassigned.** Build a causal scheduler-visible telemetry adapter;
do not dump private physical state/queue cleanup into observations before detection.
Mask greedy/exploratory actions and Bellman targets. Fold forced no-action intervals
without inventing VM actions, and log costs before the first real action as a prefix.
Accrue OPEN recovery overhead continuously and keep background power accounting to H.
Preserve original event ordering and four heuristic results. Reuse the contract's
fixed encoding directly; reject old context/checkpoint hashes. Then milestones 3-5
verify the connected adapter, train a small model, and evaluate frozen paired policies.

The schema tests establish API invariants, not live-adapter non-leakage. Paired hidden-
world tests, actual replay transitions, training and evaluation remain pending.
The enhanced physics/presentation/results guides and their existing reports remain
available; the main presentation guide now links this milestone's contract reference.


## Latest verified delivery - enhanced objective model (2026-10-26)

Owner: Codex for ashishjha20. All ten requested improvements are implemented in
A-030, a separate host/network objective contract. MI/MIPS runtime, VM-to-host MIPS/GB
budgets, host utilization/piecewise power, shared upload/output bandwidth, sampled
exponential/Weibull node outages, separate detection/repair/reboot/restart, censoring
versus deadlines, reliability metrics, paired failure-seed trials and constrained
weighted/Pareto reports are implemented. A-028/A-029 baselines remain unchanged.

**Verification:** 782 tests passed (50 new); dependency check clean. Actual artifact
audit: 132 full traces with source/workload/outage hashes, paired inputs, time/energy
conservation, resource restoration and link capacity; 142 local report links;
146 ZIP entries/CRC/byte parity; four PNG/four SVG figures. Representative figures
were visually inspected. Browser desktop/mobile layout remains a manual demo check.

**Results:** 3 workloads x 5 independent failure traces x 4 policies x 2 fault modes
plus 12 no-fault controls = 132 simulations / 7,920 task arrivals across runs.
Multi-objective recovery delivered 58.8/60 versus 55.6 without retries, with energy
40.877 versus 40.327 J and successful mean latency 19.749 versus 16.608 ms. First fit
does not consistently benefit: retries can displace new admissions. Confidence
intervals use failure-seed means conditional on the fixed three-workload panel.
No generic superiority, calibrated hardware or faithful paper reproduction is claimed.

**Presentation:** read [OBJECTIVE_MODEL_PRESENTATION.md](OBJECTIVE_MODEL_PRESENTATION.md)
for the 1-10 mapping, formulas, hand-calculated examples, workflow and talking points.
Read [OBJECTIVE_MODEL_RESULTS.md](OBJECTIVE_MODEL_RESULTS.md) for the recorded results.
Open outputs/objective-model/professor-final/objective-report.html; share the ZIP after
extraction. Generated artifacts are ignored; model, source, config, tests and the
presentation/results guides are tracked. The source hashes use the project's canonical
JSON digest of normalized source text. Choose a fresh output directory for a new plan.

**Next owner / remaining:** unassigned. No trained DQN supports the enhanced engine;
legacy checkpoints are incompatible. Next research contract: fault/network/host-aware
state/actions, delayed terminal rewards and fresh training/validation comparisons.
Measured MI/power/link/failure calibration, distribution fit, cross-node common-cause
faults, multi-hop routing, checkpoints and facility energy remain explicitly open.
No implementation item from the ten-point request remains active. Earlier entries
below describe historical versions and should not be read as current limitations.


## Latest verified delivery - actual failure/recovery (2026-10-25)

Owner: Codex for ashishjha20. A-029 implements a separate observed node-failure and
bounded restart-recovery environment, retaining all previous simulators/models.
732 tests pass (34 new); dependency check clean. Artifact checks: 69 local links,
75 ZIP entries/CRC, 60 task/resource traces and 5 PNG/5 SVG plots. Representative
figures inspected; browser layout remains a manual presentation check. Node failures affect all member VMs;
interrupted work is retried on another node with complete partial/repeated time and
energy accounting. CPU/memory releases, input immutability, no future-fault leakage,
shared outage traces and no-failure baseline parity are verified.

Actual report: 60 simulations (four rules x three modes x five workloads), fixed
250 ms energy horizon. Fog-first completed 69.8/100 without recovery vs 76.0 with it,
but energy and successful latency rise; retries also displace new admissions. No
DQN has been trained for this new fault environment; old analytical-risk checkpoints
are preserved and incompatible by design. Next: version the fault-aware state/action/
reward and delayed terminal credit, then untrained/validation experiments.

Read [FAILURE_RECOVERY.md](FAILURE_RECOVERY.md) and [RECOVERY_RESULTS.md](RECOVERY_RESULTS.md).
Open outputs/recovery/professor/recovery-report.html; share its ZIP after extraction.
Generated artifacts are ignored. Existing analytical-risk reports remain at their
old paths and must not be compared as if the energy definitions were identical.
No implementation task remains active after this delivery; source-fidelity and
hardware-calibration gaps remain explicit. Earlier records below are historical.


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


**Current stage:** v0.3.1 end-to-end live CPU DQN VERIFIED (2026-10-15).
Phase 3 is verified for the current data/configuration model. Deterministic subsets
of Phases 4-5 and seeded workload generation are verified. The full paper topology,
source-audited resource interpretation, physical allocation and paper-faithful DQN
alignment remain pending. Synthetic training/placement/evaluation are verified below.

**Working status (2026-10-13):** v0.1 and P4.3/P4.6 seeded workload generation are
complete, implemented and verified by Codex for ashishjha20. Configuration-driven
topology and explicit per-VM resource admission/reservation are VERIFIED. v0.2 is
complete for the A-017/A-020 synthetic baseline scope. The user-requested professor
demo (A-021) is complete, implemented by Codex for **1aryansharma1 / Aryan Sharma**.
**Latest delivery (2026-10-15): standard and network-aware live DQN VERIFIED,
Codex for Aravindak3 / Aravind.** Repository-local author is Aravind with
`195939576+Aravindak3@users.noreply.github.com`; credential username selects Aravindak3.
Global identity is unchanged. CPU Q-network and legal masked selector (P9 subset)
is complete, owned by Codex for Aravindak3 / Aravind (started 2026-10-14, verified
2026-10-15). User-requested fog-first completion-time scheduling is VERIFIED,
owned by Codex for Aravindak3 / Aravind. End-to-end live DQN training for both standard
and network-aware modes is VERIFIED under A-027, same owner. No implementation task
is active. Three models trained for 20 episodes each; full checkpoints and held-out
evaluation delivered. Next research task: independent initialization/validation study.

**Git publication:** the prior state/action implementation/handoff was verified on
GitHub main at `9604536` historically; collaborator updates through `5dffde0` were
fetched and merged during this delivery. Confirm the latest local/remote HEADs match
when handing off. If sandboxed authentication fails, push from a normal user terminal.
The verified v0.1 implementation commit is `993344d`.
Latest verified implementation: v0.3.1 live DQN in this delivery;
prior fog-first `2ad7e4b`;
prior Q-network `96a5407` and template fix `6f87c53`;
prior `72255bf` (v0.3.0a3 reward/transitions); normalization:
`29debfb`; state/actions:
`a7a08bb`; professor demo:
`2083faa`; resource model:
`4f4307f`; configured topology:
`e97a14d`; seeded workloads:
`7ebf8d6`. The complete remaining backlog and immediate pickup checklist are in
[NEXT_STEPS.md](NEXT_STEPS.md).
Use [HANDOFF.md](HANDOFF.md) to begin or hand over a task, and
[PROGRESS_REPORT.md](PROGRESS_REPORT.md) for the dated delivery report.

## Status legend

- NOT STARTED
- IN PROGRESS
- BLOCKED
- IMPLEMENTED - code exists but is not yet verified
- VERIFIED - the specified test/check was run and passed
- NEEDS REVISION

## Verified project state

| ID | Task | Status | Evidence / note |
|---|---|---|---|
| P0.1 | Inspect all 26 PDF pages, figures, equations, Algorithm 1 and experiment tables | VERIFIED | PDF text extracted and key visuals on pages 8-12 and 14 inspected |
| P0.2 | Build structured source extraction, parameter table, equations and ambiguity list | IMPLEMENTED | See `paper_requirements.md`; review is pending |
| P0.3 | Create paper-to-module traceability matrix | IMPLEMENTED | Requirements R001-R040 in `paper_requirements.md` |
| P0.4 | Record unresolved assumptions/contradictions without silently filling them | IMPLEMENTED | See `assumptions.md` |
| P0.5 | Confirm the first implementation boundary (smoke simulator before training) | VERIFIED | Initial implementation order and verification gates documented in `assumptions.md`; unresolved paper parameters remain open |
| P1.1 | Inspect operating system and Python/tool availability | VERIFIED | Windows, PowerShell, Python 3.11.9 and Git confirmed |
| P1.2 | Create isolated Python environment | VERIFIED | `.venv` created; project interpreter points inside `.venv` |
| P1.3 | Install compatible dependencies | VERIFIED | SimPy, NumPy, PyTorch, PyYAML and pytest installed |
| P1.4 | Run dependency/import smoke test | VERIFIED | Initial smoke tests passed; `pip check` reported no broken requirements |
| P1.5 | Check core imports and Torch device availability | VERIFIED | Core imports passed; Python 3.11.9 and dependency versions confirmed; PyTorch CPU available; CUDA unavailable |
| P1.6 | Record dependency versions and environment notes | VERIFIED | `requirements-lock.txt` is tracked and matches `pip freeze --exclude-editable`; `pip check` reported no broken requirements |
| P2.1 | Create repository/package/test directory structure | VERIFIED | Project structure created |
| P2.2 | Add package metadata and test configuration | VERIFIED | `pyproject.toml` created; package imports successfully |
| P2.3 | Create configuration schema with unresolved values explicit | VERIFIED | Configuration validation tests passed |
| P2.4 | Add package entry point or smoke test as needed | VERIFIED | `drlmots` package import succeeds |
| P2.5 | Verify pytest discovery and basic tests | VERIFIED | Initial `python -m pytest -v`: 7 passed |
| P3.1 | Define Task fields from source-supported requirements | VERIFIED | `src/drlmots/entities.py` implements `Task` and `TaskStatus`; Task tests and the full suite passed (25 total). Task-length units and exact resource schema remain unresolved |
| P3.2 | Define VM, host and resource data structures | VERIFIED | `Host`, `VM`, and `ResourceTier` implemented in `src/drlmots/entities.py`; resource validation tests pass; full suite: 54 passed. Bandwidth units and unspecified capacities remain unresolved |
| P3.3 | Define fog/cloud topology structures | VERIFIED | `topology.py`: AP/end/fog/cloud relationships, unique IDs, single resource membership, tier matching, no orphan hosts/VMs. Actual VM-to-host allocation remains unresolved |
| P3.4 | Define and test valid task-status transitions | VERIFIED | `Task.transition_to`; every status pair tested, terminal tasks cannot restart through the lifecycle API |
| P3.5 | Validate resource and configuration ranges | VERIFIED | Finite task sizes/rates/capacities; integer DQN counts; epsilon/replay consistency; separate paper experiment settings enforce 20-200 VMs; unspecified topology counts remain null |
| P4.1 | Fixed-list deterministic task factory | VERIFIED | Exact fields, unique IDs and independent resource mappings tested |
| P4.2 | Explicit-unit processing-time function | VERIFIED for byte-valued fixtures | 1000 bytes -> 2 ms; unsupported units and invalid values rejected. Original paper's task-length interpretation remains open |
| P4.3 | Seeded random arrivals and sizes | VERIFIED for synthetic model | Local RNG; uniform intervals, inclusive integer lengths, uniform data sizes; settings/bounds/count/seed validation and repeatability tests; assumption A-018 |
| P4.4-P4.5 | Paper experiment sizes and real dataset mapping | NOT STARTED | Generator accepts explicit counts, but no paper count conflict or Google Jobs dataset identity is resolved |
| P4.6 | Save workload manifest | VERIFIED for fixed and seeded inputs | Exact tasks, seed, all generator settings, distributions, units, generator/runtime versions and SHA-256; empty workloads retain provenance |
| P5.1 | Configured topology builder | VERIFIED for explicit synthetic topology | `build_topology`, strict runnable YAML schema, `RunConfig` manifest/replay and `--config`; full source-audited paper topology remains NOT STARTED |
| P5.2-P5.5 | SimPy arrivals, FCFS execution and completion timestamps | VERIFIED for deterministic baseline | Hand-calculated one-task/two-VM cases; waiting and service times; exactly-once outcomes |
| P5.6 | Queue capacity and fallback | VERIFIED for baseline policy | Explicit capacities, alternate fog-node fallback, full-cluster failure. Paper threshold/tier policy remains open |
| P5.7 synthetic subset | Explicit network-transfer delays | VERIFIED under A-026 | Per-VM Mbps/ms links, input upload readiness and FCFS overlap/head-of-line events; original-paper network model remains open |
| P5.8 | Deterministic edge cases | VERIFIED for baseline scope | Empty workloads, simultaneous arrivals/completions, missing VMs, unknown/full queues, duplicate tasks and unsupported resource requests |
| R004 / P7.3 baseline subset | Explicit per-VM demands/capacities, feasible placement and resource reservations | VERIFIED for synthetic model | Shared declared units, missing capacity never inferred, exact-fit/oversize/full-feasible-queue failures, reserve/release trace, no oversubscription/leak. Original paper resource specification and full P7 remain open |
| P13.1 synthetic subset / professor demo | Two deterministic placement rules, paired evaluation and visual presentation | VERIFIED software scope | Same inputs/hashes, accounting and restored balances; three presets, offline HTML/JSON/SVG/PNG and talk track. Browser layout not automatically verified; full P13/P15 remain open |
| P6.1/P6.2/P7.2 initial synthetic subset | Immutable decision observations, raw feature encoding, VM action mapping/masks and opt-in trace | VERIFIED under A-022 | 41 new tests; hand vectors/masks, unknown flags, event timing, detachment, all-false rejection, baseline preservation, seeded replay and CLI. Normalization/reward export delivered separately below; live agent/DQN remain pending |
| P6.3 synthetic subset | Training-only bounded maximum scaling and saved compatible artifacts | VERIFIED under A-023 | 41 new tests; hand scales, known flags, constants/extremes, immutable statistics, split/task overlap, schema/full topology checks, strict save/load and CLI. Reward delivered separately below; no DQN yet |
| Initial P8/P11 synthetic subsets | Causal reward, all-arrival return, raw transition export and forced-chain learner adapter | VERIFIED under A-024 | 50 new tests; hand values, causal forecasts, baseline/scaler preservation, terminal/empty/all-rejected runs, gamma boundaries, forced prefixes/chains/suffixes, strict config/action and actual CLI. No live agent or DQN yet |
| P9 / P10 selection synthetic subsets | CPU Q-network, legal epsilon-greedy and network snapshots | VERIFIED under A-025 | 66 new tests; 47 -> 50 -> 50 -> 3, 5,103 parameters, finite gradients/shapes, local seeds, legal ties/exploration, full artifact context, strict JSON round trip and actual held-out CLI. Replay/updates/live control pending |
| User-requested P5.7/P7 subset | Fog-first completion with origin reachability and actual transmission timing | VERIFIED under A-026 | 41 new tests; fog wins earlier/equal predicted finishes even when busy, faster cloud fallback, honest upload/queue timing, strict schema-3 replay and 3-policy demo; legacy RL capture blocked |
| Synthetic P6/P8/P10-P12/P13 subset | Contract-2 state/reward, live callbacks, replay, masked targets/Adam, target sync, resume and paired evaluation | VERIFIED under A-027 | 59 new tests; actual actions executed; full checkpoint resume equals uninterrupted run; frozen held-out evaluation and both environment modes; standard and explicitly constrained fog variants retained |

## Verification in this checkout

Historical environment entries above came from the previous checkout. This cloned
checkout now has its own ignored `.venv`, created with Python 3.11.9 on 2026-10-13.
Dependencies were installed from `pyproject.toml`; the lock export was refreshed to
match the resolved runtime versions (filelock changed to 4.1.1).

| Check | Actual outcome |
|---|---|
| Phase 3 gate: `.venv\Scripts\python.exe -m pytest -q` | 136 passed before simulator implementation |
| Final v0.1 suite: `.venv\Scripts\python.exe -m pytest -q` | 194 passed |
| Seeded-workload checkpoint suite (historical): `.venv\Scripts\python.exe -m pytest -q` | 246 passed |
| Configured-topology checkpoint suite (historical): `.venv\Scripts\python.exe -m pytest -q` | 304 passed |
| Final v0.2.0 suite: `.venv\Scripts\python.exe -m pytest -q` | 367 passed |
| v0.2.1 suite with demo extra: `.venv\Scripts\python.exe -m pytest -q` | 378 passed |
| v0.3.0a1 suite: `.venv312\Scripts\python.exe -m pytest -q` | 419 passed on Python 3.12.14 |
| v0.3.0a2 suite: `.venv312\Scripts\python.exe -m pytest -q` | 460 passed on Python 3.12.14 |
| v0.3.0a3 suite: `.venv312\Scripts\python.exe -m pytest -q` | 510 passed on Python 3.12.14 |
| v0.3.0a3 dependency check | No broken requirements found |
| v0.3.0a4 suite: `.venv312\Scripts\python.exe -m pytest -q` | 576 passed on Python 3.12.14 |
| v0.3.0a4 dependency check | No broken requirements found |
| Demo source-template opening regression | 577 tests pass; source explains missing data and links to the generated report; generated charts/controls/export still verified in DOM stub |
| v0.3.0a5 suite / dependency check | 618 tests pass in 48.53 s; no broken requirements |
| v0.3.1 suite / dependency check | 677 tests pass; no broken requirements |
| Actual professor training | 3 models x 20 episodes; optimizer updates 1878 / 1885 / 1920; checkpoints reload; greedy evaluation on workload seeds 3000/3001/3002 |
| Network CLI / demo | 65 completed/35 rejected; predicted selected finishes match actual; 9 selections/3 exports verified; 21 PNG/21 SVG figures |
| Resource fixture with `--reward-config configs/reward.yaml` | Six rewards/transitions; return 23/21; unchanged 4 completed/2 rejected, 4 ms and restored balances |
| 100-task reward fixture | 96/4; 100 records; one terminal; return 67.73152495135943; no actual finish used in reward |
| v0.3.0a2 dependency check | No broken requirements found |
| Normalizer fit/apply CLI with named fixtures | 100 train states; 100 held-out 47-feature states in [0,1]; unchanged 92 completed/8 rejected on evaluation; artifact unchanged |
| v0.3.0a1 dependency check: `.venv312\Scripts\python.exe -m pip check` | No broken requirements found |
| `--config configs/resources.yaml --capture-decisions` | Six 47-feature states; actions 0,1,0,1,null,null; 4 completed/2 failed, 4 ms and restored balances |
| `python -m drlmots.demo` | Three paired presets, 15 SVG/15 PNG figures and offline report generated |
| `node scripts/check_demo_controls.cjs` | All six scenario/policy selections, outcome tables, provenance and three JSON exports verified in DOM stub; no layout claim |
| Wheel packaging | 0.2.1 wheel includes `drlmots/templates/demo.html` |
| `.venv\Scripts\python.exe -m pip check` | No broken requirements found |
| `.venv\Scripts\python.exe -m drlmots --output outputs/metrics/smoke.json` | 3 completed, 0 failed; makespan 4 ms; average completion time 8/3 ms |
| `.venv\Scripts\python.exe -m drlmots --task-count 100 --seed 42 --output outputs/metrics/seeded-demo.json` | 96 completed, 4 explicitly failed (full queues), 100 accounted for; makespan 155.1273394932697 ms |
| `.venv\Scripts\python.exe -m drlmots --config configs/synthetic.yaml --output outputs/metrics/configured-demo.json` | 3 VMs, 6 completed, 0 failed; makespan 4 ms; average completion time 3 ms; exact config and workload checksums saved |
| `.venv\Scripts\python.exe -m drlmots --config configs/resources.yaml --output outputs/metrics/resources-demo.json` | 2 feasible/1 too-small VMs; 4 completed, 2 explicit full-feasible-queue failures; 4 ms makespan; 8 resource events; all balances restored |

The suite includes 100-task fractional and heterogeneous-resource fixtures checking
serial execution, feasibility, accounting and repeatability. It validates the documented
synthetic resource model, not physical host budgets, DQN behavior, energy, fault tolerance
or experimental improvements. Old schema-1 config and demand-free workload checksums
remain unchanged; schema 2 records explicit units/capacities/demands. Editable package
metadata is now 0.3.1 in the separate supported Python 3.12.14 `.venv312`; the earlier
3.11 `.venv` is preserved. The unchanged base lock is extended by the separate
optional `requirements-demo-lock.txt`. Representative PNG figures were visually
inspected; valid SVG namespaces/tooltips and safe embedded JSON are tested. The browser
tool disallowed local-file preview, so real browser/mobile layout remains a manual check.

## Important unresolved paper details

- Conflicting initial epsilon values.
- Conflicting episode counts.
- Conflicting task-count ranges.
- Unclear task-length units and arrival distribution.
- Unclear reward interpretation and scaling.
- Missing DQN hyperparameters.
- Unclear energy units/model and fault metric interpretation.

These must remain explicit until resolved from the paper or documented as implementation assumptions.

## Important distinction

The current 677-test suite verifies setup, entities/configuration/topology, fixed and
seeded workloads, strict runnable YAML, effective-config replay, CLI behavior and
resource admission/accounting, paired comparisons, figure exports and the initial
synthetic state/action contract, training-only normalization and synthetic reward/
transition export, CPU network inference/gradients, legal selection and compatible snapshots.
This establishes only the documented
baseline behavior. It does not reproduce the DRLMOTS algorithm or its results. The
attached DRL-MLS Kubernetes paper is a separate source and was not used to change the
DRLMOTS algorithm; see decision A-017.

## Resume protocol

Legacy contracts A-022 through A-026 remain available. A-027 delivers live contract 2,
ordered upload queues, training-only scaling, replay/targets/Adam, full resume and
actual learned placement for both environments. Read LIVE_DQN.md for commands and
results. Next research gate is a multi-initialization study with predeclared validation
and fresh test splits; do not tune against the already reported held-out seeds.
Preserve both standard and fog-preferred modes and all existing rules/contracts.
P4.3/P4.6, configured P5.1 and resource feasibility are verified for A-018/A-020 synthetic
models, not a source-audited paper topology/distribution or physical host allocation.
Retain the deterministic tests and run each applicable gate before moving on. Keep
P0.2-P0.4 source review pending until the original DRLMOTS paper is available for an
independent audit. Resolve state/action/reward, metric units and training settings before
DQN training. Do not claim that unimplemented portions of Phases 4-5 are complete.
