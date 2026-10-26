# Remaining work and collaborator pickup

## Latest delivery - enhanced masked DQN connection, milestone 2 of 5 (2026-10-26)

Owner: Codex for ashishjha20. **Milestones 1 and 2 are implemented.** The scoped
sequence is contract; masked-DQN connection; broader connected checks; small training
sanity run; frozen untrained/trained/four-rule comparison. Double DQN/PPO and larger
studies remain outside this aim.

The new opt-in `masked_dqn` path connects contract 4 to both initial and recovery
placement. It uses six VM actions, delivered periodic telemetry, legal-action masks,
actual delayed energy/time/outcome rewards, bounded replay, a target network and
context-bound JSON checkpoints. The default CLI is frozen, untrained inference only.
The four reference rules and the legacy DQN remain available on their existing paths.

Focused integration acceptance tests are present in
`tests/integration/test_enhanced_dqn.py`; this delivery has not run them. **Next is
milestone 3:** run and expand connected checks for hidden faults, delayed health notices,
initial/recovery masks, resource conservation and reward totals. Then perform milestone
4's small learning-update/behavior sanity check and milestone 5's frozen paired
comparison against the four rules. Do not report learning gains before those experiments.

See [ENHANCED_DQN_INTEGRATION.md](ENHANCED_DQN_INTEGRATION.md) for operation and
commands, [ENHANCED_DQN_CONTRACT.md](ENHANCED_DQN_CONTRACT.md) for the feature/action/
reward contract, and `assumptions.md` A-032 for telemetry and checkpoint assumptions.


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


**Updated:** 2026-10-15. **Current milestone:** v0.3.1 live synthetic CPU DQN VERIFIED.
**Owner:** completed by Codex for Aravindak3 / Aravind; no implementation task active.
Earlier professor demo by Codex for 1aryansharma1 / Aryan Sharma; earlier baseline
work for ashishjha20. Repository-local author remains Aravind.
**Publication:** collaborator documentation through `5dffde0` was fetched/merged.
Compare local HEAD with remote main before starting; do not discard another person's work.

## Delivered and retained

- Task/host/VM topology, fixed/seeded workloads, strict YAML, reproducible JSON.
- FCFS queues, resource feasibility/reservation/release and complete task accounting.
- Original least-queue/first-fit rules and offline two-policy professor demo.
- Explicit upload/propagation, origin reachability, fog-first completion rule and
  three-policy network demo. Original zero-network examples retain their outcomes.
- Preserved contract-1 state/action/scaler/reward APIs and standalone network snapshots.
- New live contract 2: ordered upload queue states, context-bound scaler/checkpoints,
  real VM-selection callbacks and causal end-to-end rewards for both environments.
- Standard DQN over all legal actions, plus explicitly constrained fog-preferred DQN.
- Bounded immutable replay, forced-chain credit, legal Bellman targets, Adam/gradient
  clipping and separate frozen target network with configured synchronization.
- Full JSON checkpoint/resume including optimizer, replay, RNG, configs, statistics
  and logs; exact resumed/uninterrupted equality tested.
- Frozen greedy held-out evaluation with paired workloads, seed/task overlap checks,
  successful timing plus rejection/return reporting and sample standard deviations.
- Three actual 20-episode models, 5683 optimizer updates; static training report,
  complete traces, five PNG/five SVG plots and a shareable offline ZIP.
- **677 tests pass** and dependency check is clean; 59 new live-training checks.
- Windows file-lock recovery tested and exercised during the actual benchmark.

Read [LIVE_DQN.md](LIVE_DQN.md) for exact commands, configurations, actual results and
limitations. `outputs/training/` and all generated checkpoints/reports/ZIPs are ignored
by Git; regenerate or obtain the bundle separately. Extract ZIP before opening HTML.
The report does not retrain when opened; placement occurred inside the Python simulator.

## Pick up next: predeclared multi-initialization/validation experiment

**Owner:** unassigned. **Status:** NOT STARTED. This is a research-quality extension;
the requested end-to-end prototype is already implemented.

1. Claim the task in progress.md. Keep standard, fog-preferred and reference rules.
2. Define several independent model-initialization/replay seeds, with identical
   environment/task seeds within each comparison. Current results used one model
   initialization (42) and three held-out workload seeds; these are different concepts.
3. Declare train/validation/fresh test splits before trials. Do not tune against the
   already reported workload seeds 3000/3001/3002. Never fit scaling on held-out data.
4. Select hyperparameters/checkpoints using validation only; evaluate final choices
   once on the reserved test set. Retain failed/underperforming runs in the report.
5. Aggregate uncertainty across independent training runs, save every run manifest/
   checkpoint/log, and test pairing/split isolation/resume determinism.
6. Make no universal-superiority claim: in the current short experiment, reference
   least-queue/fog-first rules outperform some trained policies.

**Finish criterion:** reproducible validation-selected experiments with independent
training seeds, fresh test evaluation and honest uncertainty/throughput/latency results.

## Other work still open

| Scope | Remaining deliverable |
|---|---|
| Original-source audit P0.2-P0.4 | Independently verify DRLMOTS extraction/equations/parameters and R001-R040 traceability |
| Paper task-count settings P4.4 | Resolve conflicting counts through named explicit experiment configs |
| Real dataset P4.5 / A-011 | Verify dataset version/schema/filtering/mapping before any Google Jobs label |
| Full paper topology P5.1 | Source-supported host/VM distributions and physical VM-to-host/shared-host budgets |
| Richer network P5.7 | Justified contention, output downloads, per-task origins/routes; current uploads are independent and simulated |
| Processing model | Explicit per-VM service rates if justified; current CPU quantities control admission, not speed |
| Paper P6/P7 | Source-audited state/features, priorities and threshold policy; current fog preference is user-defined |
| Paper P8 / A-006-A-008 | Resolve energy/reliability/multi-objective reward semantics and units; current reward is a separate surrogate |
| Paper P9/P10 / A-009-A-010 | Audit architecture/learning settings; current configurable choices are prototype assumptions |
| Full P13-P15 | Justified paper comparators, larger task-count sweeps, independent training seeds and raw metric/percentage audit |
| P16 | CI/clean-machine setup, applicable source traceability, final research report and explicit limitations |
| Presentation QA | Manually inspect report desktop/mobile layout in a normal browser before presenting |

Target paper: DRLMOTS cloud/fog, DOI 10.1007/s10586-024-04712-z.
The supplied DRL-MLS Kubernetes PDF (DOI 10.1016/j.future.2026.108459) is a different
paper; do not merge their definitions. Assumptions A-001 through A-027 remain the
source of explicit model decisions. Completing software tests does not close source gaps.

## Handoff rules

Update progress.md, PROGRESS_REPORT.md, HANDOFF.md and this checklist when claiming
or completing work. Record actual tests and scope, not anticipated success. Inspect/
fetch main before integration; preserve collaborator changes and normal commit history.
Do not assume ignored datasets, venvs or trained artifacts exist on another machine.
