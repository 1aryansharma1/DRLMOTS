# Remaining work and collaborator pickup

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
