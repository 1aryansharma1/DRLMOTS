# Multi-objective scheduling and efficiency reports

Current extension (2026-10-25, A-029): actual node outages, bounded restart recovery and full time/energy accounting now exist in a separate environment. See [FAILURE_RECOVERY.md](FAILURE_RECOVERY.md) and [RECOVERY_RESULTS.md](RECOVERY_RESULTS.md). The analytical-risk mode described below is retained; its DQN checkpoints are not fault-aware and are not relabelled as observed recovery results.

Updated 2026-10-25 (Asia/Calcutta). Prototype decision A-028.

The original latency-only training remains available. An optional reward contract 3
adds energy and analytical failure-risk objectives to live DQN placements. This is a
scalarized multi-objective DQN, not a multi-policy Pareto-learning algorithm.

## Source audit and interpretation

The supplied original DRLMOTS PDF (DOI 10.1007/s10586-024-04712-z), pages 9-11,
explicitly discusses makespan, energy and fault tolerance. Pages 10 and 11 were
rendered and visually inspected. Equations (17)/(18) motivate compute and input
transmission energy. Equation (19) divides energy by time; we report joules without
that division. Equations (20)/(21) concern redundant components, but no executable
failure/backup protocol or coefficients are provided. Equation (25) combines unlike
units and has inconsistent reward signs/directions. We do not silently implement
these as a well-defined, source-faithful reward.

Our documented extension uses per-task latency, active plus input energy, and an
analytical destination-failure-risk proxy. Actual makespan is also evaluated; latency
is a surrogate, not an exact makespan reward. **Actual fault tolerance, backup/retry,
failure injection and paper reproduction remain open.**

## Definitions

All coefficients in configs/objectives.yaml are illustrative synthetic inputs,
not measurements or values extracted from the paper. They deliberately create
trade-offs between fast, energy-efficient and reliable destinations.

- Compute energy J = active power W * service time ms / 1000.
- Input transmission energy J = input size decimal kB * configured J/kB.
- Total energy = compute + input transmission. Zero-network runs have no upload
  energy. Idle/queue/host power, outputs, backup work and network contention are
  excluded. Per-VM active energy is an accounting proxy, not shared-host energy.
- Estimated failure probability = 1 - exp(-hazard_per_second * latency_ms / 1000).
  The assumption is constant destination hazard over arrival-to-finish exposure,
  including upload/wait. No failures are sampled and queue execution is unchanged.
- Expected successes = sum(1 - risk) over completed tasks. Rejected tasks contribute
  zero. Expected success rate uses *all submitted tasks*, so rejecting tasks cannot
  artificially improve the reported service-success rate.
- Energy per expected success is undefined when expected successes are zero.

For admitted tasks the reward is:

    1 - [w_latency * L/(S_latency + L)
         + w_energy * E/(S_energy + E)
         + w_failure * p_failure]

The configured nonnegative weights sum to one. Default weights are .4/.4/.2;
latency scale is 10 ms, energy scale .15 J. Forced rejection keeps the configured
negative penalty (default -1). Every reward uses causal decision-time estimates;
evaluation recomputes energy/risk/utility from actual execution outcomes.
Weights and the complete coefficient catalogue are included in reward/context
hashes and full checkpoints. Changing them invalidates checkpoint compatibility.
The optional configuration is omitted entirely from legacy manifests, preserving
existing latency-only checkpoint hashes and APIs.

## Run the complete experiment

From the repository root, using the existing Python 3.12 environment:

~~~powershell
.\.venv312\Scripts\python.exe -m drlmots.efficiency --config configs/rl_network.yaml --training-config configs/efficiency_training.yaml --objectives configs/objectives.yaml --output-dir outputs/efficiency/professor-fresh --model-seeds 42 43 44
Start-Process .\outputs\efficiency\professor-fresh\efficiency-report.html
~~~

If the output already contains a matching experiment, add --resume. Completed
models are reused; incomplete models resume at an episode checkpoint. Different
plans are rejected. To start a different experiment, choose a fresh output folder.
Do not tune weights/hyperparameters against the displayed evaluation results.

The experiment trains nine networks: latency-only standard, balanced standard and
balanced fog-preferred, each at initialization/replay seeds 42, 43 and 44. Each gets
20 episodes x 100 tasks under training workload seeds 2000-2019. Calibration uses
only 2000-2002. The fixed final checkpoint is evaluated without selection/tuning.
Fresh workload seeds **4100-4104** are declared separately from the previously
published 3000-series workloads, before examining the new evaluation results.

Evaluation uses light (0-4 ms arrivals), normal (0-2 ms) and burst (simultaneous)
loads. All other task distributions and topology are the same. These are workload
stress tests, not the paper's dataset or topology-scaling experiments.

Policies compared:

1. Least queue, FCFS execution.
2. First fit, FCFS execution.
3. Fog-first earliest completion rule.
4. Objective-greedy rule: chooses the largest immediate common utility.
5. Latency-only DQN (ablation, existing reward).
6. Balanced DQN (all legal VMs).
7. Balanced fog-preferred DQN (existing explicit fog/cloud completion constraint).

The fog constraint can limit energy/reliability choices. It remains a user rule,
not a learned discovery. All seven policies receive identical task/config hashes
within each comparison, and all are evaluated using the *same* common objective
profile even though the latency-only agent trains with a different reward.

## Reports and interpretation

- efficiency-report.html: offline comparisons, training returns, trade-off plots,
  signed paired improvements and download links.
- REPORT.md: written method, actual training counts, metric tables and boundaries.
- metrics.csv: per-run makespan, mean/p95 latency, energy, completion, analytical
  reliability, common utility, occupancy and cloud fraction.
- paired-changes.csv: per-pair changes against least queue; positive means better,
  negative means worse, and zero/undefined baseline denominators remain undefined.
- evaluation.json: all summary statistics, run metrics, model evidence and plans.
- runs/*.json: actual task outcomes, resource events, decisions, Q-values and
  per-task objective breakdown. Full raw/normalized states remain in training
  traces/checkpoints; they are omitted from evaluation decisions to reduce size.
- models/*: exact resumable checkpoints, full training logs and last episode traces.
- figures/*: PNG and SVG charts.
- efficiency-report.zip: whitelisted report bundle. Extract before opening HTML.

Model-initialization SD is calculated over workload-averaged results of independent
trained models. Workload SD is calculated over model-averaged workloads. Reused rule
results are not treated as independent training runs. Neither standard deviation is
a confidence interval; no statistical significance or universal superiority is claimed.
P95 uses nearest rank within each run. Summary latency and p95 are averages of run
metrics, not pooled task-level means/quantiles. Read rejections alongside latency and
energy. The Pareto screen uses aggregate admission, latency, energy per expected
success and expected-success rate; it is descriptive, not statistical dominance.
Training returns use different reward contracts and must not be compared as if they
were the same score. The common evaluation utility permits an explicit comparison.

Generated models, traces, PDFs and bundles are ignored by Git. The tracked guide,
configurations, source, tests and MULTI_OBJECTIVE_RESULTS.md provide collaborator
context; regenerate the bundle or share it separately.

## Remaining research work

Validate coefficients against hardware/data; add actual failure/backup semantics
with their resource, latency and energy costs; use validation-only tuning before a
new untouched test set; increase initialization/workload counts; vary task/VM scales;
and complete the original paper's state, reward, topology and baseline audit.
The current implementation supports multiple objectives but does not resolve all
paper ambiguities or prove an optimal trade-off.
