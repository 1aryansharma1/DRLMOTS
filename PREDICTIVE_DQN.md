# Predictive inputs and longer basic-DQN training (A-036)

This version implements points 1 and 2 of the DQN improvement plan: useful causal
prediction features and longer training with validation-only checkpoint selection.
The original 253-input DQN, its saved initialization/16-episode checkpoint, all four
native heuristics, earlier reports and the underlying simulator remain available.

The completed study and actual results are recorded in
[PREDICTIVE_DQN_RESULTS.md](PREDICTIVE_DQN_RESULTS.md). No improvement is assumed from
the feature change or training duration; the report records observed tradeoffs.

## What changed

The new network receives **325 inputs** for the default six-VM topology:

- The same **253 observed inputs** from telemetry contract 4.
- **12 calculated features per VM**, adding 72 inputs.
- The same six VM actions and legal masks for initial and recovery placement.
- The same basic masked DQN, replay, Adam, target network and contract-4 reward.

The predictor runs inside the network before its first learned layer. Replay retains
the immutable original observations; the same deterministic feature transform is
applied to current and next states during Bellman updates. It uses only the normalized
observation and fixed public configuration. It cannot inspect simulator dictionaries,
native candidate predictions, future arrivals, outages or exact future repair times.

The derived features have their own schema version 1, manifest and hash. This is a
new network/checkpoint variant, not a change to telemetry/reward contract 4. The
checkpoint kind is predictive_masked_dqn_checkpoint, incompatible with the earlier
enhanced_masked_dqn_checkpoint. Both formats keep full network/target/Adam/replay/RNG
state for exact restoration on the recorded runtime.

## The 12 derived features for each VM

| Feature | Meaning |
|---|---|
| upload_ms | Drain reported upload/download backlog, then transmit this task's input and propagate |
| queue_wait_ms | Estimated FCFS queue tail beyond input readiness |
| compute_ms | Task MI / VM MIPS, converted to milliseconds |
| download_ms | Own output transmission and propagation time |
| latency_ms | Estimated arrival-at-scheduler to output-delivery duration |
| finish_ms | Current observed time plus estimated latency |
| incremental_energy_j | Candidate host dynamic-power increment during computation plus own input/output traffic energy |
| deadline_slack_ms | Positive remaining deadline budget after predicted latency |
| deadline_lateness_ms | Predicted latency beyond the nonnegative remaining deadline budget |
| deadline_met | Predicted completion fits the deadline, or the task has no deadline |
| failure_probability | Configured conditional failure probability over predicted latency using observed uptime age |
| estimate_available | Health, uptime age and host utilization are known and the node is observed up |

These are approximate predictions, not future simulation outcomes. The original
telemetry-age/unknown flags remain available. An unavailable estimate is identified
explicitly; masks still exclude known-down or otherwise illegal choices. Prediction
features do not introduce a new hard deadline preference or change eligibility.

## Time, energy and risk calculations

Let B be reported upload + download backlog on the VM's half-duplex link, R its
bandwidth in Mbps, P propagation in ms, I/O task input/output in decimal kB, W task
work in MI, and V VM speed in MIPS:

~~~text
input_ready_ms = 8 × (B + I) / R + P
compute_ms = 1000 × W / V
output_ms = 8 × O / R + P

tail = reported remaining running time
for each observed waiting task in FCFS order:
    ready = 8 × B / R + P       if still uploading
            remaining P        if propagating
            0                  if ready
    tail = max(tail, ready) + 1000 × queued MI / V

queue_wait_ms = max(0, tail − input_ready_ms)
latency_ms = max(tail, input_ready_ms) + compute_ms + output_ms
finish_ms = observed_now + latency_ms
~~~

Reports use sample-and-hold: a stale remaining compute/upload estimate is not silently
updated from physical state. The upload approximation charges the currently reported
backlog before the candidate; the output approximation uses its own wire time and
does not invent future competing traffic. Actual simulation still uses processor
sharing, interruptions, delivered output and all real elapsed phases.
The measured aggregate backlog can lag a locally admitted upload until the next
sensor response; its known queue entry remains among the original raw inputs.
These predictors therefore remain approximations rather than an oracle.

For energy, remove the target VM's already-running allocation from reported host
utilization to estimate other-VM load u0. Let u1 be u0 plus the candidate VM's fixed
share of host MIPS, bounded by host capacity. Using the configured host power curve:

~~~text
predicted incremental energy = [Power(u1) − Power(u0)] × compute_ms / 1000
                               + (I + O) × link joules_per_kB
~~~

Shared idle energy is excluded from this placement prediction only. The actual
reward/report continues charging all host/link states through the full horizon,
including idle, down/reboot, lost work, retransmission and coordinator overhead.

For configured uptime survival S(t), predicted failure probability is
1 − S(age + latency) / S(age), using observed uptime age. Exponential and Weibull
parameters come from public configuration; distribution none gives zero. Unknown
age uses nominal age zero together with estimate_available=0. No sampled future
failure time is an input.

Times/energy use fixed public x/(x+scale) normalization; binary/probability features
stay unchanged. Slack is split into nonnegative spare budget and lateness. Healthy
sensor transport/reliability assumptions remain those of the earlier telemetry model.
The implementation uses CPU float32 and explicitly rejects a normalized nonbinary
input rounded to 1 rather than silently clipping an unrepresentable huge value.

## Longer training and checkpoint selection

The predeclared plan is configs/predictive_training.yaml. Learner settings retain
configs/enhanced_dqn.yaml: widths [64,64], Adam .001, batch/warmup 32, replay 2000,
target synchronization every 50 updates, gamma 1, gradient clipping 10, telemetry
period 1 ms and recovery enabled. No Double DQN, prioritized replay, demonstration
training, reward tuning or architecture-width search is included.

| Item | Plan |
|---|---|
| Model seeds | 42, 43, 44 |
| Budget prefixes | 100 and 300 episodes per model; train each trajectory once to 300 |
| Work per episode | 60 arrivals with configured failures and recovery |
| New training workloads/failures | 30000-30299 / 40000-40299 |
| Validation | Workloads 84000-84002, each without faults and paired with failure seeds 94000-94002: six cases |
| Validation cadence | Before training, then every 20 episodes |
| Fresh test | Workloads 85000-85002, each without faults and crossed with failure seeds 95000-95004: 18 cases |
| Exploration | Existing 1 to .05 schedule over 2000 legal decisions; frozen validation/test epsilon 0 |

All model seeds see the same training panel, isolating initialization/exploration
variability on that panel. The 100/300 budgets are correlated prefixes of one
trajectory, not two independently trained models. The primary report uses the
predeclared 300-episode budget, regardless of which test result looks better.

Selection considers trained checkpoints only, at scheduled validation stages within
each budget. Minimize this fixed tuple in order:

1. Number of validation cases failing either service floor.
2. Mean completion/deadline service-floor shortfall.
3. Mean penalized contract-4 cost across the six cases.
4. Earlier episode, to break exact ties.

Every selected checkpoint records its episode, updates, full fingerprint, selection
rank and eligibility for promotion. A selected model with any violating validation
case is explicitly marked ineligible for promotion. Initialization is a reference,
not a selectable trained checkpoint. The last episode is not automatically preferred.

Validation is frozen: no replay insertion, optimizer step, counter or RNG change.
Full fingerprints before/after each panel verify that. All models finish and write
selection-locked.json before the actual runner generates any fresh test inputs.
Training, validation, test and previous seen panels are checked for seed overlap.
Test results cannot alter selection, budget, features or settings.

## Reports and baseline preservation

The fresh panel evaluates each predictive initialization and each budget's selected
checkpoint for all three model seeds, plus the original untrained/trained-16 DQN
and the native multi-objective rule. Identical tasks/outages are passed to every
method. This gives 216 frozen test simulations in the full plan.

The default report shows trained predictive DQN, its untrained counterpart and
without-DQN across makespan, energy and completion. The existing 40/40/20 objective
provides a combined-cost ranking, with service floors enforced. Optional details
show both budgets, each selected episode, original baselines, validation/learning
curves and paired intervals across model-seed means on the fixed test panel.

Original native rules retain their different event-loop estimates; comparisons
against them do not isolate algorithm quality under identical information. Both
features and training duration change versus the original 16-episode DQN, so that
comparison does not isolate the feature effect either. No superiority, convergence,
hardware calibration or paper-reproduction claim follows automatically.

The six baseline implementation files remain byte-for-byte unchanged. The earlier
A-034 checkpoint provenance is validated before loading, and all inference is frozen.
The old report/output directories are not overwritten. New source hashes, full
compressed training traces, per-update losses, validation/test traces, selected/final
checkpoints, metrics and freeze audits are retained in the new experiment directory.

## Commands

From the repository root in the existing supported environment, choose a NEW output
directory. The baseline directory must contain the earlier A-034 saved checkpoints;
on another checkout regenerate them using ENHANCED_DQN_TRAINING_SANITY.md first.

~~~powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.predictive_training --plan-config configs/predictive_training.yaml --output-dir outputs/objective-model/predictive-training/my-study
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.predictive_report --directory outputs/objective-model/predictive-training/my-study
Start-Process ".\outputs\objective-model\predictive-training\my-study\report.html"
~~~

To run the selected predictive model on a workload with frozen live placement:

~~~powershell
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.predictive_inference --checkpoint outputs/objective-model/predictive-training/my-study/models/seed-42/selected-budget-300.json --workload-seed 85000 --failure-seed 95000 --output outputs/objective-model/predictive-training/my-inference.json
~~~

Use --no-failures for a no-outage replay. Context/settings come from the checkpoint;
the supplied topology/model must match. Inference does not overwrite an existing
output file. Legacy inference commands still use their original checkpoints.

Generated models/traces/reports stay ignored, following the existing artifact policy.
The compact report ZIP bundles figures, summary/provenance/selection data and CSVs;
large checkpoints/full traces remain outside it. Tracked source, plan, documentation
and tests support reproduction. Cross-runtime bitwise identity is not promised.
