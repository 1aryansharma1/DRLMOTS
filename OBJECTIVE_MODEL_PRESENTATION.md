# Presentation reference: makespan, energy and fault tolerance (A-030)

Implemented 2026-10-26. Start here for the professor presentation. This is a
simulation with configured workloads and machines, not real deployed cloud VMs.

Update: [ENHANCED_DQN_CONTRACT.md](ENHANCED_DQN_CONTRACT.md) now defines the standalone
observation/action/reward contract (milestone 1 of 5). The optional basic DQN connection
is now complete (milestone 2); see [ENHANCED_DQN_INTEGRATION.md](ENHANCED_DQN_INTEGRATION.md).
Connected small-scenario verification is also complete (milestone 3); see
[ENHANCED_DQN_VERIFICATION.md](ENHANCED_DQN_VERIFICATION.md) for independent time/energy/
reward examples and hidden-failure checks. Actual enhanced training and performance
comparisons remain milestones 4-5.

## The ten improvements, in simple language

| # | Implemented change | What it improves / benefit | Evidence to show |
|---|---|---|---|
| 1 | Independent CPU work in MI, VM speed in MIPS, and input/output kB | A faster VM finishes the same calculation sooner; a large input is no longer automatically a large computation | Candidate predictions and each attempt's executed_mi / compute phase |
| 2 | Seeded exponential or Weibull uptime per node | Failures occur through an explicit statistical model instead of only hand-picked outage times; Weibull can represent changing hazard with age | Saved outage_trace and configured distribution/scale/shape |
| 3 | Separate detection, repair, reboot, task restart and backup waiting | We can explain why recovery takes time and energy; a repaired node does not magically finish a lost task | recovery_intervals, node_events and reboot power plot |
| 4 | Crossed workload/failure trials, paired across every rule | A policy is evaluated on the same work and faults as its competitors; conclusions depend less on one lucky trace | experiment.json, paired deltas and intervals |
| 5 | Explicit VM-to-physical-host mapping, MIPS and memory admission checks | Several VMs share one physical machine; host idle electricity is counted once and allocations cannot exceed host budgets | configs/objective_model.yaml and host_energy |
| 6 | Utilization-dependent piecewise host power, with idle/down/reboot states | More concurrent VM computation changes physical power; measured load/power points can replace synthetic curves | timeline-power.svg and power_segments |
| 7 | Separate application deadlines from the observation cutoff | A task still running when measurement ends is unfinished/censored, rather than automatically called a failure | Outcome categories and undefined full-batch makespan |
| 8 | Shared upload/download bandwidth, propagation and partial-byte energy | Two VMs using one link now slow each other down; a task is successful only after its output arrives | link_segments, upload_sent_kb and download_sent_kb |
| 9 | Availability, recovery delay, deadline success, rejection and execution failure | Explains which kind of reliability improved and what it cost; rejection is not hidden inside execution failure | Metrics CSV, availability and paired recovery effects |
| 10 | Fixed normalization scales, explicit weights, service floors, Pareto screening and offline sensitivity scores | Prevents a scheduler that loses most work from winning simply because it used less electricity | objective.feasible/score, Pareto eligibility and weight-sensitivity.json |

All ten are implemented in the enhanced mode. The earlier A-029 recovery mode and
A-028 analytical/DQN mode remain available as reproducible baselines. Their energy
and runtime definitions differ: compare policies within one mode, not numbers from
incompatible modes as though they were the same experiment.

## Explain the workflow first

~~~mermaid
flowchart LR
    W[Task: arrival, MI, input/output, deadline] --> C[Observed-health and resource checks]
    C --> P[Placement rule compares candidate VMs]
    P --> U[Shared-link upload and propagation]
    U --> Q[VM FCFS queue]
    Q --> X[Compute: MI divided by VM MIPS]
    X --> D[Shared-link output and propagation]
    D --> S[Output delivered: completed]
    F[Node failure: all member VMs] --> R[Detection, restart overhead, backup wait]
    U --> F
    Q --> F
    X --> F
    D --> F
    R --> C
~~~

One task occupies a VM's compute slot at a time. Resource demands must fit that VM.
During computation it uses that VM's fixed aggregate MIPS reservation; demands are
feasibility constraints, not a second multiplier on MIPS. The host supports all its
configured VM reservations simultaneously without CPU overcommit. Memory is reserved
for the running attempt and released at compute completion or interruption. Input,
queue descriptors and durable source storage are assumed outside that working-memory
budget. Output transfer can overlap the next task's computation.

## 1. How time and makespan are calculated

For task i on VM v:

- Compute time: t_compute_ms = 1000 * work_MI / VM_MIPS.
- An uncongested transfer takes 8 * size_decimal_kB / bandwidth_Mbps milliseconds.
- With n active transfers on the same bottleneck, each receives B/n Mbps. Whenever
  a transfer joins/leaves, the engine recomputes rates and the next finish event.
- Propagation is added in each direction; it consumes elapsed time, not bandwidth.
- FCFS waiting includes time behind earlier admitted tasks. Input transfers overlap
  computation, so the engine records actual phases instead of adding overlapping
  stage predictions blindly.

A successful task's latency is output_delivery_time - original_arrival_time. Across
retries it equals the sum of all attempt elapsed times plus detection/restart/backup
blocking intervals. Failed work and repeated transfers are included in that total.

**Successful makespan** is the latest successful output delivery measured from time
zero. **Full-batch makespan** has a value only if every submitted task completes.
Otherwise it is null: an incomplete batch has no observed completion time. Never
claim that a smaller successful makespan proves improvement without showing all
outcome counts. Mean latency includes successful tasks only and is labelled as such.

A deadline is an absolute application requirement per task. If output is not delivered
by that deadline, the task is deadline_missed and cancelled. At the fixed measurement
cutoff H, remaining tasks are unfinished; they may eventually finish, but this run does
not observe that. The engine stops at H, releases reservations for cleanup, and does
not claim to simulate a subsequent drain period. Delivery exactly at the deadline,
failure time or cutoff wins. External arrivals must occur strictly before H.

## 2. How each physical host accounts for energy

The VM supplies its service speed and host mapping. The **physical host**, not each
VM separately, determines electricity:

u_host(t) = sum(MIPS of currently computing VMs) / host_capacity_MIPS.

With two configured endpoints, power is linear:

P_host(u) = P_idle + (P_full - P_idle) * u.

With more load/power points, interpolate between the enclosing pair. Each event
interval has constant utilization, so integrate exactly as P * elapsed_ms / 1000.

The complete modelled system total is:

E_total = sum_host(E_idle + E_dynamic + E_down + E_reboot)
          + sum_link(E_actual_bytes + E_link_idle) + E_recovery_coordinator.

Traffic energy = actual decimal kB serialized * configured joules_per_kB, in BOTH
directions and on every attempt. Interrupted transfers pay for bytes actually sent,
even if they never arrive. Down/reboot hosts use separate configured power; network
interfaces remain powered over H. Idle host power is billed once over host uptime,
even with no tasks. Recovery coordinator power is billed per recovering task during
its detection and restart intervals; backup waiting has zero incremental coordinator
power. This is a configurable overhead model, not a measured controller architecture.

For debugging, dynamic host energy is attributed to simultaneously running attempts
in proportion to their VM MIPS, plus each attempt's traffic energy. Shared host/link
idle and repair/reboot energy remain separate system overhead. Adding only task
attributions therefore does not equal total system electricity. This avoids arbitrary
idle allocation and double counting. Failed-attempt energy is a subset, not an extra
term to add again. No facility cooling/PUE, DVFS or sleep-state optimization is claimed.

## 3. Failure probability, actual failures and recovery

A node is a failure domain: all its VMs and hosts go down together. Different nodes
have independent configured random streams; same-node VM failures are correlated.
The experiment creates the complete trace before running policies, but placement
never reads future outages or repair times.

**Exponential uptime:** R(t) = exp(-t / scale), with mean uptime = scale. Its hazard
is constant. Sample each uptime, then add configured repair and reboot durations.

**Weibull uptime:** R(t) = exp(-(t / scale)^shape). Shape = 1 matches exponential;
shape > 1 gives increasing hazard with age. After repair/reboot, the node is assumed
as good as new and its age resets. This is an explicit modelling assumption; it is
not appropriate for every software or hardware failure process.

The causal risk estimate for an already-up node of observed age a, over predicted
exposure d, is 1 - R(a+d)/R(a). This conditional probability accounts for age in the
Weibull case. It is a placement estimate, not a guarantee of the task's eventual
success after retries, and it is not substituted for observed reliability results.

Actual node failure stops uploads, queued work, computation and partially transmitted
outputs on that node. An output whose serialization has finished is already in the
network: its propagation and delivery survive a subsequent host failure. This avoids
throwing away bytes that have already left the machine. Resources release immediately because the machine has failed. The
scheduler's health information can remain stale until detection: a new task may be
sent to the physically down node and then wait for the failure notice. After detection
and restart overhead, it seeks an eligible VM on a different node, waits up to the
backup-wait limit if needed, and repeats the entire input and calculation. There is
no checkpoint/resume. The durable input source and coordinator are assumed reliable.
Retries are bounded. Repair permits future placements; it never resurrects a terminal
task. If the node repairs before a delayed failure notice, that notice does not falsely
mark the repaired node unavailable.

Observed availability = measured uptime / H. Configured long-run availability for
this alternating renewal model is mean_uptime / (mean_uptime + repair + reboot).
For Weibull, mean_uptime = scale * Gamma(1 + 1/shape). Finite runs starting all-up need
not match the steady-state value. Exponential failure-rate MLE uses failure_count /
observed_uptime_seconds, including uptime censored at the observation boundary.
Small samples are unstable. This implementation reports that estimator; it does not
fit a Weibull model or certify a distribution against real uptime logs.

## 4. What the reliability metrics mean

| Metric | Definition / interpretation |
|---|---|
| Completion rate | Delivered outputs / all arrivals; includes rejection and unfinished in the denominator |
| Known terminal failure fraction | (rejected + execution_failure + deadline_missed) / arrivals |
| Observed unsuccessful fraction | 1 - completion_rate at H; includes censored work, so it is not an eventual failure probability |
| Execution failure | Admitted task exhausts retries or cannot obtain a backup before the recovery wait limit |
| Recovery success rate | Eventually delivered interrupted tasks / unique interrupted tasks within H |
| Recovery-to-delivery delay | First interruption to successful final output delivery, among recovered tasks |
| Deadline success | Delivered tasks whose deadlines matured by H / all arrivals whose deadlines matured by H |
| Energy per completion | Full fixed-window system joules / completed tasks; undefined if none complete |

A completed task whose deadline is after H is counted as completed, but is excluded
from the matured-deadline rate. A rejected task with a matured deadline counts against
that rate. Report the eligible deadline count with the rate. Empty denominators are
null, not 0 or 100%.

## 5. Multi-objective decisions and fair comparisons

The multi_objective rule scores each currently eligible VM using:

score_candidate = w_time * predicted_remaining_latency / decision_time_scale
                + w_energy * predicted_incremental_joules / decision_energy_scale
                + w_failure * conditional_outage_probability.

Weights are nonnegative and sum to one. Scales are predeclared positive constants,
not recomputed from the best policy's test results. Ratios are not clipped at one.
If at least one candidate is predicted to meet the deadline, multi_objective first
restricts selection to those candidates. Resource and observed-health constraints
apply to every rule. Predictions approximate current queue and link work; future
arrivals, contention and failures can make actual outcomes different.

Four reference rules are compared: first fit, least queue, fog-first predicted
completion (fog wins an exact tie), and multi-objective. This is heuristic placement,
not a proof of a global multi-objective optimum. Fog-first may choose a busy fog VM
when its predicted output delivery precedes cloud delivery. It chooses cloud when
cloud is predicted to finish sooner.

At report level, the vector is successful_makespan / time_scale, total_energy /
energy_scale, and observed_unsuccessful_fraction. Both completion and matured-deadline
rates must meet the configured service floors; otherwise score is null and the run
is infeasible. A candidate score and an experiment score use different scopes and
explicit separate scales: per-task prediction versus full-run observations.

Pareto screening compares total system energy, successful mean latency and all-arrival
unsuccessful fraction among policies meeting service floors on **every** evaluated run
of that mode. A policy is dominated if another is no worse on all three and better on
at least one. This screening is empirical, limited to the four evaluated rules and
fixed test panel. Offline sensitivity files rescore existing feasible outcomes with
time-, energy- and reliability-focused weights; they do not retrain or rerun placement.

## 6. How experiments establish evidence

The default panel uses three workload seeds (6200-6202), five independently sampled
failure seeds (7100-7104), four policies and three scenarios:

- No failures: once per workload/policy, 12 runs.
- Failures without retries: every workload x failure seed x policy, 60 runs.
- The same failures with retries enabled: 60 runs.

Total: **132 actual simulations**, 7,920 task arrivals across runs. Policies share the
exact task list and fault trace in each pair, verified by SHA256. The experiment
and run traces also store hashes of the model, engine and report source files. Recovery on/off is
paired. No-failure runs are not duplicated five times and treated as new observations.

Fault-mode uncertainty first averages over the fixed workload panel within each
failure seed, then forms an approximate Student t interval across those five means.
This is uncertainty over failure traces conditional on these workloads. It is not
132 independent trials or uncertainty over every possible deployment. Only five
failure traces gives broad intervals; the report preserves them and the individual
pairs. No-failure uncertainty is across workload seeds. Increase the predeclared panel
before making stronger research claims; do not select seeds because they favor a rule.

## 7. Worked examples you can explain at the board

These examples are tested in tests/unit/test_objective_model.py; they use a tiny
fixture, not the six-VM demonstration's coefficients.

**One task, no faults.** Task = 10 MI, input = 2 kB, output = 1 kB. VM = 1000 MIPS;
link = 8 Mbps, zero propagation. Upload = 2 ms, compute = 10 ms, output = 1 ms.
Output arrives at 13 ms. Fog host capacity = 2000 MIPS with P_idle=10 W and P_full=110 W;
this one VM gives u=0.5 and P=60 W. Over H=100 ms, fog idle = 1 J and dynamic = 0.5 J.
An idle cloud host at 20 W adds 2 J. At 0.01 J/kB, traffic = 0.03 J. Total = **3.53 J**.
A 4000-MIPS cloud VM computes this same task in 2.5 ms, independently of byte sizes.

**Two fog VMs, same host and link.** Two such tasks arrive together, one on each VM.
The 8-Mbps link gives each transfer 4 Mbps: both inputs finish at 4 ms. Both compute
4-14 ms, then both outputs finish at 16 ms. Host dynamic energy = 1 J, fog idle = 1 J,
cloud idle = 2 J, traffic = 0.06 J: **4.06 J**. Charging a whole physical host per VM
would be incorrect.

**Failure and restart.** First task uploads 0-2 ms and computes 2-5 ms when fog fails.
Detection completes at 7 ms; restart overhead ends at 8 ms. Cloud uploads 8-10 ms,
computes 10-12.5 ms and delivers output at 13.5 ms. Original latency = 5 ms first
attempt + 3 ms detection/restart + 5.5 ms second attempt = **13.5 ms**. The failed
attempt's 3 ms computation and full input still consume energy. With the test's host,
traffic, down/reboot and coordinator coefficients, the full-window total is **3.665 J**.

## 8. What to show the professor, in order

1. Introduce the three goals: output completion time, system electricity and service
   reliability. Explain why saving energy by rejecting work is not a useful win.
2. Show the VM/host/link/failure configuration. Point out synthetic coefficients and
   explain what real measurements would replace them.
3. Open the enhanced report. Start with outcome counts before latency or joules.
   Compare no failures, failures without retries and failures with retries.
4. Use the attempt timeline and physical-host power chart to explain one failure,
   interruption, detection, retry and delivered output. Use the worked example above
   if the professor asks for an auditable calculation.
5. Show paired recovery effects and their uncertainty. Explain any energy/latency
   increase as a possible cost of completing extra work; do not assume recovery wins
   every metric or every trace.
6. Show service-floor feasibility and the Pareto candidates. Open one run JSON to
   show actual bytes, phase durations, resource events and host energy segments.
7. State the RL boundary clearly: live trained DQN exists for the preserved earlier
   environment. **Basic masked DQN is now connected to this enhanced host/network/outage
   environment, but no training/performance experiment has been delivered.** See
   ENHANCED_DQN_INTEGRATION.md; the recorded study here compares heuristics.

The report frontend is an offline static HTML artifact with real precomputed tables,
SVG charts and links to CSV/JSON/ZIP. Opening it does not run Python or train a model.
It has no templating variables or network fetches, so opening the generated HTML with
a file URL works. Opening a source file under src/drlmots/templates is not the report.
Download/extract the ZIP before opening its HTML on another computer.

## Run it on Windows

From the repository root, after dependencies are installed:

~~~powershell
Set-Location "C:\Users\ashis\Desktop\Auxia Software\Basic Project\chat-app"
.\.venv312\Scripts\python.exe -m drlmots.objective_report --config configs/rl_network.yaml --model-config configs/objective_model.yaml --workload-seeds 6200 6201 6202 --failure-seeds 7100 7101 7102 7103 7104 --output-dir outputs/objective-model/professor-final
Start-Process ".\outputs\objective-model\professor-final\objective-report.html"
~~~

On a new checkout, create a Python 3.12 environment and install editable dependencies:

~~~powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install -e ".[dev,demo]"
.\.venv312\Scripts\python.exe -m pytest -q
~~~

The existing configs/rl_network.yaml supplies topology, resource units and source
reachability only in the enhanced mode. Its legacy byte workload and independent
VM upload rates are not reused: the new workload, speeds and shared links come from
configs/objective_model.yaml. This intentional separation preserves legacy results.
Choose a fresh output folder when changing an experiment plan. Generated artifacts
are ignored by Git; the model/config/tests and this guide are tracked. Share the
report ZIP separately or regenerate it. See OBJECTIVE_MODEL_RESULTS.md for the
actual recorded results, not just example calculations.

## File map and verification

| File | Responsibility |
|---|---|
| src/drlmots/objective_model.py | Strict model configuration, explicit WorkItems, seeded work/fault generation, power interpolation, conditional reliability |
| src/drlmots/objective_sim.py | Event scheduling, VM admission, shared transfers, failure/recovery, deadlines, host energy and metrics |
| src/drlmots/objective_report.py | Paired experiments, failure-cluster uncertainty, charts, score sensitivity, HTML/CSV/JSON/ZIP |
| configs/objective_model.yaml | Synthetic model coefficients, distributions, physical mapping and service floors |
| tests/unit/test_objective_model.py | Hand calculations, failure boundaries, conservation, stochastic and resource checks |
| tests/integration/test_objective_report.py | Pairing, uncertainty units, offline links, complete ZIP and experiment guards |

Tests check exact small cases and randomized conservation, not proof of real-world
accuracy. Remaining research work: measured power/MI/network/failure calibration,
distribution goodness-of-fit, correlated cross-node events, multi-hop network paths,
checkpoint/migration policies, facility energy and fault-aware DQN training/validation.
None is silently represented by an invented constant or a pretrained checkpoint.

## Primary references for the modelling choices

These justify model forms, not our synthetic coefficient values or paper fidelity.

- [CloudSim Plus MI/MIPS terminology](https://javadoc.io/static/org.cloudsimplus/cloudsim-plus/8.0.0/index-all.html).
- [CloudSim Plus utilization-based linear host power](https://javadoc.io/static/org.cloudsimplus/cloudsimplus/8.5.7/org/cloudsimplus/power/models/PowerModelHostSimple.html).
- [NIST exponential reliability](https://www.itl.nist.gov/div898/handbook/apr/section1/apr161.htm).
- [NIST Weibull reliability and mean](https://www.itl.nist.gov/div898/handbook/apr/section1/apr162.htm).
- [NIST likelihood with censored reliability data and small-sample cautions](https://www.itl.nist.gov/div898/handbook/apr/section4/apr412.htm).
- [SPEC power/performance measurement methodology](https://www.spec.org/power/docs/1.12/SPEC-Power_and_Performance_Methodology.pdf).

Availability, conditional survival and energy integration formulas above follow from
these chosen models. Our sharing discipline, durable-source assumption, perfect repair,
retry policy, service floors, confidence-interval panel and weights are explicit study
design choices, not requirements quoted from the user's paper.
