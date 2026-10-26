# Enhanced DQN contract: milestone 1 of 5 (A-031)

Date: 2026-10-26. Status: executable standalone contract implemented and tested.
Milestone 1 verification: **839 full-suite tests passed, including 57 new contract checks**;
dependency check clean. This document is also a presentation reference. The enhanced simulator now also supports the optional basic masked-DQN connection
documented in [ENHANCED_DQN_INTEGRATION.md](ENHANCED_DQN_INTEGRATION.md). No enhanced
DQN training/performance experiment has been delivered. Connected scenario verification
is complete in [ENHANCED_DQN_VERIFICATION.md](ENHANCED_DQN_VERIFICATION.md).

The agreed scope is five milestones:

1. **Done:** define and test the enhanced observation/action/reward contract.
2. **Done (A-032):** connect a basic masked DQN to initial and recovery placement.
3. **Done (A-033):** verify the connected system on small scenarios and hidden-failure tests.
4. Pending: run a small training sanity check and demonstrate actual updates.
5. Pending: frozen paired evaluation of untrained DQN, trained DQN and four heuristics.

Double DQN, PPO, larger experiments and further algorithm upgrades are outside this
five-milestone scope. Existing A-028 DQN and A-030 objective results are preserved.

## Explain this to the professor in simple language

A contract is an agreement between the simulator and the learning scheduler:

- **Observation:** the information the scheduler is permitted to read right now.
- **Action:** the VM it may choose for the current placement request.
- **Reward:** a numerical account of actual consequences as time advances.

The simulator remains responsible for execution, shared bandwidth, physical power,
failures, detection and recovery. The future DQN will choose destinations. This first
milestone defines that boundary and the rules needed to test the later connection.
It does not yet demonstrate that a learned scheduler improves performance.

## Version, identity and reproducibility

Module: src/drlmots/enhanced_rl_contract.py. Contract version **4**, kind
**enhanced_scheduler_rl_contract**. This is distinct from legacy observation/reward
contracts and checkpoints. It does not change the A-030 simulator's schema version.

EnhancedRLContract.from_configs(run, model) validates the topology/model, then stores
an immutable canonical public context. It binds:

- Sorted VM action IDs, VM/host/node/link mappings and source reachability.
- VM queue limits and capacities, physical host curves and network coefficients.
- Configured failure distributions and nominal recovery settings.
- Declared task budget N and fixed measurement/training horizon H.
- Feature names, units, normalization scales, reward weights and service floors.
- Completion/deadline penalty coefficients and discount/termination semantics.

A SHA256 of the canonical context identifies the contract. Canonical context
rehydration validates version, feature layout, public model schemas and reward
semantics; extra sampled-failure fields are rejected. A future checkpoint loader
must also match the expected context freshly built from its run/model settings. Every encoded observation
and reward delta carries it. The next replay/checkpoint implementation must reject
mismatches even when tensor sizes happen to match. A manifest is a detached copy;
editing it does not mutate the contract.

Workload/failure RNG seeds, sampled outages, future tasks, future repair timestamps,
local file paths and learned weights are absent. The planned task COUNT is public;
the future tasks' identities, attributes and arrival times are not. Observation scales
come from declared configuration, with no fitting on validation/test observations.
Changing workload seeds alone leaves the contract hash unchanged. Changing public
speeds, capacities, scales, weights or floors changes it.

For an episode with a different declared task count, construct a corresponding
contract using planned_task_count. Empty training episodes are rejected.

## 1. Observation: exactly what the agent knows

For the current six-VM configuration the encoder produces **253 numbers** in a
fixed order. The dimension is derived from topology/resources/queue capacities.

| Group | Supplied information | Meaning |
|---|---|---|
| Episode | Time remaining; observed arrivals, completions and unsuccessful outcomes | Current progress, never the future workload list |
| Current task | Age, MI, input/output kB, resource demands, deadline-present/expired flags and remaining deadline | Work and urgency of this placement request |
| Recovery request | Initial/recovery flag, attempts already used, previous failed node, remaining backup-wait budget | Recovery context for this same original task |
| Node | Observed up/down/unknown, telemetry age, observed uptime age if known, detected failure count | Last known health, including stale/unknown information |
| Host | Last reported utilization, whether known, telemetry age, estimated up-state power from that utilization and the public curve | A scheduler-side power estimate, not hidden physical down/reboot power |
| Shared link | Observed upload/download flow counts and remaining bytes, telemetry age | Current reported communication contention |
| VM | Fog/cloud flag, speed, reachability, previous-node exclusion, running flag/remaining time, queue count and telemetry age | Current reported VM load and static configuration |
| Resources | Capacity-known flags, total and reported available capacity per VM | Queue admission uses total feasibility; a busy VM may still accept waiting work |
| Ordered queue slots | Present flag, upload/propagation/ready phase, remaining upload/propagation, MI and demands | FCFS order is preserved; empty slots are all zero |

Task IDs remain metadata for tracing. They are not neural-network features. Host/link/
node associations are fixed by the contract's per-VM layout and context.

All numeric nonbinary features use x/(x+s), where s is a fixed positive public scale.
Binary flags stay 0 or 1. The implementation uses an equivalent stable expression for
large x to avoid overflowing x+s. It does not fit or clip against held-out samples.
Unknown observations have explicit flags and zero placeholders; unknown is not silently
represented as a healthy idle machine. This normalization is separate from the reward's
linear fixed-scale ratios, which are not bounded or clipped.

### Telemetry boundary and information restrictions

A snapshot uses typed immutable tuples: ObservationSnapshot, PlacementRequest,
VMTelemetry, QueuedTelemetry, HostTelemetry, LinkTelemetry and NodeTelemetry. The
encoder validates exact ID coverage, finite values, phases, capacities and timestamps.
Reported timestamps cannot be later than the decision. Requests cannot arrive from the
future or be made at/after H. An expired task has an all-false placement mask.

Physical node failure may precede its observed failure notice. The future adapter
must maintain a scheduler-visible telemetry view and advance it only when information
is delivered. **Do not populate observations from the engine's hidden physical state,
current_outage, future edges or completed report.** In particular:

- Do not expose sampled next-failure or exact future repair times.
- Do not expose a new failure count before its detection notice.
- Do not reveal an undetected failure by immediately substituting hidden queue cleanup,
  actual down power or the engine's private last-up timestamp into the telemetry view.
- Keep reported workload/host/link information timestamped; stale observations may
  differ from actual physics. Observed uptime age is an estimate, not true hidden age.

The current API prevents extra raw physics fields and future timestamps from being
encoded. It cannot prove that a caller labelled its telemetry honestly. The A-032 connection supplies the live adapter and initial paired hidden-world tests;
see its integration guide. A-033 expands and verifies that connected scenario matrix.

This is an **observation**, not a claim of a fully observable Markov state. Delayed
health information and aggregated link traffic leave uncertainty. A feed-forward DQN
will be a baseline for this partially observed problem; no convergence guarantee is
claimed. History/recurrent policies can be considered after the agreed milestones.

## 2. Actions and masks

One discrete action per VM, sorted by ID:

| Action | VM in current configuration |
|---:|---|
| 0 | cloud-1 |
| 1 | cloud-2 |
| 2 | fog-a1 |
| 3 | fog-a2 |
| 4 | fog-b1 |
| 5 | fog-b2 |

A VM is eligible only when all these checks pass:

1. Reachable from the configured source.
2. Its node is observed available; unknown/down is excluded.
3. The task's demands fit its declared TOTAL capacities.
4. Its reported waiting queue has a free slot; running work is not a waiting slot.
5. For recovery, it is on a different node from the most recently failed attempt.
6. The task deadline has not expired and the episode has time remaining.

The request also validates its retry budget. Before initial placement, attempts_used=0,
previous_node=None and recovery waiting=0. Before a recovery placement, attempts_used
is the number of earlier attempts (1..max_retries), previous_node must be known, and
backup waiting remaining must be within the configured bound. The agent does not
choose the retry count, restart delay or repair duration.

Both fog and cloud remain available if eligible. There is no mandatory fog mask or
predicted-deadline filter for DQN. Predicted lateness alone does not make a VM physically
illegal. The existing multi_objective heuristic keeps its own decision rule unchanged.

validate_action rejects boolean/float/string actions, wrong contexts, wrong dimensions,
illegal indices and voluntary rejection. None is accepted only if the entire mask is
false. That is an environment-owned event: reject an initial request with no destination,
wait/expire a recovery according to existing rules, or cancel an expired deadline.
It is not a learnable seventh action. The future adapter must revalidate against its
fresh immutable observation at the same dispatch boundary.

Mask both exploratory/greedy action selection and future Bellman action values. Never
run a max over an all-false next mask. Preserve the simulator's same-time settlement,
recovery-before-new-arrival order and existing four heuristics when connecting DQN.

## 3. Reward: actual system costs and delayed outcomes

We use a system-level finite-episode objective with **gamma=1**. Each RewardSnapshot
contains cumulative, already-accrued values through its timestamp:

- Actual total joules: all host idle/dynamic/down/reboot, link traffic/idle and recovery
  coordinator costs, with shared host costs counted once.
- Observed submitted/completed/rejected/execution-failure/deadline-missed counts.
- Unfinished count only at H; matured-deadline eligible/success counts.
- Latest successful output-delivery timestamp M, or zero before any completion.

These are evaluator measurements, not extra observation features or future facts.
Physical energy is charged as it accrues. A future live adapter must include the
accrued part of OPEN detection/restart intervals, rather than waiting until a recovery
interval closes or charging its full planned duration early.

For a cumulative change from one boundary to the next:

~~~text
r_energy  = -w_E * (E_after - E_before) / E_scale
r_time    = -w_T * (M_after - M_before) / T_scale
r_failure = -w_F * (U_after - U_before) / N
~~~

U is rejected + execution_failure + deadline_missed + unfinished. Each original task
contributes at most once. An interrupted attempt that later recovers is not a terminal
unsuccessful task. Its partial/repeated energy remains in E, and its later successful
delivery affects M. No extra retry-energy term is added: that would double count.

At H add a service shortfall penalty:

~~~text
P = lambda_completion * max(0, minimum_completion_rate - completed/N)
  + lambda_deadline * max(0, minimum_deadline_success_rate - deadline_success_rate)
r_service = -P
~~~

The deadline term is omitted when no deadlines have matured. Rejections and failures
with matured deadlines count against deadline success, matching A-030 reports.
Both lambda defaults are **10**, explicit contract-bound starting hyperparameters,
not empirically optimized constants or guarantees. Report-level service feasibility
still determines whether a policy is eligible for recommendation. A finite penalty
does not mathematically enforce a hard completion/deadline constraint.

If NO task completes by H, add -w_T*H/T_scale as a defined training time surrogate.
The report still has no successful/full-batch makespan; no fabricated completion is
written back into results. This avoids treating zero delivered tasks as zero timing
cost. Known terminal failure and unfinished/censored remain separate report categories,
even though both represent undelivered service for this finite-window training cost.

For episodes with at least one completion, total undiscounted reward is exactly:

~~~text
-(w_T*M/T_scale + w_E*E/E_scale + w_F*U/N) - P
~~~

Thus for feasible A-030 runs it is the negative of the existing objective score. For
infeasible runs, where that report score is null, training retains a finite cost plus
penalty. This trains a weighted preference, not an entire Pareto frontier. The timing
term is SUCCESSFUL makespan, not mean latency or a claimed full-batch completion time.
All outcome counts must accompany timing in evaluation.

Because cumulative differences telescope, splitting an interval does not change
undiscounted return. Costs stay attached to the intervals in which they occur; no
future failure is rewarded/penalized at initial dispatch. RewardDelta records each
component, its start/end times, context, terminal flag and bootstrap discount.

### A small reward example

For N=2, H=400 ms, weights (.4 time/.4 energy/.2 failure), T_scale=400 ms and E_scale=50 J:

- By 20 ms, one task delivered at 15 ms and system energy is 1 J. Reward so far is
  -.4*(15/400) - .4*(1/50) = **-0.023**.
- By H, energy is 2 J and the other task was rejected. Both deadlines matured and
  only one succeeded. The remaining energy cost is -0.008; unsuccessful cost -0.1.
- Completion and deadline rates are .5 against floors .8, so the service penalty is
  10*(.8-.5) + 10*(.8-.5) = **6**.
- Total episode return is **-6.131**. The failed service is visible despite low energy.

This is an executable test, not a trained-model result.

## 4. Decision intervals, forced events and the episode end

The future adapter should expose decisions immediately before initial/recovery
placement after relevant same-time settlement and observed telemetry updates.
It should return each decision's actual subsequent interval reward and next eligible
observation, including intervening completions, failures, detection and restart costs.

When a request has no legal action, no invented VM action is stored in replay. Fold
environment-owned intervals and their rewards into the previous real decision until
another legal decision or H. Before the first legal decision, log costs as an explicit
episode prefix; include that prefix in the total-return audit, without fabricating a
transition for an action that never happened. An episode can have costs and zero
learnable actions; do not claim learning occurred in that case.

At H the explicitly finite-horizon TRAINING objective terminates: bootstrap discount=0.
Before H, gamma=1 and bootstrap discount=1 for a real nonterminal transition. At H all
N task outcomes must be accounted for, including unfinished. The engine must continue
accounting idle/background energy to the common H even if all tasks settle earlier.

This does not change the scientific meaning of right-censoring in reports. An external
interruption before H is not a completed finite-horizon episode: do not fabricate final
unfinished counts, service penalties or terminal targets from that interruption.

Full replay transitions, target computation, checkpoint adapters and DQN execution
are now provided by milestone 2 (A-032). This contract module defines their required
boundary; see the integration guide for the installed connection.

## 5. What is implemented and tested now

- Immutable public context/hash, typed telemetry, deterministic 253-feature encoding.
- Public fixed scales, explicit unknown flags, FCFS slot ordering and stale timestamps.
- Resource/reachability/health/queue/recovery/deadline masks and action validation.
- Cumulative reward snapshots/deltas, gamma=1, finite-horizon termination and penalties.
- Offline terminal_reward_snapshot accounting bridge for existing enhanced reports.
  This bridge validates public context and is NEVER an observation builder.
- Hand-calculated rewards, interval partition invariance, delayed failure accounting,
  cutoff semantics, malformed/future observations and immutable inputs.
- Actual reports from each of the four existing rules: feasible terminal reward equals
  negative objective score, and configuration mismatches are rejected.

These tests establish contract invariants. The A-032 integration adds a live adapter,
connected DQN and acceptance tests separately. Broader connected verification is recorded in A-033; actual
training and learned performance comparisons remain milestones 4-5.

## Minimal API example

~~~python
from drlmots.run_config import load_run_config
from drlmots.objective_model import load_objective_model
from drlmots.enhanced_rl_contract import EnhancedRLContract

contract = EnhancedRLContract.from_configs(
    load_run_config("configs/rl_network.yaml"),
    load_objective_model("configs/objective_model.yaml"),
)
print(contract.sha256)
print(contract.vm_ids)
print(len(contract.manifest()["features"]))  # 253 for this topology
~~~

The factory does not initialize a neural network, sample failures, alter simulator
placement or start training. See tests/unit/test_enhanced_rl_contract.py for complete
typed snapshot construction and reward examples.

## Run the focused tests

~~~powershell
Set-Location "C:\Users\ashis\Desktop\Auxia Software\Basic Project\chat-app"
.\.venv312\Scripts\python.exe -m pytest tests/unit/test_enhanced_rl_contract.py -q
~~~

Read [OBJECTIVE_MODEL_PRESENTATION.md](OBJECTIVE_MODEL_PRESENTATION.md) for the
underlying simulator's physical models and [OBJECTIVE_MODEL_RESULTS.md](OBJECTIVE_MODEL_RESULTS.md)
for its existing heuristic results. Existing enhanced report commands remain unchanged.
