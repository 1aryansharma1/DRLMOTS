# Enhanced masked DQN connection (A-032): milestone 2 of 5

Date: 2026-10-26. **Initial and recovery placement are now connected to a basic
masked DQN.** The four A-030 rules and all earlier simulator/DQN interfaces remain.
This milestone delivered the working connection and verification. A-034 now records a
small actual training sanity experiment in ENHANCED_DQN_TRAINING_SANITY.md; a broader
trained-model comparison remains milestone 5. The default inference command below
still initializes an untrained network unless a compatible checkpoint is supplied.

Update (A-033): connected scenario verification is complete. Read
[ENHANCED_DQN_VERIFICATION.md](ENHANCED_DQN_VERIFICATION.md) for independent totals,
hidden-failure checks, regression results and interruption/replay fixes.

Read [ENHANCED_DQN_CONTRACT.md](ENHANCED_DQN_CONTRACT.md) for the 253-feature contract,
its reward formulas and the distinction between observed and physical state.

## What now happens when a task needs a VM

1. The engine settles due events and processes delivered health information.
2. The telemetry cache builds an immutable contract-4 observation for the current task.
3. DQN emits six Q-values. Greedy or exploratory selection considers only eligible VMs.
4. The engine validates that action and dispatches the task to that VM.
5. On a fault, actual work stops immediately, while the scheduler's health view can
   remain stale. After detection/restart, the same controller selects a different
   eligible node for recovery. Retry limits, deadlines and backup waiting stay in the engine.
6. At the next real decision, actual costs since the previous decision become its
   replay transition. At H, the final transition includes remaining costs/outcomes.

The network is a CPU float32 MLP: **253 inputs -> 64 ReLU -> 64 ReLU -> 6 Q-values**
for the current configuration. Widths are explicit settings. It is ordinary DQN:
the frozen target network supplies the maximum LEGAL next-action value. Double DQN,
dueling networks, PPO and new comparative training studies are outside this milestone.

## Delivered telemetry instead of hidden physics

The optional controller path uses a periodic sensor protocol. Default period: **1 ms**.
Poll times are fixed multiples of that period; they do not depend on hidden failures.
For a physically available node, its response supplies VM/host state and that node's
link-traffic contribution. A failed node sends no response, so its last report remains
cached. The assumed healthy-node sensor response has zero transport delay and no
extra sensor-energy cost in this version; this is an explicit modelling assumption.

Only delivered observations enter the encoder. Neither the controller nor its network
receives the outage schedule, physical-state dictionaries, future repair edges, actual
hidden queue cleanup, private uptime-reset time or heuristic candidate predictions.

The cache uses sample-and-hold values: reported remaining work/utilization stays at
its last reported value, with telemetry-age features showing its freshness. It does
not silently update a down node's queues from the engine before detection. Missing
periodic responses are themselves observable; the learner may infer trouble from stale
reports. A healthy node's changed throughput on a shared link is also a legitimate
measurement, rather than a guarantee that every hidden-world observation stays equal.
Tests compare worlds that have the SAME delivered information before a sensor response.

Local dispatches reserve an observed waiting slot immediately. This is known because
the scheduler sent the task; it is not a new heartbeat. It prevents repeated admissions
from exceeding queue capacity between sensor polls. A conservative stale view may
reject/wait even when actual resources have become free. Link measurements remain
reported samples, rather than pretending that a local send is an acknowledged transfer.

Failure notices update observed health and clear the failed node's reported allocations.
Delivered service-ready notices permit future placements. An old detection notice
cannot undo a newer delivered ready notice. Observed uptime age follows delivered
health history; it does not reveal a hidden repair reset. All hosts, including unused
physical hosts, have an explicit static failure-domain map.

The three link views aggregate participating nodes' latest reports in deterministic
node order. A report from an unrelated failed node does not make another link stale.
Measured sums can conservatively include stale traffic from a failed participant until
its notice arrives. This remains a partially observed scheduling problem, not a claim
that the 253 features reveal the complete physical state.

## Actions and preserved rules

The action ordering and eligibility remain those of contract 4. Both tiers are available.
Selection masks apply during epsilon exploration, greedy inference and Bellman targets.
Recovery excludes the last failed node. Voluntary rejection/waiting is not a seventh
learnable action. All-false requests are handled by the existing environment rules.

The simulator API is opt-in:

~~~python
simulate_objectives(run, model, policy="masked_dqn", controller=controller, ...)
~~~

A controller cannot be passed with a heuristic name or omitted for masked_dqn. The
four values in MODEL_POLICIES are unchanged: least_queue_fcfs, first_fit_fcfs,
fog_first_completion and multi_objective. Their original branch has no telemetry
poll events or DQN import. Eight complete reference reports (four rules x two failure
seeds) were captured before editing and matched exactly afterward by canonical hash.
Their existing report commands retain their original meanings. New DQN polling may
split floating-point integration intervals, so comparisons to traces with another
poll cadence should use numerical tolerances for physical integrals.

## Actual rewards and replay

The controller uses contract-4 cumulative differences at actual decision times:
full host/link/coordinator energy, successful makespan increments and final unsuccessful
original tasks. Detection and restart overhead accrue while recovery intervals are
OPEN. Waiting beyond restart has no incremental coordinator power. Failed/repeated
attempt costs are already included and are never added twice. Background system
energy continues to H even after the workload settles.

Gamma is 1. Forced no-action events do not fabricate replay actions. Their costs fold
into the preceding real action until another legal decision or H. Costs before the
first legal action are an explicit prefix. If no legal action ever occurs, the episode
has a terminal prefix and zero replay items/optimizer updates. Initial and recovery
choices both use the same network, action validation and delayed feedback machinery.

Each episode checks:

~~~text
prefix reward + sum(real-decision transition rewards)
    == reward from initial zero counters to the final cumulative snapshot
~~~

This includes the no-completion surrogate and terminal service penalties defined in
the contract. Unfinished remains censored in reports even though the finite-horizon
training episode terminates with zero bootstrap at H. No mean-latency reward is silently
substituted for the specified successful-makespan objective.

Replay entries are frozen typed records with immutable observations and context hashes.
A nonterminal item must have at least one legal next action; no maximum is computed
over an all-false mask. Replay is bounded and samples with a local seeded RNG. The
learner uses Adam, Huber loss, gradient-norm clipping, warmup and explicit target sync.
Nonfinite values, mismatched contexts and invalid masks are rejected.

## Frozen evaluation and checkpoint boundaries

run_enhanced_episode(..., learn=False) disables exploration, replay insertion, optimizer
updates and training-counter changes. It hashes the learner before/after and rejects
mutation. Initialization and frozen selection leave global Python/Torch RNGs unchanged.
Selectors and replay use separate local streams. Loading a checkpoint also validates
its contract, environment and configuration before use.

The new JSON checkpoint kind is enhanced_masked_dqn_checkpoint, schema 1. It stores
online/target weights, Adam state, bounded replay, local RNGs, decision/update/episode
counters, losses and a checksum. Network dimensions/dtype, optimizer settings/state,
public context, telemetry protocol/period, host-node map and recovery mode are validated.
Old DQN checkpoints cannot be loaded into this learner. Period and recovery mode bind
the environment hash even though the contract-4 feature layout stays unchanged.

Checkpoints are allowed outside an active episode, after normal completion or an
explicit abort boundary (A-033). Mid-episode simulator resume is not implemented.
An interruption aborts only its own controller and re-raises the exception. The open
action is discarded; closed nonterminal transitions/updates are retained and completed-
episode counts do not increase. No unfinished outcome or terminal reward is fabricated.
A checkpoint after abort stores valid learner state, not the abandoned simulator.
Exact round-trip/subsequent-update equality is tested on this CPU/runtime; no cross-
hardware or cross-library-version bitwise equivalence is claimed.

## Run frozen inference now

From the repository root:

~~~powershell
Set-Location "C:\Users\ashis\Desktop\Auxia Software\Basic Project\chat-app"
.\.venv312\Scripts\python.exe -m drlmots.enhanced_dqn --config configs/rl_network.yaml --model-config configs/objective_model.yaml --dqn-config configs/enhanced_dqn.yaml --workload-seed 6200 --failure-seed 7100 --output outputs/objective-model/dqn-integration/inference.json
~~~

This initializes an **untrained** network and runs greedy frozen inference. The output
is a complete JSON trace, not a new HTML benchmark dashboard. It includes task attempts,
resources, energy, observed placement inputs/masks, Q-values, delivered telemetry,
transition reward components and the final return audit. Opening this JSON does not train.
The existing heuristic HTML/CSV/ZIP report remains available separately; the combined
untrained/trained/four-rule report is the remaining milestone 5.

To load a compatible checkpoint, add --checkpoint path/to/checkpoint.json with its
matching --dqn-config. The CLI has no training flag. Use a fresh output file if you
want to retain an earlier inference trace. Generated files stay under the ignored
outputs/objective-model/ tree; source/config/tests and this guide are tracked.

## API for the next milestone

~~~python
from drlmots.enhanced_dqn import EnhancedDQN, EnhancedDQNConfig, run_enhanced_episode

agent = EnhancedDQN.from_configs(run, model, EnhancedDQNConfig())
report = run_enhanced_episode(agent, run, model, learn=False,
                              workload_seed=6200, failure_seed=7100)
agent.save("outputs/objective-model/dqn-integration/initialization.json")
restored = EnhancedDQN.load("outputs/objective-model/dqn-integration/initialization.json",
                           agent.contract, host_nodes=agent.host_nodes)
~~~

The learning path uses learn=True. A-034 has now run a 16-episode actual training
sanity experiment: 999 optimizer updates and changed initial/recovery probe actions.
See ENHANCED_DQN_TRAINING_SANITY.md for mixed diagnostic results and the trained
checkpoint command. Earlier tiny tests stayed below warmup; their outcomes remain
software checks rather than this actual training result.

Default settings are explicit in configs/enhanced_dqn.yaml: seed 42, widths [64,64],
learning rate .001, replay 2000, batch/warmup 32, target sync every 50 optimizer updates,
gradient norm 10, epsilon 1 -> .05 over 2000 learnable decisions, telemetry period 1 ms,
recovery enabled. These are wiring defaults, not validated best hyperparameters. Sensor
cadences that exceed one million polls or lose clock precision are rejected.

## Verification and remaining scope

Focused tests cover initial/recovery network choices, resource releases, action/target
masking, stale and future failures, shared-link hidden cleanup, detection/repair notices,
open recovery energy, forced prefixes/waits, censoring, queue admission, immutable
replay, checkpoint/optimizer/RNG restoration, frozen inference, unused hosts and CLI.
These are integration acceptance checks required to deliver milestone 2. They do not
replace the additional A-033 scenario matrix documented in ENHANCED_DQN_VERIFICATION.md.
A-034 adds actual training/diagnostic curves; the broader held-out comparison remains milestone 5.

~~~powershell
.\.venv312\Scripts\python.exe -m pytest tests/integration/test_enhanced_dqn.py -q
~~~

File map:

| File | Role |
|---|---|
| src/drlmots/enhanced_telemetry.py | Delivered sensor/notice cache and local admission tracking |
| src/drlmots/enhanced_dqn.py | New masked network/learner, replay controller, checkpoints and frozen CLI |
| src/drlmots/objective_sim.py | Optional DQN dispatch, sensor delivery and actual reward cursors |
| configs/enhanced_dqn.yaml | Explicit wiring defaults |
| tests/integration/test_enhanced_dqn.py | Connected acceptance and preservation checks |

The roadmap is now: **1 contract done; 2 connection done; 3 broader connected small-
scenario verification done (A-033); 4 actual training sanity done (A-034); 5 frozen
untrained/trained/four-heuristic comparison pending.** No superiority or paper-reproduction
claim follows from connecting an untrained network.
