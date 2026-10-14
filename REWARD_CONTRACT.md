# Synthetic reward and transition contract - version 1

This v0.3.0a3 checkpoint implements A-024's causal reward and baseline transition
export. The paper's multi-objective reward, energy and reliability remain unresolved.
This does not implement an interactive agent, Q-network, replay or learning loop.

## Reward at an arrival

For a legal selected VM, use its pre-placement observation:

```
projected_wait_ms = remaining_running_service_ms + queued_service_ms
own_service_ms = explicit_byte_length * 2 / 1000
L = projected_wait_ms + own_service_ms
reward = S / (S + L)
```

S is an explicit positive scale in ms. The fixture chooses 10 ms. All-false action
masks receive `-P`; the fixture chooses penalty P=1. Both parameters are configurable,
finite and positive. Rewards are dimensionless. Admission lies in [0,1] (floating
underflow may produce zero); rejection is strictly negative. With S=10: a 2 ms task
starting now gets 5/6, the same task behind 2 ms work gets 5/7, and rejection gets -1.
The ratio is evaluated without forming potentially overflowing S+L or L/S.

The reward reads no actual finish times or future arrivals. In this serial, FCFS,
zero-network model, later tasks append behind admitted work; the projection therefore
matches its eventual latency up to floating-point arithmetic. It would need review
for preemption, transfer delay, cancellation or execution failures. No reward is added
again when a task completes or the simulator drains.

`None` is legal only for forced rejection with an all-false mask. A selected index
must be an integer legal VM action; bool, negative, out-of-range and masked indices
are rejected. There is no voluntary reject action in this contract.

## Episode accounting

The report includes every arrival's reward, admitted/rejected counts, undiscounted
sum and average over **all** arrivals. Empty runs have return 0 and null average.
All-rejected N-task runs have return `-N*P`, not an artificial latency improvement.
A matched admitted decision is better than rejecting that task. Total return is a
surrogate tradeoff: it does not guarantee every higher-throughput schedule wins if
other tasks' waits change. Read completions, failures and timing beside return.
Nonrepresentable totals/projections raise an error instead of exporting NaN/Infinity.

## One transition per arrival

Each exported record contains task/time, raw state, action/mask, reward, next raw
state/mask/time and `done`. Next means the next chronological arrival, after same-time
completions and queued starts. Simultaneous arrivals may have zero elapsed time.
Only the last arrival is terminal: its next fields are null. Empty workloads have no
transitions. Terminal means the decision horizon ended; metrics still drain running
and queued tasks afterward. Reward is already fixed at placement.

Forced rejections have a null action/VM and an explicit `forced_rejection` marker.
An all-false next mask is **not** automatically terminal. Do not gather Q-values for
a null action or take max over an empty legal set. State schema 1 is unchanged;
see STATE_ACTION_CONTRACT.md.

The pure `build_learning_transitions(decisions, gamma=...)` adapter resolves this
boundary. For each real action, it absorbs following forced steps into
`R = r0 + gamma*r1 + ...`, stopping at the next actionable state or terminal horizon.
It records the number of arrival steps and continuation factor `gamma**steps`
(zero at terminal). Gamma is required, finite and in [0,1]; it discounts **arrivals**,
not milliseconds. Prefix forced steps have no preceding action: they stay in episode
reporting but do not invent a replay action. All-forced episodes have no learning
transitions. There is still no replay buffer, optimizer or interactive control loop.

## Run and replay

```powershell
.\.venv312\Scripts\python.exe -m drlmots --config configs/resources.yaml --reward-config configs/reward.yaml --output outputs/metrics/reward-example.json
.\.venv312\Scripts\python.exe -m drlmots --config configs/normalization_train.yaml --reward-config configs/reward.yaml --output outputs/metrics/reward-seeded.json
```

`--reward-config` implies decision capture. The run config and reward config are
separate: workloads/topology hashes remain the same whether scoring is enabled.
Saved `reward.configuration` records the algorithm, parameters and SHA-256. Remove
only its `sha256` before calling `parse_reward_config` for replay. The ordinary
unscored CLI and normalizer fitting/application retain their existing behavior.

The report adds `reward` configuration/summary/transitions and a reward breakdown on
each captured decision. Both baseline policies can be scored via `--policy`.
Python APIs: `RewardConfig`, `score_action`, `simulate(..., reward_config=...)`,
`build_transitions`, `build_learning_transitions` and `summarize_rewards`. Outputs
are ignored and regenerated. A future learner should normalize both current and
nonterminal next raw states with the saved training-only scaler before tensor input.

For the resource fixture, admissions receive 5/6, 5/6, 5/7, 5/7 followed by -1,-1.
Return is 23/21, mean is 23/126. Six transitions end with a forced terminal rejection;
the earlier forced rejection is nonterminal despite its all-false next mask. The
same 4 completed/2 failed, 4 ms makespan and restored balances remain.

## Verified checkpoint

510 tests pass on Python 3.12.14, including 50 new reward/transition tests; dependency
check is clean. The seeded 100-task fixture remains 96 completed/4 failed, with
100 rewards/transitions, one terminal record and return 67.73152495135943. Tests
cover causal forecasts, normalizer/baseline preservation, strict config/index handling,
empty/all-rejected runs, return overflow, save/replay and actual CLI. The learner
adapter's forced prefix/middle/suffix cases and gamma 0/0.5/1 are hand checked.
These results establish software behavior, not policy learning or paper reproduction.
