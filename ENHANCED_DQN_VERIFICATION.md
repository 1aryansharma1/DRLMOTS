# Connected enhanced DQN verification (A-033): milestone 3 of 5

Date: 2026-10-26. This document records controlled software verification of initial
placement, recovery, actual rewards and hidden failures. It is also a presentation
reference. The enhanced DQN is connected; an actual training sanity experiment and
the frozen untrained/trained/four-rule comparison are the remaining milestones.

Verification: **949 tests passed in 159.24 seconds** (pytest console result), including
**167 focused enhanced checks: 40 existing connection + 70 new scenarios + 57 contract**.
No failures, errors or skips. Dependency and whitespace checks are clean.

## What was checked

The existing 40 connected-DQN tests and 57 contract tests are supplemented by 70 new
scenario checks in `tests/integration/test_enhanced_dqn_scenarios.py`. Deterministic
test weights make the real network prefer fog-a1, then the other eligible choices.
These explicit fixture weights are not trained weights. Random initializations are
used for the conservation matrix. New live learning-path tests stay below replay
warmup; optimizer unit tests from milestone 2 remain software checks, not a training
performance result.

| Area | Connected scenarios and required behavior |
|---|---|
| Initial placement | Select a legal VM; allow waiting behind a busy VM when total resources fit; enforce source reachability, CPU/memory fit and queue capacity |
| Recovery | Exclude the last failed domain; restart full work on a different domain; preserve the original task ID; obey retry count, backup wait, deadline and cutoff |
| Failure phases | Interrupt upload, upload propagation, computation, queued work and partial output; output already fully serialized can finish propagation after the host fails |
| Repeated faults | A task can use its two retries; a third interrupted attempt becomes one permanent failure after detection, with no fourth attempt |
| Actual costs | Independent host watt-seconds, transferred-byte joules, recovery overhead, successful makespan and final task counts agree with rewards |
| Delayed consequences | Failure costs appear after their outcome becomes known, completion costs after delivery; recovery overhead accrues while detection/restart is open |
| No legal action | Forced events create no replay action; costs fold into the previous decision or an explicit prefix; waiting ends at expiry, deadline or cutoff |
| Finite horizon | Zero terminal bootstrap at H; unfinished remains censored in reports; no-completion time surrogate and service penalties are explicit |
| Hidden failures | Equal delivered-information prefixes produce equal observations/Q-values/actions despite hidden queue/resource cleanup or different future faults/repairs |
| Legitimate evidence | A missed periodic response may change telemetry age before a failure notice; delayed old notices do not mark a repaired node down or resurrect a newer failed node |
| Replay/checkpoints | Illegal forced actions and misplaced penalties are rejected, including rehashed checkpoint replay; ordinary checkpoint/RNG restoration remains supported |
| Interruption | Exceptions and KeyboardInterrupt abort only their own episode; no invented terminal outcomes/rewards; a fresh episode can start afterward |
| Baselines | Eight complete four-heuristic reports retain their pre-fix canonical hashes |

The conservation matrix covers **36 frozen episodes**: light/burst/tight-deadline
workloads x telemetry periods 0.3/3/11 ms x recovery off/on x two paired workload/failure
seeds (6200/7100 and 6201/7101). Each episode has 12 tasks. Failure scales are shortened
in memory to 20/30/40 ms for fog A/fog B/cloud to exercise more outages during work.
Every episode checks individual reward components, outcome conservation, final resource
balances, shared-link capacity and an unchanged learner fingerprint. These are stress
checks; their counts/coefficients are not a performance benchmark or tuned model.

## Two examples calculated by hand

The controlled fixture uses H=100 ms, fog VMs at 1000 MIPS, cloud VMs at 2000 MIPS,
two 2000-MIPS fog hosts and one 4000-MIPS cloud host. Fog host power is 10+100u W;
cloud host power is 20+200u W. All three links carry 1 decimal kB/ms with zero
propagation and idle power. Fog traffic costs .01 J/kB, cloud traffic .02 J/kB.
The task needs 10 MI, uploads 2 kB and returns 1 kB. These simplified coefficients
are applied only in test memory; `configs/objective_model.yaml` stays unchanged.

**No fault:** the network chooses fog-a1. Upload takes 2 ms, computation 10 ms,
output 1 ms: successful makespan 13 ms. All hosts' idle power is counted once over H:
40 W x .1 s = 4 J. The VM uses half its host: dynamic energy 50 W x .01 s = .5 J.
Traffic costs 3 x .01 = .03 J. **Total: 4.53 J.**

**One fault and recovery:** fog A fails at 5 ms, detection arrives at 7, restart is
ready at 8, repair ends at 20 and reboot ends at 22. The first attempt has uploaded
2 kB and computed 3 MI when it stops. Recovery uses cloud-1: upload 8-10 ms, compute
10-15 ms, output 15-16 ms. Original task completion is 16 ms.

| Energy contribution | Joules |
|---|---:|
| Fog A idle over 83 ms uptime | .830 |
| Fog A partial computation, 50 W for 3 ms | .150 |
| Fog A down, 1 W for 15 ms | .015 |
| Fog A reboot, 30 W for 2 ms | .060 |
| Fog B idle | 1.000 |
| Cloud idle | 2.000 |
| Cloud full restart computation | .500 |
| First upload plus full retry upload/output | .080 |
| Detection + restart coordinator, 10 W for 3 ms | .030 |
| **Total** | **4.665** |

For both cases, weights are time=.4, energy=.4, unsuccessful=.2, with fixed scales
100 ms and 10 J. All work is delivered, so no unsuccessful fraction or service
shortfall penalty applies:

~~~text
Return without fault = -(.4 * 13/100 + .4 * 4.53/10)  = -0.2332
Return with recovery = -(.4 * 16/100 + .4 * 4.665/10) = -0.2506
~~~

The first recovery transition ends at the next actual choice, t=8 ms. Only .493 J
has accrued by then, giving energy reward -.01972; no successful completion or
permanent failure has occurred. The terminal interval contains the remaining energy
and 16-ms successful makespan. Their sum is exactly the same episode objective within
floating-point tolerance. Sensor periods .3/1/7 ms give the same physical totals for
this controlled single-task example.

With the same fault and recovery disabled, partial work/detection costs total 4.095 J.
There is no successful makespan in the report. The training time surrogate is H,
unsuccessful fraction is 1, and completion shortfall penalty is 10 x .8 = 8:
return = -(.4 + .4 x 4.095/10 + .2 + 8) = **-8.7638**. This is an explicit training
cost, not a claim that the failed task finished at H.

## Issues found and fixed

1. **Forced no-action replay:** validation previously accepted an all-false state with
   `action=None`. The live controller already folded forced events correctly, but a
   manually supplied or restored malformed sample could enter replay. Validation now
   requires a legal integer VM action before insertion or checkpoint restoration.
2. **Reward timing guards:** nonterminal replay could carry a terminal service penalty,
   and a manually supplied zero-duration interval could carry integrated energy. Both
   are now rejected. The actual reward formulas and normalization are unchanged.
3. **Interrupted episode guard:** a simulation error/KeyboardInterrupt previously left
   `episode_active=True`, blocking further runs/checkpoints. The simulator now aborts
   the controller it began and re-raises the original interruption. A refused nested
   controller cannot release a different already-active run.
4. **Fractional recovery wait:** at some valid fractional timestamps,
   (ready_ms + wait_ms) - now rounded just above wait_ms, violating the strict feature
   contract. The derived remaining wait is now bounded by [0, configured wait_ms].
   The physical dispatch/expiry timestamps and contract validation remain unchanged.
5. **Same-time energy regrouping:** closing several open recovery intervals changed
   floating-point addition order, occasionally decreasing the cumulative reward cursor
   by about 4e-16 J at the same timestamp. The cursor now sums each recovery contribution
   with math.fsum instead of successively regrouping it. Strict monotonic/zero-duration
   reward guards remain in place; no energy cost or report physics is discarded.

On abort, an outstanding action without a completed feedback interval is discarded.
Already closed nonterminal transitions and any updates remain valid; completed-episode
counts do not increase. No terminal penalty, unfinished classification or bootstrap
boundary is invented. A learner checkpoint can be saved after this explicit abort
boundary, but it cannot resume the abandoned simulator. Frozen interruption leaves
the learner fingerprint unchanged. These semantics also apply to direct
`simulate_objectives(..., controller=...)` calls.

Contract 4, six-action ordering, JSON checkpoint schema 1, model coefficients and
training defaults are unchanged. Ordinary enhanced checkpoint round trips still work;
legacy DQN checkpoints remain incompatible. The four rule branches produce the same
eight complete canonical report hashes captured from commit fac9513 before these fixes
(workload seed 6200 x failure seeds 7100/7101 x four rules). The captured hashes and
configuration identities are tracked in `tests/fixtures/enhanced_rules_fac9513.json`.

## Reproduce the verification

From the repository root in PowerShell:

~~~powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv312\Scripts\python.exe -X utf8 -m pytest tests/integration/test_enhanced_dqn.py tests/integration/test_enhanced_dqn_scenarios.py tests/unit/test_enhanced_rl_contract.py -q
.\.venv312\Scripts\python.exe -X utf8 -m pytest -q --junitxml=outputs/objective-model/dqn-verification/pytest.xml
.\.venv312\Scripts\python.exe -m pip check
~~~

The final full run writes machine-readable results to
`outputs/objective-model/dqn-verification/pytest.xml`. Generated evidence stays under
the ignored outputs directory; source tests, expected heuristic hashes and this
verification guide are tracked so a collaborator can reproduce the checks. This delivery
also saves a source-hash-bound verification-summary.json beside the XML.

The suite checks specified scenarios and invariants; it is not a formal proof for
every input or a calibration against measured cloud machines. Sample-and-hold telemetry
is partially observed. Before a new sensor response, matching delivered information
must give matching policy inputs. After a missed heartbeat or changed delivered
throughput, different observations are expected and legitimate.

For a professor: explain the two hand calculations, show a failed attempt followed by
a different-node restart, then show how reward includes the failed partial work and
the full retry. Say that the controller is connected and verified on these scenarios.
Training quality and any efficiency advantage require milestones 4 and 5.

**Next: milestone 4.** Record a small reproducible training sanity run with real
optimizer updates, weight changes, losses and frozen before/after behavior on fixed
scenarios. This will establish that learning happens, without assuming improvement.
Then milestone 5 evaluates frozen untrained/trained DQN and the four heuristics on
matching held-out work/fault traces. Double DQN/PPO remain outside the agreed scope.
