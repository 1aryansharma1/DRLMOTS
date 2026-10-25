# Observed failure and recovery: verified results

Date: 2026-10-25. Decision A-029; independent simulator/accounting contract 1.

## Delivery and verification

- 732 tests passed in 107.77 seconds; 34 new recovery/configuration/report checks. Pip check: no broken requirements.
- Hand-calculated latency and energy tests cover partial uploads, propagation, interrupted service, repeated failures, bounded waits/retries, repair, common horizons and resource release.
- Randomized scenarios verify reproducibility, complete task/attempt accounting, no future-outage leakage and unchanged input/RNG state.
- No-failure regressions preserve existing network placements and finish times for the three original reference policies, including the six-VM fixture.
- Final artifact checks: 69 valid local links, 75 unique ZIP entries with valid CRC, 60 complete task/resource traces and 5 PNG/5 SVG figures. Representative figures inspected; legend overlap corrected. Normal desktop/mobile browser layout remains a manual presentation check.
- Actual experiment: four reference rules x three modes x five workload seeds = 60 simulations, 100 tasks each. The node outage trace is the same for recovery enabled/disabled.
- Fixed common accounting horizon: 250 ms. Five outages affect fog/cloud failure domains. This is one synthetic failure trace, not a statistical study over failure processes.
- Current DQN checkpoints/rewards are preserved and explicitly not interpreted as fault-aware. Learning-state/reward adaptation remains the next step before new DQN training.

## Results

Means over workload seeds 5200-5204; timing excludes rejected/permanently failed tasks. Fixed-window energy includes active compute, partial/repeated uploads, idle uptime and coordinator recovery work.

| Policy | Mode | Completed / 100 | Rejected | Permanent failure | Recovered | Makespan ms | Successful mean ms | Total J | J / completed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| least_queue_fcfs | no_failures | 98.000 | 2.000 | 0.000 | 0.000 | 119.157 | 12.474 | 34.801 | 0.355 |
| least_queue_fcfs | failures_no_recovery | 69.600 | 12.800 | 17.600 | 0.000 | 116.255 | 12.274 | 27.251 | 0.392 |
| least_queue_fcfs | failures_with_recovery | 72.200 | 27.800 | 0.000 | 14.800 | 128.035 | 15.483 | 28.874 | 0.400 |
| first_fit_fcfs | no_failures | 97.400 | 2.600 | 0.000 | 0.000 | 117.443 | 12.796 | 36.870 | 0.379 |
| first_fit_fcfs | failures_no_recovery | 65.800 | 14.600 | 19.600 | 0.000 | 120.579 | 12.865 | 27.272 | 0.414 |
| first_fit_fcfs | failures_with_recovery | 69.600 | 29.800 | 0.600 | 15.800 | 129.947 | 16.501 | 29.757 | 0.428 |
| fog_first_completion | no_failures | 99.400 | 0.600 | 0.000 | 0.000 | 114.400 | 10.852 | 27.570 | 0.277 |
| fog_first_completion | failures_no_recovery | 69.800 | 10.600 | 19.600 | 0.000 | 116.343 | 11.480 | 24.570 | 0.352 |
| fog_first_completion | failures_with_recovery | 76.000 | 24.000 | 0.000 | 17.200 | 121.536 | 14.228 | 27.886 | 0.367 |
| energy_latency | no_failures | 98.400 | 1.600 | 0.000 | 0.000 | 116.876 | 11.917 | 26.691 | 0.271 |
| energy_latency | failures_no_recovery | 68.400 | 14.400 | 17.200 | 0.000 | 117.254 | 12.312 | 22.826 | 0.334 |
| energy_latency | failures_with_recovery | 74.200 | 25.800 | 0.000 | 17.000 | 123.159 | 15.044 | 26.706 | 0.360 |

## Interpretation

With fog-first scheduling, recovery raises mean completed tasks from 69.8 to 76.0 (+6.2 percentage points of submitted work) in this fixed outage scenario. It reduces permanent failures from 19.6 to zero, but initial admission rejections rise from 10.6 to 24.0 because recoveries have queue priority and consume capacity. The affected-task populations also change as the queues evolve; do not interpret the two modes as identical per-task histories.

Recovery increases successful mean latency from 11.480 to 14.228 ms and fixed-window energy from 24.570 to 27.886 J. This is a measured service/reliability versus energy/time trade-off, not a free improvement. Conditional recovery success does not imply every submitted task is served. Some policies still permanently fail tasks after recovery.

Makespan can appear lower when tasks are lost. No-failure energy includes more node uptime than outage cases. Use all counts and common-window energy-per-completion alongside makespan; do not claim a universally best policy. These values must not be directly compared to the older active-only analytical-risk report as if the accounting definitions were identical.

## Follow one actual task

In workload 5200 / fog-first with recovery, synthetic-000003 arrives at 2.577353 ms. It is interrupted on fog-a1 at 10 ms, restarts on fog-b2 and is interrupted again at 20 ms, then restarts on cloud-2 and finishes at 36.720785 ms. Its final latency is 34.143432 ms = 30.771647 ms in attempts + 3.371786 ms in recovery delay/backup waiting. Its task-attributed energy is 0.937743 J, including both interrupted attempts, the successful third attempt and coordinator work. VM idle energy is accounted separately. The HTML contains this actual attempt table and an outage/execution timeline.

## Boundaries and next work

Failures are observed simulation events; they are not real hardware failures. Restart assumes immediate detection and durable source input, full reupload/full recomputation, another current up node, bounded retries and bounded backup wait. No proactive replica reservation, checkpoint restore, correlated cross-node stochastic process, output transfers, shared-link contention, source/controller failure or calibrated physical-host energy is claimed.

The objective vector is exact observed makespan, fixed-window energy and all-arrival unsuccessful fraction. The existing analytical multi-objective DQN remains a different preserved baseline. Before training on failures, define a new causal fault-aware state/action/reward contract and episode boundary that includes late failures/recovery after the last arrival. Then add untrained-versus-trained comparisons and validation curves on new splits.

## Reproduce and share

See FAILURE_RECOVERY.md for the full teaching approach, formulas and command. Local report: outputs/recovery/professor/recovery-report.html; share recovery-report.zip after extraction. Tracked results/configs/source/tests provide collaborator context. Generated traces/models/reports remain ignored by Git.
