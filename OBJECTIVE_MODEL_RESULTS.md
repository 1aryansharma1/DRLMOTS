# Recorded enhanced-objective results (A-030)

Run date: 2026-10-26. These are actual synthetic simulation results, not forecasts.

Model: configs/objective_model.yaml; topology/source: configs/rl_network.yaml.
Horizon: 400 ms; 60 tasks per run; workload seeds 6200-6202; failure seeds 7100-7104.
Four reference rules, 12 no-fault runs and 120 fault runs: **132 simulations / 7,920 task arrivals across runs**.
Verification: **782 tests passed (50 new)**; dependency check clean. Final artifact audit: 132 traces, source/input hashes and paired accounting, 142 local links, 146 ZIP entries with CRC/byte parity, four PNG/four SVG figures. Representative plots inspected; browser layout remains a manual presentation check.

No DQN was trained or evaluated in this 132-run heuristic study. The later basic
DQN connection is documented in ENHANCED_DQN_INTEGRATION.md; its frozen inference
check is not part of these results. Earlier baselines are unchanged.

## Mean results across the declared panel

| Mode | Policy | Delivered / 60 | Rejected | Execution failure | Deadline missed | Total J | Successful mean ms |
|---|---|---:|---:|---:|---:|---:|---:|
| no_failures | least_queue_fcfs | 59.667 | 0.000 | 0.000 | 0.333 | 42.033 | 18.917 |
| no_failures | first_fit_fcfs | 59.333 | 0.000 | 0.000 | 0.667 | 42.265 | 29.217 |
| no_failures | fog_first_completion | 60.000 | 0.000 | 0.000 | 0.000 | 42.167 | 15.795 |
| no_failures | multi_objective | 60.000 | 0.000 | 0.000 | 0.000 | 41.655 | 16.151 |
| failures_no_recovery | least_queue_fcfs | 56.400 | 0.000 | 3.533 | 0.067 | 40.827 | 20.640 |
| failures_no_recovery | first_fit_fcfs | 56.133 | 0.267 | 2.933 | 0.667 | 41.007 | 29.635 |
| failures_no_recovery | fog_first_completion | 54.333 | 0.000 | 5.667 | 0.000 | 40.710 | 16.166 |
| failures_no_recovery | multi_objective | 55.600 | 0.000 | 4.400 | 0.000 | 40.327 | 16.608 |
| failures_with_recovery | least_queue_fcfs | 58.533 | 0.933 | 0.000 | 0.533 | 41.209 | 22.953 |
| failures_with_recovery | first_fit_fcfs | 55.933 | 2.800 | 0.000 | 1.267 | 41.101 | 31.002 |
| failures_with_recovery | fog_first_completion | 58.533 | 1.067 | 0.000 | 0.400 | 41.367 | 20.548 |
| failures_with_recovery | multi_objective | 58.800 | 0.800 | 0.000 | 0.400 | 40.877 | 19.749 |

All runs had zero unfinished tasks in this panel; censoring behavior is separately tested.
Every run met both configured 80% completion and matured-deadline service floors.
The enhanced multi-objective heuristic was the only empirical Pareto candidate with recovery in this panel. This is a result for these coefficients, policies and seeds; it does not establish global optimality or learned-policy superiority.

## Paired recovery effects: enabled minus disabled

| Policy | Extra delivered tasks | Approx. 95% CI | Extra total J | Approx. 95% CI |
|---|---:|---|---:|---|
| least_queue_fcfs | 2.133 | [0.805, 3.461] | 0.383 | [0.127, 0.638] |
| first_fit_fcfs | -0.200 | [-1.392, 0.992] | 0.094 | [-0.007, 0.194] |
| fog_first_completion | 4.200 | [2.691, 5.709] | 0.658 | [0.395, 0.921] |
| multi_objective | 3.200 | [2.203, 4.197] | 0.550 | [0.394, 0.706] |

Intervals average the same three-workload panel within each failure seed, then use five independent failure-seed means. They are approximate small-sample Student t intervals conditional on these workloads, not confidence over every deployment. Individual pairs are retained in evaluation.json.

For multi_objective, recovery raised delivered tasks from 55.600 to 58.800 of 60, while energy rose from 40.327 to 40.877 J and successful mean latency from 16.608 to 19.749 ms. It recovered extra work at a cost; it did not improve every objective simultaneously.

First fit completed slightly fewer tasks with retries on average (56.133 to 55.933): retries compete for queues and bandwidth, causing more new arrivals to be rejected. Its completion-effect interval includes zero. This counterexample is preserved rather than hidden.

## Interpret carefully

- Energy integrates all physical hosts, link idle/traffic and recovery overhead over the same 400 ms window. Failed and repeated partial work is included.
- Node uptime differs between no-fault and faulted modes, so failures may lower energy simply because hosts are down. Read delivered counts and deadline outcomes first.
- Successful makespan and successful mean latency exclude losses. Full-batch makespan is null if any task does not complete.
- Failure coefficients, MI, link rates and host power curves are synthetic demonstration values. No real hardware calibration or faithful paper replication is claimed.
- No cross-node correlated disasters, packet-level routing, checkpointing, facility cooling or fault-aware DQN training is included.

## Reproduce and present

Read [OBJECTIVE_MODEL_PRESENTATION.md](OBJECTIVE_MODEL_PRESENTATION.md) for all ten improvements, formulas, tested numeric examples and the professor walkthrough.

~~~powershell
.\.venv312\Scripts\python.exe -m drlmots.objective_report --config configs/rl_network.yaml --model-config configs/objective_model.yaml --workload-seeds 6200 6201 6202 --failure-seeds 7100 7101 7102 7103 7104 --output-dir outputs/objective-model/professor-final
Start-Process ".\outputs\objective-model\professor-final\objective-report.html"
~~~

Generated files (ignored by Git): objective-report.html, objective-report.zip, report.md, metrics.csv, experiment.json, evaluation.json, weight-sensitivity.json, 132 run traces and four PNG/four SVG plots. Extract the ZIP before opening it on another computer. Source/configuration/tests and this results summary are tracked for collaborators.
