# Multi-objective implementation: verified results

Date: 2026-10-25 (Asia/Calcutta). Decision A-028.

The latency-only scheduler is retained. The extension trains a scalarized latency, energy and analytical-risk DQN and compares it with rules. These are synthetic simulation results, not measured hardware efficiency, fault-recovery performance or paper reproduction.

## Verification and experiment

- 698 tests passed in 99.80 seconds; the final report-packaging test also passed after adding training-log downloads. Dependency check: no broken requirements. Final artifacts: 86 valid local links, 96 ZIP entries with valid CRC, 45 paired trace groups, 9 PNG/9 SVG figures; resource balances verified for all saved outcomes. Representative figures visually inspected; normal browser layout remains a manual check.
- Original supplied DRLMOTS paper: objective pages 9-11 extracted; pages 10/11 rendered and visually inspected. Equations/units and unresolved source requirements are recorded in MULTI_OBJECTIVE.md.
- Nine models: three variants x initialization/replay seeds 42, 43, 44. Each trained for 20 episodes x 100 tasks: 180 episodes / 18,000 submitted training tasks.
- Actual optimizer updates: 17062. All nine initialized weight hashes changed; complete checkpoints round-tripped.
- Training workloads 2000-2019; calibration subset 2000-2002. Fresh fixed test workloads 4100-4104, declared before examining their results; no tuning/selection on them.
- Light, normal and burst scenarios, seven policies, identical per-pair configuration/task hashes. Greedy evaluation did not change any saved training-state fingerprint.
- 195 distinct evaluation simulations (60 rule runs + 135 trained-model runs), each with 100 tasks. There are 315 paired comparison rows because rules are reused as references for each initialization, not independent replications.
- Model SD is across workload-averaged independent models; workload SD is across model-averaged workloads. Neither is a confidence interval. Three initializations are a limited pilot, not a significance study.

## Actual training

| Model | Episodes | Optimizer updates |
|---|---:|---:|
| balanced_dqn-42 | 20 | 1895 |
| balanced_dqn-43 | 20 | 1882 |
| balanced_dqn-44 | 20 | 1874 |
| fog_balanced_dqn-42 | 20 | 1917 |
| fog_balanced_dqn-43 | 20 | 1920 |
| fog_balanced_dqn-44 | 20 | 1914 |
| latency_dqn-42 | 20 | 1885 |
| latency_dqn-43 | 20 | 1887 |
| latency_dqn-44 | 20 | 1888 |

## light load

Means across the declared workloads and, for DQN, model initializations. Timing excludes rejected tasks. Energy is active compute plus input transmission only. Expected success includes rejection as zero success.

| Policy | Completed / 100 | Makespan ms | Mean latency ms | Energy J | J / expected success | Analytical risk / admitted % | Expected successes / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| least_queue_fcfs | 100.000 | 210.137 | 11.910 | 31.785 | 0.348 | 8.636 | 91.364 |
| first_fit_fcfs | 100.000 | 213.949 | 13.521 | 43.190 | 0.476 | 9.317 | 90.683 |
| fog_first_completion | 100.000 | 204.889 | 9.274 | 18.828 | 0.218 | 13.578 | 86.422 |
| objective_greedy | 100.000 | 207.896 | 11.597 | 12.218 | 0.134 | 8.904 | 91.096 |
| latency_dqn | 100.000 | 209.416 | 12.066 | 28.780 | 0.320 | 10.138 | 89.862 |
| balanced_dqn | 100.000 | 208.773 | 12.128 | 26.337 | 0.294 | 10.188 | 89.812 |
| fog_balanced_dqn | 100.000 | 207.050 | 10.590 | 16.032 | 0.182 | 11.422 | 88.578 |

Descriptive non-dominated policies: least_queue_fcfs, fog_first_completion, objective_greedy, fog_balanced_dqn. This uses aggregate admission, latency, energy per expected success and expected success rate; it is not statistical dominance.

## normal load

Means across the declared workloads and, for DQN, model initializations. Timing excludes rejected tasks. Energy is active compute plus input transmission only. Expected success includes rejection as zero success.

| Policy | Completed / 100 | Makespan ms | Mean latency ms | Energy J | J / expected success | Analytical risk / admitted % | Expected successes / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| least_queue_fcfs | 95.000 | 114.400 | 12.591 | 28.023 | 0.328 | 9.931 | 85.573 |
| first_fit_fcfs | 95.000 | 114.303 | 12.930 | 29.491 | 0.347 | 10.608 | 84.927 |
| fog_first_completion | 97.800 | 111.631 | 11.353 | 21.844 | 0.252 | 11.468 | 86.586 |
| objective_greedy | 95.200 | 112.572 | 12.596 | 21.805 | 0.258 | 10.944 | 84.787 |
| latency_dqn | 94.800 | 113.522 | 12.812 | 26.940 | 0.317 | 10.383 | 84.958 |
| balanced_dqn | 95.133 | 112.811 | 12.815 | 25.957 | 0.305 | 10.395 | 85.251 |
| fog_balanced_dqn | 97.200 | 112.654 | 12.059 | 22.750 | 0.263 | 11.061 | 86.448 |

Descriptive non-dominated policies: fog_first_completion. This uses aggregate admission, latency, energy per expected success and expected success rate; it is not statistical dominance.

## burst load

Means across the declared workloads and, for DQN, model initializations. Timing excludes rejected tasks. Energy is active compute plus input transmission only. Expected success includes rejection as zero success.

| Policy | Completed / 100 | Makespan ms | Mean latency ms | Energy J | J / expected success | Analytical risk / admitted % | Expected successes / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| least_queue_fcfs | 12.000 | 24.003 | 14.353 | 4.009 | 0.369 | 9.536 | 10.856 |
| first_fit_fcfs | 12.000 | 22.997 | 14.611 | 3.872 | 0.355 | 9.101 | 10.908 |
| fog_first_completion | 12.000 | 20.955 | 12.853 | 3.844 | 0.352 | 8.881 | 10.934 |
| objective_greedy | 12.000 | 22.352 | 13.930 | 3.906 | 0.359 | 9.325 | 10.881 |
| latency_dqn | 12.000 | 22.338 | 14.184 | 3.931 | 0.362 | 9.440 | 10.867 |
| balanced_dqn | 12.000 | 22.443 | 14.384 | 3.852 | 0.355 | 9.412 | 10.871 |
| fog_balanced_dqn | 12.000 | 21.737 | 13.692 | 3.881 | 0.356 | 9.105 | 10.907 |

Descriptive non-dominated policies: fog_first_completion. This uses aggregate admission, latency, energy per expected success and expected success rate; it is not statistical dominance.

## Normal-load paired changes against least queue

Positive means better; negative means worse. Mean of relative changes calculated within each matching workload/model pair, rather than a ratio of aggregate means.

| Policy | Energy improvement % | Mean latency improvement % | Admission improvement % | Admitted-risk improvement % | All-arrival expected-success improvement % |
|---|---:|---:|---:|---:|---:|
| latency_dqn | 3.832 | -1.794 | -0.195 | -4.578 | -0.700 |
| balanced_dqn | 7.322 | -1.804 | 0.155 | -4.744 | -0.355 |
| fog_balanced_dqn | 18.710 | 4.163 | 2.342 | -11.547 | 1.064 |
| fog_first_completion | 21.954 | 9.853 | 2.978 | -15.579 | 1.224 |

## Interpretation and next work

Fog-preferred balanced DQN reduces energy and latency versus least queue in normal load, but its risk per admitted task is higher. Improved admission partly offsets that risk in the all-arrival expected-success metric. Balanced unconstrained DQN saves energy but does not improve every other objective. The simple fog-first rule remains stronger in normal/burst aggregate comparisons. Do not claim DQN superiority or an optimal/converged trade-off.

All policies complete only 12/100 burst tasks under these queue settings: positive upload readiness means all simultaneous arrivals first consume waiting slots. Low burst energy largely reflects rejection, not evidence of high efficiency. Larger buffers/backpressure or admission/retry changes require a separate experiment.

The implemented time reward is per-task latency, a makespan-related surrogate; it does not directly optimize the exact episode makespan. Failure risk is analytical and no failures, backups, retries or recovery costs occur. Hardware power/hazard calibration, actual fault tolerance, validation-only tuning, more independent runs, topology/task-count sweeps and source-faithful paper baselines remain open.

## Reproduce and share

See MULTI_OBJECTIVE.md. Run configs/efficiency_training.yaml with configs/objectives.yaml; preserve these declared seeds/settings for this result.

Local generated report: outputs/efficiency/professor-fresh/efficiency-report.html. Bundle: efficiency-report.zip in that directory; extract before opening HTML. Generated checkpoints, raw traces and plots are ignored by Git. This tracked summary and configs give collaborators context; regenerate or transfer the ZIP separately.
