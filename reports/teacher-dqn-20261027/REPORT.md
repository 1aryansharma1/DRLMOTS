# Teacher-assisted DQN results

40 teacher rollout episodes; 20 fixed imitation epochs per model; 1800 actual RL episodes across both training branches and 3 seeds; 342 frozen test simulations.

Teacher imitation → ordinary masked DQN training → validation-selected checkpoint → frozen fresh testing. Baselines are preserved.

## failures with recovery

Best eligible among the three primary methods: **Without DQN (native rule)**.

| Method | Makespan ms ↓ | Energy J ↓ | Completion % ↑ | Combined cost ↓ | Feasible runs |
|---|---|---|---|---|---|
| Teacher + RL DQN | 232.94 | 40.862 | 89.19 | 0.640721 | 42/45 |
| Untrained DQN | 234.40 | 40.814 | 89.56 | 0.641060 | 42/45 |
| Without DQN (native rule) | 228.25 | 40.828 | 95.44 | 0.563988 | 15/15 |

Versus untrained: Makespan 0.63% lower; Energy 0.12% higher; Completion -0.37 percentage points.

Teacher warm start versus matched RL-only: combined penalized cost 6.01% lower.
Continuing RL versus imitation alone: combined penalized cost 13.54% higher.
The paired model-seed interval includes zero; a repeatable teacher benefit is not established.

### Matched controls and preserved baselines

| Method | Makespan ms ↓ | Energy J ↓ | Completion % ↑ | Combined cost ↓ | Feasible runs |
|---|---|---|---|---|---|
| Untrained DQN | 234.40 | 40.814 | 89.56 | 0.641060 | 42/45 |
| Teacher imitation only | 228.54 | 40.841 | 95.48 | 0.564300 | 45/45 |
| Teacher + RL DQN | 232.94 | 40.862 | 89.19 | 0.640721 | 42/45 |
| Matched RL-only DQN | 231.32 | 40.924 | 90.37 | 0.681677 | 40/45 |
| Original untrained DQN | 234.40 | 40.550 | 86.33 | 0.786132 | 13/15 |
| Original 16-episode DQN | 233.85 | 40.855 | 89.11 | 0.649131 | 14/15 |
| Preserved predictive DQN | 230.26 | 40.868 | 90.59 | 0.642681 | 42/45 |
| Observed multi-objective teacher | 229.15 | 40.797 | 95.33 | 0.564858 | 15/15 |
| Without DQN (native rule) | 228.25 | 40.828 | 95.44 | 0.563988 | 15/15 |

## no failures

Best eligible among the three primary methods: **Without DQN (native rule)**.

| Method | Makespan ms ↓ | Energy J ↓ | Completion % ↑ | Combined cost ↓ | Feasible runs |
|---|---|---|---|---|---|
| Teacher + RL DQN | 226.54 | 42.049 | 99.81 | 0.563304 | 9/9 |
| Untrained DQN | 230.34 | 41.891 | 99.81 | 0.565834 | 9/9 |
| Without DQN (native rule) | 223.27 | 41.613 | 100.00 | 0.556175 | 3/3 |

Versus untrained: Makespan 1.65% lower; Energy 0.38% higher; Completion +0.00 percentage points.

Teacher warm start versus matched RL-only: combined penalized cost 0.19% higher.
Continuing RL versus imitation alone: combined penalized cost 1.12% higher.
The paired model-seed interval includes zero; a repeatable teacher benefit is not established.

### Matched controls and preserved baselines

| Method | Makespan ms ↓ | Energy J ↓ | Completion % ↑ | Combined cost ↓ | Feasible runs |
|---|---|---|---|---|---|
| Untrained DQN | 230.34 | 41.891 | 99.81 | 0.565834 | 9/9 |
| Teacher imitation only | 224.24 | 41.606 | 100.00 | 0.557085 | 9/9 |
| Teacher + RL DQN | 226.54 | 42.049 | 99.81 | 0.563304 | 9/9 |
| Matched RL-only DQN | 226.60 | 41.957 | 100.00 | 0.562253 | 9/9 |
| Original untrained DQN | 227.51 | 41.802 | 100.00 | 0.561928 | 3/3 |
| Original 16-episode DQN | 228.12 | 41.818 | 100.00 | 0.562659 | 3/3 |
| Preserved predictive DQN | 228.69 | 42.014 | 100.00 | 0.564805 | 9/9 |
| Observed multi-objective teacher | 223.29 | 41.594 | 100.00 | 0.556039 | 3/3 |
| Without DQN (native rule) | 223.27 | 41.613 | 100.00 | 0.556175 | 3/3 |

## Learning and validation

| Seed | Teacher examples | Agreement before % | Agreement after % | BC updates |
|---|---|---|---|---|
| 42 | 2572 | 11.51 | 83.28 | 820 |
| 43 | 2572 | 17.92 | 77.76 | 820 |
| 44 | 2572 | 32.62 | 82.78 | 820 |

| Branch/seed | Selected RL episode | RL updates | Validation feasible |
|---|---|---|---|
| rl_only/42 | 280 | 18412 | True |
| teacher_rl/42 | 260 | 16984 | True |
| rl_only/43 | 120 | 7773 | True |
| teacher_rl/43 | 180 | 11724 | True |
| rl_only/44 | 300 | 19612 | True |
| teacher_rl/44 | 40 | 2601 | True |

## Paired teacher + RL minus control

| Mode | Control | Metric delta | Mean | Approx. 95% interval | Model seeds |
|---|---|---|---|---|---|
| failures_with_recovery | Matched RL-only DQN | successful_makespan_ms | 1.615392 | [-5.385698225368701, 8.616481382146857] | 3 |
| failures_with_recovery | Matched RL-only DQN | total_energy_j | -0.062117 | [-0.3555676904407737, 0.23133334407737333] | 3 |
| failures_with_recovery | Matched RL-only DQN | completion_rate | -0.011852 | [-0.03416370370370368, 0.010459999999999981] | 3 |
| failures_with_recovery | Matched RL-only DQN | penalized_return | 0.040956 | [-0.31173810582747985, 0.39364934558975706] | 3 |
| no_failures | Matched RL-only DQN | successful_makespan_ms | -0.057205 | [-15.490113301215336, 15.375703917336237] | 3 |
| no_failures | Matched RL-only DQN | total_energy_j | 0.092199 | [-0.2601824924248409, 0.44457999068812737] | 3 |
| no_failures | Matched RL-only DQN | completion_rate | -0.001852 | [-0.009820370370370401, 0.006116666666666685] | 3 |
| no_failures | Matched RL-only DQN | penalized_return | -0.001051 | [-0.014670078652199275, 0.012568567309231398] | 3 |
| failures_with_recovery | Teacher imitation only | successful_makespan_ms | 4.400016 | [-1.826901187798188, 10.626934018464155] | 3 |
| failures_with_recovery | Teacher imitation only | total_energy_j | 0.021170 | [-0.05909509711615795, 0.10143435820410937] | 3 |
| failures_with_recovery | Teacher imitation only | completion_rate | -0.062963 | [-0.08800998273207183, -0.03791594319385411] | 3 |
| failures_with_recovery | Teacher imitation only | penalized_return | -0.076421 | [-0.09665173925665357, -0.056190711366419795] | 3 |
| no_failures | Teacher imitation only | successful_makespan_ms | 2.306989 | [-4.725601928515754, 9.339580751338794] | 3 |
| no_failures | Teacher imitation only | total_energy_j | 0.442652 | [0.2142038348734807, 0.6711008559257674] | 3 |
| no_failures | Teacher imitation only | completion_rate | -0.001852 | [-0.009820370370370401, 0.006116666666666685] | 3 |
| no_failures | Teacher imitation only | penalized_return | -0.006219 | [-0.01569095018788834, 0.0032537930979305765] | 3 |
| failures_with_recovery | Untrained DQN | successful_makespan_ms | -1.466529 | [-4.191849036182958, 1.2587916907266863] | 3 |
| failures_with_recovery | Untrained DQN | total_energy_j | 0.048383 | [-0.6349814393218725, 0.731747069812734] | 3 |
| failures_with_recovery | Untrained DQN | completion_rate | -0.003704 | [-0.051221737535416714, 0.043814330128009324] | 3 |
| failures_with_recovery | Untrained DQN | penalized_return | 0.000339 | [-0.15454376778612855, 0.15522121860617655] | 3 |
| no_failures | Untrained DQN | successful_makespan_ms | -3.794079 | [-21.537515244860288, 13.949357287697282] | 3 |
| no_failures | Untrained DQN | total_energy_j | 0.157915 | [-0.5280394591662891, 0.8438696217294974] | 3 |
| no_failures | Untrained DQN | completion_rate | 0.000000 | [-0.013801878935127599, 0.013801878935127599] | 3 |
| no_failures | Untrained DQN | penalized_return | 0.002531 | [-0.02043201132336195, 0.025493527980019314] | 3 |

## How to read this report

- Combined cost = 0.4 × makespan/400 + 0.4 × energy/50 + 0.2 × unsuccessful fraction. Lower is better.
- Service-floor penalties are included in the displayed penalized cost. New RL candidates must have service-feasible validation selections for every model seed and be feasible in every test run to enter the overall recommendation. Read completion alongside makespan, which counts successful tasks only.
- The observation-only teacher uses the multi-objective rule's deadline filter and time/energy/risk scoring on the existing delayed predictions. It cannot see physical hidden queues, future failures, future tasks or actual outcomes. The native rule remains unchanged and has different information access.
- Energy and makespan describe simulated task execution. Workstation training energy/time and neural decision-computation overhead are outside these simulated objectives; extra teacher/BC compute is disclosed separately.
- BC and RL-only models share initialization seeds, network, reward, legal masks, RL workload/failure panel, 300-episode budget (or configured test budget), exploration and validation selection rule. BC has extra demonstration and supervised-compute cost, disclosed above. The BC-only ablation checks whether RL preserves imitation benefits.
- Teacher labels are used only in fixed offline cloning. RL replay contains only the agent's own transitions. Inference uses the neural network alone, epsilon zero, and has no teacher fallback. This is BC then DQN, not full DQfD.
- Demonstration, RL, validation and test seeds are disjoint and exclude all previous registered study panels. Test inputs are generated only after all selections are locked. There is no test-based checkpoint or hyperparameter selection.
- The three-seed intervals use paired model-seed mean effects on a fixed scenario panel, not all simulation rows as independent replicates. They do not establish general superiority. Models are averaged for reporting, not deployed as an ensemble.
- Baseline metrics here are re-evaluated on this fresh panel; compare within this report, not with previous panels. The original default scheduler is unchanged. Full traces and model checkpoints remain in the experiment directory; the compact ZIP includes only reports and summary evidence.
