# Simple DQN comparison

Best overall: Without DQN (multi-objective rule).

15 matched scenarios per method: 3 workloads × 5 failure traces. Failures + recovery; 60 tasks per run.

## Three methods, three metrics

| Method | Successful makespan ms ↓ | Energy J ↓ | Completion rate % ↑ |
|---|---:|---:|---:|
| Trained DQN | 249.73 | 41.779 | 96.78 |
| Untrained DQN | 245.15 | 41.450 | 93.11 |
| Without DQN (multi-objective rule) | 231.34 | 41.502 | 98.78 |

**Training benefit:** Trained DQN: +3.67 percentage points in completion versus untrained (93.11% → 96.78%).

It completes 2.20 more tasks per run on average, but uses 0.79% more energy and has 1.87% longer successful makespan.

## One overall ranking

Lower cost is better. Existing weights: time 40%, energy 40%, unsuccessful tasks 20%.

| Rank | Method | Combined cost ↓ |
|---|---|---:|
| 1 | Without DQN (multi-objective rule) | 0.565804 |
| 2 | Trained DQN | 0.590405 |
| 3 | Untrained DQN | 0.590531 |

Trained vs untrained overall cost differs by only 0.02%: practically tied in this small trial.

Cost = 0.4 × (makespan / 400 ms) + 0.4 × (energy / 50 J) + 0.2 × (1 − completion rate)

Both service floors passed in every run: completion ≥ 80%, deadline success ≥ 80%.

Makespan = last successful delivery from time zero; failed tasks are excluded. Completion rate includes all arrivals.

Observed averages from one small trained model. Completion-gain uncertainty includes zero; this is not a proven general improvement. Native rules and DQN use different information estimates.

[Full experiment and methodology](ENHANCED_DQN_COMPARISON.md).
