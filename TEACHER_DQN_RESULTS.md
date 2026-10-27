# Teacher-assisted DQN: actual results (A-037)

Date: 2026-10-27. **Implemented, actually trained, frozen-tested and audited.**

40 teacher rollout episodes produced 2,572 legal examples (2,321 initial, 251 recovery). Three seeds each trained teacher + RL and a matched RL-only control for 300 episodes: **1,800 RL episodes, 117,608 RL updates, 2,460 separate BC updates**. Checkpoints were selected on validation before 342 fresh frozen test simulations.

An execution-permission change stopped the original process before test generation. Complete models were reused, a partial model resumed its full checkpoint, and 10 already-complete episodes reproduced exactly. These replay checks add compute, not new distinct training episodes. Original evidence is retained; settings, seeds and budget stayed fixed. The actual completed artifact folder ends in study-20261027-resumed.

## Three methods, three metrics

Faults + recovery: 15 identical scenarios per model. DQN rows average three separately trained/initialized seeds; the native rule runs each scenario once. This is not an ensemble scheduler.

| Method | Makespan ms ↓ | Energy J ↓ | Completion % ↑ | Combined cost ↓ | Feasible test runs |
|---|---|---|---|---|---|
| Teacher + RL DQN | 232.936 | 40.862 | 89.19 | 0.640721 | 42/45 |
| Untrained DQN | 234.403 | 40.814 | 89.56 | 0.641060 | 42/45 |
| Without DQN (native rule) | 228.250 | 40.828 | 95.44 | 0.563988 | 15/15 |

**Best eligible among these three: Without DQN (native rule).** Lower combined cost is better. New trained candidates must have feasible validation selections for every seed and pass both 80% service floors in every test run. The original default remains unchanged.

Versus untrained: **Makespan: 0.63% lower; Energy: 0.12% higher; Completion: -0.37 percentage points.** Combined cost is **0.05% lower**.

Training and neural decision-computation energy/time are outside these simulated task objectives; extra BC and replay-verification compute is separate.

The cost retains 40% makespan/400 ms + 40% full energy/50 J + 20% unsuccessful fraction, plus existing service-floor penalties when applicable. Makespan includes successful tasks only; read it with completion.

## Did the teacher help?

**Main finding:** imitation-only was the strongest evaluated neural policy on the fresh fault panel: 95.48% completion, 0.564300 cost and 45/45 service-feasible runs, close to the native rule (95.44%, 0.563988). Continuing RL raised its cost by 13.54%, reduced completion by 6.30 percentage points and failed service floors in 3/45 runs. The fine-tuned model stays opt-in; preserving imitation during RL is the next research question.

- Versus matched RL-only: teacher + RL penalized cost is **6.01% lower** and energy is 0.15% lower, but makespan is 0.70% higher and completion is 1.19 percentage points lower. Service-feasible runs improve from 40/45 to 42/45; this is not an all-objective improvement.
- Versus imitation alone: continuing RL cost is **13.54% higher**.
- The paired model-seed interval includes zero: a repeatable teacher benefit is not established.
- The teacher uses the multi-objective heuristic's scoring/deadline rule on DQN's delayed observations. The unchanged native rule has different information access. The report evaluates both separately.

## Controls and preserved baselines on this same fresh panel

| Method | Makespan ms ↓ | Energy J ↓ | Completion % ↑ | Combined cost ↓ | Feasible test runs |
|---|---|---|---|---|---|
| Untrained DQN | 234.403 | 40.814 | 89.56 | 0.641060 | 42/45 |
| Imitation-only DQN | 228.536 | 40.841 | 95.48 | 0.564300 | 45/45 |
| Teacher + RL DQN | 232.936 | 40.862 | 89.19 | 0.640721 | 42/45 |
| Matched RL-only DQN | 231.321 | 40.924 | 90.37 | 0.681677 | 40/45 |
| Original untrained DQN | 234.401 | 40.550 | 86.33 | 0.786132 | 13/15 |
| Original 16-episode DQN | 233.847 | 40.855 | 89.11 | 0.649131 | 14/15 |
| Preserved predictive DQN | 230.258 | 40.868 | 90.59 | 0.642681 | 42/45 |
| Observed heuristic teacher | 229.146 | 40.797 | 95.33 | 0.564858 | 15/15 |
| Without DQN (native rule) | 228.250 | 40.828 | 95.44 | 0.563988 | 15/15 |

These are re-evaluations on new inputs; do not compare numbers directly with the previous test panels. All old implementation/checkpoint hashes stayed unchanged.

## Learning evidence and validation selection

| Seed | Training agreement before → after BC | Recovery agreement after BC | Teacher + RL selected episode | RL-only selected episode | Validation feasible: teacher / control |
|---|---|---|---|---|---|
| 42 | 11.51 → 83.28% | 85.26% | 260 | 280 | True / True |
| 43 | 17.92 → 77.76% | 80.48% | 180 | 120 | True / True |
| 44 | 32.62 → 82.78% | 84.86% | 40 | 300 | True / True |

Agreement above measures fit on teacher training examples, not held-out scheduling performance. The majority teacher label accounts for 30.56% of examples. RL continued after each selected stage through episode 300; the last model is not automatically selected. All inference uses the learned network alone, epsilon zero, with unchanged learner fingerprints. No teacher fallback, new reward weights, Double DQN or prioritized replay was added. This is behavior cloning then basic DQN, not full DQfD.

## Evidence and presentation

- **1,064 full-suite tests passed**; the reusable audit reconciled 2,758 complete reports / 165,480 task outcomes, teacher labels, training-only replay, actual energy/time/rewards, selection order, frozen evaluation, shared inputs and ZIP bytes.
- Protocol, equations and commands: [TEACHER_DQN.md](TEACHER_DQN.md).
- Portable tracked page: [report.html](reports/teacher-dqn-20261027/report.html); [download bundle](reports/teacher-dqn-20261027/report.zip); [summary data](reports/teacher-dqn-20261027/summary.json); [independent audit](reports/teacher-dqn-20261027/independent-audit.json).
- Full local models, traces, loss histories and generated reports: outputs/objective-model/teacher-training/study-20261027-resumed/. Large checkpoints/raw trajectories are not in Git or the compact ZIP. Reproduction requires the preserved A-034/A-036 baseline artifacts or regenerating them first.

For the professor: explain teacher decisions → masked imitation → actual-reward RL → validation selection → frozen comparison. Show completion with makespan and energy, then the unchanged combined cost. Show the RL-only and imitation-only controls before claiming a teacher benefit. Three model seeds, one synthetic workload family, approximate predictors and a fixed test panel limit the conclusion. Do not tune further on now-seen test seeds 87000–87002 / 97000–97004.
