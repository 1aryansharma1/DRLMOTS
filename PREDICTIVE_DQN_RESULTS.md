# Predictive DQN results: points 1 and 2

Date: 2026-10-27. **Implemented and actually run.**

Predictive training finished: 900 episodes / 54,000 task arrivals across three model seeds; 58,837 optimizer updates. All checkpoints were selected on validation before 216 fresh frozen test simulations.

The opt-in predictive network uses 325 inputs (253 observed + 72 derived predictions), with the same basic DQN, reward and legal-action rules. Original baselines/checkpoints/reports remain available.

## Three methods, three metrics

Means on failures + recovery: 15 shared scenarios per model. Predictive rows average three independent trained/initialized models; without-DQN runs each scenario once. This is not an ensemble scheduler.

| Method | Successful makespan ms ↓ | System energy J ↓ | Completion % ↑ |
|---|---:|---:|---:|
| Trained predictive DQN (300-budget selection) | 230.329 | 41.354 | 96.59 |
| Untrained predictive DQN | 233.650 | 41.271 | 96.67 |
| Without DQN: multi-objective rule | 225.269 | 41.087 | 98.67 |

**Completion change from training: -0.07 percentage points** versus the predictive initialization.

The observed mean fault-panel makespan was 1.42% lower than the same-feature initialization, while energy changed by +0.20% and completion by -0.07 percentage points. All four paired model-seed intervals include zero, so a repeatable gain is not established. The new variant did not beat the original 16-episode DQN's mean combined cost on this panel. It remains opt-in; the preserved baseline remains the default. These results do not establish an improvement on all three objectives.

**Best eligible overall: Without DQN: multi-objective rule.** Ranking retains 40% time, 40% energy and 20% unsuccessful tasks. Lower cost is better; any service-floor failure excludes a method from recommendation.

| Method | Penalized cost ↓ | Service-feasible runs |
|---|---:|---|
| Trained predictive DQN (300-budget selection) | 0.567977 | 45/45 |
| Untrained predictive DQN | 0.570481 | 45/45 |
| Without DQN: multi-objective rule | 0.556632 | 15/15 |

## Preserved baseline and budget comparison

These are fresh matched test inputs, separate from the earlier A-035 panel. Features and training duration both changed versus the 16-episode baseline, so this does not isolate the feature effect.

| Variant | Successful makespan ms | Energy J | Completion % | Penalized cost |
|---|---:|---:|---:|---:|
| Original 253-input initialization | 232.870 | 41.079 | 96.11 | 0.569284 |
| Original 16-episode DQN | 231.580 | 41.328 | 97.44 | 0.567312 |
| Predictive, 100-budget selection | 230.285 | 41.372 | 96.26 | 0.568740 |
| Predictive, 300-budget selection | 230.329 | 41.354 | 96.59 | 0.567977 |

## What was trained and selected

All three models trained to 300 episodes. Budgets 100/300 are nested prefixes of the same trajectory. The shared training panel contains 300 workload/failure pairs (18,000 generated tasks), repeated across model seeds for 54,000 arrivals.

| Model seed | Episodes | Optimizer updates | Final training epsilon |
|---|---:|---:|---:|
| 42 | 300 | 19621 | 0.05 |
| 43 | 300 | 19614 | 0.05 |
| 44 | 300 | 19602 | 0.05 |

| Seed | Budget | Selected episode | Validation all feasible |
|---|---:|---:|---|
| 42 | 100 | 40 | True |
| 42 | 300 | 220 | True |
| 43 | 100 | 20 | True |
| 43 | 300 | 260 | True |
| 44 | 100 | 20 | True |
| 44 | 300 | 20 | True |

The fixed selection rule prioritizes service-feasible validation cases, then service shortfall, penalized cost and earlier episode. Initialization is not selected as a trained checkpoint. The largest budget is the primary report regardless of test results. Seed 44 retained episode 20 because no later validation checkpoint improved its rank; longer training did not benefit every seed.

## Verification and limits

**1,033 full-suite tests passed**, including 36 predictive/training/report/inference checks. Dependency and whitespace checks are clean. The real selected seed-42 checkpoint reproduced its saved validation metrics through the inference CLI with zero evaluation updates.

Independent saved-evidence audit: 1,404 complete reports, 84,240 outcomes, 90,592 attempts; exact shared test inputs, restored resources, MI/time/energy/return accounting, selection history, source identities and archive links/bytes verified.

All six original implementation source hashes still match A-034 provenance. Every tested learner stays frozen. No feature/setting/checkpoint selection uses test results. Native-rule information access differs; three model seeds on one synthetic panel do not prove general superiority or convergence. Timing is conditional on successful deliveries; energy includes losses and the full shared horizon.

Read [PREDICTIVE_DQN.md](PREDICTIVE_DQN.md) for feature equations, assumptions, exact seed splits and commands. The generated report includes learning/validation curves, all three primary metrics, original baselines, both budgets and paired model-seed intervals.

## Open and reproduce

Actual generated artifacts (ignored by Git): outputs/objective-model/predictive-training/study-20261027/

~~~powershell
Start-Process ".\outputs\objective-model\predictive-training\study-20261027\report.html"
~~~

Share report.zip for the compact report, charts, CSVs and provenance. Large resumable checkpoints and complete traces remain in the experiment directory. prediction-example.json contains a real validation decision's 325 network inputs; validation-cli-seed42.json verifies the usable inference command.

No Double DQN, demonstrations, prioritized replay, reward reweighting, mask changes or architecture-width tuning was implemented. Optional next work should use validation and a new untouched test panel; the 85000-85002 / 95000-95004 results are now seen evidence.
