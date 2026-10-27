# Teacher-assisted DQN report snapshot

Open report.html locally. All linked report files are included. report.zip is the compact portable report; result.json and independent-audit.json provide additional provenance. Large model checkpoints and full trajectories remain in the ignored local experiment directory.

See ../../TEACHER_DQN.md for reproduction commands and ../../TEACHER_DQN_RESULTS.md for the short interpretation. Old report panels are historical; compare methods within this fresh panel. This report does not change the default scheduler.

**Main finding:** imitation-only was the strongest evaluated neural policy on the fresh fault panel: 95.48% completion, 0.564300 cost and 45/45 service-feasible runs, close to the native rule (95.44%, 0.563988). Continuing RL raised its cost by 13.54%, reduced completion by 6.30 percentage points and failed service floors in 3/45 runs. The fine-tuned model stays opt-in; preserving imitation during RL is the next research question.
