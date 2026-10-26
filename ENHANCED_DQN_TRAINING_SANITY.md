# Enhanced DQN training sanity experiment (A-034): milestone 4 of 5

Run date: 2026-10-27. **The enhanced DQN has now been trained for a small fixed
budget, with verified optimizer updates, changed weights and changed legal actions.**
This establishes that the connected learning pipeline works. It does not establish
convergence, optimal hyperparameters or superiority to the four rules. Milestone 5
is now complete in [ENHANCED_DQN_COMPARISON.md](ENHANCED_DQN_COMPARISON.md): a separate
frozen untrained/trained/four-heuristic comparison. This guide retains A-034 results.

Verification: **970 full-suite tests passed in 261.86 seconds**, including 21 new
runner checks. Dependency/whitespace checks are clean. An independent saved-artifact
audit checked 32 traces, six source hashes, CSV record counts and the trained CLI's
matching metrics; two PNG/two SVG plots were generated and the PNGs visually inspected.

## What ran, in simple language

The agent starts with random network weights. For each initial or recovery placement,
it encodes the existing 253 features, masks ineligible VMs, and uses the same basic
DQN to choose among six destinations. Training sometimes explores a random LEGAL VM.
Actual energy, successful-makespan increments and final task outcomes arrive through
the verified delayed reward transitions. After replay warmup, Adam updates the online
network from sampled minibatches using Huber loss and masked target-network values.

The fixed run used `configs/rl_network.yaml`, `configs/objective_model.yaml` and the
existing `configs/enhanced_dqn.yaml` without changing any coefficients or learner
defaults. Its budget/seed split is in the new `configs/enhanced_sanity.yaml`:

| Setting | Actual value |
|---|---|
| Model initialization seed | 42 |
| Training | 16 episodes x 60 tasks = 960 task arrivals |
| Workload/failure seeds | 10000-10015 / 20000-20015, paired by episode |
| Diagnostic generated cases | 81000/91000 and 81001/91001 |
| Constructed recovery probe | Separate workload seed 81002 plus a fixed explicit outage trace |
| Frozen validation stages | Before training and after episodes 4, 8, 12 and 16 |
| Architecture | CPU float32, 253 -> 64 ReLU -> 64 ReLU -> 6 Q-values; 20,806 parameters |
| Learning | Adam .001, batch/warmup 32, replay capacity 2000, gradient clip 10 |
| Target sync / discount | Every 50 optimizer updates; gamma=1 from contract 4 |
| Training exploration | Epsilon 1 -> .05 over 2000 legal decisions; actual endpoint .51075 |
| Frozen inference | Epsilon=0; no replay insertion, updates or counter changes |
| Checkpoint retained | Episode 16, the predeclared final episode |

Diagnostic seeds are disjoint from training seeds. Validation never changes the
training budget, optimizer, features or selected checkpoint. The recovery probe gives
the first task an upload long enough to remain active during a common node outage,
then delays the other tasks until after reboot. It makes recovery observable in a
controlled diagnostic; it is not sampled reliability evidence and never enters replay.

## Actual learning evidence

- **1,030 real decision transitions:** 913 initial and 117 recovery. The other 47
  initial arrivals were forced rejections and therefore created no fake replay action.
  These are collected request kinds, not per-minibatch sampling counts.
- **999 optimizer updates**, all finite losses; **19 target-network synchronizations**.
  Replay contains all 1,030 transitions, below its 2,000-entry capacity.
- Parameter L2 distance from initialization: **4.6359729004**. Online network digests
  changed from `c97b3d29f22037709b4ceb4a17af90ef458ea63c07072366de12b5e1225028bc`
  to `2be87c37aaea0ab967909db62663a7b1186075984ec27d7140bd935233aef1b6`.
- Frozen actions changed on **87/156 initial** and **5/20 recovery** saved observations.
  Each comparison uses the EXACT same normalized vector and legal mask as before
  training. This isolates the learned network change from different live queues.
- Every validation stage preserved the full learner fingerprint. The trained JSON
  checkpoint loaded exactly and reproduced its final recovery-probe trace. The
  existing frozen inference CLI also loaded it and reproduced generated-1 metrics
  with optimizer updates still 999.
- Mean Huber loss for the first/last 50 updates: **.0010115304 / .0005958198**.
  Replay contents, sampled tasks and target weights change during training, so this
  lower ending average is not evidence of convergence or uniformly better scheduling.

Training outcomes across the 960 arrivals were 884 completions, 47 rejections and
29 deadline misses; no final execution failures or unfinished tasks. There were
82 recovered completions. These exploratory training outcomes are not frozen policy
benchmark results. Training ended at epsilon .51075 because the declared small budget
ended; it was not silently extended to reach the epsilon floor.

## Frozen before/after behavior on the same diagnostics

Higher return means a smaller configured cost, including service shortfall penalties.
All cases use identical workload and outage hashes before/after. Done is out of 60;
makespan covers successful deliveries only. Read service feasibility alongside energy
and timing.

| Diagnostic | Stage | Done | Return | Total J | Successful makespan ms | Feasible |
|---|---|---:|---:|---:|---:|---|
| generated-1 | Untrained | 42 | -2.602847 | 36.653442 | 249.619124 | No |
| generated-1 | Final, episode 16 | 42 | -2.589081 | 36.523241 | 236.894708 | No |
| generated-2 | Untrained | 58 | -.587300 | 41.476892 | 248.817710 | Yes |
| generated-2 | Final, episode 16 | 58 | -.594195 | 41.412114 | 256.231700 | Yes |
| recovery-probe | Untrained | 44 | -1.938263 | 40.161009 | 230.308116 | No |
| recovery-probe | Final, episode 16 | 47 | -.937041 | 40.362339 | 237.475847 | No |

The result is mixed. Generated-1 keeps the same completion count while using slightly
less energy and finishing its last successful task earlier. Generated-2 has a slightly
worse return: a little less energy but a later successful makespan. The constructed
probe completes three more tasks while using more energy and time; reduced failure
and service-shortfall costs improve its return. It still falls below the .8 completion
floor (47/60), as does generated-1. There is no general efficiency or reliability claim.

Intermediate diagnostic returns were also recorded. Episode 12 looked better on some
cases than episode 16. It was NOT selected afterward: this run retains episode 16 to
avoid choosing a favorable result from the diagnostic panel. Training and validation
curves are under `figures/` in the saved run directory.

## Evidence and commands

The actual saved run is:

~~~text
outputs/objective-model/dqn-sanity/milestone4-20261027/
  experiment.json                 configuration, plan, runtime and source hashes
  initialization.json            original network/learner checkpoint
  trained-final.json             episode-16 network, target, Adam, replay and RNG state
  result.json                    checks, stage results, losses and behavior changes
  training.csv / losses.csv      real per-episode / per-update records
  validation.csv                 all frozen stage metrics and input hashes
  fixed-observation-probes.json  identical before/after state vectors and masks
  training/ / validation/        full simulation traces and stage probe results
  figures/learning.png/.svg      training returns, losses and action changes
  figures/validation.png/.svg    fixed diagnostic returns
  REPORT.md                      generated results with charts
  checkpoint-inference.json      existing CLI frozen checkpoint check
  sanity-audit.json              independent saved-artifact accounting/source audit
~~~

Generated models/traces stay in the existing ignored output tree. This guide, runner,
plan, tests and progress/handoff updates are committed; another collaborator can
regenerate the checkpoints. Choose a NEW output directory: existing evidence is never
overwritten. Plot generation uses matplotlib from the project's existing demo extra,
which is installed in the recorded environment. From the repository root in PowerShell:

~~~powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.enhanced_sanity --plan-config configs/enhanced_sanity.yaml --output-dir outputs/objective-model/dqn-sanity/reproduction-1
~~~

To use the actual trained model through the existing frozen inference interface:

~~~powershell
.\.venv312\Scripts\python.exe -m drlmots.enhanced_dqn --dqn-config configs/enhanced_dqn.yaml --checkpoint outputs/objective-model/dqn-sanity/milestone4-20261027/trained-final.json --workload-seed 81000 --failure-seed 91000 --output outputs/objective-model/dqn-sanity/loaded-example.json
~~~

The inference CLI still starts an untrained network when --checkpoint is omitted.
The earlier heuristic HTML report does not automatically load this trained checkpoint.
The sanity runner is separate from earlier simulators, DQN interfaces and all four
rule paths. Contract 4, masking, physical accounting and checkpoint schema 1 stay intact.

To verify the runner and repository:

~~~powershell
.\.venv312\Scripts\python.exe -X utf8 -m pytest tests/integration/test_enhanced_sanity.py -q
.\.venv312\Scripts\python.exe -X utf8 -m pytest -q --junitxml=outputs/objective-model/dqn-sanity/pytest.xml
~~~

The new checks cover real optimizer activity, exact repeatability on this CPU/runtime,
restored frozen traces, identical-probe masks, disjoint seeds, strict plan parsing,
fixed final-episode evaluation, too-small-budget failure reporting, plot exports and
protection of prior outputs. Cross-library/hardware bitwise equality is not promised.

For the professor: show the weight/update counts, then the identical-observation
action changes, then the frozen return curves. Explain the mixed diagnostic results
and service failures. **Point 5 is now complete:** [ENHANCED_DQN_COMPARISON.md](ENHANCED_DQN_COMPARISON.md)
compares frozen untrained DQN, this trained DQN and all four heuristics on matching
held-out scenarios. It reports mixed learned tradeoffs and better native-rule mean
results on the panel, with information-access and statistical limits. These later
results do not replace this historical diagnostic panel.
