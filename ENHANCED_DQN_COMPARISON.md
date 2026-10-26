# Frozen six-policy comparison (A-035): milestone 5 of 5

Date: 2026-10-27. **All five agreed enhanced-DQN milestones are complete.** This
delivery compares the actual A-034 initialization and final trained checkpoint with
all four preserved native heuristics. It does not retrain or select a model using
these results. Earlier simulator, rule, DQN and report commands remain available.

The small trained DQN completes more work than its untrained initialization on the
fault panel, with greater energy expenditure and a later last successful finish.
Its combined return is almost unchanged. The native multi-objective rule performs
better than this trained DQN on the three mean objectives in both modes. This is a
comparison of these implementations on a synthetic panel, not a general ranking of
RL against heuristics or a reproduction of the paper's results.

## What was compared

| Policy | Meaning |
|---|---|
| Untrained DQN | Exact saved seed-42 initialization, frozen greedy legal choices |
| Trained DQN | Exact final episode-16 A-034 checkpoint, frozen greedy legal choices |
| Least queue | Existing least-running-plus-waiting-load placement, FCFS execution |
| First fit | Existing first eligible VM placement, FCFS execution |
| Fog-first completion | Existing fog preference with predicted completion comparison |
| Multi-objective rule | Existing constrained, weighted candidate-cost heuristic |

The training record is unchanged: 16 episodes, 960 arrivals, 999 optimizer updates,
19 target synchronizations, 913 initial and 117 recovery transitions. The final
episode was fixed before this experiment; a better-looking intermediate checkpoint
was not substituted. Read [ENHANCED_DQN_TRAINING_SANITY.md](ENHANCED_DQN_TRAINING_SANITY.md)
for the actual learning curves, update checks and mixed earlier diagnostics.

Evaluation uses **epsilon 0 and learn=False**. There are no optimizer updates, replay
insertions, RNG/counter changes or adaptive choices between evaluation scenarios.
Both checkpoints keep recovery enabled in both modes; in the no-failure mode there
is simply no outage to trigger recovery. Both initial and recovery VM choices use
the existing network and legal masks.

The runner verifies checkpoint/environment/configuration identity and all six A-034
implementation source hashes before loading. A different model or modified training
runtime is rejected. Both complete learner fingerprints were identical before and
after all evaluation runs:

~~~text
Untrained: 4a8bbe3e8531347ae7bedd6a797340a4b148240ac1499163b6b4213a021e9849
Trained:   d0759cbeaa42a49fc5862100764a267d6e56953dbd20080c4d56420801a3382c
~~~

## Matched workloads and failures

The predeclared plan is [configs/enhanced_comparison.yaml](configs/enhanced_comparison.yaml):

- Three workload seeds: **82000, 82001, 82002**, each with 60 arrivals.
- Five failure seeds: **92000-92004**, using the configured exponential/Weibull
  uptime, detection, repair and reboot model.
- One no-failure case per workload, plus all five fault traces per workload.
- Six policies per case: **18 shared input scenarios, 108 simulations, 6,480 arrivals**.
- Same topology, demands, deadlines, 400 ms observation horizon, power/link model,
  recovery settings, objective weights and service floors for all policies.

Tasks and outages are generated once per case and explicitly passed to all six
policies. Workload/outage hashes and full inputs are retained and checked. No-fault
cases are not duplicated five times. Training seeds 10000-10015 / 20000-20015 and
sanity diagnostic seeds 81000/81001/81002 / 91000/91001 are excluded. The strict plan
and loader reject overlap with the recorded training/diagnostic seeds.

**Information access is not identical across all six implementations.** Both DQNs
use the same periodic delivered telemetry and local admission cache. Native rules
retain their existing event-loop queue/traffic/risk estimates; those estimates can
reflect physical queue cleanup before delayed failure detection. Matching tasks and
outages therefore does not establish an equal-information, algorithm-only contest.
The trained-versus-untrained DQN comparison does share the same information protocol.
No heuristic was changed to make it perform differently in this experiment.

## Actual results

Each numeric metric below is a **mean over scenarios**. Done is out of 60 per run.
All six policies passed both .8 service floors in every run on this panel: 3/3
no-fault and 15/15 faulted runs per policy. This does not erase the service failures
on the different A-034 diagnostic scenarios.

| Mode | Policy | Done / 60 | Energy J | Successful makespan ms | Mean successful latency ms | Penalized return |
|---|---|---:|---:|---:|---:|---:|
| No failures | Untrained DQN | 59.667 | 42.261 | 239.603 | 26.543 | -.578800 |
| No failures | Trained DQN | 60.000 | 42.434 | 239.256 | 22.531 | -.578731 |
| No failures | Least queue | 60.000 | 42.556 | 231.186 | 19.922 | -.571633 |
| No failures | First fit | 57.333 | 42.627 | 241.292 | 29.473 | -.591199 |
| No failures | Fog-first completion | 60.000 | 42.654 | 224.527 | 15.852 | -.565757 |
| No failures | Multi-objective rule | 60.000 | 42.080 | 228.969 | 16.838 | -.565607 |
| Failures + recovery | Untrained DQN | 55.867 | 41.450 | 245.153 | 30.066 | -.590531 |
| Failures + recovery | Trained DQN | 58.067 | 41.779 | 249.728 | 26.093 | -.590405 |
| Failures + recovery | Least queue | 58.533 | 41.921 | 240.058 | 23.543 | -.580316 |
| Failures + recovery | First fit | 53.200 | 41.585 | 245.929 | 31.933 | -.601274 |
| Failures + recovery | Fog-first completion | 58.800 | 42.073 | 233.111 | 19.882 | -.573691 |
| Failures + recovery | Multi-objective rule | 59.267 | 41.502 | 231.342 | 19.502 | -.565804 |

The generated HTML/Markdown report also shows every outcome category, full-batch
counts, service feasibility, pooled deadline/recovery success, Pareto membership and
every paired trained-minus-baseline comparison. Do not present only energy or timing.

On the fault panel, mean outcome counts are:

| Policy | Done | Rejected | Execution failure | Deadline missed | Unfinished | Pooled recovery success |
|---|---:|---:|---:|---:|---:|---:|
| Untrained DQN | 55.867 | 2.400 | 0 | 1.733 | 0 | .779 |
| Trained DQN | 58.067 | .933 | 0 | 1.000 | 0 | .841 |
| Least queue | 58.533 | .667 | 0 | .800 | 0 | .857 |
| First fit | 53.200 | 2.667 | .067 | 4.067 | 0 | .734 |
| Fog-first completion | 58.800 | .667 | 0 | .533 | 0 | .914 |
| Multi-objective rule | 59.267 | .533 | 0 | .200 | 0 | .952 |

Recovery success is recovered completed tasks / interrupted tasks, pooled across
runs. A task that completes after a restart counts once, while the failed attempt's
partial computation/traffic and the retry still consume time and energy.

## What the paired differences support

Positive completion/return deltas favor trained DQN; negative energy deltas favor
trained DQN. Makespan must always be read with the number of successful tasks.

| Trained minus untrained | No failures: mean [approx. 95% interval] | Failures + recovery: mean [approx. 95% interval] |
|---|---|---|
| Completed tasks | +.333 [-1.101, +1.768] | +2.200 [-.427, +4.827] |
| Energy J | +.173670 [+.001392, +.345948] | +.329095 [+.238956, +.419234] |
| Successful makespan ms | -.347 [-21.168, +20.474] | +4.575 [-9.041, +18.191] |
| Penalized return | +.000069 [-.017127, +.017265] | +.000126 [-.018487, +.018738] |

The completion increase is an observed mean, with uncertainty spanning zero. Energy
increased consistently across this panel's fault-seed means. Combined return barely
changed, and its interval spans zero. Lower mean successful latency alongside higher
successful makespan is possible: these measure average arrival-to-delivery time and
the last successful finish from time zero, respectively.

For fault mode, average paired deltas over the same three workloads **within each
failure seed**, then form an approximate Student-t interval from the five seed means
(df=4, critical value 2.776). Do not count 15 crossed runs as independent replications.
No-fault intervals use three workload seeds (df=2). They are conditional on this small
workload panel and one trained initialization, with exploratory multiple comparisons;
they do not establish robustness across independently trained models. JSON records
all individual pairs, valid contributing counts and undefined timing explicitly.

Among policies meeting service floors in every run, empirical mean Pareto candidates
are fog-first/multi-objective in no-fault mode and untrained/multi-objective in fault
mode. The untrained faulted DQN remains a candidate because it spends slightly less
energy while losing more work; this is a tradeoff, not a recommendation that it is
best. The multi-objective rule has the best mean configured return in both modes and
dominates trained DQN's mean energy/makespan/unsuccessful fraction on this panel,
subject to the information-access caveat above.

## Accounting and report interpretation

Every policy is scored offline with the **same contract-4 terminal return**:

~~~text
R = -[.4 * successful_makespan_ms / 400
      + .4 * total_system_energy_j / 50
      + .2 * (rejected + execution_failure + deadline_missed + unfinished) / 60]
    - 10 * max(0, .8 - completion_rate)
    - 10 * max(0, .8 - deadline_success_rate)
~~~

The deadline penalty applies when there are deadline-eligible arrivals. If none
complete, the reward uses the 400 ms surrogate while reported successful makespan
stays undefined. The common scorer checks DQN's live cumulative return. Heuristics
do not learn from this score. A-030 feasible objective score is separate from the
penalized return; only feasible runs receive an objective score/recommendation.

System energy counts all host states, link idle/actual bytes and coordinator
overhead through the common horizon, including lost partial attempts and retries.
Failures can reduce energy by keeping hosts down. A low energy result alone does
not establish efficiency. Full-batch makespan exists only when all tasks complete.
No physical coefficients were tuned against this test panel.

## Reproduce, open and share

From the repository root, with the existing Python environment and demo extra:

~~~powershell
Set-Location "C:\Users\ashis\Desktop\Auxia Software\Basic Project\chat-app"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.enhanced_comparison --plan-config configs/enhanced_comparison.yaml --training-dir outputs/objective-model/dqn-sanity/milestone4-20261027 --output-dir outputs/objective-model/frozen-comparison/my-comparison
Start-Process ".\outputs\objective-model\frozen-comparison\my-comparison\comparison-report.html"
~~~

Use a NEW output directory; existing evidence is never overwritten. On another
checkout, first regenerate A-034 using its unchanged configurations and current
compatible runtime, then pass that directory to --training-dir:

~~~powershell
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.enhanced_sanity --plan-config configs/enhanced_sanity.yaml --output-dir outputs/objective-model/dqn-sanity/my-training
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.enhanced_comparison --training-dir outputs/objective-model/dqn-sanity/my-training --output-dir outputs/objective-model/frozen-comparison/my-heldout
~~~

For a new environment, use the repository's existing Python 3.12 setup and install
the project with its demo/dev extras. Do not replace the existing environments or
load legacy checkpoints into the enhanced network. Bitwise cross-runtime/hardware
reproduction is not promised; configuration/source compatibility is enforced.

Actual evidence is under:

~~~text
outputs/objective-model/frozen-comparison/milestone5-20261027-release/
  comparison-report.html        offline compiled dashboard: open this file
  comparison-report.zip         portable dashboard, all figures, data and traces
  REPORT.md                     full generated tables and paired intervals
  metrics.csv                   108 simulation rows
  paired-deltas.csv             90 matched trained-minus-baseline rows
  evaluation.json               summaries, rates, counts and individual paired deltas
  experiment.json               plan, provenance, runtime and implementation hashes
  frozen-audit.json              complete learner fingerprints before/after
  artifacts.json                SHA256 manifest of report evidence
  runs/                         108 complete simulation traces
  figures/                      three PNG and three SVG charts
  independent-audit.json        separate post-packaging accounting/integrity audit
~~~

The ZIP contains local figures and data, including every linked trace; extract it
before opening on another computer. It requires no server, JavaScript, CDN or template
rendering. The large checkpoint files are referenced by hashes rather than bundled.
Generated files stay ignored, following the existing artifact policy; this guide,
source, plan, tests and status updates are committed for collaborators. The separate
post-packaging audit is outside the ZIP. Report-layout-only reruns preserved all
108 metric rows and paired results; no model, budget, workload or failure selection
was changed to improve results.

## Verification and presentation

Verification: **997 full-suite tests passed in 309.88 seconds**, including 27 new comparison checks.
The checks cover actual frozen checkpoint loading, six-way pairing, native-rule full
report preservation, independently calculated rewards, failure-seed cluster counts,
missing timing, service-floor screening, seed leakage, provenance rejection, existing
output protection and all local HTML/ZIP links and evidence hashes. The final report
adjustments also passed all 27 focused comparison checks in 25.93 seconds. The
dependency check found no broken requirements; whitespace checks were clean. All
three actual PNG figures were visually inspected with no clipped titles/labels.

The independent saved-evidence audit passed for **6,480 outcomes, 6,889 attempts
and 88,095 link segments**. It checked shared full inputs across all 18 scenarios,
restored resources, attempt/outcome time partitions, host/link energy sums, transfer
capacity, recomputed returns, frozen counters and current implementation hashes.
It also verified 108 CSV rows, 90 paired rows, 118 local HTML links, all 121 evidence
file hashes and CRC/byte parity for all 122 ZIP entries. Final metric summaries and
paired differences exactly match the initial panel run before report layout fixes.

~~~powershell
.\.venv312\Scripts\python.exe -X utf8 -m pytest tests/integration/test_enhanced_comparison.py -q
.\.venv312\Scripts\python.exe -X utf8 -m pytest -q --junitxml=outputs/objective-model/frozen-comparison/pytest.xml
.\.venv312\Scripts\python.exe -m pip check
~~~

For the professor, open comparison-report.html and explain:

1. All six policies receive exactly the same arrivals and outage traces; show one
   paired workload/outage hash and the checkpoint freeze audit.
2. Start with delivered/lost outcomes, then compare energy, successful makespan and
   service-feasible counts. Show failure-free and faulted modes separately.
3. Show trained-minus-untrained intervals. Learning happened in A-034, but this small
   budget does not yet provide a convincing combined-return improvement on A-035.
4. Explain why multi-objective heuristics remain competitive and disclose their
   different information access. Show one trace's failed attempt and recovery costs.
5. Download/share the ZIP or CSVs so the evidence can be inspected independently.

The five-point implementation/evaluation aim is finished. Optional future research
is an equal-information rule adapter, then predeclared multi-initialization and
training-budget experiments with a new untouched test panel. These reported test
seeds are now seen evidence and must not become tuning/selection targets. Double
DQN/PPO, calibrated hardware and a source-audited paper reproduction remain outside
this completed scope; no further training was performed in this milestone.
