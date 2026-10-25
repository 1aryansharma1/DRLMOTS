# Node failure, restart recovery and observed objectives

Updated 2026-10-25. Versioned simulator/accounting contract 1, decision A-029.

## What this version completes

The simulator now actually takes fog/cloud nodes down, interrupts work, and attempts
bounded restart recovery on another failure domain. The measured objective vector is
makespan (ms), fixed-window energy (J), and the observed unsuccessful fraction of all
submitted tasks. Completion, rejection, execution failure and recovery counts remain
separate. This replaces analytical risk only in the new simulator's observed metrics;
the previous analytical-risk simulator, trained DQN models and reports are preserved.

This is an environment/accounting baseline before the next training study. It does
**not** claim that the old DQN has been trained for failures. Legacy DQN policies are
rejected by this mode: their state/reward/episode boundary does not represent outages,
retries, pending recovery or delayed execution failure. A fault-aware learning contract
is the next step, followed by untrained-versus-trained and validation curves.

## Teaching example

A 2500-byte task needs 5 ms service. Its 24 kB input takes 3 ms to upload to fog.
The fog node fails at time 5 ms, after 2 ms service. Detection is immediate. A 1 ms
recovery delay is followed by a fresh 4 ms cloud upload and 5 ms full restart:

- Attempt 1: dispatch 0, upload ready/start 3, interrupted 5.
- Recovery coordinator: time 5-6.
- Attempt 2: dispatch 6, upload ready/start 10, completes 15.
- Final latency: 15 ms = 5 ms first attempt + 1 ms recovery + 9 ms second attempt.
- At fog 100 W / .01 J per kB, cloud 200 W / .02 J per kB, coordinator 10 W:
  .2 + .24 + 1 + .48 + .01 = **1.93 J**, excluding idle power in this test.

Tests also verify interruption during serialization/propagation, queued task failure,
retry exhaustion, repair, backup waiting and horizon termination using hand calculations.

## Failure domains and fair scenarios

VM-to-node membership comes from the existing topology. IDs use fog:<fog_node_id>
and cloud:<datacenter_id>. All VMs on that node fail together. Physical host budgets
are still not modeled. A cloud datacenter is one failure domain in this fixture.

configs/failure_recovery.yaml declares the full outage trace before execution. All
policies receive identical tasks and outages. Future outage/repair times are never
provided to the scheduler. It observes current up/down status only.

The optional sample_outages API builds an alternating exponential-up/fixed-repair
trace using local, seed-derived per-node streams. It materializes the trace first,
independently of policy actions. Adding a node does not alter existing node streams.
The delivered professor experiment uses one explicit fixed synthetic trace and five
workload seeds. Its standard deviations describe workload variation, not independent
failure scenarios. Failure coefficients are not real hardware measurements.

## Restart and retry semantics

- Original tasks are immutable inputs. Each dispatch creates a unique attempt.
- A task has exactly one final outcome: completed, rejected at initial admission, or
  permanently failed after admission. Attempts have their own interrupted/deadline
  outcomes; an interrupted attempt is not a second failed original task.
- A down node loses running, waiting and uploading tasks. Running reservations are
  released exactly once. Queued/uploading work has no CPU/memory reservation.
- Full input remains available at an assumed durable source. No VM checkpoint is
  available, so each restart uploads the complete input and repeats all processing.
- Backups must be up, reachable, resource-feasible and have waiting queue space. They
  must belong to a different node than the just-failed attempt. A previously used
  node may be selected after repair on a later retry if it differs from the last one.
- max_retries is the number of extra execution attempts after the initial attempt.
  No spare capacity or simultaneous replica is reserved: this is on-demand passive
  restart, not active replication or fault-free recovery guarantees.
- After recovery_delay_ms the scheduler tries placement. If no eligible backup exists,
  it waits for capacity/repair until recovery_wait_ms expires. It reacts to actual
  events; it does not inspect future repair times. No infinite retry/wait loop exists.
- Recoveries have dispatch priority over new arrivals, after existing queued work
  starts. They can consume capacity that would otherwise serve later arrivals.
- Repair admits new work but never resurrects an already finalized task.
- Every input arrives strictly before the fixed observation horizon. At that horizon,
  unfinished attempts/recoveries become permanent failures; resources are released.

## Same-time event contract

At a timestamp: finish completed services; apply every node down/up event; start
existing ready FCFS work; dispatch eligible recoveries; handle external arrivals in
stable input order. A completion exactly at node failure succeeds. A node failure
exactly at upload readiness interrupts the task before it starts. A backup becoming
available at its wait deadline may be admitted. Service completed exactly at the
horizon succeeds; other work is finalized at the horizon. No future fault information
is used for placement, reward or candidate forecasts.

## Time and energy accounting

For each attempt, elapsed time is split into disjoint components:

    upload elapsed + waiting after upload readiness + actual executed service

Uploads overlap pre-existing execution, so full queue delay is not blindly added to
full upload time. A task's terminal latency equals the sum of its attempt durations
plus all recovery-blocked intervals. Completed-task latency is measured from the
original arrival, not the retry arrival. Every retry has its own resource/energy trace.

The common observation window is [0,horizon_ms], even if every task settles earlier:

- Active compute J = total running VM power W * executed milliseconds / 1000.
  Interrupted partial service is charged; discarded progress is not refunded.
- Idle J = idle power W * (node uptime - VM execution time) / 1000. No double count
  of idle and active power. Uploading/waiting VMs draw idle power. Down power is zero.
- Input energy J = serialized decimal kB * J/kB. Serialization occurs before fixed
  propagation. A failure mid-serialization charges only transmitted bytes; failure
  during propagation has already spent the full input transmission energy.
- Coordinator J = per-task recovery power * the actual elapsed part of the fixed
  recovery delay / 1000. Backup waiting after that delay has no active coordinator
  charge. Coordinator resource contention itself is not simulated.
- Total energy = compute + transmission + idle + coordinator. All failed attempts,
  successful retries and permanently failed tasks contribute. Retry/failed-attempt
  energy are informative subsets and must not be added again to total energy.
- Task-attributed energy sums all its attempts/coordinator work. VM idle energy is
  separate; task totals plus idle equal the whole-run total.
- energy_latency chooses by current completion estimate and marginal active-minus-idle
  compute/input cost. It does not see future failures or use analytical risk values.

Energy per completed task is undefined when nothing completes. Makespan is the last
successful finish, so low makespan with many failures is not evidence of efficiency.
Settlement time and failed/rejected counts are included to make that boundary visible.
No output transfers, idle network equipment, restart checkpoint storage, source faults,
controller failures, live migration, host-sharing power or link contention are included.
These explicit assumptions do not establish a faithful reproduction of the paper.

## Run and inspect

~~~powershell
Set-Location -LiteralPath "C:\Users\ashis\Desktop\Auxia Software\Basic Project\chat-app"
.\.venv312\Scripts\python.exe -m drlmots.recovery_report --config configs/rl_network.yaml --objectives configs/objectives.yaml --recovery-config configs/failure_recovery.yaml --seeds 5200 5201 5202 5203 5204 --output-dir outputs/recovery/professor
Start-Process .\outputs\recovery\professor\recovery-report.html
~~~

Use a fresh output folder for changed configurations; a matching report can be
regenerated without affecting earlier models or reports. The report compares four
rules across three modes: no failures, failures without recovery, and identical
failures with recovery. Five workloads yield 60 actual simulations, not DQN training.

Outputs: recovery-report.html, REPORT.md, metrics.csv, evaluation.json, experiment.json,
60 detailed run JSON files, comparison/timeline PNG+SVG figures, recovery-report.zip.
The offline HTML includes an actual recovered-task breakdown. Extract the ZIP before
opening HTML. Generated artifacts are ignored by Git; share the ZIP or regenerate.
The tracked RECOVERY_RESULTS.md contains measured results for collaborators.

## Remaining before a fault-aware DQN study

Version the state/action/reward contract to include observed node availability,
attempt budget, pending recovery, partial progress and actual incurred costs. Permit
learning decisions on new tasks and retry dispatches. End episodes after drain/horizon,
not merely at the last external arrival, so late faults receive appropriate credit.
Define objective normalization and scalarization explicitly without mixing old
analytical-risk rewards with observed failure outcomes. Keep future outages hidden.
Then add untrained comparisons/validation curves and reserve new untouched test seeds.
