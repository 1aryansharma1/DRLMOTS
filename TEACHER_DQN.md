# Teacher-assisted masked DQN (A-037)

This adds an opt-in **behavior-cloning warm start followed by ordinary masked DQN**. Existing simulator physics, observation contract, predictive features, objectives, masks, heuristics and saved baselines are unchanged. Actual results are recorded separately in `TEACHER_DQN_RESULTS.md`.

## What happens

1. An observation-only version of the multi-objective heuristic schedules 40 training workloads with failures and recovery.
2. Each legal decision becomes a training example: the 253 observed inputs, legal VM mask and the teacher's selected VM. Recovery decisions are included. Forced rejection events have no artificial teacher label.
3. A fresh predictive DQN imitates those examples for a fixed 20 epochs. It still uses the existing 325-input network: 253 observed values plus 72 deterministic predictions.
4. The cloned weights initialize ordinary DQN. Its target is synchronized and its RL optimizer, replay, exploration counters and RL episode count begin fresh. The separate cloning optimizer/history are documented in `cloning.json`.
5. DQN schedules its own workloads for 300 episodes and learns from actual energy, successful makespan, unsuccessful tasks and terminal service penalties. Teacher labels do not enter this RL replay.
6. Every 20 episodes it is evaluated without learning on validation workloads. A checkpoint is selected using the unchanged service-feasibility-first ranking.
7. After all seeds and both training branches finish, selections are locked. Fresh test scenarios are generated and every model is evaluated without learning or exploration.

At inference, the trained neural network selects a legal VM. There is **no teacher fallback** and no heuristic override. DQN can disagree with the teacher. Learning to imitate a rule does not guarantee learning to outperform it; RL can also forget useful imitation behavior.

## What the teacher knows

The native `multi_objective` policy uses event-loop estimates. Feeding its private instantaneous estimates to a learner that sees delayed reports would create inconsistent labels. This implementation retains its scoring approach but supplies the **same observed estimates available to the predictive DQN**.

Both teacher and DQN see contract-4 sample-and-hold observations: current task, locally tracked admissions, delayed VM queues, shared-link work, host utilization, known node health and uptime age, plus public configuration. Neither receives the outage trace, future jobs, actual future completion times or undisclosed failures.

The teacher uses the unchanged predictor in `predictive_dqn.py`; its approximations and limitations in `PREDICTIVE_DQN.md` still apply. In particular, link backlog is a delayed aggregate estimate; it is not an exact future shared-link simulation. Unknown node ages use the predictor's nominal zero-age estimate and availability indicator.

For each legal candidate VM, the teacher calculates:

```text
score = 0.4 × predicted task latency / 50 ms
      + 0.4 × predicted incremental energy / 0.2 J
      + 0.2 × conditional probability of failure during that latency
```

Coefficients and scales come from the configured objective, not hard-coded reward changes. The existing predictor includes upload backlog, propagation, FCFS waiting, compute time, output transfer, host dynamic energy, traffic energy and exponential/Weibull conditional failure risk.

If any legal VM is predicted to meet the task deadline, the teacher considers those VMs first. It chooses the minimum score; exact ties prefer fog, then VM ID. If none can meet the deadline, it selects the minimum score among all legal VMs. With no legal VM it returns no action. Resource feasibility, queue capacity, reachability, known health and recovery's failed-node exclusion remain enforced by the original mask.

This is an **observed multi-objective teacher**, not an assertion that it exactly reproduces native-rule actions. Both the observed teacher and unchanged native rule are evaluated separately in the report.

## How imitation and RL differ

Imitation minimizes masked cross entropy:

```text
L_BC = −log softmax(Q(observation), restricted to legal VMs)[teacher VM]
```

The output values act as classification logits during cloning; their absolute values are not calibrated expected returns. Adam with learning rate 0.001, batch size 64, 20 epochs and seeded local shuffles performs cloning. Gradient clipping uses the existing bound of 10. Data are shuffled normally; there is no recovery oversampling or prioritized replay. Agreement is reported separately for initial placement and recovery, with counts. Training-set agreement is a fit check, not a generalization metric.

The cloned online weights are copied into the target network. The cloning optimizer is discarded; the ordinary RL Adam optimizer starts empty. RL then uses the unchanged Huber Bellman loss, masked next-action maximum, target synchronization, uniform replay and epsilon schedule. RL updates/counters are counted separately from BC updates. The saved `cloned.json` and RL checkpoints use the existing strict predictive checkpoint format and work with the existing inference command.

This is a deliberately small demonstration-assisted extension. It is **not full DQfD**: it does not implement its combined demonstration/TD training, prioritized demonstration replay or additional loss terms. Research motivation: [Hester et al., Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732). That paper's results are not evidence that this scheduler improves.

## Controlled experiment

The fixed plan is `configs/teacher_training.yaml`:

| Purpose | Workload seeds | Failure seeds |
|---|---|---|
| Teacher demonstrations: 40 paired cases | 50000–50039 | 60000–60039 |
| RL training: 300 paired cases | 51000–51299 | 61000–61299 |
| Validation: 3 no-fault + 3 paired fault cases | 86000–86002 | 96000–96002 |
| Fresh test: 3 no-fault + 15 crossed fault cases | 87000–87002 | 97000–97004 |

Model seeds: 42, 43 and 44. Each independently trains both branches for the same 300 RL episodes. That is 1,800 RL episodes / 108,000 task arrivals, plus 40 teacher episodes / 2,400 arrivals. RL uses 300 unique workloads repeated across six model/branch runs; these are not 108,000 unique tasks.

The teacher-assisted branch has extra demonstration-generation and supervised-compute costs. The RL-only control matches the RL budget, not total compute. Both branches share initialization seeds, architecture, reward, masks, epsilon schedule, RL work/fault inputs and validation protocol. Different actions create different visited states and decision counts; identical training seeds do not imply identical trajectories.

Validation selection minimizes, in order: number of infeasible validation runs, mean service-floor shortfall, mean penalized objective cost, and earlier episode. Only RL checkpoints at episodes 20…300 are selectable. The BC-only model is retained as a separate ablation, not substituted after looking at test performance. Validation-ineligible models remain visible rather than being silently dropped.

No old diagnostic/test inputs are reused as new training, validation or fresh test inputs. Plan validation also excludes A-036's training/validation/test ranges. Tests are generated only after `selection-locked.json` is written for every branch and model seed. Hyperparameters and budget are fixed before testing. Test results cannot extend training or select a better-looking checkpoint.

Fresh comparisons include:

- Teacher + RL, untrained predictive DQN and without-DQN native rule: the main three-method report.
- Matched RL-only DQN: the effect of the teacher warm start under the same RL budget.
- Cloned-only DQN: whether subsequent RL preserves or improves imitation.
- Observation-only teacher: how much information/approximation differs from the native rule.
- All three previously selected A-036 predictive models and the original A-034 untrained/trained pair: frozen preserved baselines, re-evaluated on this same new panel.

Three model seeds are averaged for reporting, not deployed as an ensemble. The original 16-episode baseline has one model seed. Native comparisons retain different information access. Approximate 95% paired intervals use three model-seed mean effects on the fixed test panel; they do not treat all test rows as independent training replications or establish general deployment superiority.

## Metrics and recommendation

The same three metrics and combined score remain in use:

```text
cost = 0.4 × successful makespan / 400 ms
     + 0.4 × full system energy / 50 J
     + 0.2 × unsuccessful fraction
```

The actual RL return also includes the unchanged service-floor penalties when completion or deadline success falls below 80%. The report displays penalized cost and feasible-run counts; a method failing a floor in any run is excluded from the recommendation. Successful makespan excludes unsuccessful tasks, so read it alongside completion. Full system energy includes idle/down/reboot states, all traffic, failed partial work and recovery through the same horizon.

Green highlights indicate observed improvements versus untrained; tradeoffs remain visible. A lower mean alone is not a statistically established or universal gain. The report retains all controls even if imitation or RL performs poorly. No new variant is silently promoted into the default scheduler.

## Run and inspect

From the repository in PowerShell:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv312\Scripts\python.exe -u -X utf8 -m drlmots.teacher_training --output-dir outputs/objective-model/teacher-training/my-study
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.teacher_report --directory outputs/objective-model/teacher-training/my-study
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.teacher_audit --directory outputs/objective-model/teacher-training/my-study
Start-Process '.\outputs\objective-model\teacher-training\my-study\report.html'
```

The default baseline directories must contain the preserved A-034 and A-036 runs. On another checkout regenerate those first using their guides, or provide their original directories with `--baseline-dir` and `--predictive-dir`. Source hashes, settings, selected fingerprints and baseline file hashes are checked. The output must be empty. The main training CLI starts a fresh study. The separate teacher_resume command recovers an interrupted pre-test study into a fresh directory, reuses complete models, restores a complete learner checkpoint and verifies exact canonical report replay against preserved complete episodes. It rejects already-tested studies and changed settings/sources. Original evidence stays untouched.

Frozen inference uses the same existing loader:

```powershell
.\.venv312\Scripts\python.exe -X utf8 -m drlmots.predictive_inference --checkpoint outputs/objective-model/teacher-training/study-20261027-resumed/models/teacher_rl/seed-42/selected.json --workload-seed 87000 --failure-seed 97000 --output outputs/objective-model/teacher-training/new-inference.json
```

This replays a now-seen test scenario; it is not a new evaluation panel. Choose a new output filename to preserve evidence.

The experiment directory stores full compressed demonstration and RL reports, validation/test reports, BC histories, actual RL losses, strict checkpoints, locked selections, input/source fingerprints, CSVs, HTML, charts, compact ZIP and report evidence hashes. Large generated artifacts remain under ignored `outputs/`; the tracked results document records the actual findings and reproduction commands.

## Files

| File | Responsibility |
|---|---|
| `src/drlmots/teacher_dqn.py` | Observed teacher, legal demonstration labels, masked cloning, agreement checks |
| `src/drlmots/teacher_training.py` | Strict plan, disjoint panels, both training branches, validation selection, frozen baseline comparison |
| `src/drlmots/teacher_report.py` | Simple report, controls, learning charts, paired effects, ZIP/hashes |
| `src/drlmots/teacher_audit.py` | Independent saved-trajectory energy/time/reward, labels, replay, selection and report audit |
| `configs/teacher_training.yaml` | Fixed experiment protocol |
| `tests/integration/test_teacher_dqn.py` | Teacher legality, hidden failures, cloning and checkpoint behavior |
| `tests/integration/test_teacher_training.py` | Complete pipeline, selection, data separation, reporting and inference |

## Controlled recovery after an external interruption

The execution-permission update terminated the first long run before held-out testing. The actual study was continued in `outputs/objective-model/teacher-training/study-20261027-resumed/`; the original `study-20261027/` directory is retained. Both complete seed-42 models were reused. Seed 43 resumed its full episode-120 checkpoint, including optimizer, replay, RNG and counters. Previously completed later episodes must reproduce exactly before new work proceeds. The budget, seeds, teacher dataset, hyperparameters, objective and selection rule remain fixed. Replayed verification episodes are additional compute, not additional distinct training data.

```powershell
.\.venv312\Scripts\python.exe -u -X utf8 -m drlmots.teacher_resume --source outputs/objective-model/teacher-training/study-20261027 --output-dir outputs/objective-model/teacher-training/my-resumed-study
```

Use this only for an interrupted pre-test source and a new empty destination. Recovery adds a `latest.json` boundary checkpoint and flushes trace/loss buffers at each validation stage. `teacher_resume.py` is included in the recovered experiment's source provenance. The tests compare resumed weights, optimizer/replay state and frozen scenario results with an uninterrupted small experiment.
## Accounting boundary for the professor

Energy and makespan measure the configured task simulator: hosts, shared links and recovery over the fixed horizon. They do not include the workstation's offline BC/RL training energy or wall time, or neural decision-computation overhead. Simulated time does not advance while Python computes a placement. Extra teacher-generation, BC and replay-verification compute is disclosed separately; it is not counted as a task-energy saving.