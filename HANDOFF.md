# DRLMOTS collaborator handoff

## Latest verified delivery - actual failure/recovery (2026-10-25)

Owner: Codex for ashishjha20. A-029 implements a separate observed node-failure and
bounded restart-recovery environment, retaining all previous simulators/models.
732 tests pass (34 new); dependency check clean. Artifact checks: 69 local links,
75 ZIP entries/CRC, 60 task/resource traces and 5 PNG/5 SVG plots. Representative
figures inspected; browser layout remains a manual presentation check. Node failures affect all member VMs;
interrupted work is retried on another node with complete partial/repeated time and
energy accounting. CPU/memory releases, input immutability, no future-fault leakage,
shared outage traces and no-failure baseline parity are verified.

Actual report: 60 simulations (four rules x three modes x five workloads), fixed
250 ms energy horizon. Fog-first completed 69.8/100 without recovery vs 76.0 with it,
but energy and successful latency rise; retries also displace new admissions. No
DQN has been trained for this new fault environment; old analytical-risk checkpoints
are preserved and incompatible by design. Next: version the fault-aware state/action/
reward and delayed terminal credit, then untrained/validation experiments.

Read [FAILURE_RECOVERY.md](FAILURE_RECOVERY.md) and [RECOVERY_RESULTS.md](RECOVERY_RESULTS.md).
Open outputs/recovery/professor/recovery-report.html; share its ZIP after extraction.
Generated artifacts are ignored. Existing analytical-risk reports remain at their
old paths and must not be compared as if the energy definitions were identical.
No implementation task remains active after this delivery; source-fidelity and
hardware-calibration gaps remain explicit. Earlier records below are historical.


## Latest verified delivery - 2026-10-25

Owner: Codex for Aravindak3 / Aravind. Multi-objective extension A-028 is VERIFIED
for its documented synthetic scope. Standard latency-only scheduling is retained.
Optional live reward contract 3 uses weighted latency, active/input energy and
analytical destination risk. Actual failure/backup/recovery simulation is still open.

**Verification:** 698 tests pass; final packaging test rerun passed; pip check clean.
Nine actual 20-episode models: 180 episodes / 18,000 training tasks / 17,062 optimizer
updates. Three independent model seeds, five fresh workload seeds 4100-4104 and three
load scenarios; 195 distinct evaluation simulations, represented in 315 paired rows.
All checkpoints round-tripped and evaluation left training fingerprints unchanged.

**Results:** normal-load fog-balanced DQN versus least queue: mean paired modeled
energy reduction 18.710%, successful mean latency reduction 4.163%; admitted-task
analytical failure risk is worse (11.547% relative increase). The fog-first heuristic
still performs better in normal/burst aggregate comparisons. Burst admission is only
12/100 across all policies; do not interpret low energy from rejection as efficiency.

**Reports:** outputs/efficiency/professor-fresh/efficiency-report.html and
 efficiency-report.zip (extract first), plus CSV/JSON/Markdown and 9 PNG/9 SVG plots.
All 86 local links, 96 ZIP entries/CRC, 45 paired trace groups and resource balances
verified; representative figures inspected and overlapping labels corrected. Normal
browser desktop/mobile layout remains a manual presentation check. Generated outputs
are ignored; share the ZIP or regenerate. Source/config/tests and the tracked results
summary provide the collaborator handoff.

Read [MULTI_OBJECTIVE.md](MULTI_OBJECTIVE.md) and
[MULTI_OBJECTIVE_RESULTS.md](MULTI_OBJECTIVE_RESULTS.md) first. Run:

~~~powershell
python -m drlmots.efficiency --config configs/rl_network.yaml --training-config configs/efficiency_training.yaml --objectives configs/objectives.yaml --output-dir outputs/efficiency/professor-fresh --model-seeds 42 43 44
~~~

Use --resume for matching existing output; choose a fresh folder for a different plan.
Next work: calibrated coefficients and real failure/backup behavior with its costs;
validation-only tuning and a new untouched test set; larger task/VM sweeps and paper
baseline/source fidelity. Independent initializations are implemented, but this fixed
budget study has no validation tuning or significance claim. No implementation task
is currently active.

## Earlier milestone context (2026-10-15)


**Updated:** 2026-10-15. **Current:** v0.3.1 end-to-end synthetic CPU DQN VERIFIED.
**Tests:** 677 pass with demo dependencies; pip check clean.
**Owner:** Codex for Aravindak3 / Aravind; no task active.
Earlier professor demo: Codex for 1aryansharma1 / Aryan Sharma. Earlier baseline work:
ashishjha20. Preserve contributors' history and the user-selected repo-local identity.
**Next research task:** independent model-initialization/validation study in NEXT_STEPS.md.

## What now works

The network actually chooses VMs inside the SimPy arrival loop. Standard zero-network
mode, network-aware standard DQN and network-aware fog-preferred DQN all work end to
end, with training-only scaling, replay/targets/Adam, frozen held-out evaluation and
full JSON checkpoint resume. Old reference rules, schema-1/2 behavior, schema-3
fog-first rule, legacy raw traces and standalone Q-network artifacts are retained.

Read LIVE_DQN.md first for complete commands/contracts/results; then NEXT_STEPS.md,
progress.md, PROGRESS_REPORT.md, assumptions.md (A-027) and implementation_plan.md.
Older STATE_ACTION_CONTRACT.md, NORMALIZATION.md, REWARD_CONTRACT.md and Q_NETWORK.md
describe preserved contract-1 APIs; the live pipeline uses a separate contract 2.

## Reproduce

Follow README setup using supported Python 3.11/3.12 and install the dev/demo extras.
This agent checkout preserves its original .venv and uses ignored .venv312 (Python
3.12.14); the Store Python was inaccessible. Neither venv is supplied by Git.

```powershell
python -m pip install -e '.[dev,demo]'
python -m pytest -q
python -m pip check
python -m drlmots.train benchmark --standard-config configs/normalization_train.yaml --network-config configs/rl_network.yaml --training-config configs/training.yaml --output-dir outputs/training/professor
```

Open outputs/training/professor/training-demo.html. The offline ZIP must be extracted
before opening its HTML. Both link all paired runs, actual task traces, logs and
resumable checkpoints. Outputs/training is ignored: regenerate or obtain the ZIP;
do not assume another contributor has local artifacts. Fixed smoke workloads are
unsuitable for held-out training experiments because different seeds can yield the
same exact task list; that overlap is rejected.

If interrupted, rerun the benchmark with --resume. Matching completed models are
retained; incomplete ones continue without reinitialization/refitting. Individual
train/resume/evaluate commands are in LIVE_DQN.md. Resume requires the saved exact
Python/Torch runtime (this run: Python 3.12.14, Torch 2.14.1+cpu as recorded by the
checkpoint; inspect its actual runtime string rather than guessing an install build).
The implementation tests deterministic resumed versus uninterrupted equality.

## Actual results

Each model trained for 20 episodes (100 tasks each), initialization seed 42,
training workloads 2000–2019 and calibration subset 2000–2002:

| Model | Updates | Held-out mean completed / 100 |
|---|---:|---:|
| standard-zero | 1878 | 93.00 |
| standard-network | 1885 | 94.33 |
| fog-preferred-network | 1920 | 96.00 |

Held-out workload seeds: 3000/3001/3002. Within the zero-network experiment least queue
completes 95.67 on average; within the network experiment fog-first rule completes
97.67. Do not claim the trained agents beat all rules, converged or reproduced paper
results. One initialization plus three workload seeds is not an independent training
seed study. Full timing/returns/sample standard deviations are in JSON/LIVE_DQN.md.

## Code map

| File | Responsibility |
|---|---|
| entities.py / topology.py / resources.py | Validated data, memberships and per-VM reservations; no physical host budget |
| workload.py / run_config.py | Reproducible tasks, explicit units, strict schema 1/2/3 and effective manifests |
| environment.py | Standard zero-network event loop; optional validated live callback |
| fog_cloud.py | Explicit uploads/readiness/reachability, network event loop, fog-first estimates and callback |
| control.py | Frozen detached task/VM/ordered-queue decision frames; illegal callback choices rejected |
| rl_state.py | Version-2 encoding/layout, context identity, standard/fog preference masks and causal reward |
| normalization.py | Existing frozen scaler, now also supports versioned extra flag/placeholder layout |
| q_network.py | Existing CPU float32 network, selector and validated composable JSON weight artifacts |
| dqn.py | Hyperparameters, immutable replay, legal targets, target sync/Adam and online forced-chain credit |
| train.py | Calibration, train/resume/evaluate, exact-task split checks, full JSON checkpoint and atomic saves |
| training_report.py | Three-model benchmark, actual training/held-out figures, static report and ZIP |
| demo.py / templates/demo.html | Preserved two/three-reference-policy demos, separate from trained results |
| configs/training.yaml | Explicit prototype optimizer/exploration/replay/seed settings |
| configs/rl_network.yaml | Six-VM network fixture providing real within-tier learned choices |

New tests: tests/unit/test_dqn.py, tests/integration/test_live_dqn.py,
tests/integration/test_training_cli.py. They cover actual placement, targets,
independent target sync, forced chains, frozen scaling/evaluation, context/split
guards, corrupt checkpoints, exact resume, CLI/ZIP/report links and Windows locks.
Earlier tests remain regression gates.

## Contracts to preserve

- Callback frames arrive after same-time completions/queued ready starts, before
  each task placement. Network starts wait for input upload and FCFS predecessors.
- One running task per VM; only execution reserves CPU/memory. Uploading tasks count
  against waiting capacity. All inputs remain unchanged and every task is accounted for.
- Standard DQN may choose any eligible VM. Fog-preferred DQN has an explicit user
  constraint: fog choices must be no slower than best cloud when fog is preferred;
  otherwise cloud choices must beat best fog. Do not present the constraint as learned.
- Contract 2 retains ordered waiting-slot readiness/service/demand, reachable flags,
  input upload and known remaining arrival budget. Future task values are hidden.
- Normalization fits training calibration only and stays frozen. Context binds full
  topology/units/links/model/action variant/reward/scaler; equal dimensions are insufficient.
- Reward is immediate causal S/(S+end-to-end duration), rejection -P. Forced rewards
  fold once into the previous real action; terminal/no-action states never bootstrap.
- Checkpoints are checksummed non-executable JSON with online/target/optimizer/replay/
  RNG/config/statistics/history. Logs save before the atomic checkpoint commit marker.
  Windows sharing locks retry briefly without deleting the previous checkpoint.
- Evaluation is greedy, has no updates/refit and rejects seed/exact-task overlap.
  Pair inputs within environments; never conflate zero-network/network timings.

## Git and remaining science

Work is only in 1aryansharma1/DRLMOTS, main. Inspect status and fetch before edits;
collaborator documentation through 5dffde0 was merged during this delivery. Preserve
normal commits/history and verify remote main equals local HEAD after publishing.
Git Credential Manager may require an authorised unsandboxed push or normal terminal.
Repo-local author is Aravind with 195939576+Aravindak3@users.noreply.github.com.

The repo targets DRLMOTS cloud/fog (DOI 10.1007/s10586-024-04712-z). The supplied
DRL-MLS Kubernetes PDF (DOI 10.1016/j.future.2026.108459) is different. Original-source
audit, real data, host mapping, richer network/service models, energy/reliability,
paper reward/thresholds and paper comparators remain open. See NEXT_STEPS.md.
