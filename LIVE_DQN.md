# End-to-end live DQN: standard and network-aware modes

Version **0.3.1**, assumption A-027. The Q-network now controls actual VM placement
inside the SimPy event loop and learns from those placements. All earlier reference
policies, YAML schemas, raw contract-1 traces and demos remain available.

## Run the complete professor experiment

From an installed checkout with the demo extra (`pip install -e '.[dev,demo]'`):

```powershell
python -m drlmots.train benchmark --standard-config configs/normalization_train.yaml --network-config configs/rl_network.yaml --training-config configs/training.yaml --output-dir outputs/training/professor
Start-Process outputs/training/professor/training-demo.html
```

This trains and reloads three separate models:

| Model | Environment | Action selection |
|---|---|---|
| standard-zero | Original zero-network simulator | DQN among all resource/queue-legal VMs |
| standard-network | Explicit uploads/propagation, six eligible-capacity VMs | Same unconstrained DQN with network-aware state |
| fog-preferred-network | Exactly the same six-VM network environment | DQN inside the explicit fog/cloud completion preference |

The new six-VM fixture gives the constrained agent real within-tier choices. If best
fog finishes no later than best cloud, allowed actions are fog VMs no slower than
best cloud. Otherwise they are cloud VMs faster than best fog. With only one available
tier all its legal VMs are allowed. This constraint is a user rule, **not a learned
discovery**. Standard DQN has no such constraint. Both variants update their networks.

The benchmark compares standard-zero against its two original baselines, and the
two network agents against least-queue, first-fit and fog-first under identical
network inputs. Zero-network and network results are separate experiments.

## Train, resume and evaluate separately

```powershell
python -m drlmots.train train --config configs/rl_network.yaml --training-config configs/training.yaml --variant standard --output-dir outputs/training/my-standard
python -m drlmots.train train --config configs/rl_network.yaml --training-config configs/training.yaml --variant fog_preferred --output-dir outputs/training/my-fog
python -m drlmots.train resume --checkpoint outputs/training/my-fog/checkpoint.json --additional-episodes 10 --output-dir outputs/training/my-fog
python -m drlmots.train evaluate --checkpoint outputs/training/my-standard/checkpoint.json --checkpoint outputs/training/my-fog/checkpoint.json --config configs/rl_network.yaml --seeds 3100 3101 3102 --output outputs/training/network-comparison.json
```

`train` refuses an existing checkpoint. Resume it explicitly instead. If the full
benchmark is interrupted, rerun its command with `--resume`: matching completed
models are retained and incomplete ones continue without refitting the scaler. A
different environment, architecture or hyperparameters is rejected. Increasing only
the episode budget is supported. Resume requires the saved Python/Torch versions.
An evaluation-only command never updates weights, replay, RNG or normalization.

## Implemented training contract

- **Live callback:** both simulators supply immutable decision frames after same-time
  completions/ready queued starts and before placement. The callback's VM is checked
  against real eligibility and actually executed. Missing/illegal/voluntary reject
  actions fail clearly; forced rejection requires None.
- **State 2:** task/time/demand and VM resource fields plus known remaining arrival
  budget, reachability, current input upload, queue-tail duration and each waiting
  slot's presence/upload readiness/service/demands in FCFS order. This represents
  idle VMs with uploading queues. No future task sizes/arrivals are exposed; a general
  real-world Markov-completeness proof is not claimed. State dimensions are 72 in the
  standard fixture and 168 in the six-VM network fixture.
- **Identity:** full topology/units, queue layout, source/network model, action variant,
  feature schema, fitted scaler and reward are bound into checkpoint context. Equal
  vector dimensions alone never establish compatibility. Contract 1 remains unchanged.
- **Scaling:** fit maxima using the first three declared training workload seeds
  under least-queue calibration. Those seeds are also part of network training.
  Extra presence/reachability flags pass through; empty/unknown values are guarded.
  Training/application use frozen statistics and clip numeric values to [0,1].
- **Reward 2:** immediate S/(S+causal predicted completion duration), including upload
  and FCFS waiting, with S=10 ms; forced rejection -1. No duplicate completion/drain
  bonus. This is a synthetic surrogate, not paper Eq. (25).
- **Transitions:** the next actionable arrival closes the previous action. Following
  forced rejections accumulate as `r0 + gamma*r1 + ...`; bootstrap discount is
  `gamma**arrival_steps`, already computed once. Forced prefixes stay in episode
  return without fake replay actions. Final actionable transition is terminal; an
  all-forced episode has no replay. Physical VM queues still drain for final metrics.
- **Replay/update:** bounded immutable CPU transitions, local seeded sampling without
  replacement, masked target maximum, terminal reward-only targets, Smooth L1 loss,
  Adam and gradient clipping. Target weights are independent/frozen and copied only
  on the configured update interval. No max over empty masks or gather of null actions.
- **Exploration:** epsilon 1 -> .05 over 2000 actionable training choices, greedy at
  evaluation; all sampling is legal and local-seeded. CPU thread changes are scoped
  and restored. Explicit settings are in configs/training.yaml, not paper defaults.

Default run: 20 episodes/model, 100 tasks/episode; training workload seeds 2000–2019,
calibration subset 2000–2002, initialization seed 42; held-out workload seeds
3000/3001/3002. Seed ranges and actual task hashes must be disjoint. Exact task overlap
is rejected even if seed labels differ. No held-out tuning/checkpoint selection is
performed: the fixed-budget final checkpoint is evaluated. A workload without legal
placements fails before training; a warmup-only run reports no optimizer update and
leaves a checkpoint for explicit continuation instead of claiming successful learning.

## Persistence and recovery

Each completed episode atomically saves actual episode/log data followed by the
checkpoint as a commit marker. The JSON checkpoint includes online/target weights,
Adam moments/step, replay, both local RNG states, scaler/provenance, full configs,
action/context identity, counters and loss/history. No executable pickle is loaded.
Checksums, versions, shapes, finite values and counter/discount/accounting consistency
are validated. A checksum detects accidental changes, not malicious authenticity.

Windows may briefly lock a destination during scanning. Atomic replacement retries
only sharing/access-lock errors within a bounded interval; the old checkpoint is
never deleted to force a save. The actual professor run exercised checkpoint resume
after such a lock. Unit/integration tests prove resumed training equals uninterrupted
training, including optimizer/replay/RNG and final checkpoint checksum.

## Actual verified run, 2026-10-15

**677 tests pass**, with 59 new live-training tests, and dependency check is clean.
Tests cover targets, independent target sync, actual parameter changes, immutable
replay/eviction, forced prefixes/middles/suffixes at gamma 0/.5/1, live placements,
resource balances, split/context guards, corrupt checkpoints, exact resume, CLI,
Windows lock retry and a complete offline report/ZIP with valid local links.

| Model | Episodes | Optimizer updates |
|---|---:|---:|
| standard-zero | 20 | 1878 |
| standard-network | 20 | 1885 |
| fog-preferred-network | 20 | 1920 |

All three changed their initialized weights and round-tripped full checkpoints.
Three held-out workload seeds give these means (100 submitted tasks per seed):

| Environment / policy | Completed | Rejected | Successful mean ms |
|---|---:|---:|---:|
| Zero / least queue | 95.67 | 4.33 | 3.27 |
| Zero / first fit | 93.00 | 7.00 | 3.62 |
| Zero / standard DQN | 93.00 | 7.00 | 3.62 |
| Network / least queue | 94.33 | 5.67 | 12.96 |
| Network / first fit | 93.00 | 7.00 | 13.38 |
| Network / fog-first rule | 97.67 | 2.33 | 11.84 |
| Network / standard DQN | 94.33 | 5.67 | 13.07 |
| Network / fog-preferred DQN | 96.00 | 4.00 | 12.26 |

Latency is the mean of per-seed successful means, not a pooled statistic. Raw JSON
also records sample standard deviations and all-arrival return. These short runs do
**not** beat every heuristic and are not claimed as converged/optimal. Three workload
seeds are not three independent model-initialization runs. The next research gate is
a predeclared multi-initialization study with separate validation-controlled tuning.

## Deliverables and boundaries

`outputs/training/professor/training-demo.html` is a static offline result report:
no JavaScript or runtime training is required to display it. It links five PNG/five
SVG plots, complete paired reports, actual training traces and resumable checkpoints.
`professor-demo.zip` contains that bundle; **extract it before opening the HTML**.
Browser layout still requires a normal desktop/mobile preview; representative figures
and report local links/ZIP integrity were checked automatically. The old two-policy
and network-rule demos are preserved as reference views.

Outputs/training is ignored by Git: teammates regenerate with the command or receive
the ZIP separately. Do not assume checkpoints/venvs/generated outputs come from Git.
Current code: control.py / rl_state.py / dqn.py / train.py / training_report.py.

This remains a synthetic SimPy implementation. Shared network contention, downloads,
physical host budgets, CPU-dependent processing rates, energy/reliability, dataset
provenance and source-faithful paper equations/baselines remain separate research work.
