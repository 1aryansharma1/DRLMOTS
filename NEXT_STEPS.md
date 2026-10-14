# Remaining work and pickup checklist

**Updated:** 2026-10-14
**Latest verified implementation:** `29debfb`, v0.3.0a2 training-only normalization.
State/action foundation `a7a08bb`; prior demo `2083faa`; resource foundation `4f4307f`.
**Working state:** completed by Codex for Aravindak3 / Aravind; next task unassigned.
**Publication check:** prior state/action work was confirmed on main at `9604536`.
Compare local HEAD with the remote reference before coordinating new work. The
sandbox may require the user to push from a normal Aravindak3-authenticated terminal.
**Milestone:** v0.1/v0.2 synthetic baseline and v0.2.1 demo software checks verified;
v0.3 partially implemented; v1.0 pending. Demo delivered for 1aryansharma1 / Aryan Sharma.

## What already works

- [x] Task, Host and VM models with validated values and task lifecycle transitions.
- [x] Explicit fog/cloud topology and AP/end links with unique IDs and checked references.
- [x] Deterministic SimPy execution, FCFS waiting queues, fallback and recorded failures.
- [x] Fixed and seeded synthetic workloads with units/settings/checksums recorded.
- [x] Strict runnable YAML, configurable topology/workload, JSON reports and config replay.
- [x] Explicit demand/capacity/unit model, feasible placement and distinct failure reasons.
- [x] Running-task resource reservation/release, trace and final balances without leaks.
- [x] Verified suite with demo dependencies: 460 tests pass; dependency check is clean.
- [x] Runnable fixture: six tasks on three VMs, six completions, 4 ms makespan.
- [x] Resource fixture: four complete/two full-feasible-queue failures, 4 ms makespan,
  eight resource events; old baseline outcomes/config/workload fingerprints preserved.
- [x] Paired least-queue/first-fit comparisons with shared input checks and safe resources.
- [x] Offline professor report, three scenarios, 15 SVG/15 PNG figures and talk track.
- [x] Offline JavaScript control/export test in a DOM stub; wheel contains HTML template.
- [x] Initial observation/feature/action contract documented in STATE_ACTION_CONTRACT.md.
- [x] Immutable snapshots, raw encoder, stable masks/actions and opt-in decision traces
  verified; 41 targeted tests and all 419 full-suite tests pass on Python 3.12.14.
- [x] Training-only maximum scaling, raw/binary/unknown semantics, frozen statistics
  and separate held-out application; another 41 tests pass (460 total).
- [x] Strict JSON artifact persistence, source provenance, topology/schema compatibility
  and two runnable 100-task train/evaluation fixtures; no simulator behavior change.
- [ ] Manually preview the generated HTML in a normal browser at desktop/mobile widths
  before presenting. The agent browser tool disallowed local-file preview; this check
  is independent of starting state/action work. See PROFESSOR_DEMO.md.

These checks establish the current synthetic baseline, not the paper's full algorithm
or experimental results. See [PROGRESS_REPORT.md](PROGRESS_REPORT.md) for evidence.

## Pick up this task next - initial synthetic reward/metric contract

**Owner:** unassigned. **Status:** NOT STARTED.

The A-022 raw contract and A-023 normalization are verified. Next define a causal,
explicitly synthetic reward before wiring an agent/replay/training loop:

1. Claim this task in `progress.md` and record your name/handle and IN PROGRESS status.
2. Read P8, A-006/A-007/A-008, STATE_ACTION_CONTRACT.md and NORMALIZATION.md.
3. Record a provisional formula's units, sign, success/rejection behavior and scales.
   Do not label the source's ambiguous energy/reliability combination as resolved.
4. Define the reward timestamp and credit for queued/delayed completion; use causal
   information and keep held-out outcomes out of fitted reference statistics.
5. Test hand values, shorter waits, full/infeasible queues, all-rejected/empty and
   terminal cases. Reject timing-only comparisons that reward dropping more work.
6. Save formula/config/version in provenance; preserve raw outcomes/masks/scaling and
   state what the later agent transition API must provide before replay/training.
7. Run the relevant tests and full suite. Record actual results, update this checklist
   and HANDOFF with the next task, then commit the reviewed change.

**Finish criterion:** a documented synthetic reward and timing contract with finite
hand-checked behavior. Paper reward fidelity, energy/reliability, full agent transitions
and DQN training remain later tasks.

## Everything still remaining

The ordering below is the pickup order. Paper review can progress alongside software
work, but its unresolved scientific decisions must be resolved or documented before
paper-faithful training/evaluation. Detailed microtasks remain in implementation_plan.

| Order | Scope / plan reference | Remaining deliverable | Completion evidence |
|---|---|---|---|
| 1 | Initial synthetic P8 subset | Explicit scalar reward, metric/rejection semantics and causal timing/credit contract | Hand values, direction/bounds, no future leakage and formula provenance |
| 2 | Paper-specific placement, P7.1/P7.4/P7.5 | Decide priority/tier thresholds and fallback beyond the implemented all-tier least-queue baseline | Source-supported or labelled policy; ordering/threshold/no-capacity tests |
| 3 | P0.2-P0.4 / R001-R040 | Independently audit the original DRLMOTS paper and extraction; classify each source-supported vs assumed behavior | Paper-to-code/test traceability and updated assumptions; no invented requirement |
| 4 | P4.4 | Handle the paper's conflicting task-count ranges explicitly in experiment configs | Exact requested counts; separate named experiment settings; manifests |
| 5 | P4.5 / A-011 | Identify dataset version/schema/filtering/mapping before any Google Jobs label | Verified provenance and ingestion tests, or explicitly unavailable with synthetic alternatives |
| 6 | Full P5.1 | Configure the source-audited full paper topology and VM distribution | Counts/memberships match documented source or a labelled assumption |
| 7 | P5.7 / A-014 | Decide and implement network/transfer delay and its units, if supported | Hand-calculated timestamps; clearly labelled inclusion/exclusion |
| 8 | Full P6.4 / A-001 | Extend the initial provisional observation with source-audited feature/boundary coverage if justified | Explicit schema evolution; finite stable shapes; saturation/unknown checks |
| 9 | P7 / A-003-A-004 | Extend masks/mapping for the chosen paper tier policy after baseline actions are verified | No illegal selection; no-action outcome; policy/encoding compatibility |
| 10 | P8 / A-006-A-008 | Audit makespan semantics, compute/transmission energy, reliability/fault metric and scalar reward | Hand-calculated units/sign/scales; reward direction verified |
| 11 | P9 / A-010 | Confirm DQN width/layers and implement CPU network | Correct input/action dimensions, finite outputs, parameter checks |
| 12 | P10 / A-009 | Replay, valid-action exploration, terminal targets, target synchronization and learning settings | Capacity/sampling, numeric targets, epsilon bounds and finite gradient/update tests |
| 13 | P11 | Agent/environment transition boundary, one experience per decision, terminal handling | Tiny complete episode; consistent state/action/reward/next-state/done; no task loss |
| 14 | P12 | Seeded training loop, logs, checkpoints and bounded CPU smoke training | No NaNs; checkpoint reload matches outputs; complete run manifest |
| 15 | Full P13 / A-012 | Extend verified simple paired comparisons to justified paper comparators, learned policy and repeated-seed evaluation | Shared workloads/resources/metrics/seeds; approximate baselines clearly labelled |
| 16 | P14 / A-015 | Stable raw task/run result tables and independent percentage audit | Row accounting, provenance, correct arithmetic and repeated-seed summaries |
| 17 | Full P15 | Extend demo figures into experiment/task-count sweeps and analysis | Regenerated from saved data, correct units/sample counts and limitations |
| 18 | P16 | Clean setup, repeatable end-to-end checks, complete requirement audit and final report | All applicable gates verified or explicitly open; no unsupported reproduction claim |

## Source and training decisions still open

The repository targets DRLMOTS cloud/fog scheduling, DOI 10.1007/s10586-024-04712-z.
The supplied DRL-MLS Kubernetes PDF is a different paper, DOI 10.1016/j.future.2026.108459.
Do not combine their state/action/reward/resource definitions. The original DRLMOTS
PDF was not independently audited during these implementation checkpoints.

Open paper items include task length units and the value 700,000; arrival distribution;
priority/threshold equations; task/VM resource capacities and physical mapping;
reward sign/normalization; energy units/coefficients; fault-tolerance interpretation;
initial epsilon, episode counts and unspecified DQN settings; real dataset identity;
exact CNN/LSTM/GGCN baselines; task-count conflicts and reported improvement arithmetic.
The synthetic per-VM resource model does not resolve the paper's physical capacities
or mapping. See assumptions A-001 through A-023 and retain the unresolved entries.

## Before handing off again

- Update `progress.md` with owner, status, exact tested scope and actual verification.
- Update `PROGRESS_REPORT.md` with delivered behavior, limitations and implementation commit.
- Update `HANDOFF.md` and this checklist with one concrete next task and acceptance criteria.
- Preserve other contributors' work; inspect/fetch `main` before integration and use normal
  commits/pushes. A task is complete only after its code and applicable checks pass.
- Local `.venv`, PDFs/datasets and generated outputs are not supplied by the repository.
  Follow README setup and regenerate results; never assume another machine's environment.
