# DRLMOTS — Phased Implementation Roadmap

This roadmap is intentionally incremental. Each implementation microtask will be delivered in a later response with exact file contents/changes, exact command(s), expected output, verification, troubleshooting, and a completion criterion. Do not skip a verification gate or mark a task VERIFIED without actual evidence.

## Version milestones (2026-10-15)

The user requested practically achievable versions before attempting the complete paper.
The original phases below remain the full reproduction roadmap. Version gates apply
to the subset actually used: optional random workloads/dataset ingestion and paper-scale
topology are not prerequisites for a fixed-list, tiny software baseline. Their unfinished
microtasks stay pending. This staged scope is documented in assumptions A-017.

| Version | Scope | Gate / current evidence |
|---|---|---|
| 0.1 | P3.3-P3.5; P4.1-P4.2; fixed-list P4.6; deterministic P5.2-P5.6/P5.8 | VERIFIED: Phase 3 passed 136 tests before simulation; final suite 194 passed; exact 3-task example; `pip check` clean |
| 0.2 | P4.3 workload/manifest; configurable topology; explicit capacities and documented feasibility/placement policy | VERIFIED for A-020 synthetic baseline: package 0.2.0, 367 tests pass; no paper/physical-model claim |
| 0.2.1 | User-requested professor demo: paired simple policies, offline report, figures and walkthrough | VERIFIED software checkpoint `2083faa`: 378 tests; browser layout manual check; A-021 |
| 0.3.0a1 | Initial P6.1/P6.2/P7.2: immutable observations, pure raw encoding and legal actions | VERIFIED `a7a08bb` under A-022; 419 tests and clean dependency check |
| 0.3.0a2 | P6.3: training-only scaling, schema/topology-bound statistics, held-out application | VERIFIED `29debfb` under A-023; 460 tests and clean dependency check |
| 0.3.0a3 | Initial synthetic P8/P11: causal reward, arrival transitions and forced-chain adapter | VERIFIED `72255bf` under A-024; 510 tests and clean dependency check |
| 0.3.0a4 | Synthetic P9 / P10 selection subset: CPU Q-network, legal epsilon-greedy and compatible snapshots | VERIFIED under A-025; 576 tests, exact reload and held-out inference; see Q_NETWORK.md |
| 0.3.0a5 | User-requested P5.7/P7 subset: explicit uploads, local fog reachability and fog-preferred finish-time selection | VERIFIED under A-026; 618 tests, network-aware CLI/three-policy demo; legacy RL capture intentionally guarded |
| 0.3 | P6-P12: documented state/action/reward, network/replay, CPU learning and checkpoints | Partial: network/selection delivered; next replay/targets; live control/training pending |
| 1.0 | Original-paper audit, experiment sizes, appropriate baselines and evaluation | Numerical reproduction only where source/data/definitions support it |

v0.1 does not complete P4.3-P4.5, the full P5.1 topology builder, P5.7 transfer delays,
or paper-specific placement/reward. `progress.md` is the authority for actual completed
checks. The source extraction is for DRLMOTS; the supplied DRL-MLS Kubernetes PDF is
a different paper and has not been combined with its definitions.

The v0.2 workload checkpoint completes P4.3 and the seeded-input portion of P4.6
under A-018. It adds configurable CLI seed/count/ranges; topology is still the small
fixture. P4.4-P4.5 and full P5.1/P5.7 remain pending. The first arrival is 0 ms;
subsequent intervals are synthetic uniform draws, with explicit byte lengths.

The configured-topology checkpoint adds `build_topology`, a strict synthetic run YAML,
`--config`, effective configuration/checksum and replay. A-019 documents this scoped
P5.1 delivery: counts/resources/links are explicit and validated; full paper topology,
physical VM placement and resource feasibility remain open. Six tasks/three VMs finish
in exactly 4 ms; that checkpoint passed 304 tests (before the resource model below).

The final v0.2.0 checkpoint (`4f4307f`) adds explicit unit catalogues, per-VM capacities,
task demand profiles, componentwise eligibility and running-task reservation/release.
Schema 2 preserves these constraints in replay; schema 1 and generator-v1 checksums
remain compatible. A-020 records the baseline approximation: all tiers may be eligible,
FCFS/least-load/VM-ID rules remain, CPU units do not change service speed, and no physical
host budget or paper-specific threshold is inferred. The resource fixture gives four
completions/two explicit failures with all balances restored; 367 tests pass. This meets
the scoped v0.2 milestone, not the full original Phase 7/paper reproduction gates.

Before v0.3 the user requested a professor-facing demonstration. A-021 scopes this
early P13.1 synthetic comparison: least queue and first fit run identical workloads;
the report includes failures beside timings, occupancy, outcomes and input hashes.
Three presets and reusable SVG/PNG figures support a five-minute walkthrough in
PROFESSOR_DEMO.md. This completes the demonstration checkpoint, not the full P13
paper comparisons or P15 multi-seed/task-count analysis. Subsequent prereleases below
implement state/actions, normalization and a synthetic reward; paper energy/reliability
and DQN remain pending.

The initial v0.3 prerelease implements A-022 arrival-time snapshots and a fixed
`3+R+V*(6+4R)` raw vector. VM/resource order, units and schema hash accompany an
opt-in trace. Legal masks use total capacity and waiting slots; all-false masks retain
explicit rejection. This is a provisional synthetic observation, with no claim of
paper fidelity or Markov sufficiency. Training-only normalization is now verified
under A-023 with saved statistics, compatibility checks and no held-out fitting.
The initial synthetic reward/transition export is now verified under A-024. It uses
causal projected latency and explicit rejection penalties, with a forced-chain adapter
requiring gamma per arrival. Full source metrics/reward, interactive agent control,
replay and learning remain. State/action design
does not complete the full P6-P12 milestone.

## Project directory (target)

```text
DRLMOTS/
├── README.md
├── paper_requirements.md
├── implementation_plan.md
├── assumptions.md
├── progress.md
├── pyproject.toml
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                 # externally obtained data; do not commit raw data by default
│   └── processed/
├── src/
│   └── drlmots/
│       ├── __init__.py
│       ├── config.py
│       ├── entities.py      # Task, VM, fog/cloud nodes, host
│       ├── topology.py      # topology construction/invariants
│       ├── workload.py      # synthetic/workload-file input
│       ├── priority.py      # task priority; paper gap kept explicit
│       ├── placement.py     # threshold/tier and valid-VM selection
│       ├── environment.py   # SimPy environment and step/transition semantics
│       ├── state.py         # state feature encoding/normalization
│       ├── actions.py       # valid actions, action-index ↔ VM mapping
│       ├── reward.py        # isolated, configurable reward calculation
│       ├── metrics.py       # makespan, energy, fault metric
│       ├── baselines.py     # FCFS/sanity policies and paper comparator variants
│       ├── dqn/
│       │   ├── __init__.py
│       │   ├── network.py
│       │   ├── replay_buffer.py
│       │   └── agent.py
│       ├── training.py
│       ├── evaluation.py
│       ├── plotting.py
│       └── reproducibility.py
├── scripts/
│   ├── run_smoke.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
└── outputs/
    ├── logs/
    ├── metrics/
    ├── figures/
    └── checkpoints/
```

Only create a module when its first microtask needs it. Do not create empty files for the entire structure in advance. Raw Google workload data is not bundled because the paper's dataset identity/schema is not sufficiently explicit.

## Phase 0 — Paper analysis and requirements (source analysis completed; scope still has open issues)

**Objective:** Map every implementable requirement to paper evidence, a module and a test. **Prerequisites:** uploaded 26-page paper. **Deliverables:** `paper_requirements.md`, `assumptions.md`, `progress.md`. **Verification:** review equation/page references, check the R001–R040 traceability table, and make sure ambiguous facts are marked OPEN rather than silently filled. **Completion gate:** extraction artifacts exist and no unresolved source gap is represented as a fact.

| Microtask | Ordered action | Output / verification | Status |
|---|---|---|---|
| P0.1 | Read all pages, including figures, Algorithm 1, Tables 3–19, and conclusion/data statement | Full-page/source inspection; key diagrams inspected | VERIFIED |
| P0.2 | Extract topology, state, action, reward, Q-update, training values, metrics and workloads | `paper_requirements.md`; compare all facts to page/section references | IMPLEMENTED |
| P0.3 | Write unique paper requirement IDs and module/test mapping | R001–R040 traceability table | IMPLEMENTED |
| P0.4 | Record contradictions/missing parameters in a decision log | `assumptions.md`; no unsupported value silently filled | IMPLEMENTED |
| P0.5 | Confirm the first implementation boundary (smoke simulator before training) | Initial decision and version scopes recorded in `assumptions.md` | VERIFIED |

## Phase 1 — Development environment

**Objective:** Discover the existing system and set up an isolated, reproducible Python environment. **Prerequisites:** none. **Files:** later `pyproject.toml`; no source-code files yet. **Tests:** Python/version/platform checks, package installation check, import smoke test. **Completion gate:** command output shows the selected interpreter, environment location and installed dependency versions; `pip check` and imports pass. Never install until P1.1 confirms the environment.

| Microtask | Ordered action | Inputs/files | Verification/completion criterion |
|---|---|---|---|
| P1.1 | Collect OS, shell, architecture, Python launcher/version/path, Git availability, current directory | Terminal only | User pastes output; interpreter choice is confirmed |
| P1.2 | Choose a supported Python version and create a project folder and virtual environment | Project directory chosen with user; command depends on OS | Activate env; `python --version` and `python -m pip --version` point to env |
| P1.3 | Add only necessary direct dependencies to `pyproject.toml`: SimPy, NumPy, PyTorch; Pandas/Matplotlib/pytest where needed | `pyproject.toml` | Dependency versions install without resolver error |
| P1.4 | Install dependencies inside the active environment | Terminal, project root | `python -m pip list` has expected packages; no global installation relied on |
| P1.5 | Check core imports and Torch device availability (CPU first; accelerator optional) | One temporary smoke command/script | Script exits 0, prints versions; record CPU/device result |
| P1.6 | Record dependency versions and environment notes | `progress.md`, `README.md`, lock/export file as appropriate | A clean re-run reproduces the same environment setup |

## Phase 2 — Minimal project skeleton and config

**Objective:** Create a small Python package with one configuration source and a test runner. **Prerequisites:** P1 VERIFIED. **Files:** `pyproject.toml`, `configs/default.yaml`, `src/drlmots/__init__.py`, test directory. **Verification:** editable install/import, config schema/default checks, pytest discovery. **Gate:** tests pass from repository root.

| Microtask | Ordered action | Verification |
|---|---|---|
| P2.1 | Create repository root and package/test/output/data directories | Directory listing matches only the agreed initial structure |
| P2.2 | Add package metadata and test configuration | Build/import metadata parses; no unexplained dependencies |
| P2.3 | Create config schema for seed, topology, workload and DQN placeholders with unresolved values explicitly tagged | Required keys validate; invalid types fail loudly |
| P2.4 | Add package entry point and a `python -m drlmots` or smoke script only if useful | Import smoke test succeeds |
| P2.5 | Create a first trivial pytest test and verify test discovery | `pytest` collects and passes that test |

## Phase 3 — Data structures and parameter validation

**Objective:** Represent tasks, VMs, hosts, fog nodes and the cloud using consistent IDs and explicit units. **Prerequisites:** P2 VERIFIED. **Files:** `entities.py`, `config.py`, tests. **Verification:** constructors, uniqueness, resource capacities and invalid-input tests. **Gate:** no duplicate IDs, negative durations or invalid capacities can silently enter the model.

| Microtask | Ordered action | Verification |
|---|---|---|
| P3.1 | Define `Task` fields (ID, arrival time, length/data size, required resources, status) from source-supported fields only | Type/value tests; missing attributes not fabricated |
| P3.2 | Define `VM`/host fields (ID, tier, capacity, queue, power only once power parameters are resolved) | Capacity and ID validation tests |
| P3.3 | Define FogNode, CloudDatacenter and AP/EndNode structures needed by the chosen topology | Parent-child/topology relationship tests |
| P3.4 | Add enums/status transitions for created, queued, running, completed, failed | Invalid transitions reject with descriptive errors |
| P3.5 | Validate configuration ranges, including 20–200 VMs and host specs, while preserving the task-count contradiction as a separate experiment setting | Config validation tests |

## Phase 4 — Workload generation and data ingestion

**Objective:** Generate small deterministic workloads first, then configurable random workloads, and keep any Google trace mapping separate. **Prerequisites:** P3 VERIFIED. **Files:** `workload.py`, optional `data/processed/` manifest, workload tests. **Verification:** bounds, arrivals, unique IDs, fixed-seed repeatability, processing-time hand calculations. **Gate:** every generated task is accounted for and all units are logged.

| Microtask | Ordered action | Verification |
|---|---|---|
| P4.1 | Implement a fixed-list task factory for deterministic unit tests | Input list round-trips with exact IDs/fields |
| P4.2 | Implement the paper's Eq. (8) calculation as a pure function with explicit units | 1000-byte example yields 2 ms; unit mismatch is rejected or documented |
| P4.3 | Implement seeded random arrivals and task sizes in separately configured ranges | Same seed gives same tasks/arrival sequence; sizes remain within configured bounds |
| P4.4 | Add small/medium/large experiment workload sizes and explicitly resolve 300–3000 vs 3300–4500 config handling | Generated counts match requested test-case values |
| P4.5 | Inspect/identify a real workload source and mapping before labelling any file “Google Jobs” | Dataset schema and provenance documented; otherwise mark unavailable and use synthetic data transparently |
| P4.6 | Save a workload manifest containing seed, generator settings and data-source checksum/metadata | Re-run can identify exact workload |

## Phase 5 — Discrete-event environment and topology

**Objective:** Model arrivals, queues, resource availability, execution and completion with SimPy. **Prerequisites:** P3 and P4 VERIFIED. **Files:** `topology.py`, `environment.py`, tests. **Verification:** deterministic one-task and two-VM hand-traced timelines, no lost/duplicate tasks, queue/capacity invariants. **Gate:** tiny workload produces exact expected start/finish timestamps.

| Microtask | Ordered action | Verification |
|---|---|---|
| P5.1 | Build the paper's four-device/AP/end-node/fog-node/cloud topology in config | Topology counts match requested configuration |
| P5.2 | Add one VM and one task to the smallest SimPy environment | One task starts and completes once |
| P5.3 | Add task arrival events and per-VM queues | Simultaneous arrivals are ordered by the selected queue policy, documented |
| P5.4 | Add service-time processes from the pure Eq. (8) function | Completion time matches a hand calculation |
| P5.5 | Track completion timestamps and running/queued tasks | Every task ends in one terminal state, no task disappears |
| P5.6 | Add VM-capacity and full-queue behavior, including “try another eligible fog VM” | Full queue triggers exact specified fallback or an explicit no-capacity outcome |
| P5.7 | Add network-transfer delays only if a formula/parameter is made explicit; keep the default separately identifiable | Tests show whether transfer delay is included/excluded |
| P5.8 | Add deterministic edge cases: empty workload, simultaneous arrivals, resource unavailable, queue full, invalid VM, environment termination | All cases pass without swallowed exceptions |

## Phase 6 — State representation

**Objective:** Convert current task/environment into a numeric, stable feature vector. **Prerequisites:** P5 VERIFIED and A-001 decision documented. **Files:** `state.py`, config, unit tests. **Verification:** fixed ordering, fixed dimensions, finite values and normalization tests. **Gate:** same environment snapshot always maps to the same vector shape/values.

| Microtask | Ordered action | Verification |
|---|---|---|
| P6.1 | Decide the exact source-supported state fields and record each unit | `assumptions.md` contains state schema and rationale |
| P6.2 | Implement a pure state encoder returning numeric NumPy array/tensor-ready data | Hand-built environment matches expected vector |
| P6.3 | Add normalization with documented fit/use rules (no evaluation-data leakage) | Training-only normalization tests pass |
| P6.4 | Test empty queues, saturated VMs, out-of-range values and non-finite inputs | Encoder fails or handles edge cases explicitly; dimension invariant holds |

## Phase 7 — Action space and placement/scheduling decision

**Objective:** Map each legal action index to exactly one VM and preserve the chosen tier-threshold policy as a traceable decision. **Prerequisites:** P5/P6 VERIFIED; A-002–A-004 decisions logged. **Files:** `actions.py`, `placement.py`, `priority.py`, tests. **Verification:** one-to-one action mapping, no illegal full/unavailable VM selection, threshold boundary tests. **Gate:** every accepted action has a clear scheduling outcome.

| Microtask | Ordered action | Verification |
|---|---|---|
| P7.1 | Implement and test task-priority policy as a separate replaceable function | Hand-calculated priority ordering follows documented rule; approximation labelled |
| P7.2 | Implement valid VM list and action-index conversions | Round-trip index↔VM identity tests pass |
| P7.3 | Implement resource/queue feasibility and invalid action rejection | Full/unavailable VM cannot be accepted silently |
| P7.4 | Implement threshold/tier placement policy behind an interface | Boundary tests and decision log align with chosen interpretation |
| P7.5 | Test fallback across fog nodes and “no eligible resource” handling | No task loss; explicit wait/reject/cloud fallback behavior is observed |

## Phase 8 — Metrics and reward

**Objective:** Implement metrics independently of reward, then derive a configurable reward decision from the paper and ambiguity log. **Prerequisites:** P5 VERIFIED; A-006–A-008 decisions documented before agent training. **Files:** `metrics.py`, `reward.py`, tests. **Verification:** hand-calculated examples and unit/direction tests. **Gate:** reported metric definitions and units are documented; reward responds in the intended direction for each objective.

| Microtask | Ordered action | Verification |
|---|---|---|
| P8.1 | Implement makespan Eq. (16) from task completion timestamps | Multi-VM hand example has exact expected makespan |
| P8.2 | Implement compute-energy `P×T` as a pure function with explicit units | Hand calculation matches exactly |
| P8.3 | Implement transmission-energy calculation separately; do not select `E_data` without a documented source/default | Unit tests and dimension audit pass |
| P8.4 | Implement the selected fault/reliability probability equation only after resolving its meaning | Values stay within [0,1] where applicable; semantic name matches formula |
| P8.5 | Independently compute paper Eq. (25) examples and expose its known scale/direction problem | No silent unit mixing; assumptions documented |
| P8.6 | Add scalar reward with configurable terms/signs only after agreement/decision log | Tests prove better/worse cases get intended reward ordering |

## Phase 9 — DQN network

**Objective:** Build the network shape described in Algorithm 1 and verify tensor dimensions before training. **Prerequisites:** P6 and P7 VERIFIED; state length/action count fixed for the configured run. **Files:** `dqn/network.py`, config and tests. **Verification:** forward pass, output dimension, finite Q values, CPU inference. **Gate:** network output has one Q value per configured legal-action slot.

| Microtask | Ordered action | Verification |
|---|---|---|
| P9.1 | Confirm hidden-layer width choice in `assumptions.md` (paper says 50 neurons, precise interpretation uncertain) | Decision has cited source and status |
| P9.2 | Implement input → Dense/ReLU → Dense/ReLU → Q-output network | Output shape `[batch, n_actions]` for batch size 1 and >1 |
| P9.3 | Test dtype/device and finite outputs | CPU forward pass succeeds; invalid dimensions fail clearly |
| P9.4 | Inspect parameter count/layer widths against configured architecture | Test assertion confirms chosen widths and action dimension |

## Phase 10 — Replay buffer, exploration and target network

**Objective:** Implement experience storage, epsilon-greedy action selection and fixed target-network bootstrap. **Prerequisites:** P9 VERIFIED. **Files:** `dqn/replay_buffer.py`, `dqn/agent.py`, tests. **Verification:** replay sampling, terminal target, non-terminal target, epsilon limits and target synchronization. **Gate:** hand-computed update target matches implementation.

| Microtask | Ordered action | Verification |
|---|---|---|
| P10.1 | Resolve initial epsilon (1.0 vs comment 0.1) and decay interpretation in config | Decision log contains chosen value/rule |
| P10.2 | Implement bounded replay buffer for `(s,a,r,s_next,done)` | Capacity, insertion, sampling and shape tests pass |
| P10.3 | Implement epsilon-greedy with only valid actions selectable | Forced explore/exploit tests pass; epsilon remains in bounds |
| P10.4 | Implement online/target networks and terminal-aware target equation | Hand-calculated terminal and non-terminal cases pass |
| P10.5 | Resolve/install learning rate, optimizer, batch size, buffer capacity and target sync interval as explicit implementation values | All values printed in run config; no “paper specified” claim for missing values |
| P10.6 | Implement one gradient-update step with shape and finite-loss assertions | Parameters change on a valid update; gradients/loss are finite |

## Phase 11 — Agent/environment integration

**Objective:** Connect a single DQN action to one environment transition without dropping or duplicating tasks. **Prerequisites:** P5–P10 VERIFIED. **Files:** `environment.py`, `agent.py`, integration tests. **Verification:** deterministic one-step and complete tiny-episode tests. **Gate:** each task assignment creates one consistent transition and reward record.

| Microtask | Ordered action | Verification |
|---|---|---|
| P11.1 | Define transition boundary (when observation starts/ends and which event closes a decision) | Design note and one-task timeline agree |
| P11.2 | Connect encoded state → action → placement → next state → metrics/reward | Integration trace logs all five transition fields |
| P11.3 | Store experience exactly once per decision | Experience count equals decision count |
| P11.4 | Handle done/terminal states and empty workload | Terminal target masks future-Q term correctly |
| P11.5 | Test full queues, simultaneous arrivals and unavailable VMs with agent decisions | Invariants remain true under decision policy |

## Phase 12 — Training loop and checkpoints

**Objective:** Train from replay, log each run, save/load weights and keep evaluation separate. **Prerequisites:** P11 VERIFIED; critical reward/metric assumptions resolved. **Files:** `training.py`, `reproducibility.py`, `scripts/train.py`, tests. **Verification:** one tiny episode, several update steps, checkpoint reload and deterministic inference. **Gate:** smoke training finishes without NaNs and produces a reloadable checkpoint; this alone does not establish paper reproduction.

| Microtask | Ordered action | Verification |
|---|---|---|
| P12.1 | Add seed setup for Python, NumPy, PyTorch and workload generator | Same seed reproduces initial workload and initial agent decisions under deterministic CPU settings |
| P12.2 | Add one-episode training function with bounded work | Tiny run exits and records episode outcome |
| P12.3 | Add learning/update schedule and epsilon decay; log all hyperparameters | Log manifest includes all values and decisions |
| P12.4 | Add checkpoint save/load | Loaded model reproduces pre-save Q outputs within defined tolerance |
| P12.5 | Add small training smoke test and loss/gradient sanity checks | No NaN/inf; update counter changes as expected |
| P12.6 | Scale to the documented experiment repetitions only after smoke tests pass | Config supports paper's range and explicitly records 100/150 resolution |

## Phase 13 — Baselines and evaluation protocol

**Objective:** Establish fair comparators and evaluate all schedulers on the same workloads and topology. **Prerequisites:** P12 smoke test VERIFIED. **Files:** `baselines.py`, `evaluation.py`, tests. **Verification:** same task IDs/workload seed across policies, same metric functions, baseline action validity. **Gate:** comparison manifest proves comparable resource/workload settings.

| Microtask | Ordered action | Verification |
|---|---|---|
| P13.1 | Implement a simple deterministic baseline (e.g. FCFS or random) strictly as a simulator sanity baseline, not as a paper comparator | Valid allocations and repeatability tests pass |
| P13.2 | Document that CNN/LSTM/GGCN internals are not fully specified by the paper | Assumption entry prevents false exact-reproduction claim |
| P13.3 | Implement approximate CNN/LSTM/GGCN variants only if their roles/action mapping can be stated clearly | Same workload and fair parameter protocol documented; variants labelled approximate |
| P13.4 | Run all policies against paired workload manifests | Workload hashes and topology configs match across policies |
| P13.5 | Add fixed-VM Case 1 and linear-VM Case 2 experiment configs | Each experiment produces a complete run manifest |

## Phase 14 — Metrics, raw results and audit

**Objective:** Calculate per-task/per-run metrics using one code path and keep raw results available. **Prerequisites:** P13 VERIFIED. **Files:** `metrics.py`, `evaluation.py`, output CSV schema and tests. **Verification:** hand calculations, row counts, no missing terminal tasks, exact raw results preserved. **Gate:** results are traceable to config, seed, workload and policy.

| Microtask | Ordered action | Verification |
|---|---|---|
| P14.1 | Define CSV schema for run-level and task-level outcomes | Schema/unit tests and stable column names |
| P14.2 | Write raw metric records and configuration/seed manifest | Every row links to an experiment ID |
| P14.3 | Recompute percentage improvements from raw baseline and proposed values | Audit detects the paper's mismatched example and reports arithmetic transparently |
| P14.4 | Add confidence summaries across repeated seeds if repeat data exists | Sample count and summary formula appear in outputs |

## Phase 15 — Plots and result analysis

**Objective:** Produce clear plots/tables for makespan, energy, fault metric and scale cases without assuming claims are true. **Prerequisites:** P14 VERIFIED. **Files:** `plotting.py`, `outputs/figures/`, `outputs/metrics/`. **Verification:** plotting smoke test, labels/units, expected file paths. **Gate:** all figures are regenerated from saved CSVs and their source configs are recorded.

| Microtask | Ordered action | Verification |
|---|---|---|
| P15.1 | Plot makespan versus task count for fixed-VM/linear-VM cases | Figures exist with units, legend and case labels |
| P15.2 | Plot compute/transmission/total energy only where definitions/units are valid | Figure labels reflect metric semantics |
| P15.3 | Plot fault/reliability metric with correct direction and explicit name | No misleading “higher is better” unless formula justifies it |
| P15.4 | Create a comparison table against the raw paper values and note incomparable/missing settings | Every value has provenance; no fabricated result values |

## Phase 16 — Debugging, validation, reproducibility and final verification

**Objective:** Run the full test hierarchy, eliminate silent task/resource bugs, and document reproduction limits. **Prerequisites:** earlier phases complete. **Files:** tests, `README.md`, docs and final results manifest. **Verification:** test output archived; clean setup and end-to-end run reproduced. **Gate:** every applicable definition-of-done criterion is passed or explicitly marked blocked with evidence.

| Microtask | Ordered action | Verification |
|---|---|---|
| P16.1 | Run all unit tests | Test report recorded; no failing tests ignored |
| P16.2 | Run boundary/invalid-input and deterministic simulation tests | Empty queues, simultaneous arrivals, unavailable resources, invalid actions and termination covered |
| P16.3 | Run agent forward/learning/terminal/checkpoint tests | Shapes, gradients and checkpoint tests pass |
| P16.4 | Run integration and smoke training tests | Agent interacts end-to-end; no task loss/duplication |
| P16.5 | Run clean-environment setup from README | No hidden local dependency required |
| P16.6 | Re-run the same workload/config/seed and compare outputs | Determinism or expected nondeterminism is documented |
| P16.7 | Audit requirement statuses R001–R040 and unresolved assumptions | Each requirement is VERIFIED, OPEN, BLOCKED or not applicable with a reason |
| P16.8 | Publish final README, plots, metrics, trained checkpoint and limitation report | Definition of done is checked item-by-item; no unsupported reproduction claim |

## Global verification rules

- Use `pytest` after each meaningful code change, not only at the end.
- Start with one or two tasks and a small number of VMs; scale only after deterministic tests pass.
- Use assertions for unique task/VM IDs, legal queue/resource state, task completion count, finite state/reward/tensors and legal action masks.
- Never catch every exception or return arbitrary defaults to hide a bug.
- Every training/evaluation result must save config, seed, workload ID/hash, code version if available, episode/repetition count, metric units and baseline definition.
- A passing smoke run means the software executes; it does not prove the algorithm matches the paper or reproduces results.
