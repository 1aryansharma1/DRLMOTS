# DRLMOTS — Assumptions and Decision Log

Do not treat any item below as an explicit specification from the paper. Each choice remains **PENDING** until a microtask explicitly resolves it.

## A-025 — Synthetic CPU Q-network and selection

- **Date/owner:** 2026-10-14, Codex for Aravindak3 / Aravind.
- **Status:** DECIDED for prototype P9 only; A-009/A-010 remain unresolved for replication.
- **Architecture:** schema-sized input -> Linear(50)/ReLU -> Linear(50)/ReLU ->
  Linear(number of stable VM actions), CPU float32. Hidden widths are configurable;
  (50,50) is a provisional interpretation of the existing extraction, not an audited
  paper architecture. Q outputs are unbounded dimensionless reward estimates.
- **Initialization:** PyTorch Linear's default initialization within a saved/restored
  CPU RNG context, explicit seed (fixture 42). No mutation of caller CPU/Python RNG.
  Repeatability is within a recorded Torch/runtime version, not across all versions.
- **Selection:** epsilon-greedy over legal indices only, local Python Random seed;
  greedy ties choose the smallest stable action index. All-false masks return None.
  Epsilon is explicit; no decay schedule or paper training setting is inferred.
- **Compatibility:** bind network/checkpoint to full topology/state schema, fitted
  normalizer checksum and reward checksum. Inputs must be normalized finite float32
  CPU batches with explicit matching context. A tensor cannot prove its provenance;
  the observation inference path derives context and scaling from the actual artifacts.
- **Persistence:** strict checksummed JSON architecture/context/runtime/float32 weights,
  no executable pickle. This is a network snapshot, not a resumable training checkpoint;
  optimizer/replay/exploration RNG state are deferred to training.
- **Gate:** hand mask/ties, local seed/RNG isolation, finite shapes and parameters,
  invalid tensors/context, exact checkpoint round trip and real held-out observations.

## A-001 — State-vector encoding

* **Question:** Which exact numeric features and order represent task state, VM state, queue load, resources, and network conditions?
* **Paper says:** Eq. (23), p. 11, joins task state with VM state; §5.4 p. 13 lists tasks, their attributes, resource capacity, and network information.
* **Status:** PENDING.
* **Decision rule:** Start with the smallest vector justified by verified simulator variables; document each feature, unit, normalization and dimension. Avoid inventing unobservable fields.

## A-002 — Priority formula

* **Question:** How are tasks prioritized?
* **Paper says:** Based on task length and processing capacity; Fig. 4 p. 14 refers to equations (10)–(12), but those equations describe VM load and a threshold in the body.
* **Status:** BLOCKED for faithful priority replication.
* **Decision rule:** Implement an explicitly named provisional priority policy only if necessary to continue the simulator; label it as an implementation approximation and keep it replaceable.

## A-003 — Cloud/fog action ordering

* **Question:** Does the threshold first select a tier and DQN then choose a VM inside the tier, or does DQN choose from all VMs?
* **Paper says:** Eq. (24) action set contains VMs; prose and Fig. 4 separately describe a fog/cloud threshold.
* **Status:** PENDING.
* **Decision rule:** Compare source wording and diagram; if not resolvable, implement one documented policy and retain tests isolating this choice.

## A-004 — Threshold equations

* **Question:** Exact variables/conditions for Eqs. (11)–(13) and their units.
* **Paper says:** Dynamic thresholds use resource capacity, interval workload, and average resource requirement; rendered equations have damaged/incomplete notation.
* **Status:** PENDING.
* **Decision rule:** Do not put a guessed equation under the paper's equation number. Write an approximation in separate named code with a paper-uncertainty comment.

## A-005 — Task length and processing-time units

* **Question:** What unit does “700,000” represent, and how does it relate to 20–150 kB task data sizes?
* **Paper says:** Eq. (8) gives `length × 2 / 1000` and says 1000 bytes require 2 ms; Table 3 lists task length 700,000 with no unit; narrative says task data sizes 20–150 kB.
* **Status:** PENDING.
* **Decision rule:** Use byte-based conversion only in the deterministic formula tests and do not use 700,000 as a runtime input until its meaning is documented.

## A-006 — Reward direction/combination

* **Question:** How do makespan, fault tolerance and energy form one scalar reward?
* **Paper says:** Eq. (25) is the minimum of those three, but prose says minimize each and is inconsistent about reward sign; units and scales differ.
* **Status:** BLOCKED for paper-faithful reward replication. A-024 explicitly selects
  a separate synthetic prototype surrogate; this does not resolve Eq. (25).
* **Decision rule:** Implement metrics independently first. When the reward is needed, make an explicit configurable interpretation, write down all signs/scaling, and test simple cases. Never present the chosen interpretation as the paper's unambiguous formula.

## A-007 — Fault tolerance meaning

* **Question:** Is the reported quantity reliability, probability of system success, failure probability, or an inverse score?
* **Paper says:** Eq. (21) resembles `1 − product\_i P(F\_i)`, but Table 13–15 values for DRLMOTS are lower than baseline values while text calls this an improvement. The prose also says “minimizing fault tolerance.”
* **Status:** BLOCKED before interpreting the results.
* **Decision rule:** Code the probability calculation only after defining independent component-failure probabilities and retain both “raw score” and its name as defined in the implementation. Clearly mark any comparison with paper's table values as not semantically validated.

## A-008 — Energy units/model

* **Question:** What are power `P`, duration `T`, transmission data `D`, and unit energy `E\_data` for each resource?
* **Paper says:** Eqs. (17)–(19), p. 10, but no per-host power values or data-energy coefficient. Eq. (19) divides energy by time.
* **Status:** PENDING.
* **Decision rule:** Write unit tests for E=P×T and E\_transmission=D×E\_data separately; do not claim exact paper energy reproduction until values and Eq. (19)'s intended units are resolved.

## A-009 — DQN hyperparameters

* **Question:** Learning rate, optimizer, replay capacity, batch size, target sync interval, episode count, random seed.
* **Paper says:** These values are missing or conflicting. Gamma 0.95, epsilon minimum 0.01 and decay 0.01 are given. Initial epsilon appears as both 1.0 and 0.1. Text gives both 100 and 150 runs/episodes.
* **Status:** PENDING.
* **Decision rule:** Do not train until these have explicit defaults in `config.py` and are reported in every run manifest. For conflicting values, the decision note must cite both source locations.

## A-010 — Network shape

* **Question:** Does “50 neurons ... in the model hidden layers” mean 50 per hidden layer?
* **Paper says:** Algorithm 1 gives two ReLU dense layers; §5.4 p. 13 mentions 50 neurons in the hidden layers.
* **Status:** PENDING.
* **Decision rule:** Prefer 50 units per hidden layer as a clearly labelled inference if no further evidence appears; preserve configurability.

## A-011 — Dataset identity

* **Question:** Which Google trace/file, version, columns, filtering and workload mapping were used?
* **Paper says:** Calls it “Google Jobs” and mentions 1,000 tasks plus generated remaining tasks; the data-availability statement says no datasets were generated or analysed.
* **Status:** BLOCKED for exact workload reproduction.
* **Decision rule:** Do not download or fabricate a substitute under the label “Google Jobs.” First identify a source and column mapping; otherwise run a synthetic-workload experiment clearly labelled synthetic.

## A-012 — Baseline models

* **Question:** Exact CNN/LSTM/GGCN architecture, training data, optimizer and scheduling action mapping.
* **Paper says:** These are comparator names only; sufficient implementation detail is not provided.
* **Status:** PENDING.
* **Decision rule:** If comparison is attempted, label our baseline implementations as approximate variants and use matched workloads/resources/seeds; do not claim exact baseline reproduction.

## A-013 — Queue capacity and resource limits

* **Question:** Queue size, per-VM CPU/memory/storage, and what happens when no fog VM is eligible.
* **Paper says:** A full VM queue should cause search for another fog VM; queue and individual VM capacities are absent.
* **Status:** PENDING.
* **Decision rule:** Keep queue capacity and VM capacities configurable. Add a defined overflow/no-resource behavior and tests before training.

## A-014 — Event semantics and makespan

* **Question:** How do arrivals, queue waiting, service time, network delay and task completion contribute to Eq. (16)?
* **Paper says:** SimPy, millisecond execution units, random arrivals, and a max of per-machine accumulated processing time. Network latency formula is not specified.
* **Status:** PENDING.
* **Decision rule:** Make event timestamps explicit and calculate makespan from observed finish times. Report whether arrival and network delays are included.

## A-015 — Reported-result integrity

* **Issue:** The percentage-improvement tables do not consistently match raw metric tables; fault-tolerance directions also conflict.
* **Status:** OPEN audit item.
* **Decision rule:** Keep the source's raw numbers and independently recalculated percentages side-by-side in a later result-audit artifact. Do not tune the simulator to reproduce suspicious values.

## A-016 — Bandwidth unit interpretation

- **Question:** What precise unit is intended by the paper's host bandwidth notation `1000 MBPS` and VM bandwidth notation `200 MBPS`?
- **Paper says:** The reported simulation setup gives host bandwidth as `1000 MBPS` and VM bandwidth as `200 MBPS`.
- **Status:** PENDING.
- **Decision rule:** Preserve the reported numeric values and original notation in the source record. Do not silently reinterpret the capitalization or convert the values to another unit. If calculations require a specific unit, document and test that interpretation separately before using it in simulation.

## Decision log format

Whenever a decision is resolved, add: date, microtask ID, the exact selected rule, reason, paper references, tests affected, and whether it is a paper requirement / inference / implementation approximation. Do not delete the original unresolved note; update its status and preserve the history.

## A-024 - Causal projected-latency reward and arrival transitions

- **Date:** 2026-10-14. **Status:** DECIDED for a synthetic prototype only.
- **Scope:** Initial P8 reward/metric subset and P11 transition-export subset;
  A-006/A-007/A-008 remain unresolved for paper reproduction.
- **Reward:** At placement, projected wait is the selected VM's remaining running
  service plus queued service. Add this task's byte-fixture service time to get
  projected completion latency L. Reward admission by S/(S+L), with explicit positive
  scale S in ms; all-false-mask rejection receives -P, explicit positive penalty P.
  Evaluate the ratio in a numerically safe form. This is dimensionless and prefers
  shorter projected latency. Admission is in [0,1], allowing underflow to zero;
  rejection is negative. It is not makespan/energy/reliability or paper Eq. (25).
- **Timing:** Read only the pre-placement observation. No actual future finish or
  evaluation-fitted statistics enter the reward. Under the serial FCFS/zero-network
  model, future arrivals append behind this task and cannot delay its predicted work.
  No extra completion/terminal reward is added; simulator drain remains for metrics.
- **Accounting:** One reward and transition per arrival, including forced rejections.
  Episode reporting sums rewards without discounting and averages over all arrivals;
  empty runs have sum 0 and null average. Compare counts and timings alongside return.
  More throughput is not universally guaranteed to increase return when other waits
  change; this is an explicit surrogate with a configurable tradeoff.
- **Transitions:** Next state/mask is the next arrival observation. Only the last
  arrival is terminal; terminal next fields are null. Forced rejections have action
  and VM null, not a fake VM or terminal marker. A pure learner adapter absorbs forced
  chains with explicit gamma per arrival before bootstrapping; terminal continuation
  is zero, and forced prefix costs remain in metrics without fake actions. Gamma is
  required and is not inferred as a paper hyperparameter. No raw max over an empty
  legal-action set is valid.
- **Boundary:** Opt-in rewards imply decision capture. Existing unscored results,
  normalizer statistics and raw feature schema stay unchanged. Exporting baseline
  transitions does not implement interactive agent control, replay or DQN training.
- **Verification:** 510 full-suite tests pass on Python 3.12.14, including 50 new
  reward/transition tests; dependency check clean. Hand rewards, causal prefixes,
  unchanged simulation/scaling, empty/all-rejected cases, terminal boundaries,
  forced-chain discounts, strict configs/indices, replay and actual CLI verified.
  Six-task fixture: return 23/21; 100-task fixture: 96 admitted/4 rejected, return
  67.73152495135943; these are synthetic software results, not learned performance.

## A-023 - Training-only maximum scaling

- **Date:** 2026-10-14. **Status:** DECIDED for provisional synthetic model inputs.
- **Scope:** P6.3 prerelease v0.3.0a2. This is an implementation approximation, not
  the original paper's specified preprocessing or proof of a sufficient state.
- **Scaling:** For each numeric column, fit its maximum using declared training
  traces only. Exclude unknown resource amounts from fitting. Preserve binary tier,
  running and known flags unchanged. Divide nonnegative values by their positive
  training maximum; use scale 1 for all-zero or never-observed columns. Clip held-out
  values above scale to 1 before division, avoiding overflow even for subnormal scales.
  Constant positive columns map to 1; lower held-out values retain their ratio.
- **Unknowns:** Unknown amount placeholders remain zero, with known flags zero;
  known available zero retains flag one. Binary identity comes from schema positions,
  never a user-supplied unit string such as "binary".
- **Isolation:** Fitting accepts only batches labelled train; transform is pure and
  cannot update statistics. The CLI has separate fit/apply commands. Held-out batch
  application rejects declared training-source/exact-task overlap. Labels still
  depend on the caller's honest split declaration; no general data-governance claim.
- **Persistence:** Save feature names/order/units, schema hash, canonical full-topology
  hash, per-column maxima/counts, training provenance and an artifact checksum.
  Reject malformed/incompatible artifacts even when vector dimensions match.
- **Verification:** 460 full-suite tests passed on Python 3.12.14, including 41 new
  normalization tests; dependency check clean. Hand ratios, unknown flags, extremes,
  held-out isolation, same-task overlap rejection, full topology compatibility,
  save/load, strict artifact validation, CLI and baseline invariants verified.
  The fixtures fit 100 training states and transform 100 held-out 47-feature states
  into [0,1], retaining 92 completions/8 rejections and leaving fitted data unchanged.

## A-022 - Initial synthetic state/action contract

- **Date:** 2026-10-14. **Status:** DECIDED for a provisional synthetic observation.
- **Scope:** Initial P6.1/P6.2/P7.2 milestone only. A-001 original-paper encoding remains
  pending; no training normalization, reward or DQN is selected here.
- **Decision:** Use detached immutable arrival-time snapshots after same-time completions
  and queued starts, before placement; earlier simultaneous decisions are visible.
  Stable lexicographic VM IDs/resources define actions and numeric column order.
- **Encoding:** Raw finite floats, dimension `3+R+V*(6+4R)`, with explicit known flags
  for capacity/available amounts. Unknown is not known zero. Exact feature units/order
  and boundary are in STATE_ACTION_CONTRACT.md and captured schema provenance.
- **Legality:** Total-capacity componentwise fit plus a free waiting slot (or idle VM).
  Available resources describe the running reservation, not eligibility to wait.
  All-false masks retain explicit rejection with no selected VM/action.
- **Boundary:** Optional trace records one observation/state/mask/selection per arrival.
  Baseline policies still drive execution; no agent step/transition interface yet.
  Queued service aggregate is provisional; Markov sufficiency has not been established.
- **Verification:** 419 tests pass on Python 3.12.14, including 41 new state/action
  tests. Hand-checked busy/full/unknown and same-time cases, masks, fixed dimensions,
  detached snapshots, baseline-preserving capture, seeded replay and actual CLI pass.
  Dependency check is clean; the professor demo/control checks still pass.

## A-021 - Paired synthetic policies and professor demonstration

- **Date:** 2026-10-13. **Status:** DECIDED for a synthetic demonstration only.
- **Scope:** User-requested v0.2.1 professor checkpoint, early P13.1 sanity comparison;
  no completion of full paper baselines or experimental plots is implied.
- **Policies:** After A-020 resource/queue filtering, least queue minimizes running
  plus waiting count and ties by VM ID; first fit picks the minimum eligible VM ID.
  Both drain each VM FCFS, preserve completion-before-arrival ordering and reserve
  only running tasks. These policies are neither learned nor paper comparator replicas.
- **Fairness:** Generate each preset from the same effective config/settings for each
  policy; check identical workload/config hashes, exactly-once task accounting and
  restored resource balances. Policy identity is saved separately from input hashes.
- **Presets:** Preserve config topology/unit catalogue/demand profile. Use seed 42:
  worked = six simultaneous 1000-byte/20-kB tasks; seeded = 100 tasks, intervals
  uniform 0-3 ms, lengths integer-uniform 500-2000 bytes, data uniform 20-150 kB;
  pressure = 25 simultaneous tasks with the seeded length/data ranges.
- **Metrics:** Makespan is latest successful finish from time zero. Average completion
  and waiting times use successful tasks only; no successes means null averages.
  VM occupancy is sum of service durations divided by makespan (zero if makespan is
  zero), not physical CPU utilization. Show rejection counts alongside every timing
  comparison; different rejected work can lower makespan without better service.
- **Artifacts:** Offline HTML switches precomputed reports; regeneration reruns the
  simulator. Export raw JSON and SVG/PNG figures; no RL training or live cloud claim.
- **Evidence:** 378 tests passed; dependency check clean; paired seed-42 resource preset
  gives 96/4 least-queue and 91/9 first-fit completions/failures. DOM-stub control/export
  check passed; representative PNGs inspected. Actual browser/mobile layout requires
  manual preview because the browser tool disallowed local-file navigation.
- **Remaining:** Original-source audit, state/action/reward, DQN, real dataset,
  energy/reliability, physical host budgets and repeated-seed evaluation stay open.

## A-020 - Explicit per-VM resource feasibility and accounting

- **Date:** 2026-10-13.
- **Status:** DECIDED as a synthetic software model; paper resource capacities stay open.
- **Scope:** v0.2 resource-aware baseline, scoped R004/P7.3 feasibility and no-capacity handling.
- **Units:** One explicit `resource_units` catalogue gives each resource name its unit
  for the entire run. Task demands and VM capacities refer to these same names/units.
  No implicit conversion or per-VM unit override is performed. Missing unit declarations
  are invalid input; missing/null capacity is unknown and cannot satisfy a positive demand.
- **Amounts:** Resource names/units are nonempty strings and amounts are finite/positive.
  Per-VM capacities are supplied explicitly; none are derived from physical hosts.
- **Placement:** Filter to VMs whose total capacity satisfies every requested component.
  Among those with a waiting slot, retain least-running-plus-waiting placement and VM-ID
  ties. Empty demands preserve the old baseline behavior. No feasible VM means a FAILED
  outcome; feasible VMs with full queues means a distinct full-queue FAILED outcome.
- **Execution:** Each VM still runs one task at a time. Queued tasks reserve no resources.
  At start, reserve that task's demands; release them at completion before starting the
  next queued task or processing same-time arrivals. Record reservation/release events,
  available balances and verify all reservations are released at termination.
- **Service time:** Capacities control eligibility only; byte-valued Eq. (8) fixture
  service time is unchanged. CPU units are abstract admission units, not MIPS or a speed
  model. No physical host budget, multi-task execution or cross-VM sharing is claimed.
- **Configuration:** Preserve schema 1 and its existing examples/checksums. Schema 2
  adds an explicit unit catalogue, each VM's resource_capacities (null may mean unknown),
  and workload required_resources. A synthetic generator uses a constant explicit demand
  profile for every generated task; hand-built tasks may have different demand profiles.
- **Provenance:** Include demands, capacities, units, admission/queue policy, resource
  events and final balances. Demand-free generator-v1 manifests remain unchanged; adding
  a demand profile uses generator version 2 and records it in generator settings.
- **Example:** `configs/resources.yaml` has two VMs fitting 2 abstract CPU units/4 GB
  and a third too small. Six simultaneous 2 ms tasks should give four completions, two
  explicit full-feasible-queue failures, 4 ms makespan and fully restored balances.
- **Remaining paper gaps:** Physical VM/host mapping, true resource values, priority/
  threshold/tier semantics, state/reward, energy/reliability and DQN remain open.
- **Verification:** 367 tests passed, including existing fixture compatibility,
  exact-fit/oversize/unknown cases, componentwise fit, queued-vs-running reservations,
  release ordering, 100-task heterogeneous accounting and resource-profile replay.
  Resource fixture: four completed, two explicit queue-full failures, 4 ms makespan;
  eight reserve/release events and all per-VM balances restored.

## A-019 - Explicit runnable topology configuration

- **Date:** 2026-10-13.
- **Status:** DECIDED for the synthetic software baseline only.
- **Microtask:** Configurable portion of P5.1 for v0.2; full paper topology remains open.
- **Schema boundary:** A separate `kind: synthetic_baseline`, `schema_version: 1`
  YAML describes a runnable experiment. `configs/default.yaml` remains the unresolved
  paper-planning configuration and is not implicitly converted into a run.
- **Topology:** Supply all six lists explicitly: hosts, VMs, fog nodes, cloud
  datacenters, access points and end nodes. Counts derive from list lengths, never
  from guessed defaults. Explicit empty lists are allowed if structurally valid.
- **Resources:** Every host/VM field and VM waiting-queue capacity is supplied.
  Unknown/null queue capacity cannot be used. Existing identifier, single-membership,
  tier and reference invariants apply. No VM-to-host allocation is inferred.
- **Workload:** Supply every A-018 generator setting, the uniform_interarrival generator
  name, byte length unit and millisecond arrival unit. CLI workload overrides cannot
  be combined with a run file; this prevents ambiguous configuration precedence.
- **Provenance:** Save normalized effective run configuration plus its SHA-256,
  topology counts, actual topology, workload/settings checksum and outcomes. YAML
  formatting and the file's local pathname do not affect the effective config checksum.
- **Validation:** Reject missing fields, unknown fields, invalid YAML/types, dangling
  references, orphan or duplicate resources and wrong-tier memberships before output.
- **Example:** `configs/synthetic.yaml` contains three VMs (two fog, one cloud), one
  waiting slot each, and six simultaneous 1000-byte tasks. All six should finish,
  with a 4 ms makespan. These are verification fixtures, not paper experiment claims.
- **Remaining limits:** CPU/memory feasibility, physical placement, network delay,
  priority/threshold rules, energy, reliability and learned scheduling remain pending.

## A-018 - Seeded synthetic arrivals and workload provenance

- **Date:** 2026-10-13.
- **Status:** DECIDED as a synthetic workload model, not an inferred paper distribution.
- **Microtasks:** P4.3 and the random-workload extension of P4.6.
- **Source boundary:** The existing extraction R009 says random arrival intervals;
  the arrival distribution remains unspecified. This decision does not resolve the
  original dataset identity, task-length units or conflicting experiment task counts.
- **Selected rule:** A caller supplies a nonnegative integer task count and seed,
  a finite nonnegative interarrival range in milliseconds, an inclusive positive
  integer length range in bytes, and a finite positive data-size range in kB.
- **Arrivals:** The first task arrives at 0 ms. Later intervals are independent uniform
  draws within the supplied range, accumulated in input order. Zero intervals are
  allowed and represent simultaneous arrivals. Reject non-finite accumulated times.
- **Task attributes:** Lengths use inclusive integer-uniform draws, data sizes use
  continuous uniform draws; these fields are independent. IDs are unique within each
  workload. Required resources remain empty for the existing baseline.
- **Randomness:** Use a local `random.Random(seed)` instance, leaving global RNG state
  unchanged. For each task draw its interval (except the first), then length, then
  data size. Record generator version and Python runtime version; repeatability is
  guaranteed for identical settings within the recorded runtime/generator version.
- **Provenance:** The manifest includes source `synthetic_seeded`, seed, distribution,
  all bounds/counts/units, first-arrival rule, RNG identity/version, exact tasks and a
  SHA-256 over inputs plus settings. An empty workload still records its settings.
- **CLI example:** Random generation is requested explicitly and requires an explicit
  seed and count. CLI range defaults are labelled software-demo settings, not values
  inferred from the paper. The original fixed three-task smoke example stays available.
- **Verification gate:** Bounds, invalid inputs, zero tasks, unchanged global RNG,
  repeated outputs/checksums, provenance sensitivity and simulator task accounting.

## A-017 - Versioned implementation and v0.1 fixture semantics

- **Date:** 2026-10-13.
- **Status:** DECIDED for the software baseline only; original source gaps remain open.
- **Microtasks:** P3.3-P3.5, P4.1-P4.2, fixed-list P4.6, baseline P5.2-P5.6/P5.8.
- **Decision type:** Implementation scope and explicit test-fixture assumptions.
- **User direction:** Build practically achievable versions before attempting the complete paper.
- **Source boundary:** Existing DRLMOTS requirements reference DOI 10.1007/s10586-024-04712-z.
  The supplied 16-page PDF is instead DRL-MLS, DOI 10.1016/j.future.2026.108459, about
  GPU-aware Kubernetes scheduling. Continue the repository's existing DRLMOTS baseline
  for this version; do not mix either paper's algorithm into the other. Independent
  review of the original DRLMOTS PDF remains pending.
- **Selected topology:** Explicit AP -> end node -> fog-node links, cloud datacenters,
  and single node membership for each host/VM. Physical VM-to-host placement and link
  latencies are deliberately unspecified. Numeric host/VM fields retain the extraction's
  reported storage/memory/bandwidth; bandwidth is not converted or used for transfer.
- **Selected workload:** Fixed synthetic inputs; explicit byte-valued task lengths and
  millisecond arrival/service timestamps. Apply the recorded Eq. (8) interpretation
  `duration_ms = length_bytes * 2 / 1000` only to such inputs. This does not resolve
  A-005 or the unitless table value 700,000. Data size is not converted into task length.
- **Selected scheduling rule:** Least queued eligible VM, considering running plus
  waiting count; VM ID breaks ties. Each VM executes one task at a time and drains its
  waiting queue FCFS. All configured tiers are eligible; no paper threshold is inferred.
- **Capacity and overflow:** Require positive explicit waiting-queue capacities. The
  running task is excluded from that count. Try another eligible VM when a queue is
  full; reject with a recorded FAILED outcome when every queue is full. This is the
  baseline's overflow policy, not a resolution of the paper's unspecified behavior.
- **Task lifecycle:** CREATED -> QUEUED -> RUNNING -> COMPLETED; failure is allowed
  from CREATED/QUEUED/RUNNING. Terminal states cannot restart via `transition_to`.
- **Time ordering:** Complete all finishing tasks before same-time arrivals; simultaneous
  arrivals preserve input order. Record every successful start and finish.
- **Metrics:** Makespan is the last successful finish from time zero (arrival gaps count).
  Average completion time is finish minus arrival for successful tasks only. Report
  failed counts separately; empty workload average is null. This baseline definition
  does not establish equivalence with the original paper's Eq. (16).
- **Out of scope:** CPU/memory feasibility, network delay (fixture uses zero), power/
  energy, fault tolerance, priorities/thresholds, state/reward, DQN or paper results.
  Nonempty required_resources are rejected; unknown queue limits are not treated as
  unlimited. These restrictions keep missing model inputs visible.
- **Version gates:** Verify Phase 3 before simulation; require fixed-list workload tests
  and hand-calculated timelines for v0.1. Random generation and real dataset ingestion
  remain pending for later versions, rather than being falsely marked complete.
- **Tests affected:** topology, lifecycle, configuration, workload and integration tests.
  Final v0.1 suite: 194 passed. Example: three completions, zero failures, makespan
  4 ms, average completion time 8/3 ms; repeated runs agree.

This entry does not resolve the scientific questions in A-001 through A-016. Later
resource, reward, energy, network and DQN decisions require their own evidence and tests.



\## P0.5 — Initial implementation boundary



\- \*\*Date:\*\* 2026-10-10

\- \*\*Status:\*\* DECIDED — initial scope only; paper-specific ambiguities remain open.

\- \*\*Microtask:\*\* P0.5

\- \*\*Decision type:\*\* Implementation-process decision, not a claim made by the paper.



\### Selected implementation order



1\. Implement the minimum task and resource data structures using source-supported fields. Do not add unsupported paper attributes as if they were established facts.

2\. Add validation and unit tests for identifiers, capacities, valid values, and state transitions.

3\. Build a small, deterministic simulator with explicitly defined test fixtures. Test fixtures are for software verification, not claims about the paper's experimental settings.

4\. Add workload generation only after the data structures pass their tests.

5\. Add SimPy execution and verify event ordering, task completion, resource use, and deterministic behavior before scaling up.

6\. Define and test state representation, valid actions, reward, and evaluation metrics before implementing or training the DQN.

7\. Keep unresolved paper parameters configurable and documented. Do not silently choose values to make results resemble the paper.



\### Explicitly out of scope for the initial implementation



\- Long DQN training runs or hyperparameter tuning.

\- Claims of exact reproduction of the paper's results.

\- Treating an assumed Google Jobs dataset schema as verified.

\- Finalizing reward semantics, energy calculations, fault-tolerance interpretation, or ambiguous training settings without documenting the chosen interpretation.



\### Reason and verification plan



This sequence isolates basic software errors before introducing stochastic workloads and reinforcement learning. It makes each stage independently testable and prevents unverified simulator behavior from contaminating training results.



\- \*\*Tests affected:\*\* unit tests for entities and configuration; topology tests; later deterministic simulation and metric tests.

\- \*\*Paper references:\*\* `paper\_requirements.md` requirements R001–R040 and the unresolved items already listed in this file.

\- \*\*Verification criterion:\*\* each stage must pass its own tests before the next stage begins. A passing test establishes only the behavior tested, not exact paper reproduction.



\### Decision history



The paper's unresolved items remain OPEN/PENDING unless separately resolved with evidence. This decision defines the implementation order only; it does not close those scientific or methodological questions.
