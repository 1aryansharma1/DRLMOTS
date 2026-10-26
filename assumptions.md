# DRLMOTS — Assumptions and Decision Log

Do not treat any item below as an explicit specification from the paper. Each choice remains **PENDING** until a microtask explicitly resolves it.

## A-027 — End-to-end CPU DQN, retained standard mode and fog-preferred variant

- **Date/owner:** 2026-10-15, Codex for Aravindak3 / Aravind; requested live training,
  placement, resume and evaluation across both existing and new environments.
- **Status:** DECIDED for the synthetic prototype; paper A-009/A-010 remain open.
- **Contract 2:** retain contract-1 encoder/checkpoints. New causal frames contain
  ordered queued upload readiness/service/resource demands, reachable VM flags,
  input upload and queue-tail duration plus the known remaining arrival budget.
  Future task values remain hidden. No proof of general real-world Markov completeness
  or audited paper state is claimed. New schema/context includes full topology,
  links/origin/model, queue capacities, action variant and reward version.
- **Standard DQN:** choose among all resource/queue/reachability-legal VMs.
- **Fog-preferred DQN:** apply the user's deterministic completion preference as an
  explicit action constraint; among qualifying VMs DQN learns placement. If best fog
  <= best cloud, only fog VMs no slower than best cloud are allowed; otherwise only
  cloud VMs faster than best fog. Do not call this constraint a learned discovery.
- **Reward:** immediate S/(S+causal end-to-end duration), forced rejection -P, with
  S=10 ms and P=1 by default. No completion/drain bonus. All-arrival return includes
  forced prefixes; learner transitions fold following forced rewards with gamma per
  arrival and gamma**steps bootstrap. Only final actionable transition is terminal.
- **Learning:** configurable (50,50) ReLU CPU float32 network, Adam lr .001,
  betas (.9,.999), eps 1e-8, no weight decay; Smooth L1 loss, norm clip 10,
  gamma .95, replay 2000, batch 32, target hard-copy every 50 updates. These are
  explicit prototype settings, not verified paper hyperparameters.
- **Exploration:** epsilon linearly decreases 1 -> .05 across 2000 actionable training
  choices; evaluation epsilon 0. Local seeded sampling/selection; forced choices
  consume no action RNG. CPU execution uses one thread inside a restored context.
- **Splits:** default 20 episode seeds starting at 2000, first 3 used only as the
  training calibration fit. Held-out seeds 3000/3001/3002 never fit/update/select a
  checkpoint. Reject overlap in seeds and exact task hashes, even across seed labels.
  No tuning/model selection on test results; report all requested variants honestly.
- **Persistence:** checksummed JSON online/target weights, Adam moments, replay,
  local RNG states, scaler/provenance, counters/config and losses/history. Resume
  requires matching Python/Torch runtime for deterministic continuation. JSON is not
  executable pickle; checksum is corruption detection, not an authenticity signature.
- **Evaluation:** pair baselines and trained agents on identical inputs within each
  environment/seed; zero-network and network timings are separate experiments. No
  guaranteed DQN superiority; report completed/rejected tasks, successful latency,
  all-arrival return, seed counts and sample standard deviations.

## A-026 — User-requested fog-first completion-time scheduling

- **Date/owner:** 2026-10-15, Codex for Aravindak3 / Aravind.
- **Status:** DECIDED for the requested synthetic policy, not the paper's threshold.
- **Origin/reachability:** one explicitly configured source end node per run; its
  linked fog nodes define local fog candidates. All configured cloud VMs are remote
  candidates. No nearest-by-distance or physical routing is inferred.
- **Network:** explicit per-candidate bandwidth in decimal Mbps and one-way fixed
  propagation in ms. Input data_size_kb is decimal kB. Upload time is
  propagation_ms + 8 * data_size_kb / bandwidth_mbps. Independent uploads overlap
  without bandwidth contention; response downloads, energy and outages are excluded.
  Reported paper bandwidth fields are not used as network parameters.
- **FCFS timing:** accepted uploads occupy a waiting slot until execution starts;
  only running tasks reserve CPU/memory. A VM runs one task at a time. For each
  preceding queued task, finish = max(previous finish, its upload-ready time) +
  service. An arriving task starts at max(queue-tail finish, its upload-ready time).
  Head-of-line upload waiting is preserved; uploads overlap prior execution.
- **Preference:** compare the best eligible local fog VM with the best cloud VM.
  Choose fog if its predicted finish is <= cloud's (exact ties prefer fog); otherwise
  choose cloud. Within a tier, finish time then VM ID breaks ties. Busy fog is eligible
  while its waiting queue has room. No arbitrary fog slack or deadline is invented.
- **Scope:** schema-3 placement settings are explicit/replayable. Earlier schema-1/2
  zero-network baselines remain unchanged. Legacy state/normalizer/reward traces do
  not model uploading/head-of-line idle states, so their capture/training path rejects
  network-aware runs until that contract is extended. Network decision estimates and
  actual transfer/execution timestamps are exported separately.

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


## A-028 - Explicit multi-objective reward and paired efficiency experiments

- Date: 2026-10-25. Status: IMPLEMENTED; verification results in MULTI_OBJECTIVE_RESULTS.md.
- User direction: extend the latency-focused scheduler and produce efficiency reports.
- Original source: supplied s10586-024-04712-z.pdf, pp. 9-11, with pp. 10/11 rendered
  and inspected. Confirms makespan, energy and fault tolerance, and the previously
  identified unit/direction ambiguities. This does not close the entire source audit.
- Decision: retain legacy rewards and add optional live reward contract 3: normalized
  weighted latency/active+input-energy/analytical destination risk; all coefficients
  and weights explicit and checkpoint-bound. Default .4/.4/.2; synthetic values.
- Energy in joules, using seconds for W*time; do not repeat Eq. (19)'s energy/time
  label. Exclude idle/host/network-contention/output/backup energy explicitly.
- Risk: constant destination hazard over arrival-to-finish; expected successes and
  rejections reported separately. No fault sampling, redundancy or recovery claims.
- Latency is a makespan-related surrogate. Actual makespan is reported separately.
- Experiment: three independent model seeds (42/43/44), fixed 20-episode budget,
  training seeds 2000-2019, calibration 2000-2002, fresh tests 4100-4104; no held-out
  tuning/selection. Light/normal/burst workloads, seven policies, exact paired hashes.
- Reports: task-level energy/risk ledger, completion/makespan/mean/p95/energy/expected
  success/common utility, separate model/workload SD, signed paired changes and
  descriptive Pareto screen. No guaranteed superiority or statistical significance.
- Initial exploratory run using old 3000-series evaluation settings was interrupted
  before evaluation. It is not the delivered experiment; final run uses the declared
  fresh-seed configuration. Existing generated outputs were preserved, not deleted.
- Source/data/failure/physical-host/paper-baseline gaps remain open. See MULTI_OBJECTIVE.md.


## A-029 - Observed node failures, bounded passive restart and accounting

Date: 2026-10-25. User requested failure/recovery and time/energy reports before the
next training study. Failure domains follow fog/cloud topology membership. A fixed,
exogenous node outage trace is shared across policies; optional pre-sampling uses
local per-node streams. Immediate detection, durable source input, restart from zero
on another current up/reachable/feasible node, bounded retries and backup waits.
Completions precede same-time faults. Recovery dispatch precedes new arrivals.
All tasks settle by a common fixed observation horizon; unfinished work fails.

Joules use actual partial service and bytes serialized, idle uptime excluding service
and downtime, and explicit coordinator active delay. Retry costs are never erased.
Task latency partitions into attempt time plus recovery blocking. Metrics distinguish
rejection, permanent execution failure and recovered completion. No active replicas,
checkpoint migration, real power calibration or physical-host allocation is implied.

Legacy simulator/DQN/reward contracts remain unchanged. The new observed objective
vector is makespan/total energy/unsuccessful fraction. Old analytical risk coefficients
are not sampled as failures or used as future fault information. Fault-aware DQN state,
actions and delayed rewards are explicitly the next contract before new training.
See FAILURE_RECOVERY.md and RECOVERY_RESULTS.md for tests, commands and actual scope.


## A-030 - Explicit compute, physical host power, stochastic reliability and shared traffic

Date: 2026-10-26. User requested all ten objective-model improvements and a
presentation reference. Implemented as a separate host_network_objectives schema-1
contract; A-028 analytical/DQN and A-029 restart baselines are retained unchanged.

Work is explicit MI with VM aggregate MIPS, independent input/output decimal kB,
absolute optional task deadlines and a fixed observation cutoff. One compute slot per
VM; fixed reservations fit physical host MIPS and GB memory budgets without overcommit.
Each VM maps to exactly one host within its topology node. Host power is integrated
once using a piecewise linear utilization/power curve, with explicit idle/down/reboot
power. Task energy attribution includes dynamic host share and actual transferred
bytes, keeping shared host/link overhead separate from task allocations.

Shared half-duplex bottlenecks split capacity equally among active uploads/downloads.
Propagation releases bandwidth. Fully serialized output already in propagation survives
a subsequent host failure; partial output is lost. There is one configured bottleneck
per VM route, not packet-level or multi-hop routing. Input source/controller are durable
and reliable. Working-memory reservations cover computation, with input/queue/durable
storage outside that budget.

Nodes have independent per-node seeded renewal streams, exponential or Weibull uptime,
deterministic repair/reboot, perfect age reset and simultaneous member-VM outages.
Detection, restart overhead and backup waiting are separate. Health can remain stale;
physical failures interrupt work immediately, but retry dispatch waits for detection.
Restart on another node repeats full work/input, bounded by retries/wait/deadline.
No checkpointing, cross-node common-cause faults, DVFS, host sleep or facility cooling.

Deadline failure is distinct from right-censoring at cutoff. Full-batch makespan is
undefined unless all tasks complete. Availability, matured-deadline success, rejection,
execution failure, successful recovery delay and all-arrival completion are separate.
Conditional Weibull survival uses current node age; analytical estimates do not reveal
the sampled future trace or replace observed results.

Fixed-scale weighted ratios and predeclared completion/deadline floors prevent cheap
reject-all winners. Scores are null for infeasible runs; empirical Pareto recommendations
require feasibility in every evaluated run. Candidate estimates are causal heuristics,
not exact future completion forecasts or a global optimizer.

Default study: 3 workload seeds x 5 failure seeds x 4 policies x 2 fault/recovery modes,
plus 12 nonduplicated no-fault controls = 132 simulations. Paired hashes and effects;
intervals use five independent failure-seed means conditional on the fixed workload
panel, rather than treating crossed/duplicated runs as independent evidence.

All bundled coefficients are explicitly synthetic. Forms are motivated by NIST
reliability, CloudSim Plus MI/MIPS/host power and SPEC measurement methodology; their
use does not establish calibration, exact paper fidelity or real VM deployment.
No DQN checkpoint is compatible with this environment. Fault-aware state/action/reward
and delayed terminal credit require a new contract and training/validation study.
See OBJECTIVE_MODEL_PRESENTATION.md and OBJECTIVE_MODEL_RESULTS.md.


## A-031 - Enhanced RL observation/action/reward contract, milestone 1 only

Date: 2026-10-26. User limited the next aim to five milestones and requested starting
with the first. Implemented standalone contract 4 in enhanced_rl_contract.py; no live
DQN/controller/replay/training integration is installed in the enhanced simulator.

Public immutable context binds observed telemetry schema, sorted VM actions, physical
model parameters, fixed public feature scales, task budget/horizon, objective weights,
service floors and penalty coefficients. Seeds and sampled future events are excluded.
253 features for the current six-VM topology: task/episode, health/telemetry ages, host
utilization estimates, shared traffic, resource information and ordered FCFS slots.
Unknown health is masked; busy VMs may queue if total resources fit. Recovery excludes
the latest failed node. None is only an environment-owned no-legal-action outcome.

Observations use delivered telemetry, not physical fault truth; no adapter exists yet.
Timestamp/allowlist tests establish API invariants, not engine-level non-leakage.
Actual causal telemetry and paired hidden-world tests remain milestones 2/3.

Reward uses cumulative actual system joules, successful-makespan increments and
all-arrival unsuccessful outcomes, gamma=1, plus explicit terminal service shortfalls.
Default shortfall coefficients are 10 each, configurable starting hyperparameters.
No completions uses H as a training time surrogate without inventing a report makespan.
Finite-horizon training termination has zero bootstrap at H; report unfinished remains
censored. Prefix/forced-event rewards must be preserved without invented VM actions.
No hard-constraint guarantee follows from a finite penalty; evaluation floors remain.

Hand calculations, interval partition equality, delayed costs, strict context guards,
masking and terminal reward parity with actual reports from all four rules are tested.
All earlier simulator, DQN, reward, report and checkpoint interfaces remain unchanged.
Read ENHANCED_DQN_CONTRACT.md for formulas and the five-milestone handoff.


## A-032 - Optional basic masked DQN for enhanced initial/recovery placement

Date: 2026-10-26. User requested milestone 2, keeping rules and documentation.
Enhanced_dqn.py and enhanced_telemetry.py connect contract 4 to an opt-in masked_dqn
path in objective_sim.py. Four original rules take the original path without sensor
poll events; eight full baseline reports matched pre-edit canonical hashes exactly.

Telemetry protocol: fixed public polling period (default 1 ms), zero-latency successful
reports from up nodes, absent responses from down nodes, held report values/ages,
explicit detection and service-ready notices. Sensor traffic has no extra power/latency
cost in this version. Local sends reserve observed waiting slots immediately. Link
observations aggregate only participating nodes' latest reports. Hidden physical cleanup,
private uptime reset and future fault edges never enter observations. Missing responses
and actual changed throughput in delivered healthy-node reports are legitimate signals.
This remains a partially observed environment, not an exact current-state oracle.

System rewards accrue true host/link/OPEN recovery costs between real decisions;
forced intervals fold into prior actions, with a separate prefix before the first.
Finite H terminates with zero bootstrap and all task outcomes accounted. Both initial
and recovery requests use the same six-action network/masks. Bounded replay, local
RNGs, Adam/Huber/clipping, warmup and target sync are wired; frozen inference fingerprints
all learning state. New JSON checkpoints bind contract, host map, telemetry period and
recovery mode and save complete between-episode learner state. No mid-episode resume.
Legacy checkpoints are rejected. CPU/runtime resume equality is tested, not promised
across hardware/library versions.

No tuned training study or learned-performance improvement is claimed. The delivered
runnable example is frozen UNTRAINED inference. Tests exercise synthetic optimizer
updates and tiny below-warmup live replay plumbing only. Broader connected verification,
actual training sanity run and frozen comparative evaluation remain milestones 3-5.
See ENHANCED_DQN_INTEGRATION.md for scope, commands and explicit telemetry limitations.


## A-033 - Connected DQN scenario verification and boundary fixes

Date: 2026-10-26. User requested milestone 3: test initial/recovery placement, rewards
and hidden failures, fix exposed issues, and document the evidence. The new 70 cases
include closed-form host/network/recovery energy, phase and repeated faults, resource/
reachability masks, frozen scenario sweeps and hidden-world prefixes. See
ENHANCED_DQN_VERIFICATION.md for exact fixtures, results and commands.

Replay requires a legal integer VM action. Forced events are folded by the controller,
not restored as action=None experiences. Service shortfall penalties occur only at
terminal H; integrated energy cannot change over a zero-time feedback interval.
These guards apply to replay insertion and checkpoint loading. Reward formulas,
weights, normalization, contract version 4 and checkpoint schema 1 are unchanged.

An error or KeyboardInterrupt aborts the controller begun by that simulator call,
then re-raises. Its pending action without completed feedback is dropped. Closed
nonterminal samples and existing updates remain valid; completed-episode counters do
not advance, and no terminal outcomes/rewards are fabricated. A refused nested call
cannot abort an existing active owner. Checkpoints can save learner state after an
explicit abort boundary, but cannot resume an abandoned physical simulation.

At recovery dispatch now >= ready_ms, so remaining wait is mathematically bounded by
the public configured wait budget. Clamp subtraction roundoff to [0, wait_ms] before
encoding; this prevents valid fractional-clock simulations from failing a strict
contract check. It changes no physical time, event ordering or heuristic choice.
The reward cursor uses math.fsum across individual recovery contributions so moving
an interval from open to closed cannot create a same-time energy decrement through
regrouping. Strict cumulative reward checks and physical report ledgers remain intact.

Eight full reference-report canonical hashes captured from fac9513 are retained as
fixtures; all four heuristic branches must match them exactly. Verification covers
declared synthetic scenarios, not all possible inputs or measured deployment behavior.
No enhanced training or comparative learned-efficiency claim follows from this work.


## A-034 - Fixed small enhanced-DQN training sanity experiment

Date: 2026-10-27. User requested milestone 4: run actual training, verify updates and
behavior changes, and document results. The new enhanced_sanity.py runner uses an
explicit configs/enhanced_sanity.yaml plan, existing model/config/learner defaults,
and a fixed final-episode checkpoint. It never selects a checkpoint or extends its
budget from diagnostic scores. Earlier simulator/rule/DQN interfaces are preserved.

Actual run: 16 episodes x 60 tasks, model seed 42, training work/fault seeds
10000-10015 / 20000-20015; diagnostic generated pairs 81000/91000, 81001/91001;
a separate constructed recovery probe uses work seed 81002 and explicit outages.
No diagnostic observation enters training replay, fitting or optimization. Validation
is greedy and frozen before training and after episodes 4/8/12/16. Full workload and
outage hashes match across these stages. The constructed probe is not statistical
reliability evidence. Source hashes/runtime/configuration are saved in experiment.json.

There were 1,030 collected transitions (913 initial, 117 recovery), 999 optimizer
updates and 19 target synchronizations. Parameter L2 change was 4.6359729004. Actions
changed on 87/156 identical initial and 5/20 identical recovery observation probes.
The final JSON checkpoint restored exactly and reproduced a full frozen diagnostic
trace, and loaded through the normal inference CLI without updates. All losses were
finite. Training stopped at epsilon .51075, before the .05 floor, as predeclared.
Collected request counts are not per-minibatch sampling coverage counts.

Results are mixed: generated-1 retains 42/60 completions with a slightly better return;
generated-2 retains 58/60 but its return worsens; the probe improves 44/60 to 47/60.
Generated-1 and the probe still fail service floors. Episode 12 had some better
diagnostics but was not chosen after inspection. Lower ending average loss, changed
weights and actions do not establish convergence or general efficiency. This actual
16-episode experiment was run once; exact repetition is tested on a smaller fixture
on this CPU/runtime, not promised across library/hardware versions or as statistical
evidence across model seeds. Milestone 5's four-rule comparison remains pending.

Verification: 970 full-suite tests passed, including 21 new runner checks. Dependencies
and whitespace are clean; 32 saved training/frozen traces, six source hashes, CSV
record counts and trained CLI metrics were audited. Two PNG/two SVG figures were
generated from actual data and the PNGs visually inspected. Generated artifacts stay
under ignored outputs/objective-model/dqn-sanity/milestone4-20261027; tracked sources,
plan and ENHANCED_DQN_TRAINING_SANITY.md let collaborators regenerate them.
