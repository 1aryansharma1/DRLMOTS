# DRLMOTS — Assumptions and Decision Log

Do not treat any item below as an explicit specification from the paper. Each choice remains **PENDING** until a microtask explicitly resolves it.

## A-001 — State-vector encoding
- **Question:** Which exact numeric features and order represent task state, VM state, queue load, resources, and network conditions?
- **Paper says:** Eq. (23), p. 11, joins task state with VM state; §5.4 p. 13 lists tasks, their attributes, resource capacity, and network information.
- **Status:** PENDING.
- **Decision rule:** Start with the smallest vector justified by verified simulator variables; document each feature, unit, normalization and dimension. Avoid inventing unobservable fields.

## A-002 — Priority formula
- **Question:** How are tasks prioritized?
- **Paper says:** Based on task length and processing capacity; Fig. 4 p. 14 refers to equations (10)–(12), but those equations describe VM load and a threshold in the body.
- **Status:** BLOCKED for faithful priority replication.
- **Decision rule:** Implement an explicitly named provisional priority policy only if necessary to continue the simulator; label it as an implementation approximation and keep it replaceable.

## A-003 — Cloud/fog action ordering
- **Question:** Does the threshold first select a tier and DQN then choose a VM inside the tier, or does DQN choose from all VMs?
- **Paper says:** Eq. (24) action set contains VMs; prose and Fig. 4 separately describe a fog/cloud threshold.
- **Status:** PENDING.
- **Decision rule:** Compare source wording and diagram; if not resolvable, implement one documented policy and retain tests isolating this choice.

## A-004 — Threshold equations
- **Question:** Exact variables/conditions for Eqs. (11)–(13) and their units.
- **Paper says:** Dynamic thresholds use resource capacity, interval workload, and average resource requirement; rendered equations have damaged/incomplete notation.
- **Status:** PENDING.
- **Decision rule:** Do not put a guessed equation under the paper's equation number. Write an approximation in separate named code with a paper-uncertainty comment.

## A-005 — Task length and processing-time units
- **Question:** What unit does “700,000” represent, and how does it relate to 20–150 kB task data sizes?
- **Paper says:** Eq. (8) gives `length × 2 / 1000` and says 1000 bytes require 2 ms; Table 3 lists task length 700,000 with no unit; narrative says task data sizes 20–150 kB.
- **Status:** PENDING.
- **Decision rule:** Use byte-based conversion only in the deterministic formula tests and do not use 700,000 as a runtime input until its meaning is documented.

## A-006 — Reward direction/combination
- **Question:** How do makespan, fault tolerance and energy form one scalar reward?
- **Paper says:** Eq. (25) is the minimum of those three, but prose says minimize each and is inconsistent about reward sign; units and scales differ.
- **Status:** BLOCKED before agent training.
- **Decision rule:** Implement metrics independently first. When the reward is needed, make an explicit configurable interpretation, write down all signs/scaling, and test simple cases. Never present the chosen interpretation as the paper's unambiguous formula.

## A-007 — Fault tolerance meaning
- **Question:** Is the reported quantity reliability, probability of system success, failure probability, or an inverse score?
- **Paper says:** Eq. (21) resembles `1 − product_i P(F_i)`, but Table 13–15 values for DRLMOTS are lower than baseline values while text calls this an improvement. The prose also says “minimizing fault tolerance.”
- **Status:** BLOCKED before interpreting the results.
- **Decision rule:** Code the probability calculation only after defining independent component-failure probabilities and retain both “raw score” and its name as defined in the implementation. Clearly mark any comparison with paper's table values as not semantically validated.

## A-008 — Energy units/model
- **Question:** What are power `P`, duration `T`, transmission data `D`, and unit energy `E_data` for each resource?
- **Paper says:** Eqs. (17)–(19), p. 10, but no per-host power values or data-energy coefficient. Eq. (19) divides energy by time.
- **Status:** PENDING.
- **Decision rule:** Write unit tests for E=P×T and E_transmission=D×E_data separately; do not claim exact paper energy reproduction until values and Eq. (19)'s intended units are resolved.

## A-009 — DQN hyperparameters
- **Question:** Learning rate, optimizer, replay capacity, batch size, target sync interval, episode count, random seed.
- **Paper says:** These values are missing or conflicting. Gamma 0.95, epsilon minimum 0.01 and decay 0.01 are given. Initial epsilon appears as both 1.0 and 0.1. Text gives both 100 and 150 runs/episodes.
- **Status:** PENDING.
- **Decision rule:** Do not train until these have explicit defaults in `config.py` and are reported in every run manifest. For conflicting values, the decision note must cite both source locations.

## A-010 — Network shape
- **Question:** Does “50 neurons ... in the model hidden layers” mean 50 per hidden layer?
- **Paper says:** Algorithm 1 gives two ReLU dense layers; §5.4 p. 13 mentions 50 neurons in the hidden layers.
- **Status:** PENDING.
- **Decision rule:** Prefer 50 units per hidden layer as a clearly labelled inference if no further evidence appears; preserve configurability.

## A-011 — Dataset identity
- **Question:** Which Google trace/file, version, columns, filtering and workload mapping were used?
- **Paper says:** Calls it “Google Jobs” and mentions 1,000 tasks plus generated remaining tasks; the data-availability statement says no datasets were generated or analysed.
- **Status:** BLOCKED for exact workload reproduction.
- **Decision rule:** Do not download or fabricate a substitute under the label “Google Jobs.” First identify a source and column mapping; otherwise run a synthetic-workload experiment clearly labelled synthetic.

## A-012 — Baseline models
- **Question:** Exact CNN/LSTM/GGCN architecture, training data, optimizer and scheduling action mapping.
- **Paper says:** These are comparator names only; sufficient implementation detail is not provided.
- **Status:** PENDING.
- **Decision rule:** If comparison is attempted, label our baseline implementations as approximate variants and use matched workloads/resources/seeds; do not claim exact baseline reproduction.

## A-013 — Queue capacity and resource limits
- **Question:** Queue size, per-VM CPU/memory/storage, and what happens when no fog VM is eligible.
- **Paper says:** A full VM queue should cause search for another fog VM; queue and individual VM capacities are absent.
- **Status:** PENDING.
- **Decision rule:** Keep queue capacity and VM capacities configurable. Add a defined overflow/no-resource behavior and tests before training.

## A-014 — Event semantics and makespan
- **Question:** How do arrivals, queue waiting, service time, network delay and task completion contribute to Eq. (16)?
- **Paper says:** SimPy, millisecond execution units, random arrivals, and a max of per-machine accumulated processing time. Network latency formula is not specified.
- **Status:** PENDING.
- **Decision rule:** Make event timestamps explicit and calculate makespan from observed finish times. Report whether arrival and network delays are included.

## A-015 — Reported-result integrity
- **Issue:** The percentage-improvement tables do not consistently match raw metric tables; fault-tolerance directions also conflict.
- **Status:** OPEN audit item.
- **Decision rule:** Keep the source's raw numbers and independently recalculated percentages side-by-side in a later result-audit artifact. Do not tune the simulator to reproduce suspicious values.

## Decision log format

Whenever a decision is resolved, add: date, microtask ID, the exact selected rule, reason, paper references, tests affected, and whether it is a paper requirement / inference / implementation approximation. Do not delete the original unresolved note; update its status and preserve the history.
