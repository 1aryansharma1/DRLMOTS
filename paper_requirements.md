# DRLMOTS — Paper Requirements and Technical Extraction

**Primary source:** Prashanth Choppara and Sudheer Mangalampalli, “An efficient deep reinforcement learning based task scheduler in cloud-fog environment,” *Cluster Computing* 28:67 (2025), DOI: 10.1007/s10586-024-04712-z. PDF pages are cited below using the page numbers printed in the paper.

**Purpose:** Turn the paper into an implementation specification while keeping explicit paper statements separate from inference and unresolved decisions. This is an extraction, not evidence that an implementation or reproduction has passed tests.

## 1. Scope in one page

Implement a Python discrete-event simulation of a three-layer IoT–fog–cloud environment. IoT devices generate tasks; gateways/access points and end nodes collect/prioritize them; the scheduler allocates each task to a VM associated with fog/cloud resources. The proposed decision policy is a Deep Q-Network (DQN), using task/environment state, VM choice as an action, a multi-objective reward, epsilon-greedy exploration, replay memory, and a target network. SimPy is explicitly named for simulation. The paper compares makespan, energy consumption, and a metric called fault tolerance against CNN, LSTM, and GGCN baselines.

**Implementation target:** first reproduce the *specified behavior* on deterministic toy cases; then use the reported network/workload settings; finally compare results where the source provides enough information. Do not claim numerical reproduction when the paper leaves key definitions or data unavailable.

## 2. System architecture and end-to-end workflow

The architecture figures and prose (Fig. 1, p. 6; Fig. 2, p. 9; Fig. 3, p. 11; Fig. 4, p. 14) imply this workflow:

1. IoT devices generate heterogeneous tasks.
2. Gateways/access points collect task requests; end nodes gather and prioritize them based on task length and processing capacity. The priority formula itself is not explicitly given in a consistent, recoverable form.
3. The resource manager tracks task requests, VM queue load, capacity, storage, memory, and network conditions.
4. A threshold/load rule sends low-processing-time tasks toward cloud and higher-processing-time tasks toward fog, as stated in the prose and the decision diagram. Other examples in the paper describe high-computation tasks as cloud candidates, so this rule must be checked against the exact intended meaning before freezing it.
5. A DQN agent observes state and chooses a VM from the action set.
6. Tasks are queued and executed; if a chosen fog VM queue is full, the paper says to find another fog node with a non-full queue.
7. The environment reports the next state and reward. The transition tuple is stored in replay memory; the online network is trained and the target network is updated.
8. Evaluate makespan, energy, and the paper's fault-tolerance metric over different workload sizes and VM configurations; compare with CNN, LSTM, and GGCN.

### Entity/module inventory

| Entity/component | Responsibility | Source |
|---|---|---|
| IoT device/task generator | Generate tasks with lengths/data sizes and arrival times | pp. 5–8, 14–15 |
| Gateway/access point | Aggregate device tasks | pp. 5–8, Fig. 1 |
| End node / task manager | Collect requests and prioritize jobs | pp. 7–8, Fig. 2 |
| Fog node / fog controller | Nearby execution, lower-latency but limited resource tier | pp. 5–9 |
| Cloud datacenter | Cloud execution tier | pp. 5–9, Fig. 2 |
| Physical host / VM | Compute, storage/memory/resource capacity and a task queue | pp. 6–10 |
| Scheduler | Use priority, queue/load and DQN action to map a task to a VM | pp. 7–12 |
| Simulation environment | Drive arrivals, execution and transitions in SimPy | pp. 13–15 |
| DQN online network | Estimate Q values for possible VM actions | pp. 10–13, Algorithm 1 p. 12 |
| Target network / replay memory | Stabilize Q learning using stored `(state, action, reward, next_state, done)` experience | Algorithm 1 p. 12 and §5.4 p. 13 |
| Metrics/evaluator | Makespan, energy and fault-tolerance calculations; two VM scaling cases | pp. 9–10, 15–23 |
| Baseline policies/models | CNN, LSTM, GGCN comparison | pp. 14–25 |

## 3. Algorithm and reinforcement-learning formulation stated by the paper

### 3.1 State

Equation (23), §5.1 p. 11, writes the state as a combination/union of a task state and a VM state. The prose says state should capture the number of tasks, fog/cloud resource availability, and network conditions. Later prose in §5.4 p. 13 also mentions active tasks, task attributes, fog/cloud capacity, and real-time network information.

**Not specified:** an actual ordered feature vector, each feature's unit, its normalization, its dimension, and how the “union” is represented numerically. The implementation must define these only after recording them as design decisions.

### 3.2 Action

Equation (24), §5.2 p. 11:

\[
A=\{vm_1,vm_2,\ldots,vm_n\}.
\]

An action is selecting a VM for the task. It is not clear whether the action list includes cloud VMs alongside fog VMs, or whether the fog/cloud threshold rule runs before the DQN and restricts the legal actions.

### 3.3 Reward

Equation (25), §5.3 p. 11:

\[
r_t=\min(m_t,\; fault\_tolerance_t,\; E_t).
\]

The prose says rewards should be negative to encourage the agent to minimize makespan, fault tolerance, and energy, while elsewhere it says positive rewards are good. The formula takes a minimum across quantities with potentially different scales/units. The paper does not provide normalization or weights. Its use of “fault tolerance” as a quantity to minimize conflicts with standard interpretation and with its own claims of improved fault tolerance. Therefore, do **not** silently implement this as a well-defined scalar reward without documenting a chosen interpretation.

### 3.4 DQN/Q-learning update

Equation (22), p. 10, gives a Q-learning style update, readable in standard notation as:

\[
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right].
\]

Algorithm 1, p. 12, separately states a terminal/non-terminal target: terminal target is `reward`; non-terminal target is `reward + gamma * max_a Q_target(next_state, a)`. This is the DQN target form. A neural network represents Q-values because a tabular table would not scale to the state/action space.

The paper names replay memory, a fixed target network, epsilon-greedy selection, and two ReLU hidden layers. Algorithm 1 shows an input layer, hidden layers `h1 = ReLU(Dense(state,...))` and `h2 = ReLU(Dense(h1,...))`, and a Dense output layer returning Q values for actions. §5.4 p. 13 says “50 neurons” are used in the hidden layers; whether that means 50 in each hidden layer is not explicitly precise.

### 3.5 Exploration/learning values stated in Algorithm 1, p. 12

| Parameter | Paper value | Status/interpretation |
|---|---:|---|
| Discount factor `gamma` | 0.95 | Explicit |
| Initial epsilon | Algorithm line says `1.0`; comment says “initial set to 0.1 for full exploration” | Conflict in the paper; unresolved |
| Minimum epsilon | 0.01 | Explicit |
| Epsilon decay | 0.01 | Explicit numeric value, but additive vs multiplicative schedule unspecified |
| Hidden-layer width | 50 neurons “in the model hidden layers” | Explicit but layer-by-layer reading uncertain |
| Scheduling decision time | 10 ms | Explicit statement in §5.4 p. 13; simulator interpretation unclear |
| Learning frequency | “one” | Explicit but unit/meaning not adequately explained |
| Learning rate `alpha` | Symbol appears; no numeric value found | Missing |
| Replay buffer capacity | Mentioned, no numeric capacity found | Missing |
| Batch size | Mentioned, no numeric value found | Missing |
| Target-network update frequency | Target update is named; frequency not stated | Missing |
| Optimizer | Not identified | Missing |
| Number of episodes/iterations | 150 in one paragraph; 100 episodes in same experiment section; most result descriptions say 100 iterations | Conflict |

## 4. Equations/definitions to track

The equations below are reported by the paper. Several are not sufficiently clear to implement literally without a decision; those are flagged rather than repaired silently.

| Equation | Paper statement / useful reading | Source | Implementation status |
|---|---|---|---|
| (1) | Total devices `d = sum_r d_r` | p. 7 | Straightforward aggregation |
| (2) | Total tasks `t = sum_r t_r` | p. 7 | Straightforward aggregation |
| (3) | `env = sum_r (d_r + t_r)` | p. 7 | Reported as written; dimension/meaning of combined count is unclear |
| (4)–(5) | Access-point aggregation of devices and tasks | p. 7 | Equation typesetting/indices are unclear; need a data-model decision |
| (6) | Total task length as a sum over task categories | p. 7 | Exact index semantics unclear |
| (7) | Total end-node process time `EN = sum_i EN_i` | p. 7 | Straightforward aggregation, subject to Eq. (8) units |
| (8) | Processing time `EN_n = length × 2 / 1000`; text says 1000 bytes needs 2 ms | pp. 7–8 | Use only after confirming length unit is bytes; paper also lists task length 700,000 without its unit in Table 3 |
| (9) | Total fog nodes `FN = sum_i FN_i` | p. 8 | Straightforward aggregation |
| (10) | VM queue load `lo_VMq = sum_n lo^n` | p. 8 | The exact meaning/unit of `lo^n` is not specified |
| (11)–(12) | Dynamic threshold `d_(j,p+1)` based on resource/load conditions, current interval and average requirement `Re_j_avg`; first-interval branch uses `R_(j,p)` | p. 9 | Source notation is partially garbled and variable definitions are incomplete |
| (13) | Relationship between VM queue load and physical-host load | p. 9 | Equation/indices and normalization unclear |
| (14) | Find a fog node `f` with a VM queue that is not full | p. 9 | Logical constraint is recoverable; queue capacity not specified |
| (15) | Append task `T` to selected node's queue | p. 9 | Straightforward queue operation once queue semantics are set |
| (16) | `Makespan(M) = max_i sum_(j=1..m_i) t_ij` | p. 9 | Implement as maximum accumulated processing/completion time across machines, documenting arrival-time treatment |
| (17) | Compute energy `E = P × T` | p. 10 | Power and time units/host values missing |
| (18) | Transmission energy `E_transmission = D × E_data` | p. 10 | Data energy coefficient and definition of D are missing |
| (19) | `E_total = (E + E_transmission) / T` | p. 10 | Labelled energy but divided by time; this is dimensionally inconsistent if numerator is energy |
| (20) | `FT = 1 − product(...)` but product term is missing/malformed in rendered equation | p. 10 | Cannot implement literally from Eq. (20) alone |
| (21) | `Faulttolerance(FT) = 1 − product_i P(F_i)`; `P(F_i)` is probability component i fails | p. 10 | Probability independence/redundancy interpretation not fully stated; table outputs appear to move opposite to the prose claim |
| (22) | Q-learning update with `alpha`, reward and discounted best next Q-value | p. 10 | Standard Q update can be mapped; DQN target-network form is clarified in Algorithm 1 |
| (23) | State combines task state and VM state | p. 11 | Feature schema absent |
| (24) | Action set is the set of VMs | p. 11 | Candidate/valid action filtering not specified |
| (25) | Reward `r_t = min(m_t, fault_tolerance_t, E_t)` | p. 11 | Objective directions, signs, units and normalization conflict; unresolved |

## 5. Experimental setup extracted from the paper

### Topology and task generation

- Three conceptual tiers: IoT devices, fog nodes, and cloud datacenter (pp. 5–6; Fig. 1 p. 6).
- Simulation description lists four access points, four end nodes, four fog nodes, four IoT devices and one cloud datacenter (p. 13).
- Each fog node is assigned a set of VMs; the count of VMs is varied from 20 to 200 in Table 3 (p. 15). Exact distribution of VMs over fog/cloud nodes is not stated.
- Task data size is stated as 20–150 kB (pp. 13–14). Table 3 separately lists “Task lengths” as 700,000, with no unit or distribution (p. 15).
- Tasks arrive at random intervals; execution time is measured in milliseconds (p. 14).
- A paragraph says the Google Jobs workload has 1,000 tasks and the remaining task sets are fabricated/randomized. The exact dataset URL/version, schema, preprocessing and mapping of data fields to task length/arrival time are not given. The final data-availability statement says no datasets were generated or analysed, which conflicts with the claimed Google Jobs use (pp. 15, 24–25).
- Workload scales reported in tables: small 300–1,500; medium 1,800–3,000; large 3,300–4,500. Table 3 instead says total tasks 300–3,000 (pp. 15–17).

### Table 3 — stated simulation settings (p. 15)

| Setting | Stated value |
|---|---|
| Total number of tasks | 300–3,000 (inconsistent with large-dataset tables through 4,500) |
| Task lengths | 700,000 (unit unspecified) |
| Host storage | 3 TB |
| Host RAM | 32 GB |
| Bandwidth | 1000 MBPS |
| Number of VMs | 20–200 |
| VM bandwidth | 200 MBPS (listed twice in table) |
| Processor | Intel i7 |
| Operating system | Linux |

The narrative also says 150 episodes/iterations in one place and 100 episodes in another; result-table descriptions repeatedly say 100 iterations (p. 14–15). No random seed is stated.

### Evaluation cases and reported comparator methods

- **Case 1:** VM count fixed while tasks increase; Tables 4–6 / Figs. 5–7 cover makespan.
- **Case 2:** VM count increases linearly while task count changes; Tables 7–9 / Figs. 8–10 cover makespan.
- Energy: Tables 10–12 / Figs. 11–13.
- Fault tolerance: Tables 13–15 / Figs. 14–16.
- Baselines: CNN, LSTM, GGCN. Their architectures, optimizer, inputs, training regime and mapping from predictions to scheduling actions are not specified sufficiently to reproduce them exactly.

## 6. Explicit vs inferred vs missing

### Explicitly specified

- Use SimPy for the discrete-event simulation (pp. 2, 13–15).
- Use a DQN/deep Q-learning scheduler; the action is a VM choice (pp. 10–12).
- Include replay memory, target model, and epsilon-greedy choice (Algorithm 1, p. 12).
- `gamma = 0.95`, `epsilon_min = 0.01`, `epsilon_decay = 0.01`; the initial epsilon is internally contradictory (p. 12).
- Network has an input layer, two ReLU dense hidden layers, and an output layer for Q values. Text reports 50 hidden neurons but not unambiguously for each layer (pp. 12–13).
- Stated topology and machine/workload parameters as tabulated above.
- Report makespan, energy, fault-tolerance metric and compare with CNN/LSTM/GGCN.

### Reasonable inference (must be logged as such)

- Each decision maps one newly considered task to one valid VM; action masking may be needed where resources/queues are unavailable.
- The output layer size should match the number of VM actions for a fixed topology/configuration.
- The DQN experience tuple should be `(state, action_index, reward, next_state, done)`, as named in Algorithm 1.
- To test the simulator incrementally, use tiny deterministic workloads before randomly generated full workloads.

### Missing or ambiguous

1. Exact state feature vector, feature order, units, normalization and dimension.
2. Actual task-priority formula; Fig. 4 refers to Eqs. (10)–(12) as priorities, but those numbered equations describe load/threshold in the body.
3. How VM selection and the fog/cloud threshold interact; prose and examples are not fully aligned.
4. Whether the DQN acts over every fog/cloud VM or after tier selection.
5. VM/host CPU speeds, per-VM memory/storage, queue limits, number of VMs per node and number of cloud VMs.
6. Threshold variables and Eq. (11)–(13) semantics.
7. Task length unit/distribution, task arrival distribution, and Google workload preprocessing.
8. Reward scaling/sign; how to jointly combine unlike metrics; whether “fault tolerance” means reliability, failure probability, or an inverse.
9. Power draw for each resource type, transmission energy-per-data-unit, units for energy, and exact use of Eq. (19).
10. Failure probabilities, redundancy policy and when/where failures occur.
11. Learning rate, optimizer, replay size, batch size, target-network update frequency, training episode count and seed policy.
12. Meaning of “10 ms scheduling choice” and “learning frequency at one.”
13. Baseline architecture/hyperparameters and fair training/evaluation conditions.
14. Exact Google Jobs dataset identity and download/preprocessing procedure.

## 7. Contradictions and result-integrity checks

These are source issues, not implementation bugs:

- **Initial epsilon:** Algorithm 1 says `epsilon = 1.0`, with a comment saying `0.1 for full exploration`.
- **Episode count:** 150 is mentioned on p. 14; 100 episodes/iterations appear elsewhere.
- **Task-count range:** Table 3 reports 300–3,000, while tables include 3,300–4,500.
- **Fault-tolerance direction:** The paper describes minimizing fault tolerance in places and calls the values “fault tolerance,” yet tables show DRLMOTS values lower than all baselines and the prose says fault tolerance improves. This metric cannot be interpreted safely without a documented choice.
- **Reward:** Eq. (25) takes a minimum of makespan, fault tolerance and energy, despite different scales/directions; text alternates between negative and positive rewards.
- **Energy:** Eq. (19) divides energy by time while calling the result total energy.
- **Priority formula:** Fig. 4 points to Eqs. (10)–(12) for task priority, but Eq. (10) describes VM queue load and Eqs. (11)–(12) describe a threshold.
- **Claimed improvements vs raw result tables:** Table 4 reports at 300 tasks CNN makespan 1012.5 and DRLMOTS 687.1, a direct reduction of about 32.1%; Table 16 reports 42.12% for the same task count/case label. These values do not reconcile under the usual `(baseline − DRLMOTS)/baseline` calculation.
- **Dataset availability:** the narrative claims a Google Jobs dataset; the declaration on pp. 24–25 says no datasets were generated or analysed.
- **Terminology:** the acronym is expanded in different ways, including “Deep Reinforcement Learning based Multi Objective Task Scheduler” in the abstract and “Dynamic Resource Allocation and Load Monitoring System” in the conclusion. Use DRLMOTS as the paper's scheduler name.

We will retain original table values as source benchmarks but not hard-code them as expected outputs for our simulator. First reproduce definitions and behavior, then investigate reported numbers separately.

## 8. Traceability matrix

Status meanings: **EXTRACTED** = found in the paper; **OPEN** = paper lacks sufficient implementation detail; **PLANNED** = module/test will be added; **VERIFIED** is reserved for a test that has actually been run and passed.

| ID | Paper requirement | Source | Implementation component | Verification method | Status |
|---|---|---|---|---|---|
| R001 | Three-tier IoT/fog/cloud architecture | Fig. 1, p. 6 | `topology.py`, `entities.py` | Topology invariant test | EXTRACTED |
| R002 | Four IoT devices, APs, end nodes and fog nodes plus cloud datacenter | p. 13 | `config.py`, `topology.py` | Configuration assertion | EXTRACTED |
| R003 | VMs are scheduling resources; experiment range 20–200 | Table 3, p. 15 | `entities.py`, `config.py` | VM count test at lower/upper values | EXTRACTED |
| R004 | VM/host compute, storage and memory affect feasibility | Table 2 and §4, pp. 6–9 | `entities.py`, `environment.py` | Reject infeasible allocation | OPEN (capacities incomplete) |
| R005 | Host storage 3 TB, host RAM 32 GB | Table 3, p. 15 | `config.py` | Config-value test | EXTRACTED |
| R006 | Host bandwidth 1000 MBPS; VM bandwidth 200 MBPS | Table 3, p. 15 | `config.py`, transfer model | Config test; units check | EXTRACTED |
| R007 | Data sizes stated as 20–150 kB | pp. 13–14 | `workload.py` | Range/distribution test | EXTRACTED |
| R008 | Table 3 task length 700,000 | Table 3, p. 15 | `workload.py` | Config/reporting test | OPEN (unit/meaning missing) |
| R009 | Arrivals at random intervals | p. 14 | `workload.py` | Seeded arrival test and bounds | EXTRACTED (distribution OPEN) |
| R010 | 1000 tasks from Google Jobs/Google workload are mentioned | pp. 15, 24–25 | `data/` import pipeline | Dataset schema/records validation | OPEN (identity and mapping missing) |
| R011 | Processing time based on length: length × 2 / 1000; 1000 bytes → 2 ms | Eqs. (7)–(8), pp. 7–8 | `workload.py` / `environment.py` | Hand-calculated conversion tests | EXTRACTED (unit OPEN) |
| R012 | Task manager determines job priorities | §4, pp. 7–8; Fig. 4 p. 14 | `priority.py` | Known-input priority test | OPEN (formula not recoverable) |
| R013 | Track VM queue load | Eq. (10), p. 8 | `environment.py` | Queue-load invariant test | EXTRACTED (load unit OPEN) |
| R014 | Dynamic resource threshold directs cloud/fog assignment | Eqs. (11)–(12), pp. 8–9 | `placement.py` | Hand-calculated threshold tests | OPEN (equations/semantics damaged) |
| R015 | Find another fog node when chosen VM queue is full | Eq. (14), p. 9 | `placement.py` | Full-queue fallback test | EXTRACTED (capacity OPEN) |
| R016 | Append task to selected queue | Eq. (15), p. 9 | `environment.py` | Exactly-once queue test | EXTRACTED |
| R017 | State contains task and VM/environment context | Eq. (23), p. 11; §5.4 p. 13 | `state.py` | Feature shape/type tests | OPEN (features missing) |
| R018 | Action is selecting a VM | Eq. (24), p. 11 | `actions.py`, `agent.py` | Action-to-VM mapping test | EXTRACTED (valid action set OPEN) |
| R019 | Reward accounts for makespan, fault tolerance, energy | Eq. (25), p. 11 | `reward.py` | Hand-calculated reward tests | OPEN (sign/scaling conflict) |
| R020 | Q-learning temporal-difference update | Eq. (22), p. 10 | `agent.py` | Numeric target/update test | EXTRACTED |
| R021 | Two ReLU dense hidden layers and Q output | Algorithm 1, p. 12 | `dqn/network.py` | Shape and forward-pass tests | EXTRACTED (width interpretation OPEN) |
| R022 | 50 hidden neurons reported | §5.4 p. 13 | `config.py`, `dqn/network.py` | Architecture inspection/test | OPEN (per-layer ambiguity) |
| R023 | `gamma = 0.95` | Algorithm 1, p. 12 | `config.py`, `agent.py` | Config assertion | EXTRACTED |
| R024 | Epsilon-greedy policy; epsilon min 0.01; decay 0.01 | Algorithm 1, p. 12 | `agent.py` | Policy probabilities/bounds test | OPEN (initial epsilon conflict/schedule) |
| R025 | Replay experience `(s,a,r,s',done)` | Algorithm 1, p. 12; §5.4 p. 13 | `dqn/replay_buffer.py` | Insert/sample/terminal tests | EXTRACTED |
| R026 | Fixed target network and target-based bootstrap | Algorithm 1, p. 12 | `agent.py` | Target computation and sync test | EXTRACTED (sync cadence OPEN) |
| R027 | Decision overhead described as 10 ms | §5.4, p. 13 | event/scheduler timing | Timing accounting test | OPEN (simulated vs wall-clock meaning) |
| R028 | Makespan equation | Eq. (16), p. 9 | `metrics.py` | Hand-calculated multi-VM example | EXTRACTED |
| R029 | Compute energy as `P×T` | Eq. (17), p. 10 | `metrics.py` | Hand-calculated energy test | OPEN (power values/units missing) |
| R030 | Transmission energy `D×E_data` and Eq. (19) | Eqs. (18)–(19), p. 10 | `metrics.py` | Unit/dimension check | OPEN |
| R031 | Fault-tolerance metric from component failure probabilities | Eqs. (20)–(21), p. 10 | `metrics.py` | Probability-bound tests | OPEN (Eq. 20 malformed/meaning conflict) |
| R032 | Test fixed VM count and increasing task count | Case 1, pp. 15–17 | `evaluation.py` | Configured experiment manifest | EXTRACTED |
| R033 | Test linearly increasing VM count | Case 2, pp. 17–19 | `evaluation.py` | VM/task scaling manifest | EXTRACTED (slope missing) |
| R034 | Small/medium/large workload sizes | Tables 4–15, pp. 15–22 | `workload.py`, `evaluation.py` | Workload count checks | EXTRACTED (Table 3 conflicts) |
| R035 | Compare with CNN, LSTM and GGCN | §6, pp. 14–25 | `baselines.py` | Same-workload evaluation protocol | EXTRACTED (model specs missing) |
| R036 | Record experiment repetitions/iterations | pp. 14–15 and result tables | `training.py`, `evaluation.py` | Manifest/log check | OPEN (100 vs 150 conflict) |
| R037 | Results and plots for makespan, energy, fault tolerance | Tables/Figs. 4–19, pp. 15–24 | `metrics.py`, `plots.py` | CSV/figure existence + schema check | EXTRACTED |
| R038 | Reproducible configuration and random seeds | Not stated; needed for verification | `seeding.py`, run manifest | Repeat-run deterministic check | IMPLEMENTATION REQUIREMENT (not specified by paper) |
| R039 | Preserve and audit source results/claims | Tables 4–19 and abstract/conclusion | `results_audit.py` or validation notebook | Recalculate percentages from raw tables | IMPLEMENTATION REQUIREMENT; source inconsistencies noted |
| R040 | Document unavailable data and approximations | Data availability pp. 24–25 | `assumptions.md`, `README.md` | Audit each OPEN item | IMPLEMENTATION REQUIREMENT |
