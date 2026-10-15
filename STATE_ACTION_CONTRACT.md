# State and action contract - schema 1

This preserved legacy contract is separate from the live schema-2 training pipeline
delivered in v0.3.1. See [LIVE_DQN.md](LIVE_DQN.md) for network-ready states, actual
agent-controlled placement, replay/updates and resumable checkpoints. Historical
"not yet implemented" statements below refer to the original schema-1 checkpoint.

This is the initial v0.3 state/action milestone (A-022), not the paper's confirmed
state vector, a trained policy or a complete learning environment.

## Observation boundary

Observe each arriving task after every completion at that timestamp and all queued
successors have started, immediately before this arrival is placed or rejected.
Simultaneous arrivals retain input order; later arrivals see earlier placements.
There is one decision record per arrival, including rejected tasks. Input task/VM
objects stay unchanged. Snapshots use frozen dataclasses and tuples, detached from
runtime queues/ledger. Captured decisions remain unchanged after resource release.

VM snapshots contain tier, running flag, remaining running service time, waiting task
count/capacity, total queued service time, total capacities and available balances.
Task fields are ID, decision time, byte length, data size in kB and resource demands.
Absent task demand means zero. Missing/null VM resource capacity means unknown.
Neither future arrivals nor task IDs enter the numeric state. The queue aggregate
does not encode queued-task identities/profiles; it is a provisional observation,
not a proof of the Markov property for a future learning setup.

## Fixed numeric ordering

VM actions use lexicographically sorted VM IDs. Resource columns use lexicographically
sorted names from the run's explicit unit catalogue, including currently unused names.
Both mappings are fixed for a configured run. New topology/catalogue means a new
schema and incompatible feature/action dimensions; do not reuse a model silently.

The pure encoder returns a tuple of finite Python floats in this order:

| Block | Ordered features | Units |
|---|---|---|
| Task | decision time, task length, data size | ms, bytes, kB |
| Task resources | demand for each resource name | catalogue unit |
| Each VM | cloud flag, running flag, remaining service time, waiting count, waiting capacity, queued service time | binary, binary, ms, tasks, tasks, ms |
| Each VM/resource | capacity-known flag, total capacity, available-known flag, available amount | binary, catalogue unit, binary, catalogue unit |

Dimension = `3 + R + V * (6 + 4 * R)`. Unknown amounts use zero with a zero known flag;
known available zero has a one known flag. Fog/cloud is binary 0/1. Values are raw,
unscaled doubles in the raw trace. v0.3.0a2 adds separate training-only scaling and
saved artifacts; see NORMALIZATION.md. v0.3.0a3 adds a separate synthetic reward and
arrival-transition/forced-chain contract; see REWARD_CONTRACT.md. Float32 model
conversion and interactive agent control remain later tasks. Feature names and units,
VM/resource order, dimension and schema version accompany a captured trace.

## Legal actions

Action `i` maps to exactly one VM ID; negative, boolean, noninteger and out-of-range
indices are invalid. A boolean mask follows that same VM order. An action is legal
when every positive demand fits a known total VM capacity and the VM is idle or has
a free waiting slot. An idle VM starts immediately. A busy VM may queue a task even
when its currently available capacity is zero; only running tasks reserve resources.
The mask filters every VM independently and permits both fog/cloud tiers under A-020.

All-false masks do not add a fake rejection VM/action. The decision records
`selected_action: null`, `selected_vm_id: null` and the existing explicit failure
reason. Zero demands fit unknown capacities, preserving the legacy baseline.

## Capture and replay

```powershell
.\.venv\Scripts\python.exe -m drlmots --config configs/resources.yaml --capture-decisions --output outputs/metrics/decisions.json
```

The optional `decision_trace` report contains `schema` and `decisions`, each with raw
`observation`, `state`, `action_mask`, selected index/ID and rejection reason. Capture
is opt-in to avoid storing every vector in ordinary reports. It does not change
placement, timings, input hashes or resource balances. Empty traces still include
their schema. Replay with the saved config and policy, also enabling capture.

Python entry points: `StateSchema.from_topology`, `encode_state`, `ActionMap`,
`action_mask`; `simulate(..., capture_decisions=True)` returns its captured decisions.
The environment still runs baseline policies to completion. Opt-in rewards add
raw arrival transitions and a pure learner adapter; there is no learned action
callback, interactive step API, experience replay or DQN training yet.

For the resource fixture: V=3, R=2, dimension=47, VM order cloud/fog-a/fog-b. Six
simultaneous arrivals see the masks below (T = legal, F = illegal, in VM order):

| Arrival | Mask (cloud, fog-a, fog-b) | Selected action | Selected VM |
|---|---|---|---|
| 1 | TTF | 0 | cloud |
| 2 | TTF | 1 | fog-a |
| 3 | TTF | 0 | cloud |
| 4 | FTF | 1 | fog-a |
| 5 | FFF | null | none (rejected) |
| 6 | FFF | null | none (rejected) |

The normal 4 completed/2 rejected, 4 ms makespan and restored balances are  preserved.
