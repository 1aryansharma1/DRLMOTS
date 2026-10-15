# Fog-first completion-time policy

Version 0.3.0a5 implements the user's requested policy, `fog_first_completion`, under
A-026. It is a deterministic scheduling rule; it is not learned RL or the paper's
unresolved fog/cloud threshold formula.

## Decision rule

1. Use the configured source end node's linked fog nodes as local candidates, and
   the cloud datacenters' VMs as cloud candidates. Host placement/distance is not inferred.
2. Keep only VMs whose total resource capacity fits the task and whose waiting queue
   has a slot. A running VM is eligible; uploading tasks also occupy waiting slots.
3. Predict completion for every candidate using current work and explicit link data.
4. Compare the earliest-finishing fog VM with the earliest-finishing cloud VM.
   Choose fog when its finish is earlier or equal. Choose cloud when it is faster
   or no fog VM is eligible. VM ID resolves equal finishes within a tier.
5. Reject explicitly if neither tier has an eligible VM. Record candidates, estimates,
   selection reason and actual timestamps in JSON.

For decimal kB and Mbps:

```text
upload_ms = propagation_ms + 8 * task.data_size_kb / bandwidth_mbps
upload_ready = arrival_time + upload_ms

tail_finish = running_task_finish, or current_time if idle
for each already queued task in FCFS order:
    tail_finish = max(tail_finish, queued_upload_ready) + queued_service_ms

predicted_start  = max(tail_finish, upload_ready)
predicted_finish = predicted_start + own_service_ms
completion_duration = predicted_finish - arrival_time
```

Uploads overlap existing VM execution, so adding full upload time to full queue delay
would double-count overlap. FCFS is maintained even if a later task uploads sooner.

Example: a task arrives at 1 ms with 1 ms upload and 2 ms service. Fog's previous work
finishes at 3 ms, so it starts at max(3,2)=3 and finishes at 5 ms. Cloud needs 5 ms
upload and finishes at 8 ms. **Busy fog is selected.** If fog instead finishes at
23 ms, cloud is selected. Equal predicted finishes always select fog.

## Configuration and execution

`configs/fog_cloud.yaml` is a complete runnable schema-3 synthetic configuration.
Its `placement` section explicitly names the origin, units, independent-upload model
and one link for each reachable fog/cloud VM. Missing/duplicate/unknown link IDs,
unknown origin, nonpositive bandwidth and negative/nonfinite delays are rejected.
Reported paper bandwidth values are separate and never silently reused.

From a configured repository environment:

```powershell
python -m drlmots --config configs/fog_cloud.yaml --output outputs/metrics/fog-cloud.json
python -m drlmots.demo --config configs/fog_cloud.yaml --output-dir outputs/demos/fog-cloud
Start-Process outputs/demos/fog-cloud/drlmots-demo.html
```

Schema 3 defaults to `fog_first_completion`; `--policy least_queue_fcfs` or
`--policy first_fit_fcfs` runs those rules under the same network model. Schema 1/2
defaults and zero-network results are preserved. The new policy without explicit
schema-3 network settings fails clearly. Saved effective configs include placement
settings in the config checksum and can be replayed through `parse_run_config`.

The optional network demo compares three policies under identical inputs/links,
includes 21 PNG/21 SVG figures, and defaults to fog-first. Expand **Fog/cloud decisions
and transmission times** to inspect best tier totals, selected upload time, predicted
finish and reason. The original baseline demo still compares two policies/15 figures.
Scenario/policy selectors show precomputed runs; no browser training occurs.

## Verification and results

All **618 tests pass** on Python 3.12.14; dependency check is clean. Forty-one new tests
cover busy/idle fog, faster cloud, ties, data-dependent choices, queued uploads,
overlap, full queues, resource failure, source reachability, same-time events,
causal forecasts versus actual finishes, input immutability, strict config/replay,
CLI, RL compatibility guards and network-demo generation.

The seed-42 100-task fixture produces 65 completions/35 rejections, makespan
165.06162267920254 ms and successful mean completion 10.524262058797573 ms under
fog-first. Least queue/first fit each complete 62/reject 38 on those same inputs.
These are one synthetic fixture's observations, not general policy superiority.
Do not compare these directly against the old zero-network 96/4 fixture: links and
queue settings differ. Both workload counts and successful latency must be reported.

Both demo modes pass their Node DOM-stub checks (9/6 selections and 3 JSON exports
each); raw-template startup is safe. Representative network PNGs were visually
inspected. Actual desktop/mobile browser layout remains a manual check.

## Boundaries and RL follow-up

This is simulated one-way input upload plus fixed propagation. Uploads are independent,
with no shared bandwidth contention, packet routing, output download, failures or energy.
One source end node is configured per run; physical nearest-fog selection is not modeled.
Per-VM resource accounting remains serial; host budgets and CPU-dependent speed are open.

The original observation/reward contract assumes zero network delay and cannot encode
idle VMs with uploading queues. Network-aware runs therefore **reject legacy decision
capture/reward/normalization**, rather than silently emitting incompatible learning
data. Placement estimates/transfers are exported in a separate report section.
The existing zero-network RL prerequisites and untrained Q-network remain available.
Before training on this model, version the observation/action/normalizer context and
reward/transitions to include origin, link/upload readiness and FCFS upload waiting.
Replay/target updates and live agent control/training then follow.
