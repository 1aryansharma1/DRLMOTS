# Synthetic CPU Q-network and legal action selection

Version 0.3.0a4 implements the P9 prototype subset under A-025. It produces
untrained Q-values and legal selections. Replay, Bellman targets, optimizer updates,
live agent control and training remain subsequent milestones. This is separate from
paper-faithful architecture/hyperparameter decisions A-009/A-010.

## Model and compatibility

`NetworkConfig(hidden_widths=(50,50), seed=42)` selects two Linear/ReLU hidden layers
and a linear Q output. Both widths are configurable positive integers. Input count
comes from the state schema; output indices follow its sorted VM IDs. The resource
fixture has 47 inputs, 3 outputs and 5,103 parameters. There is no output activation:
Q-values are unbounded dimensionless estimates, not probabilities.

All layers are explicitly CPU float32. Initialization uses PyTorch Linear defaults
inside a restored CPU RNG context; it leaves caller CPU/Python RNG unchanged and does
not seed CUDA. Repeatability applies within the recorded runtime. Different Torch
versions or numerical batch kernels may differ; single/batch inference is compared
with float32 tolerance. Exact reload is checked with the same batch/runtime.

`NetworkContext.from_artifacts(normalizer,reward)` binds the full state schema,
topology hash, ordered actions, fitted-normalizer checksum and reward checksum.
`QNetwork(context,config)` requires this context again on every forward call:

```python
q = network(states, context=context)  # [batch, features] -> [batch, VM actions]
```

The tensor must have two dimensions, exact feature count, CPU float32 dtype, finite
values and bounds [0,1]. Empty batches are supported. Context disagreement is rejected
even with equal dimensions. Nonfinite/moved/retyped weights and overflowing Q outputs
are rejected. Forward preserves gradients for the later optimizer milestone.

Tensor values cannot prove which scaler produced them; the explicit context is a
caller assertion. Prefer `network.infer(observation, normalizer=scaler,reward=reward)`
for one actual `DecisionObservation`: it encodes, scales and derives artifact identity,
then returns a detached Q vector. The caller still declares/validates data splits;
the CLI uses `transform_batch` to reject training-source/exact-task overlap.

## Legal selection

```python
selector = MaskedSelector(seed=42)
action = selector.select(q, mask, epsilon=0.1)
```

Supply a one-dimensional finite CPU float32 Q vector and an equally sized boolean
mask from `action_mask(observation)`. Masked indices are never selected. Greedy ties
choose the first legal index in stable schema order; epsilon=1 samples uniformly from
legal indices with a local Python RNG. Intermediate epsilon samples the explore/greedy
choice locally. Greedy and forced-rejection calls consume no RNG draws. All-false
returns None, including during exploration, so no fake VM action is introduced.
Epsilon must be explicit and finite in [0,1]; there is no decay schedule yet.

## Network snapshots

`network.save(path)` writes strict checksummed JSON containing architecture, seed,
context, Torch version and named float32 weights. `QNetwork.load(path,context=context)`
requires exact artifact context and validates version, keys, checksum, architecture,
parameter names/shapes and finite representable numbers. It uses no executable pickle.
Snapshots round-trip weights exactly and preserve CPU RNG. The checksum detects
accidental changes; it is not an authenticity signature. A snapshot is not a resumable
training checkpoint: optimizer, replay and exploration RNG state are not saved.

## Reproduce the inference smoke example

From the repository root, install the package as in README (local agent interpreter
here is `.venv312\Scripts\python.exe`; use your own supported venv):

```powershell
python -m drlmots.normalization fit --train-config configs/normalization_train.yaml --output outputs/metrics/q-normalizer.json
python -m drlmots.q_network --config configs/normalization_evaluation.yaml --normalizer outputs/metrics/q-normalizer.json --reward-config configs/reward.yaml --checkpoint outputs/checkpoints/q-network.json --output outputs/metrics/q-inference.json --seed 42 --hidden-widths 50 50 --epsilon 0
```

The first command fits only 100 seed-42 training observations. The second uses 100
held-out seed-314159 observations collected under `least_queue_fcfs`, saves and reloads
the untrained network, and verifies identical Q-values. Its report records context,
workload/config provenance, checkpoint checksum, epsilon/selection seed, Q-values,
masks, action indices and VM IDs. There are 92 legal selections and 8 forced None
selections on this trace. Those selections are **not executed**: the 92/8 baseline
outcome is not a learned-policy result and no Q-policy return is claimed. Repeating
the same command/runtime yields the same artifact and decisions.

Outputs are ignored and regenerated per checkout. Report/checkpoint destinations
must differ and cannot overwrite input configs or the normalizer.

## Next gate: replay and target updates

Implement bounded replay/local sampling and hand-calculated Bellman targets from
the forced-aware `LearningTransition` adapter. Apply the frozen normalizer to both
current and nonterminal next states. Use its already computed discounted reward and
`bootstrap_discount`; do not discount forced chains twice. Bootstrap only over legal
next actions and never gather a null action. Add an independent target network,
explicit optimizer/gamma/batch/capacity/sync settings, finite updates and target-copy
tests. Interactive simulator control and end-to-end training follow that gate.
