# Training-only normalization - version 1

This v0.3.0a2/P6.3 checkpoint builds on synthetic state schema 1 (A-022). A-023
defines the scaling approximation. It does not add rewards, agent transitions or DQN.
v0.3.0a3 implements rewards/transition export separately in REWARD_CONTRACT.md;
normalizer fitting/application still use unscored traces and do not change statistics.

## Formula and boundary cases

For each nonbinary feature, use the maximum across explicitly declared training
traces. Resource amount columns contribute only when their known flag is one.
Use scale 1 when the training maximum is zero or no known samples exist. Transform
as `min(value / scale, 1)` for known nonnegative values. Implementation checks
`value >= scale` before dividing, so tiny scales and huge held-out values cannot
overflow. Constant positive columns map to 1. Held-out values below that constant
retain their ratio; all-zero training columns use the documented scale-1 fallback.

Tier/running/known flags remain exactly 0/1. Unknown amount placeholders stay zero;
a known zero available balance still has known flag one. Binary column positions
come from the state layout, not catalogue unit labels. All output values are finite
in [0,1]. Clipping deliberately loses magnitude above the training range; it does
not prove good training behavior. Raw decision vectors and action masks stay intact.

## Fit and apply separately

From a configured repository root (use `.venv312` in the current agent checkout):

```powershell
.\.venv312\Scripts\python.exe -m drlmots.normalization fit --train-config configs/normalization_train.yaml --output outputs/metrics/normalizer.json
.\.venv312\Scripts\python.exe -m drlmots.normalization apply --config configs/normalization_evaluation.yaml --normalizer outputs/metrics/normalizer.json --output outputs/metrics/normalized-evaluation.json
```

Both configs use the same resource topology and 100-task variable workload. Training
uses seed 42; the named evaluation fixture uses seed 314159. They are synthetic
software fixtures, not a training/experiment recommendation. Repeat `--train-config`
to explicitly fit multiple training configurations. `--policy` defaults to
least_queue_fcfs; an artifact records the generating training policy. Apply can use
another baseline on a compatible topology without changing fitted statistics.

Fit accepts only training batches. Apply uses a saved artifact and labels its run
test (or `--split validation`); it has no fitting step. Held-out application rejects
training-source/exact-task overlap, including replaying the same tasks under another
policy. Library users may transform a batch labelled train for training inspection.
Caller-supplied labels/provenance are explicit declarations, not independent proof
of dataset independence. Full real-dataset/split design remains future work.

## Artifact and compatibility

The JSON artifact records algorithm/version, schema feature names/order/units and
hash, full canonical topology hash, maxima, known numeric sample counts, fallback
scales, total training-vector count and per-source configuration/workload/generator/
software provenance. An SHA-256 catches accidental corruption; it is not a signature.
Binary columns have count/max 0 and scale 1 because they pass through unchanged.

The topology hash includes VM IDs/tiers/capacities/queue limits, host metadata and
node memberships/links. List order is canonicalized, so list permutations preserving
the same relationships are compatible. Changed IDs, units, resources, capacities,
queue limits, tier or links are incompatible even if dimensions match. Numeric
JSON representations are conservative: changing integer metadata to float metadata
can change the fingerprint. Workload seed/count/length settings may differ between
training and evaluation; they are not part of topology/schema compatibility.

Load checks version, required fields, finite nonnegative statistics, counts, scale
consistency and checksum; duplicate JSON keys and NaN/Infinity are rejected.
Apply verifies the current context before transforming. The fitted object is frozen;
transforms cannot alter maxima, counts, sources or the saved file.

## Python APIs

`NormalizationContext.from_topology(topology, resource_units)` binds a state schema
to its full topology. `TraceBatch` holds immutable raw vectors, explicit split and
source provenance. `capture_batch(config, split=..., policy=...)` creates a batch
and its ordinary captured simulator report. `fit_normalizer(training_batches)`
returns a frozen `StateNormalizer`; use `transform`, `transform_batch`, `save` and
`StateNormalizer.load(path, context=...)` with explicit compatibility checks.

The application output preserves the original raw report and adds normalized vectors
beside each decision plus artifact identity. It does not change simulation placement,
failures, timings, workload/config hashes, resource traces or masks. Empty held-out
traces are supported; fitting requires at least one training vector. Generated
artifacts/reports stay ignored by Git and are regenerated by these commands.

## Verified checkpoint

At the v0.3.0a2 checkpoint the full suite passed 460 tests on Python 3.12.14,
including 41 normalization tests;
dependency check is clean. The named fixtures fit 100 training observations and
transform 100 held-out observations, each with 47 features and values in [0,1].
The held-out simulator still completes 92 and rejects 8; training stays at 96/4.
These counts verify unchanged software behavior and are not learned-policy results.
Hand ratios, zero/constant/unseen columns, unknown flags, subnormal scales and huge
inputs, save/load, topology changes at equal dimensions, malformed artifacts,
training overlap and actual CLI paths are covered.
