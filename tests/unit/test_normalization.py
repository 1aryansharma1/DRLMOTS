from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math

import pytest

from drlmots.entities import Host, ResourceTier, VM
from drlmots.normalization import NormalizationContext, StateNormalizer, TraceBatch, fit_normalizer
from drlmots.state import DecisionObservation, VMObservation, encode_state
from drlmots.topology import FogNode, Topology


def context():
    return NormalizationContext.from_topology(Topology((), ()))


def batch(rows=((0, 2, 4), (2, 4, 8)), *, split="train", source="training"):
    return TraceBatch(context(), rows, split, source)


def resource_context():
    topology = Topology((Host("h", 3, 32, 1000),),
                        (VM("v", ResourceTier.FOG, 200, queue_capacity=1, resource_capacities={"cpu": 2}),),
                        fog_nodes=(FogNode("f", ("h",), ("v",)),))
    # A resource's custom unit spelling must not turn its amounts into flags.
    return NormalizationContext.from_topology(topology, {"cpu": "binary", "memory": "GB"})


def resource_state(ctx, known=True):
    vm = VMObservation("v", ResourceTier.FOG, True, 2, 0, 1, 0,
                       (2, None) if known else (None, None), (0, None) if known else (None, None))
    obs = DecisionObservation("t", 0, 1000, 20, ctx.schema.resource_units, (1, 0), (vm,))
    return encode_state(ctx.schema, obs)


def test_hand_calculated_ratios_and_saturation():
    normalizer = fit_normalizer([batch()])
    assert normalizer.maxima == normalizer.scales == (2, 4, 8)
    assert normalizer.counts == (2, 2, 2)
    assert normalizer.transform(context(), (1, 2, 4)) == (0.5, 0.5, 0.5)
    assert normalizer.transform(context(), (100, 8, 0)) == (1, 1, 0)


def test_zero_and_positive_constant_columns_have_explicit_fallback():
    normalizer = fit_normalizer([batch(((0, 2, 4), (0, 2, 4)))])
    assert normalizer.scales == (1, 2, 4)
    assert normalizer.transform(context(), (0.25, 1, 4)) == (0.25, 0.5, 1)
    assert normalizer.transform(context(), (2, 4, 8)) == (1, 1, 1)


def test_unknown_flags_known_zero_and_custom_binary_unit_are_preserved():
    ctx = resource_context()
    known, unknown = resource_state(ctx), resource_state(ctx, False)
    normalizer = fit_normalizer([TraceBatch(ctx, (known, unknown), "train", "mixed-known")])
    fields = {name: index for index, (name, _) in enumerate(ctx.schema.features)}
    a, b = normalizer.transform(ctx, known), normalizer.transform(ctx, unknown)
    assert normalizer.counts[fields["vm[0].capacity.cpu"]] == 1
    assert normalizer.counts[fields["vm[0].capacity.memory"]] == 0
    assert normalizer.counts[fields["vm[0].available.cpu"]] == 1
    assert normalizer.counts[fields["vm[0].available_known.cpu"]] == 0
    assert normalizer.scales[fields["vm[0].capacity.cpu"]] == 2
    assert a[fields["vm[0].capacity.cpu"]] == 1
    assert a[fields["vm[0].available.cpu"]] == b[fields["vm[0].available.cpu"]] == 0
    assert a[fields["vm[0].available_known.cpu"]] == 1
    assert b[fields["vm[0].available_known.cpu"]] == 0
    assert a[fields["vm[0].running"]] == b[fields["vm[0].running"]] == 1


def test_held_out_extremes_do_not_change_fit_or_artifact(tmp_path):
    normalizer = fit_normalizer([batch()])
    before = normalizer.manifest()
    path = tmp_path / "normalizer.json"
    normalizer.save(path)
    original = path.read_bytes()
    held_out = batch(((1e308, 1e308, 1e308),), split="test", source="held-out")
    assert normalizer.transform_batch(held_out) == ((1, 1, 1),)
    assert normalizer.manifest() == before and path.read_bytes() == original
    with pytest.raises(FrozenInstanceError):
        normalizer.maxima = (1, 1, 1)


def test_subnormal_scale_does_not_overflow():
    normalizer = fit_normalizer([batch(((5e-324, 1e-320, 2e-320),))])
    assert normalizer.transform(context(), (1e308, 1e308, 1e308)) == (1, 1, 1)
    values = normalizer.transform(context(), (0, 5e-321, 1e-320))
    assert values == pytest.approx((0, 0.5, 0.5))
    assert all(math.isfinite(v) for v in values)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_fit_rejects_evaluation_splits(split):
    with pytest.raises(ValueError, match="only train"):
        fit_normalizer([batch(), batch(split=split, source="held-out")])


def test_empty_training_duplicate_sources_and_empty_evaluation():
    with pytest.raises(ValueError, match="at least one"):
        fit_normalizer([])
    with pytest.raises(ValueError, match="at least one training state"):
        fit_normalizer([batch(())])
    with pytest.raises(ValueError, match="duplicate training source"):
        fit_normalizer([batch(), batch()])
    fitted = fit_normalizer([batch()])
    assert fitted.transform_batch(batch((), split="test", source="empty-held-out")) == ()
    with pytest.raises(ValueError, match="overlaps"):
        fitted.transform_batch(batch(split="test"))


def test_multiple_training_batches_aggregate_only_training_statistics():
    normalizer = fit_normalizer([batch(), batch(((1, 8, 4),), source="second")])
    assert normalizer.maxima == (2, 8, 8)
    assert normalizer.counts == (3, 3, 3)
    assert normalizer.manifest()["training_vector_count"] == 3
    assert len(normalizer.manifest()["training_sources"]) == 2


def test_batch_copies_rows_and_canonicalizes_provenance():
    rows = [[0, 2, 4]]
    example = TraceBatch(context(), rows, "train", "t", '{"z":2,"a":1}')
    rows[0][1] = 900
    assert example.states == ((0, 2, 4),)
    assert example.provenance_json == '{"a":1,"z":2}'


@pytest.mark.parametrize("row", [(0, 1), (-1, 2, 4), (float("nan"), 2, 4),
                                (float("inf"), 2, 4), (True, 2, 4), (10**400, 2, 4)])
def test_invalid_raw_states_rejected(row):
    with pytest.raises((TypeError, ValueError)):
        batch((row,))


def test_invalid_binary_and_unknown_placeholder_rejected():
    ctx = resource_context()
    fields = {name: index for index, (name, _) in enumerate(ctx.schema.features)}
    for name, value in (("vm[0].running", 0.5), ("vm[0].capacity.memory", 3)):
        state = list(resource_state(ctx))
        state[fields[name]] = value
        with pytest.raises(ValueError):
            TraceBatch(ctx, (state,), "train", "bad")


def test_deterministic_save_load_preserves_transform(tmp_path):
    model = fit_normalizer([batch()])
    path = tmp_path / "nested" / "normalizer.json"
    model.save(path)
    loaded = StateNormalizer.load(path, context=context())
    assert loaded == model
    assert loaded.transform(context(), (1, 2, 4)) == (0.5, 0.5, 0.5)
    original = path.read_bytes()
    loaded.save(path)
    assert path.read_bytes() == original


def rehash(payload):
    payload["sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in payload.items() if k != "sha256"}, sort_keys=True,
        allow_nan=False, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    return payload


@pytest.mark.parametrize("key,value", [("schema_version", True), ("algorithm", "other"),
    ("counts", [99, 2, 2]), ("counts", [True, 2, 2]), ("maxima", [-1, 4, 8]),
    ("scales", [1, 1, 1]), ("training_vector_count", 999), ("training_sources", [])])
def test_malformed_artifact_rejected_even_with_updated_checksum(tmp_path, key, value):
    payload = fit_normalizer([batch()]).manifest()
    payload[key] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(rehash(payload)))
    with pytest.raises((ValueError, TypeError)):
        StateNormalizer.load(path, context=context())


def test_checksum_duplicate_keys_and_nonfinite_json_rejected(tmp_path):
    payload = fit_normalizer([batch()]).manifest()
    payload["maxima"][0] += 1
    path = tmp_path / "bad.json"
    for text in (json.dumps(payload), '{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            StateNormalizer.load(path, context=context())


def test_wrong_context_rejected_during_fit_transform_and_load(tmp_path):
    fitted = fit_normalizer([batch()])
    other = replace(context(), topology_sha256="f" * 64)
    with pytest.raises(ValueError, match="incompatible"):
        fit_normalizer([batch(), TraceBatch(other, ((0, 2, 4),), "train", "other")])
    with pytest.raises(ValueError, match="incompatible"):
        fitted.transform(other, (0, 2, 4))
    path = tmp_path / "model.json"
    fitted.save(path)
    with pytest.raises(ValueError, match="incompatible"):
        StateNormalizer.load(path, context=other)
