from dataclasses import replace
import json
import random

import pytest
import torch

from drlmots.normalization import NormalizationContext, TraceBatch, _digest, capture_batch, fit_normalizer
from drlmots.q_network import MaskedSelector, NetworkConfig, NetworkContext, QNetwork
from drlmots.reward import RewardConfig
from drlmots.run_config import load_run_config
from drlmots.topology import Topology
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def artifacts():
    batch, _ = capture_batch(load_run_config(ROOT / "configs/normalization_train.yaml"), split="train")
    normalizer = fit_normalizer([batch])
    reward = RewardConfig(10, 1)
    return normalizer, reward, NetworkContext.from_artifacts(normalizer, reward)


@pytest.mark.parametrize("size", [0, 1, 7])
def test_cpu_float32_shapes_and_parameter_count(artifacts, size):
    _, _, context = artifacts
    net = QNetwork(context)
    states = torch.zeros((size, 47), dtype=torch.float32, device="cpu")
    result = net(states, context=context)
    assert result.shape == (size, 3)
    assert result.dtype == torch.float32 and result.device.type == "cpu"
    assert torch.isfinite(result).all()
    assert sum(p.numel() for p in net.parameters()) == 5103
    assert [layer.out_features for layer in net.layers if isinstance(layer, torch.nn.Linear)] == [50, 50, 3]


def test_hand_forward_and_gradient_through_all_layers(artifacts):
    _, _, context = artifacts
    net = QNetwork(context, NetworkConfig((1, 1), 0))
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(1)
        net.layers[-1].weight.copy_(torch.tensor([[1], [2], [-1]], dtype=torch.float32))
    result = net(torch.zeros((1, 47)), context=context)
    assert result.tolist() == [[3, 5, -1]]  # 0 -> 1 -> 2 -> [3,5,-1], no output ReLU
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_seed_repeatability_isolation_and_caller_default_dtype(artifacts):
    _, _, context = artifacts
    cpu_state, python_state = torch.random.get_rng_state().clone(), random.getstate()
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        a = QNetwork(context)
        b = QNetwork(context)
        c = QNetwork(context, NetworkConfig(seed=43))
    finally:
        torch.set_default_dtype(dtype)
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    assert python_state == random.getstate()
    assert all(torch.equal(a.state_dict()[k], v) and v.dtype == torch.float32 for k, v in b.state_dict().items())
    assert any(not torch.equal(a.state_dict()[k], v) for k, v in c.state_dict().items())


@pytest.mark.parametrize("states", [
    [[0] * 47], torch.zeros(47), torch.zeros((1, 46)), torch.zeros((1, 1, 47)),
    torch.zeros((1, 47), dtype=torch.float64), torch.zeros((1, 47), dtype=torch.int64),
    torch.full((1, 47), float("nan")), torch.full((1, 47), float("inf")),
    torch.full((1, 47), -0.1), torch.full((1, 47), 1.01),
    torch.empty((1, 47), device="meta"),
])
def test_bad_input_tensor_rejected(artifacts, states):
    with pytest.raises((ValueError, TypeError)):
        QNetwork(artifacts[2])(states, context=artifacts[2])


@pytest.mark.parametrize("change", ["normalizer", "reward", "topology", "vm_ids", "units"])
def test_same_dimension_context_mismatch_is_rejected(artifacts, tmp_path, change):
    normalizer, reward, context = artifacts
    if change == "normalizer":
        altered = replace(context, normalizer_sha256="0" * 64)
    elif change == "reward":
        altered = NetworkContext.from_artifacts(normalizer, replace(reward, latency_scale_ms=11))
    elif change == "topology":
        altered = replace(context, normalization=replace(context.normalization, topology_sha256="0" * 64))
    else:
        schema = context.normalization.schema
        schema = (replace(schema, vm_ids=("other-a", "other-b", "other-c")) if change == "vm_ids"
                  else replace(schema, resource_units=tuple((k, v + " changed") for k, v in schema.resource_units)))
        altered = replace(context, normalization=replace(context.normalization, schema=schema))
    assert altered.normalization.schema.dimension == 47
    net = QNetwork(context)
    with pytest.raises(ValueError, match="incompatible"):
        net(torch.zeros((1, 47)), context=altered)
    path = tmp_path / "network.json"
    net.save(path)
    with pytest.raises(ValueError, match="incompatible"):
        QNetwork.load(path, context=altered)


def test_nonfinite_or_changed_dtype_parameters_rejected(artifacts, tmp_path):
    net = QNetwork(artifacts[2])
    with torch.no_grad():
        net.layers[0].weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="parameters"):
        net(torch.zeros((1, 47)), context=artifacts[2])
    with pytest.raises(ValueError, match="weights"):
        net.save(tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
    net = QNetwork(artifacts[2]).double()
    with pytest.raises(ValueError, match="parameters"):
        net(torch.zeros((1, 47)), context=artifacts[2])


def test_finite_weights_with_overflowing_outputs_rejected(artifacts):
    net = QNetwork(artifacts[2])
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="Q outputs"):
        net(torch.ones((1, 47)), context=artifacts[2])


@pytest.mark.parametrize("config", [
    {"hidden_widths": ()}, {"hidden_widths": (1, 0)}, {"hidden_widths": (True, 2)},
    {"hidden_widths": (1, 2, 3)}, {"seed": True}, {"seed": -1}, {"seed": 2**63},
])
def test_invalid_network_configuration(config):
    with pytest.raises(ValueError):
        NetworkConfig(**config)


def test_zero_vm_network_rejected():
    context = NormalizationContext.from_topology(Topology((), ()))
    normalizer = fit_normalizer([TraceBatch(context, ((0, 1, 2),), "train", "empty-topology")])
    with pytest.raises(ValueError, match="at least one VM"):
        NetworkContext.from_artifacts(normalizer, RewardConfig(10, 1))


@pytest.mark.parametrize("mask,expected", [
    ((True, True, True), 0), ((False, True, True), 1), ((False, False, True), 2),
    ((False, False, False), None),
])
def test_greedy_uses_only_legal_actions_and_stable_ties(mask, expected):
    q = torch.tensor([100, 3, 3], dtype=torch.float32)
    assert MaskedSelector(42).select(q, mask, epsilon=0) == expected


def test_exploration_seed_legal_coverage_and_global_rng_isolation():
    before = random.getstate()
    a, b, c = MaskedSelector(42), MaskedSelector(42), MaskedSelector(43)
    q, mask = torch.tensor([100, -4, -5], dtype=torch.float32), (False, True, True)
    left = [a.select(q, mask, epsilon=1) for _ in range(100)]
    assert left == [b.select(q, mask, epsilon=1) for _ in range(100)]
    assert left != [c.select(q, mask, epsilon=1) for _ in range(100)]
    assert set(left) == {1, 2}
    assert random.getstate() == before


def test_mixed_epsilon_and_forced_or_greedy_calls_do_not_consume_rng():
    a, b = MaskedSelector(42), MaskedSelector(42)
    q = torch.tensor([1, 2, 3], dtype=torch.float32)
    for _ in range(10):
        assert a.select(q, (False, False, False), epsilon=0.5) is None
        assert a.select(q, (True, True, True), epsilon=0) == 2
    left = [a.select(q, (True, True, True), epsilon=0.5) for _ in range(100)]
    assert left == [b.select(q, (True, True, True), epsilon=0.5) for _ in range(100)]
    assert set(left) == {0, 1, 2}


@pytest.mark.parametrize("epsilon", [True, -0.01, 1.01, float("nan"), float("inf"), "0"])
def test_bad_epsilon_rejected_even_for_forced_rejection(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        MaskedSelector(42).select(torch.zeros(3), (False,) * 3, epsilon=epsilon)


@pytest.mark.parametrize("q,mask", [
    (torch.zeros(3), (True, False)), (torch.zeros(3), (1, 0, 0)),
    (torch.zeros((1, 3)), (True,) * 3), (torch.zeros(3, dtype=torch.float64), (True,) * 3),
    (torch.tensor([0, float("nan"), 0]), (True, False, True)),
    ([1, 2, 3], (True,) * 3),
])
def test_bad_selection_inputs_rejected(q, mask):
    with pytest.raises((ValueError, TypeError)):
        MaskedSelector(42).select(q, mask, epsilon=0)


def test_checkpoint_exact_round_trip_and_no_rng_mutation(artifacts, tmp_path):
    context = artifacts[2]
    net = QNetwork(context, NetworkConfig((7, 9), 99))
    states = torch.linspace(0, 1, 47).repeat(3, 1)
    path = tmp_path / "nested/network.json"
    checksum = net.save(path)
    before = torch.random.get_rng_state().clone()
    restored = QNetwork.load(path, context=context)
    assert torch.equal(before, torch.random.get_rng_state())
    assert restored.config == net.config
    assert torch.equal(net(states, context=context), restored(states, context=context))
    assert json.loads(path.read_text())["sha256"] == checksum
    assert restored.save(tmp_path / "copy.json") == checksum
    assert all(torch.equal(net.state_dict()[k], v) for k, v in restored.state_dict().items())


@pytest.mark.parametrize("mutation", ["version", "fields", "architecture", "name", "shape", "bool", "overflow", "runtime"])
def test_rechecksummed_malformed_checkpoints_rejected(artifacts, tmp_path, mutation):
    path = tmp_path / "network.json"
    QNetwork(artifacts[2]).save(path)
    data = json.loads(path.read_text())
    name = "layers.0.weight"
    if mutation == "version":
        data["schema_version"] = True
    elif mutation == "fields":
        data["extra"] = 0
    elif mutation == "architecture":
        data["architecture"]["activation"] = "sigmoid"
    elif mutation == "name":
        del data["weights"][name]
    elif mutation == "shape":
        data["weights"][name][0].pop()
    elif mutation == "bool":
        data["weights"][name][0][0] = True
    elif mutation == "overflow":
        data["weights"][name][0][0] = 1e308
    else:
        data["runtime"] = {"torch": 42}
    data["sha256"] = _digest({k: v for k, v in data.items() if k != "sha256"})
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)):
        QNetwork.load(path, context=artifacts[2])


def test_corrupted_checksum_duplicate_keys_and_nonfinite_json(artifacts, tmp_path):
    path = tmp_path / "bad.json"
    QNetwork(artifacts[2]).save(path)
    data = json.loads(path.read_text())
    data["weights"]["layers.0.bias"][0] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        QNetwork.load(path, context=artifacts[2])
    for text in ('{"schema_version":1,"schema_version":1}', '{"x":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            QNetwork.load(path, context=artifacts[2])
