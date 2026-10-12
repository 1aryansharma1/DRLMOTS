
from copy import deepcopy
from pathlib import Path

import pytest

from drlmots.config import ConfigError, load_config, validate_config, validate_paper_vm_count


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def test_default_config_loads():
    config = load_config(CONFIG_PATH)

    assert config["schema_version"] == 1
    assert config["dqn"]["gamma"] == 0.95
    assert config["workload"]["task_size_kb_range"] == [20, 150]


def test_unresolved_parameters_remain_explicit():
    config = load_config(CONFIG_PATH)

    assert config["seed"] is None
    assert config["dqn"]["epsilon_start"] is None
    assert config["dqn"]["learning_rate"] is None
    assert config["unresolved"]


def test_invalid_task_count_is_rejected():
    config = load_config(CONFIG_PATH)
    invalid = deepcopy(config)
    invalid["workload"]["task_count"] = "many"

    with pytest.raises(ConfigError, match="task_count"):
        validate_config(invalid)


def test_invalid_task_size_range_is_rejected():
    config = load_config(CONFIG_PATH)
    invalid = deepcopy(config)
    invalid["workload"]["task_size_kb_range"] = [150, 20]

    with pytest.raises(ConfigError, match="task_size_kb_range"):
        validate_config(invalid)


def test_missing_required_section_is_rejected():
    config = load_config(CONFIG_PATH)
    invalid = deepcopy(config)
    del invalid["dqn"]

    with pytest.raises(ConfigError, match="missing keys"):
        validate_config(invalid)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_task_size_is_rejected(value):
    config = load_config(CONFIG_PATH)
    config["workload"]["task_size_kb_range"] = [20, value]
    with pytest.raises(ConfigError, match="task_size_kb_range"):
        validate_config(config)


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf"), 0, -1])
@pytest.mark.parametrize("field", ["replay_capacity", "batch_size", "episodes", "target_sync_interval"])
def test_dqn_counts_require_positive_integers(field, value):
    config = load_config(CONFIG_PATH)
    config["dqn"][field] = value
    with pytest.raises(ConfigError, match=field):
        validate_config(config)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0, -1])
def test_learning_rate_must_be_finite_and_positive(value):
    config = load_config(CONFIG_PATH)
    config["dqn"]["learning_rate"] = value
    with pytest.raises(ConfigError, match="learning_rate"):
        validate_config(config)


def test_inconsistent_exploration_and_replay_parameters():
    config = load_config(CONFIG_PATH)
    config["dqn"]["epsilon_start"] = 0
    with pytest.raises(ConfigError, match="epsilon_min"):
        validate_config(config)
    config["dqn"]["epsilon_start"] = None
    config["dqn"]["batch_size"] = 10
    config["dqn"]["replay_capacity"] = 5
    with pytest.raises(ConfigError, match="replay_capacity"):
        validate_config(config)


@pytest.mark.parametrize("count", [20, 200])
def test_paper_vm_range_boundaries(count):
    assert validate_paper_vm_count(count) == count


@pytest.mark.parametrize("count", [None, True, 20.5, 19, 201])
def test_paper_vm_range_rejects_invalid_counts(count):
    with pytest.raises(ConfigError):
        validate_paper_vm_count(count)


@pytest.mark.parametrize("count", [20, 200])
def test_paper_experiment_config_accepts_vm_boundaries(count):
    config = load_config(CONFIG_PATH)
    config["paper_experiment"]["vm_count"] = count
    validate_config(config)


@pytest.mark.parametrize("count", [19, 201, True, 20.5])
def test_paper_experiment_config_rejects_vm_counts(count):
    config = load_config(CONFIG_PATH)
    config["paper_experiment"]["vm_count"] = count
    with pytest.raises(ConfigError, match="VM count"):
        validate_config(config)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
@pytest.mark.parametrize("field", [
    "host_storage_tb", "host_memory_gb", "host_bandwidth_reported", "vm_bandwidth_reported"
])
def test_invalid_paper_resource_settings(field, value):
    config = load_config(CONFIG_PATH)
    config["paper_experiment"][field] = value
    with pytest.raises(ConfigError, match=field):
        validate_config(config)

