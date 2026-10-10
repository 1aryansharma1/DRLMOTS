
from copy import deepcopy
from pathlib import Path

import pytest

from drlmots.config import ConfigError, load_config, validate_config


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

