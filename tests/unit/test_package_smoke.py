import numpy as np
import pytest
import simpy
import torch
import drlmots


def test_core_dependencies_import():
    assert simpy.__version__
    assert np.__version__
    assert torch.__version__
    assert drlmots.__file__ is not None


def test_torch_cpu_tensor():
    tensor = torch.tensor([1, 2, 3])
    assert tensor.tolist() == [1, 2, 3]
    assert tensor.device.type == "cpu"
