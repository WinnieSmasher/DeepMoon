from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch

from deepmoon.utils.seed import is_deterministic_experiment, seed_everything


def test_seed_everything_makes_python_numpy_and_torch_repeatable() -> None:
    seed_everything(2026, deterministic=True)
    first_python = random.random()
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    seed_everything(2026, deterministic=True)
    second_python = random.random()
    second_numpy = np.random.rand(3)
    second_torch = torch.rand(3)

    assert first_python == second_python
    assert np.allclose(first_numpy, second_numpy)
    assert torch.allclose(first_torch, second_torch)


def test_is_deterministic_experiment_reads_flag_from_config() -> None:
    config = SimpleNamespace(experiment=SimpleNamespace(deterministic=True))
    assert is_deterministic_experiment(config) is True


def test_is_deterministic_experiment_defaults_to_false() -> None:
    assert is_deterministic_experiment(SimpleNamespace()) is False
