from __future__ import annotations

import pytest

from src.training.train import make_synthetic_records


@pytest.fixture()
def synthetic_records():
    return make_synthetic_records(n_structures=6, n_timesteps=16, seed=42)


@pytest.fixture()
def small_records():
    return make_synthetic_records(n_structures=2, n_timesteps=8, seed=1)
