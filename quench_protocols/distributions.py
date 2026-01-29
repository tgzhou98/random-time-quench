"""Time sampling distributions for quench protocols."""

from __future__ import annotations

from typing import Callable

import numpy as np


def sample_times_uniform(rng: np.random.Generator, T: float, size: int | tuple[int, ...]) -> np.ndarray:
    """Sample i.i.d. times uniformly from [0, T]."""
    if T < 0:
        raise ValueError("T must be non-negative.")
    return rng.uniform(0.0, float(T), size=size)


def get_time_sampler(name: str, **kwargs: object) -> Callable[[np.random.Generator, float, int | tuple[int, ...]], np.ndarray]:
    """Return a time sampler by name.

    Supported names: "uniform".
    """
    name_lower = name.lower()
    if name_lower == "uniform":
        def _sampler(rng: np.random.Generator, T: float, size: int | tuple[int, ...]) -> np.ndarray:
            return sample_times_uniform(rng, T, size)

        return _sampler
    raise ValueError(f"Unknown sampler name: {name}")
