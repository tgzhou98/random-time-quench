"""Time sampling distributions for quench protocols."""

from __future__ import annotations

from typing import Callable

import numpy as np


def sample_times_uniform(
    rng: np.random.Generator,
    T: float,
    size: int | tuple[int, ...],
    t0: float = 0.0,
) -> np.ndarray:
    """Sample i.i.d. times uniformly from [t0, t0 + T]."""
    if T < 0:
        raise ValueError("T must be non-negative.")
    t0_f = float(t0)
    return rng.uniform(t0_f, t0_f + float(T), size=size)


def sample_times_hann(rng: np.random.Generator, T: float, size: int | tuple[int, ...]) -> np.ndarray:
    """Sample i.i.d. times from the Hann (raised cosine) window on [0, T]."""
    if T <= 0:
        raise ValueError("T must be positive for Hann sampling.")

    if isinstance(size, tuple):
        total = int(np.prod(size))
    else:
        total = int(size)
        size = (total,)

    out = np.empty(total, dtype=float)
    filled = 0
    while filled < total:
        batch = max(1024, total - filled)
        u = rng.uniform(0.0, float(T), size=batch)
        r = rng.uniform(0.0, 1.0, size=batch)
        accept = r < (1.0 - np.cos(2.0 * np.pi * u / T)) / 2.0
        if not np.any(accept):
            continue
        accepted = u[accept]
        take = min(accepted.size, total - filled)
        out[filled : filled + take] = accepted[:take]
        filled += take
    return out.reshape(size)


def sample_times_kaiser(
    rng: np.random.Generator,
    T: float,
    size: int | tuple[int, ...],
    beta: float = 8.0,
) -> np.ndarray:
    """Sample i.i.d. times from a Kaiser window on [0, T]."""
    if T <= 0:
        raise ValueError("T must be positive for Kaiser sampling.")
    if beta < 0:
        raise ValueError("beta must be non-negative for Kaiser sampling.")

    if isinstance(size, tuple):
        total = int(np.prod(size))
    else:
        total = int(size)
        size = (total,)

    denom = np.i0(beta) if beta != 0 else 1.0
    out = np.empty(total, dtype=float)
    filled = 0
    while filled < total:
        batch = max(1024, total - filled)
        u = rng.uniform(0.0, float(T), size=batch)
        x = (2.0 * u / T) - 1.0
        x = np.clip(x, -1.0, 1.0)
        w = np.i0(beta * np.sqrt(1.0 - x * x)) / denom
        r = rng.uniform(0.0, 1.0, size=batch)
        accept = r < w
        if not np.any(accept):
            continue
        accepted = u[accept]
        take = min(accepted.size, total - filled)
        out[filled : filled + take] = accepted[:take]
        filled += take
    return out.reshape(size)


def get_time_sampler(name: str, **kwargs: object) -> Callable[[np.random.Generator, float, int | tuple[int, ...]], np.ndarray]:
    """Return a time sampler by name.

    Supported names: "uniform", "hann", "kaiser".
    """
    name_lower = name.lower()
    if name_lower == "uniform":
        # For a shifted uniform distribution, pass t0 (or T0) in sampler_kwargs.
        t0 = float(kwargs.get("t0", kwargs.get("T0", 0.0)))

        def _sampler(rng: np.random.Generator, T: float, size: int | tuple[int, ...]) -> np.ndarray:
            return sample_times_uniform(rng, T, size, t0=t0)

        return _sampler
    if name_lower == "hann":
        def _sampler(rng: np.random.Generator, T: float, size: int | tuple[int, ...]) -> np.ndarray:
            return sample_times_hann(rng, T, size)

        return _sampler
    if name_lower == "kaiser":
        beta = float(kwargs.get("beta", 8.0))

        def _sampler(rng: np.random.Generator, T: float, size: int | tuple[int, ...]) -> np.ndarray:
            return sample_times_kaiser(rng, T, size, beta=beta)

        return _sampler
    raise ValueError(f"Unknown sampler name: {name}")
