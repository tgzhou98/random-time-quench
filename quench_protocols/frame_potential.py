"""Monte Carlo estimators for the frame potential.

Example:
    import numpy as np
    from quench_protocols.protocols import TwoTimeProtocol
    from quench_protocols.frame_potential import estimate_frame_potential

    rng = np.random.default_rng(0)
    prot = TwoTimeProtocol(H1, H2, assume_hermitian=True)
    F2 = estimate_frame_potential(prot, T=10.0, k=2, num_pairs=2000, rng=rng)
    print(F2)
"""

from __future__ import annotations

import math
from typing import Callable, Protocol, Sequence

import numpy as np

from .distributions import get_time_sampler


class _Protocol(Protocol):
    def sample_unitary(
        self,
        rng: np.random.Generator,
        T: float,
        sampler: Callable[[np.random.Generator, float, int | tuple[int, ...]], np.ndarray],
    ) -> np.ndarray:
        ...


def trace_overlap_power(U: np.ndarray, V: np.ndarray, k: int) -> float:
    """Return |Tr(U^† V)|^(2k)."""
    if k < 1:
        raise ValueError("k must be a positive integer.")
    trace_val = np.trace(U.conj().T @ V)
    abs_trace = float(np.abs(trace_val))
    if abs_trace == 0.0:
        return 0.0
    exponent = (2.0 * k) * math.log(abs_trace)
    max_log = math.log(np.finfo(float).max)
    if exponent < max_log:
        return float(abs_trace ** (2.0 * k))
    return float(math.exp(exponent))


def estimate_frame_potential(
    protocol: _Protocol,
    T: float,
    k: int,
    num_pairs: int,
    rng: np.random.Generator,
    sampler_name: str = "uniform",
    sampler_kwargs: dict[str, object] | None = None,
    t0: float | None = None,
) -> float:
    """Estimate the k-th frame potential by Monte Carlo sampling."""
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if k < 1:
        raise ValueError("k must be a positive integer.")

    merged_kwargs: dict[str, object] = dict(sampler_kwargs or {})
    if t0 is not None:
        merged_kwargs["t0"] = float(t0)

    sampler = get_time_sampler(sampler_name, **merged_kwargs)
    values = np.empty(num_pairs, dtype=float)
    for idx in range(num_pairs):
        U = protocol.sample_unitary(rng, T, sampler)
        V = protocol.sample_unitary(rng, T, sampler)
        values[idx] = trace_overlap_power(U, V, k)
    estimate = float(np.mean(values))
    print("Max value of FP:", values.max())
    print("Std of FP:", np.std(values))
    print("Estimate of FP:", estimate)
    return estimate


def estimate_frame_potential_list(
    protocol: _Protocol,
    T: float,
    k_list: Sequence[int],
    num_pairs: int,
    rng: np.random.Generator,
    sampler_name: str = "uniform",
    sampler_kwargs: dict[str, object] | None = None,
    t0: float | None = None,
) -> dict[int, float]:
    """Estimate frame potentials for multiple k values using shared samples."""
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if not k_list:
        raise ValueError("k_list must be non-empty.")

    ks = [int(k) for k in k_list]
    if any(k < 1 for k in ks):
        raise ValueError("k_list entries must be positive integers.")

    merged_kwargs: dict[str, object] = dict(sampler_kwargs or {})
    if t0 is not None:
        merged_kwargs["t0"] = float(t0)

    sampler = get_time_sampler(sampler_name, **merged_kwargs)
    abs_traces = np.empty(num_pairs, dtype=float)
    for idx in range(num_pairs):
        U = protocol.sample_unitary(rng, T, sampler)
        V = protocol.sample_unitary(rng, T, sampler)
        abs_traces[idx] = float(np.abs(np.trace(U.conj().T @ V)))

    log_abs = np.full_like(abs_traces, -np.inf)
    np.log(abs_traces, out=log_abs, where=abs_traces > 0)

    results: dict[int, float] = {}
    for k in ks:
        exponent = (2.0 * k) * log_abs
        results[k] = float(np.exp(exponent).mean())
    return results
