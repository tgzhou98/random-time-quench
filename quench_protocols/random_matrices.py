"""Random matrix generators for quench protocols."""

from __future__ import annotations

import numpy as np


def sample_gue(rng: np.random.Generator, D: int, scale: str | None = "wigner") -> np.ndarray:
    """Sample a GUE Hermitian matrix with optional Wigner scaling."""
    if D <= 0:
        raise ValueError("D must be a positive integer.")
    real = rng.normal(0.0, 1.0, size=(D, D))
    imag = rng.normal(0.0, 1.0, size=(D, D))
    X = real + 1j * imag
    H = (X + X.conj().T) / 2.0
    if scale == "wigner":
        H = H / np.sqrt(D)
    elif scale is None:
        pass
    else:
        raise ValueError(f"Unknown scale: {scale}")
    return np.asarray(H, dtype=np.complex128)
