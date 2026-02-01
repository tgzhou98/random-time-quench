"""Spin-1/2 random interaction Hamiltonians for quench protocols."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def sample_spin_couplings(rng: np.random.Generator, N: int, J: float) -> np.ndarray:
    """Sample all-to-all Gaussian couplings with variance 4 J^2 / N.

    Returns a symmetric matrix with zero diagonal.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if J <= 0:
        raise ValueError("J must be positive.")

    scale = np.sqrt(4.0 * (J ** 2) / float(N))
    couplings = np.zeros((N, N), dtype=float)
    iu = np.triu_indices(N, k=1)
    couplings[iu] = rng.normal(0.0, scale, size=iu[0].size)
    couplings = couplings + couplings.T
    return couplings


def _pauli_matrices() -> dict[str, np.ndarray]:
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return {"x": sx, "y": sy, "z": sz}


def _site_operator(N: int, index: int, op: np.ndarray) -> np.ndarray:
    factors = []
    for site in range(N):
        factors.append(op if site == index else np.eye(2, dtype=np.complex128))
    out = factors[0]
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out


def build_spin_hamiltonian(J: np.ndarray, xi: Iterable[float]) -> np.ndarray:
    """Build H = sum_{i<j,alpha} J_ij * xi_alpha * S_i^alpha S_j^alpha."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be a square matrix.")
    if not np.allclose(J, J.T):
        raise ValueError("J must be symmetric.")
    if not np.allclose(np.diag(J), 0.0):
        raise ValueError("J diagonal must be zero.")

    xi_vals = tuple(float(x) for x in xi)
    if len(xi_vals) != 3:
        raise ValueError("xi must have three entries.")

    N = J.shape[0]
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)

    paulis = _pauli_matrices()
    s_ops = {key: val / 2.0 for key, val in paulis.items()}

    site_ops = {
        (i, key): _site_operator(N, i, op)
        for key, op in s_ops.items()
        for i in range(N)
    }

    for i in range(N):
        for j in range(i + 1, N):
            if J[i, j] == 0.0:
                continue
            for (key, xi_val) in zip(("x", "y", "z"), xi_vals):
                if xi_val == 0.0:
                    continue
                term = site_ops[(i, key)] @ site_ops[(j, key)]
                H += (J[i, j] * xi_val) * term
    return H


def sample_spin_hamiltonian(
    rng: np.random.Generator,
    N: int,
    J: float,
    xi: Iterable[float] = (1.0, 1.0, -2.0),
) -> np.ndarray:
    """Sample couplings and return the corresponding Hamiltonian."""
    couplings = sample_spin_couplings(rng, N=N, J=J)
    return build_spin_hamiltonian(couplings, xi=xi)
