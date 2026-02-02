"""Complex-fermion SYK (q=4) Hamiltonian construction."""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Dict, Tuple

import numpy as np

CouplingDict = Dict[Tuple[int, int, int, int], complex]


def generate_couplings(rng: np.random.Generator, N: int, J: float) -> CouplingDict:
    """Sample complex Gaussian SYK couplings in a canonical index set."""
    if N <= 0 or N % 2 != 0:
        raise ValueError("N must be a positive even integer.")
    if N < 4:
        raise ValueError("N must be at least 4 for q=4 SYK.")
    if J <= 0:
        raise ValueError("J must be positive.")

    variance = 6.0 * (J ** 2) / (N ** 3)
    sigma = np.sqrt(0.5 * variance)
    couplings: CouplingDict = {}
    for i, j in combinations(range(N), 2):
        for k, l in combinations(range(N), 2):
            if len({i, j, k, l}) < 4:
                continue
            if (k, l) < (i, j):
                continue
            real = rng.normal(0.0, sigma)
            imag = rng.normal(0.0, sigma)
            couplings[(i, j, k, l)] = real + 1j * imag
    return couplings


def _canonical_pair(i: int, j: int) -> tuple[int, int, int]:
    if i == j:
        return i, j, 0
    if i < j:
        return i, j, 1
    return j, i, -1


def _canonical_key(
    i: int, j: int, k: int, l: int
) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    i1, j1, s1 = _canonical_pair(i, j)
    k1, l1, s2 = _canonical_pair(k, l)
    sign = s1 * s2
    if (k1, l1) < (i1, j1):
        return (k1, l1, s2), (i1, j1, s1), sign
    return (i1, j1, s1), (k1, l1, s2), sign


def lookup_coupling(couplings: CouplingDict, i: int, j: int, k: int, l: int) -> complex:
    """Lookup J_{ijkl} with antisymmetry in i<->j and k<->l."""
    if len({i, j, k, l}) < 4:
        return 0.0 + 0.0j
    (i1, j1, s1), (k1, l1, s2), sign = _canonical_key(i, j, k, l)
    value = couplings.get((i1, j1, k1, l1))
    if value is None:
        return 0.0 + 0.0j
    return sign * value


def _apply_annihilation(state: int, idx: int) -> tuple[int | None, int]:
    if ((state >> idx) & 1) == 0:
        return None, 0
    mask = (1 << idx) - 1
    parity = (state & mask).bit_count() % 2
    sign = -1 if parity else 1
    return state & ~(1 << idx), sign


def _apply_creation(state: int, idx: int) -> tuple[int | None, int]:
    if ((state >> idx) & 1) == 1:
        return None, 0
    mask = (1 << idx) - 1
    parity = (state & mask).bit_count() % 2
    sign = -1 if parity else 1
    return state | (1 << idx), sign


def build_hamiltonian(couplings: CouplingDict, N: int, charge_sector: str = "half") -> np.ndarray:
    """Build the dense Hamiltonian in the half-filled charge sector."""
    if N <= 0 or N % 2 != 0:
        raise ValueError("N must be a positive even integer.")
    if charge_sector != "half":
        raise ValueError("Only half-filling is supported.")
    if N < 4:
        raise ValueError("N must be at least 4 for q=4 SYK.")

    n_particles = N // 2
    basis = [state for state in range(1 << N) if state.bit_count() == n_particles]
    index = {state: idx for idx, state in enumerate(basis)}
    dim = comb(N, n_particles)
    H = np.zeros((dim, dim), dtype=np.complex128)

    for (i, j, k, l), value in couplings.items():
        if len({i, j, k, l}) < 4:
            continue
        for col, state in enumerate(basis):
            st, s1 = _apply_annihilation(state, l)
            if st is None:
                continue
            st, s2 = _apply_annihilation(st, k)
            if st is None:
                continue
            st, s3 = _apply_creation(st, j)
            if st is None:
                continue
            st, s4 = _apply_creation(st, i)
            if st is None:
                continue
            row = index[st]
            sign = s1 * s2 * s3 * s4
            H[row, col] += value * sign

    return H + H.conj().T


def sample_syk_hamiltonian(
    rng: np.random.Generator,
    N: int,
    J: float,
    charge_sector: str = "half",
) -> np.ndarray:
    """Sample couplings and build the corresponding SYK Hamiltonian."""
    couplings = generate_couplings(rng, N=N, J=J)
    return build_hamiltonian(couplings, N=N, charge_sector=charge_sector)
