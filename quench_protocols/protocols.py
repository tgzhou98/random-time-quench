"""Two-time and three-time quench protocol unitaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import expm


def _validate_square_matrix(name: str, H: np.ndarray) -> np.ndarray:
    H_arr = np.asarray(H, dtype=np.complex128)
    if H_arr.ndim != 2 or H_arr.shape[0] != H_arr.shape[1]:
        raise ValueError(f"{name} must be a square 2D array.")
    return H_arr


def _validate_same_dim(*matrices: np.ndarray) -> int:
    dims = {mat.shape[0] for mat in matrices}
    if len(dims) != 1:
        raise ValueError("Hamiltonians must have the same dimension.")
    return matrices[0].shape[0]


def _exp_from_eig(E: np.ndarray, V: np.ndarray, t: float) -> np.ndarray:
    phase = np.exp(-1j * E * t)
    return (V * phase) @ V.conj().T


def _extract_times(times: np.ndarray, expected: int) -> tuple[float, ...]:
    flat = np.asarray(times, dtype=float).reshape(-1)
    if flat.size != expected:
        raise ValueError(f"Sampler returned {flat.size} times, expected {expected}.")
    return tuple(float(x) for x in flat)


@dataclass
class TwoTimeProtocol:
    """Two-time protocol U(t1, t2) = exp(-i H2 t2) exp(-i H1 t1)."""

    H1: np.ndarray
    H2: np.ndarray
    assume_hermitian: bool = False

    def __post_init__(self) -> None:
        self.H1 = _validate_square_matrix("H1", self.H1)
        self.H2 = _validate_square_matrix("H2", self.H2)
        self.dim = _validate_same_dim(self.H1, self.H2)
        self._eig1: tuple[np.ndarray, np.ndarray] | None = None
        self._eig2: tuple[np.ndarray, np.ndarray] | None = None
        if self.assume_hermitian:
            self._eig1 = np.linalg.eigh(self.H1)
            self._eig2 = np.linalg.eigh(self.H2)

    def unitary(self, t1: float, t2: float) -> np.ndarray:
        """Return U(t1, t2)."""
        if self.assume_hermitian and self._eig1 and self._eig2:
            E1, V1 = self._eig1
            E2, V2 = self._eig2
            U1 = _exp_from_eig(E1, V1, t1)
            U2 = _exp_from_eig(E2, V2, t2)
        else:
            U1 = expm(-1j * self.H1 * t1)
            U2 = expm(-1j * self.H2 * t2)
        return U2 @ U1

    def sample_unitary(
        self,
        rng: np.random.Generator,
        T: float,
        sampler: Callable[[np.random.Generator, float, int | tuple[int, ...]], np.ndarray],
    ) -> np.ndarray:
        """Sample a unitary by drawing times from the provided sampler."""
        t1, t2 = _extract_times(sampler(rng, T, 2), 2)
        return self.unitary(t1, t2)


@dataclass
class ThreeTimeProtocol:
    """Three-time protocol U(t1, t2, t3) = exp(-i H3 t3) exp(-i H2 t2) exp(-i H1 t1)."""

    H1: np.ndarray
    H2: np.ndarray
    H3: np.ndarray
    assume_hermitian: bool = False

    def __post_init__(self) -> None:
        self.H1 = _validate_square_matrix("H1", self.H1)
        self.H2 = _validate_square_matrix("H2", self.H2)
        self.H3 = _validate_square_matrix("H3", self.H3)
        self.dim = _validate_same_dim(self.H1, self.H2, self.H3)
        self._eig1: tuple[np.ndarray, np.ndarray] | None = None
        self._eig2: tuple[np.ndarray, np.ndarray] | None = None
        self._eig3: tuple[np.ndarray, np.ndarray] | None = None
        if self.assume_hermitian:
            self._eig1 = np.linalg.eigh(self.H1)
            self._eig2 = np.linalg.eigh(self.H2)
            self._eig3 = np.linalg.eigh(self.H3)

    def unitary(self, t1: float, t2: float, t3: float) -> np.ndarray:
        """Return U(t1, t2, t3)."""
        if self.assume_hermitian and self._eig1 and self._eig2 and self._eig3:
            E1, V1 = self._eig1
            E2, V2 = self._eig2
            E3, V3 = self._eig3
            U1 = _exp_from_eig(E1, V1, t1)
            U2 = _exp_from_eig(E2, V2, t2)
            U3 = _exp_from_eig(E3, V3, t3)
        else:
            U1 = expm(-1j * self.H1 * t1)
            U2 = expm(-1j * self.H2 * t2)
            U3 = expm(-1j * self.H3 * t3)
        return U3 @ U2 @ U1

    def sample_unitary(
        self,
        rng: np.random.Generator,
        T: float,
        sampler: Callable[[np.random.Generator, float, int | tuple[int, ...]], np.ndarray],
    ) -> np.ndarray:
        """Sample a unitary by drawing times from the provided sampler."""
        t1, t2, t3 = _extract_times(sampler(rng, T, 3), 3)
        return self.unitary(t1, t2, t3)
