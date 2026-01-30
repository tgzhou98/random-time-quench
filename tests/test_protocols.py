import numpy as np
import pytest

from quench_protocols.distributions import get_time_sampler, sample_times_hann, sample_times_kaiser
from quench_protocols.protocols import ThreeTimeProtocol, TwoTimeProtocol
from quench_protocols.random_matrices import sample_gue


def test_two_time_unitary_is_unitary() -> None:
    rng = np.random.default_rng(0)
    H1 = sample_gue(rng, 4)
    H2 = sample_gue(rng, 4)
    protocol = TwoTimeProtocol(H1, H2, assume_hermitian=True)
    U = protocol.unitary(0.3, 0.7)
    ident = np.eye(4, dtype=np.complex128)
    np.testing.assert_allclose(U.conj().T @ U, ident, atol=1e-10)


def test_three_time_unitary_is_unitary() -> None:
    rng = np.random.default_rng(1)
    H1 = sample_gue(rng, 6)
    H2 = sample_gue(rng, 6)
    H3 = sample_gue(rng, 6)
    protocol = ThreeTimeProtocol(H1, H2, H3, assume_hermitian=True)
    U = protocol.unitary(0.2, 0.4, 0.6)
    ident = np.eye(6, dtype=np.complex128)
    np.testing.assert_allclose(U.conj().T @ U, ident, atol=1e-10)


def test_dimension_mismatch_raises() -> None:
    rng = np.random.default_rng(2)
    H1 = sample_gue(rng, 4)
    H2 = sample_gue(rng, 5)
    with pytest.raises(ValueError):
        TwoTimeProtocol(H1, H2)


def test_sampling_is_reproducible_with_seed() -> None:
    rng = np.random.default_rng(3)
    H1 = sample_gue(rng, 4)
    H2 = sample_gue(rng, 4)
    protocol = TwoTimeProtocol(H1, H2, assume_hermitian=True)
    sampler = get_time_sampler("uniform")

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    U1 = protocol.sample_unitary(rng1, 1.5, sampler)
    U2 = protocol.sample_unitary(rng2, 1.5, sampler)
    np.testing.assert_allclose(U1, U2, atol=1e-12)


def test_hann_sampler_bounds() -> None:
    rng = np.random.default_rng(0)
    samples = sample_times_hann(rng, 2.0, size=1000)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 2.0)


def test_kaiser_sampler_bounds() -> None:
    rng = np.random.default_rng(1)
    samples = sample_times_kaiser(rng, 3.0, size=1000, beta=6.0)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 3.0)
