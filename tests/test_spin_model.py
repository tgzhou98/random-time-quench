import numpy as np

from quench_protocols.spin_model import build_spin_hamiltonian, sample_spin_couplings


def test_spin_couplings_reproducible_and_symmetric() -> None:
    rng = np.random.default_rng(123)
    J1 = sample_spin_couplings(rng, N=6, J=1.5)
    rng = np.random.default_rng(123)
    J2 = sample_spin_couplings(rng, N=6, J=1.5)
    assert np.allclose(J1, J2)
    assert np.allclose(J1, J1.T)
    assert np.allclose(np.diag(J1), 0.0)


def test_spin_hamiltonian_shape_and_hermitian() -> None:
    rng = np.random.default_rng(0)
    J = sample_spin_couplings(rng, N=3, J=1.0)
    H = build_spin_hamiltonian(J, xi=(1.0, 1.0, -2.0))
    assert H.shape == (2**3, 2**3)
    assert np.allclose(H, H.conj().T)
