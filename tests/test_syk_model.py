import numpy as np

from quench_protocols.syk_model import (
    build_hamiltonian,
    generate_couplings,
    lookup_coupling,
    sample_syk_hamiltonian,
)


def test_syk_couplings_reproducible() -> None:
    rng = np.random.default_rng(0)
    J1 = generate_couplings(rng, N=6, J=1.0)
    rng = np.random.default_rng(0)
    J2 = generate_couplings(rng, N=6, J=1.0)
    assert J1 == J2


def test_syk_antisymmetry_lookup() -> None:
    rng = np.random.default_rng(1)
    J = generate_couplings(rng, N=6, J=1.0)
    val = lookup_coupling(J, 0, 1, 2, 3)
    assert np.allclose(lookup_coupling(J, 1, 0, 2, 3), -val)
    assert np.allclose(lookup_coupling(J, 0, 1, 3, 2), -val)
    assert np.allclose(lookup_coupling(J, 1, 0, 3, 2), val)


def test_syk_hamiltonian_shape_and_hermitian() -> None:
    rng = np.random.default_rng(2)
    J = generate_couplings(rng, N=4, J=1.0)
    H = build_hamiltonian(J, N=4, charge_sector="half")
    assert H.shape == (6, 6)
    assert np.allclose(H, H.conj().T)


def test_sample_syk_hamiltonian_matches_components() -> None:
    rng = np.random.default_rng(3)
    H1 = sample_syk_hamiltonian(rng, N=4, J=1.0, charge_sector="half")
    rng = np.random.default_rng(3)
    couplings = generate_couplings(rng, N=4, J=1.0)
    H2 = build_hamiltonian(couplings, N=4, charge_sector="half")
    assert np.allclose(H1, H2)
