import numpy as np

from quench_protocols.frame_potential import estimate_frame_potential, estimate_frame_potential_list
from quench_protocols.protocols import TwoTimeProtocol
from quench_protocols.random_matrices import sample_gue


def _make_protocol() -> TwoTimeProtocol:
    rng = np.random.default_rng(10)
    H1 = sample_gue(rng, 4)
    H2 = sample_gue(rng, 4)
    return TwoTimeProtocol(H1, H2, assume_hermitian=True)


def test_fp_k1_nonnegative() -> None:
    protocol = _make_protocol()
    rng = np.random.default_rng(0)
    value = estimate_frame_potential(protocol, T=2.0, k=1, num_pairs=50, rng=rng)
    assert value >= 0.0


def test_fp_reproducible() -> None:
    protocol = _make_protocol()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    value1 = estimate_frame_potential(protocol, T=3.0, k=2, num_pairs=40, rng=rng1)
    value2 = estimate_frame_potential(protocol, T=3.0, k=2, num_pairs=40, rng=rng2)
    np.testing.assert_allclose(value1, value2, atol=0.0)


def test_fp_list_reproducible() -> None:
    protocol = _make_protocol()
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    ks = [1, 2, 3]
    values1 = estimate_frame_potential_list(protocol, T=2.5, k_list=ks, num_pairs=30, rng=rng1)
    values2 = estimate_frame_potential_list(protocol, T=2.5, k_list=ks, num_pairs=30, rng=rng2)
    assert values1 == values2
