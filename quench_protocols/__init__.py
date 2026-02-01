"""Quench protocols and frame potential estimators."""

from .distributions import (
    get_time_sampler,
    sample_times_hann,
    sample_times_kaiser,
    sample_times_uniform,
)
from .frame_potential import (
    estimate_frame_potential,
    estimate_frame_potential_list,
    trace_overlap_power,
)
from .protocols import ThreeTimeProtocol, TwoTimeProtocol
from .random_matrices import sample_gue
from .spin_model import (
    build_spin_hamiltonian,
    sample_spin_couplings,
    sample_spin_hamiltonian,
)

__all__ = [
    "ThreeTimeProtocol",
    "TwoTimeProtocol",
    "estimate_frame_potential",
    "estimate_frame_potential_list",
    "get_time_sampler",
    "sample_times_hann",
    "sample_times_kaiser",
    "sample_gue",
    "sample_times_uniform",
    "trace_overlap_power",
    "build_spin_hamiltonian",
    "sample_spin_couplings",
    "sample_spin_hamiltonian",
]
