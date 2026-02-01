"""Command line interface for quick frame potential estimates."""

from __future__ import annotations

import argparse

import numpy as np

from .frame_potential import estimate_frame_potential
from .protocols import ThreeTimeProtocol, TwoTimeProtocol
from .random_matrices import sample_gue
from .spin_model import sample_spin_hamiltonian
from .syk_model import build_hamiltonian as build_syk_hamiltonian, generate_couplings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate frame potentials for quench protocols.")
    parser.add_argument("--protocol", choices=["two", "three"], default="two")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=["gue", "spin", "syk"], default="gue")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-spins", type=int, default=None)
    parser.add_argument("--n-fermions", type=int, default=None)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--xi-x", type=float, default=1.0)
    parser.add_argument("--xi-y", type=float, default=1.0)
    parser.add_argument("--xi-z", type=float, default=-2.0)
    parser.add_argument("--assume-hermitian", action="store_true")
    parser.add_argument("--sampler", type=str, default="uniform")
    parser.add_argument("--sampler-beta", type=float, default=None, help="Kaiser beta parameter.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    if args.model == "spin":
        if args.dim is not None:
            raise ValueError("--dim is not supported when --model spin is selected.")
        if args.n_fermions is not None:
            raise ValueError("--n-fermions is only supported when --model syk is selected.")
        n_spins = 8 if args.n_spins is None else args.n_spins
        xi = (args.xi_x, args.xi_y, args.xi_z)
        H1 = sample_spin_hamiltonian(rng, N=n_spins, J=args.J, xi=xi)
        H2 = sample_spin_hamiltonian(rng, N=n_spins, J=args.J, xi=xi)
    elif args.model == "syk":
        if args.dim is not None:
            raise ValueError("--dim is not supported when --model syk is selected.")
        if args.n_spins is not None:
            raise ValueError("--n-spins is only supported when --model spin is selected.")
        n_fermions = 6 if args.n_fermions is None else args.n_fermions
        J1 = generate_couplings(rng, N=n_fermions, J=args.J)
        H1 = build_syk_hamiltonian(J1, N=n_fermions, charge_sector="half")
        J2 = generate_couplings(rng, N=n_fermions, J=args.J)
        H2 = build_syk_hamiltonian(J2, N=n_fermions, charge_sector="half")
    else:
        if args.n_spins is not None:
            raise ValueError("--n-spins is only supported when --model spin is selected.")
        if args.n_fermions is not None:
            raise ValueError("--n-fermions is only supported when --model syk is selected.")
        dim = 8 if args.dim is None else args.dim
        H1 = sample_gue(rng, dim)
        H2 = sample_gue(rng, dim)

    if args.protocol == "two":
        protocol = TwoTimeProtocol(H1, H2, assume_hermitian=args.assume_hermitian)
    else:
        if args.model == "spin":
            H3 = sample_spin_hamiltonian(rng, N=n_spins, J=args.J, xi=xi)
        elif args.model == "syk":
            J3 = generate_couplings(rng, N=n_fermions, J=args.J)
            H3 = build_syk_hamiltonian(J3, N=n_fermions, charge_sector="half")
        else:
            H3 = sample_gue(rng, dim)
        protocol = ThreeTimeProtocol(H1, H2, H3, assume_hermitian=args.assume_hermitian)

    estimate = estimate_frame_potential(
        protocol=protocol,
        T=args.T,
        k=args.k,
        num_pairs=args.pairs,
        rng=rng,
        sampler_name=args.sampler,
        sampler_kwargs={"beta": args.sampler_beta} if args.sampler_beta is not None else None,
    )
    print(estimate)


if __name__ == "__main__":
    main()
