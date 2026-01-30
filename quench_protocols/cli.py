"""Command line interface for quick frame potential estimates."""

from __future__ import annotations

import argparse

import numpy as np

from .frame_potential import estimate_frame_potential
from .protocols import ThreeTimeProtocol, TwoTimeProtocol
from .random_matrices import sample_gue


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate frame potentials for quench protocols.")
    parser.add_argument("--protocol", choices=["two", "three"], default="two")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--assume-hermitian", action="store_true")
    parser.add_argument("--sampler", type=str, default="uniform")
    parser.add_argument("--sampler-beta", type=float, default=None, help="Kaiser beta parameter.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    H1 = sample_gue(rng, args.dim)
    H2 = sample_gue(rng, args.dim)
    if args.protocol == "two":
        protocol = TwoTimeProtocol(H1, H2, assume_hermitian=args.assume_hermitian)
    else:
        H3 = sample_gue(rng, args.dim)
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
