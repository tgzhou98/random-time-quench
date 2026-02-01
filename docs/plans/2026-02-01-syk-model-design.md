# Complex-Fermion SYK Quench Integration Design

Date: 2026-02-01

## Summary
Add a complex-fermion SYK (q=4) Hamiltonian generator in the half-filled charge sector and integrate it into the existing two-time/three-time quench pipeline. Provide dense Hamiltonian construction in the half-filled basis, reproducible coupling generation, and a CLI switch to select the SYK model.

## Physics Model
We model N complex fermions with Hamiltonian:

H = sum_{i<j, k<l, all distinct} J_ijkl c_i^dagger c_j^dagger c_k c_l + h.c.

Couplings are complex Gaussian with zero mean and covariance:

< J_ijkl J_ijkl* > = (3! J^2 / N^3)

Antisymmetry is enforced within each pair: J_{jikl} = -J_{ijkl}, J_{ijlk} = -J_{ijkl}. Pair-of-pairs are identified so (i,j,k,l) and (k,l,i,j) share the same coupling. Hermiticity is enforced by including the explicit h.c. term.

Charge sector: half-filling (N/2 fermions). Require N even.

## Architecture
1. Add `quench_protocols/syk_model.py` with:
   - `generate_couplings(rng, N, J)` -> dict of canonical couplings keyed by (i,j,k,l).
   - `build_hamiltonian(couplings, N, charge_sector="half")` -> dense complex128 Hamiltonian in half-filled basis.
   - Helper `canonicalize_indices` and `coupling_lookup` for antisymmetry/signs.

2. Reuse existing protocol classes (TwoTimeProtocol, ThreeTimeProtocol) unchanged.

3. Extend CLI with `--model syk` and SYK-specific arguments.

## Data Flow
- Couplings: sample only canonical tuples with i<j, k<l, all four indices distinct. For variance, sample Re and Im as Normal(0, sigma^2) with sigma^2 = (3! J^2 / N^3) / 2 to match the target complex covariance.
- Basis: enumerate bitstrings with popcount N/2, build index map.
- Operator action: apply c_i^dagger c_j^dagger c_k c_l right-to-left with fermionic sign rules based on the number of occupied modes below each index. Accumulate contributions in dense H. After filling the non-Hermitian part, add the explicit h.c. term.

## Error Handling
- Enforce N even and N >= 4 for SYK.
- Validate charge_sector == "half" (raise otherwise).
- Validate coupling keys for distinct indices and canonical ordering where applicable.
- Use complex128 throughout.

## CLI
Add `--model syk` alongside existing models.
- New args: `--n-fermions` (even N), `--J` (coupling scale).
- When `--model syk`, build H1/H2 (and H3 for three-time) via SYK generator in half-filled sector.

## Tests
Add `tests/test_syk_model.py` covering:
- Reproducibility of `generate_couplings` with a fixed seed.
- Antisymmetry lookup behavior for swapped indices.
- Hermiticity of built Hamiltonian for small N (e.g., N=4 or 6).
- Shape equals comb(N, N/2).

## Non-Goals
- No sparse Hamiltonian support.
- No analytic large-N or long-time limits.
- No projection from full 2^N Hilbert space.
