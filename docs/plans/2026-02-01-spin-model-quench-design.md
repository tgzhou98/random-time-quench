# Spin-Model Quench Integration Design

Date: 2026-02-01

## Summary
Add a dense spin-model Hamiltonian generator for an all-to-all random interacting spin-1/2 model and integrate it into the existing two-time and three-time quench pipeline. Keep current protocol and frame-potential APIs unchanged. Provide CLI selection between GUE and spin model and add a tutorial notebook demonstrating the spin-model workflow.

## Physics Model
We target a spin-1/2 Heisenberg-like model with anisotropy vector xi = (xi_x, xi_y, xi_z):

H = sum_{i<j} sum_{alpha in {x,y,z}} J_ij * xi_alpha * S_i^alpha S_j^alpha,

where S^alpha = sigma^alpha / 2 and J_ij are all-to-all random couplings.

Couplings: J_ij ~ Normal(0, 4 J^2 / N) for i < j, symmetric with zero diagonal.

## Architecture
1. Add a new module `quench_protocols/spin_model.py` with:
   - `sample_spin_couplings(rng, N, J)` -> dense symmetric J with zero diagonal.
   - `build_spin_hamiltonian(J, xi)` -> dense complex128 Hermitian H of size 2^N.
   - `sample_spin_hamiltonian(rng, N, J, xi)` -> convenience wrapper that samples J and builds H.

2. Reuse existing protocol classes (TwoTimeProtocol, ThreeTimeProtocol) without changes.

3. Extend the CLI with a `--model {gue,spin}` switch and spin-specific options.

## Data Flow
- For GUE: existing flow unchanged, using `sample_gue` for H1/H2/H3.
- For spin model:
  - draw J_ij with `sample_spin_couplings` (independent per quench step);
  - build H with `build_spin_hamiltonian` using fixed xi;
  - pass H1/H2/H3 into protocol class;
  - run Monte Carlo frame-potential estimator as-is.

Implementation details for operator construction:
- Precompute Pauli matrices (sigma_x, sigma_y, sigma_z) and S^alpha = sigma^alpha/2.
- Build site operators S_i^alpha using Kronecker products.
- For each pair (i, j), form S_i^alpha @ S_j^alpha and sum with J_ij * xi_alpha.
- Use complex128 throughout. For N ~ 8 (dim 256), dense matrices are acceptable.

## Error Handling
- Validate N >= 1, J > 0, xi length is 3.
- If user supplies J directly, validate shape (N, N), symmetry (within tolerance), and diagonal zeros (raise on nonzero or force to zero; pick one consistently).

## CLI
- Add `--model` with default `gue` to preserve current behavior.
- For spin model: use `--n-spins`, `--J`, `--xi-x`, `--xi-y`, `--xi-z` with default xi = (1, 1, -2).
- If `--model spin` is selected, infer dim = 2**N internally and ignore `--dim`.

## Notebook
Create a tutorial notebook `output/jupyter-notebook/spin-model-quench-demo.ipynb`:
- Sample J_ij for N=8.
- Build H for xi = (1, 1, -2).
- Run TwoTimeProtocol and estimate F_k.
- Optionally compare with GUE for a quick sanity check.

## Tests
Add lightweight tests:
- `sample_spin_couplings` is reproducible with fixed seed, symmetric, and has zero diagonal.
- `build_spin_hamiltonian` returns correct shape and is Hermitian for small N (e.g., N=2 or 3).
- Existing protocol unitarity and frame-potential tests remain unchanged.

## Non-Goals
- No analytic long-time limits or Haar bounds.
- No sparse evolution or state-only propagation.
- No heavy statistical tests of spectra or distributions.
