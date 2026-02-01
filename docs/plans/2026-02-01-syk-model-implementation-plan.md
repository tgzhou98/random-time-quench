# Complex-Fermion SYK Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement complex-fermion SYK (q=4) couplings and half-filled Hamiltonian construction, and integrate SYK into the CLI for quench/FP demos.

**Architecture:** Add a `quench_protocols/syk_model.py` module with coupling generation and half-filled Hamiltonian builder. Build the Hamiltonian directly in the half-filling basis using bitstring states and fermionic sign rules. Integrate SYK into the CLI via a `--model syk` option.

**Tech Stack:** Python 3, numpy, scipy (existing), pytest.

### Task 1: Add SYK model tests

**Files:**
- Create: `tests/test_syk_model.py`

**Step 1: Write the failing tests** (@superpowers:test-driven-development)

```python
import numpy as np

from quench_protocols.syk_model import (
    build_hamiltonian,
    generate_couplings,
    lookup_coupling,
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
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=$(pwd) pytest tests/test_syk_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quench_protocols.syk_model'`.

**Step 3: Commit**

```bash
git add tests/test_syk_model.py
git commit -m "test: add SYK model coverage"
```

### Task 2: Implement SYK coupling and Hamiltonian module

**Files:**
- Create: `quench_protocols/syk_model.py`
- Modify: `quench_protocols/__init__.py`

**Step 1: Write minimal implementation to satisfy tests** (@superpowers:test-driven-development)

```python
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Dict, Tuple

import numpy as np

CouplingDict = Dict[Tuple[int, int, int, int], complex]


def generate_couplings(rng: np.random.Generator, N: int, J: float) -> CouplingDict:
    if N <= 0 or N % 2 != 0:
        raise ValueError("N must be a positive even integer.")
    if N < 4:
        raise ValueError("N must be at least 4 for q=4 SYK.")
    if J <= 0:
        raise ValueError("J must be positive.")
    variance = 6.0 * (J ** 2) / (N ** 3)
    sigma = np.sqrt(0.5 * variance)
    couplings: CouplingDict = {}
    for i, j in combinations(range(N), 2):
        for k, l in combinations(range(N), 2):
            if len({i, j, k, l}) < 4:
                continue
            if (k, l) < (i, j):
                continue
            real = rng.normal(0.0, sigma)
            imag = rng.normal(0.0, sigma)
            couplings[(i, j, k, l)] = real + 1j * imag
    return couplings


def _canonical_pair(i: int, j: int) -> tuple[int, int, int]:
    if i == j:
        return i, j, 0
    if i < j:
        return i, j, 1
    return j, i, -1


def _canonical_key(i: int, j: int, k: int, l: int) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    i1, j1, s1 = _canonical_pair(i, j)
    k1, l1, s2 = _canonical_pair(k, l)
    sign = s1 * s2
    if (k1, l1) < (i1, j1):
        return (k1, l1, s2), (i1, j1, s1), sign
    return (i1, j1, s1), (k1, l1, s2), sign


def lookup_coupling(couplings: CouplingDict, i: int, j: int, k: int, l: int) -> complex:
    if len({i, j, k, l}) < 4:
        return 0.0 + 0.0j
    (i1, j1, s1), (k1, l1, s2), sign = _canonical_key(i, j, k, l)
    value = couplings.get((i1, j1, k1, l1))
    if value is None:
        return 0.0 + 0.0j
    return sign * value


def _apply_annihilation(state: int, idx: int) -> tuple[int | None, int]:
    if ((state >> idx) & 1) == 0:
        return None, 0
    mask = (1 << idx) - 1
    parity = bin(state & mask).count("1") % 2
    sign = -1 if parity else 1
    return state & ~(1 << idx), sign


def _apply_creation(state: int, idx: int) -> tuple[int | None, int]:
    if ((state >> idx) & 1) == 1:
        return None, 0
    mask = (1 << idx) - 1
    parity = bin(state & mask).count("1") % 2
    sign = -1 if parity else 1
    return state | (1 << idx), sign


def build_hamiltonian(couplings: CouplingDict, N: int, charge_sector: str = "half") -> np.ndarray:
    if N <= 0 or N % 2 != 0:
        raise ValueError("N must be a positive even integer.")
    if charge_sector != "half":
        raise ValueError("Only half-filling is supported.")
    if N < 4:
        raise ValueError("N must be at least 4 for q=4 SYK.")
    n_particles = N // 2
    basis = [state for state in range(1 << N) if state.bit_count() == n_particles]
    index = {state: idx for idx, state in enumerate(basis)}
    dim = comb(N, n_particles)
    H = np.zeros((dim, dim), dtype=np.complex128)
    for (i, j, k, l), value in couplings.items():
        if len({i, j, k, l}) < 4:
            continue
        for col, state in enumerate(basis):
            st, s1 = _apply_annihilation(state, l)
            if st is None:
                continue
            st, s2 = _apply_annihilation(st, k)
            if st is None:
                continue
            st, s3 = _apply_creation(st, j)
            if st is None:
                continue
            st, s4 = _apply_creation(st, i)
            if st is None:
                continue
            row = index[st]
            sign = s1 * s2 * s3 * s4
            H[row, col] += value * sign
    return H + H.conj().T
```

**Step 2: Run tests**

Run: `PYTHONPATH=$(pwd) pytest tests/test_syk_model.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add quench_protocols/syk_model.py quench_protocols/__init__.py
git commit -m "feat: add SYK coupling and Hamiltonian builder"
```

### Task 3: Integrate SYK model into CLI

**Files:**
- Modify: `quench_protocols/cli.py`
- Create: `tests/test_cli_syk_model.py`

**Step 1: Write failing CLI test** (@superpowers:test-driven-development)

```python
import sys

from quench_protocols.cli import _parse_args


def test_cli_parses_syk_model_args(monkeypatch) -> None:
    argv = [
        "prog",
        "--model",
        "syk",
        "--protocol",
        "two",
        "--n-fermions",
        "6",
        "--J",
        "1.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = _parse_args()
    assert args.model == "syk"
    assert args.n_fermions == 6
    assert args.J == 1.0
```

**Step 2: Run test to verify failure**

Run: `PYTHONPATH=$(pwd) pytest tests/test_cli_syk_model.py -v`
Expected: FAIL with unknown args / missing field.

**Step 3: Implement CLI integration**

```python
parser.add_argument("--model", choices=["gue", "spin", "syk"], default="gue")
parser.add_argument("--n-fermions", type=int, default=None)
```

Then in `main()`, add a SYK branch:
- Require `--n-fermions` (even N), set default (e.g., 6 if None).
- Use `generate_couplings` + `build_hamiltonian` for each H1/H2/H3.
- Keep `dim` and `n-spins` validation consistent with other models.

**Step 4: Run tests**

Run: `PYTHONPATH=$(pwd) pytest tests/test_cli_syk_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add quench_protocols/cli.py tests/test_cli_syk_model.py
git commit -m "feat: add SYK option to CLI"
```

### Task 4: Full test pass

**Step 1: Run full suite**

Run: `PYTHONPATH=$(pwd) pytest -v`
Expected: PASS

**Step 2: Commit (if any changes)**

```bash
git add -A
git commit -m "test: verify full suite"
```
