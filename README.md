# Random Time Quench Protocols

This project implements two-time and three-time quench protocols with a Monte Carlo estimator for the k-th frame potential.

## Quick start

Create an environment with `numpy` and `scipy`, then run the CLI:

```bash
python -m quench_protocols.cli --protocol two --k 2 --T 10 --pairs 2000 --seed 0 --dim 8
```

### What is the result?
The CLI prints a single floating-point estimate of the frame potential:

```
53.420397783696785
```

This value is **stochastic** because it is a Monte Carlo estimate. It will generally change if you change `--seed`, `--pairs`, `--dim`, or `--T`. With the same seed and parameters, the output should be reproducible (up to tiny floating-point differences across platforms).

### Alternative time samplers
You can swap the time distribution with `--sampler`:

```bash
python -m quench_protocols.cli --protocol three --k 4 --T 100 --pairs 1000 --seed 42 --dim 200 --sampler hann
python -m quench_protocols.cli --protocol three --k 4 --T 100 --pairs 1000 --seed 42 --dim 200 --sampler kaiser --sampler-beta 8.0
```

## Running tests

```bash
python -m pytest -q
```

## Python API example

```python
import numpy as np
from quench_protocols.protocols import TwoTimeProtocol
from quench_protocols.frame_potential import estimate_frame_potential
from quench_protocols.random_matrices import sample_gue

rng = np.random.default_rng(0)
H1 = sample_gue(rng, 8)
H2 = sample_gue(rng, 8)

protocol = TwoTimeProtocol(H1, H2, assume_hermitian=True)
F2 = estimate_frame_potential(protocol, T=10.0, k=2, num_pairs=2000, rng=rng)
print(F2)
```
