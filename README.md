# symplie (v1)

A small, test-driven library for **structure-preserving Lie-group integration** on **SO(3)** / **SE(3)**.

Version 1 implements a **torque-free rigid body** integrator on **SO(3)** with:
- SO(3) updates using the exponential map (no quaternion drift)
- a discrete variational / Moser–Veselov style step
- regression tests for:
  - rotation validity ($R^TR \approx I$)
  - **spatial momentum conservation** ($L = R\pi$)
  - bounded energy drift
  - SO(3) exponential/logarithm round trips

## What’s implemented in v1

State is $(R, \pi)$:
- $R \in SO(3)$ orientation
- $\pi \in R^3$ body angular momentum ($\pi = J\omega$)

One step computes a relative rotation $F \in SO(3)$ then updates:
- $R_{k+1} = R_k F$
- $\pi_{k+1} = F^T \pi_k$

This implies spatial momentum is conserved:
$R_{k+1} \pi_{k+1} = R_k \pi_k$.

### SO(3) exponential and logarithm maps

`symplie.so3` provides complementary maps between rotation vectors and
rotation matrices:

```python
import jax.numpy as jnp
from symplie.so3 import exp as expSO3
from symplie.so3 import log as logSO3

w = jnp.array([0.1, -0.2, 0.3])
R = expSO3(w)
w_recovered = logSO3(R)
```

`logSO3(R)` returns the principal rotation vector $w \in \mathbb{R}^3$ whose
direction is the rotation axis and whose magnitude is the angle in radians,
with principal angle in $[0, \pi]$. The implementation uses separate numerical
paths near the identity and near $\pi$, where the standard closed-form formula
is ill-conditioned. Both maps are compatible with JAX JIT compilation.

For rotation vectors on the principal branch, the test suite checks
`logSO3(expSO3(w)) \approx w`. It also checks
`expSO3(logSO3(R)) \approx R`, including rotations close to $\pi$.

## Install

CPU-only install for JAX is typically just:

```bash
pip install -U jax
```

(That installs jax and a matching jaxlib wheel.)
See JAX installation docs for GPU/TPU variants.

For development:

```bash
pip install -e ".[dev]"
```

## Quickstart
```python
import jax.numpy as jnp
from symplie.integrators import simulate_free_rigid_body
from symplie.invariants import spatial_momentum, energy

J  = jnp.diag(jnp.array([0.6, 1.0, 1.8]))
R0 = jnp.eye(3)
pi0 = jnp.array([0.2, 0.7, 1.0])

Rs, pis = simulate_free_rigid_body(R0, pi0, J, dt=1e-2, steps=2000, newton_iters=6)

L0 = spatial_momentum(Rs[0], pis[0])
E0 = energy(pis[0], J)
```

## Run tests

```
pytest
```

Generate example plots

```
python scripts/make_plots.py --out artifacts
```

## Outputs:

```
    artifacts/energy_drift.png
    artifacts/spatial_momentum_error.png
```

## Roadmap (next)

 -  $SE(3)$ dynamics (translation + forces)

 -  IMU-driven propagation (bias/noise models)

 -  C++ reference implementation (pybind11) + benchmarks

 -  constraint support (discrete multipliers)
