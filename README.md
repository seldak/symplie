# SympLie

[![CI](https://github.com/seldak/symplie/actions/workflows/ci.yml/badge.svg)](https://github.com/seldak/symplie/actions/workflows/ci.yml)

SympLie is an experimental JAX package for differentiable Lie-group
rigid-body dynamics and control.

It currently provides:

- Numerically stable, batched $SO(3)$ and $SE(3)$ exponential and
  logarithm maps.
- Torque-free and forced Moser–Veselov integration, including a composable
  step for custom JAX scans and zero-order-hold control loops.
- Fixed-target geometric PD attitude control and gyroscope propagation with
  optional bias compensation.
- JIT compilation, automatic differentiation, and per-step solver diagnostics.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For tests and figure generation:

```bash
python -m pip install -e ".[dev]"
pytest
```

### Optional CUDA support

The default installation uses CPU-backed JAX. To use a CUDA backend instead,
install the matching JAX package:

```bash
python -m pip install --upgrade "jax[cuda13]"
python -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
```

Use `jax[cuda12]` when required by the installed driver or CUDA installation.
Consult the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for current compatibility requirements. The test suite uses the active JAX
backend.

## Quickstart

```python
import jax.numpy as jnp
from symplie import rigid_body_step

J = jnp.diag(jnp.array([0.6, 1.0, 1.8]))
R = jnp.eye(3)
pi = jnp.array([0.2, 0.7, 1.0])
torque = jnp.array([0.01, -0.02, 0.03])

R_next, pi_next, info = rigid_body_step(
    R,
    pi,
    J,
    torque,
    torque,
    dt=0.01,
)
assert info.converged, info.residual_norm
```

`rigid_body_step` advances attitude and body angular momentum by one forced
timestep and reports whether its nonlinear solve converged.

The following closed-loop example uses geometric PD control to regulate a target
attitude while damping angular velocity:

```bash
python examples/attitude_control.py
```

## Documentation

The [documentation](https://seldak.github.io/symplie/) covers installation,
frame and torque-sampling conventions, numerical validation, and the complete
public API.

## Scope

SympLie deliberately focuses on rotational dynamics and control; it is not a
replacement for jaxlie, Sophus, Drake, or Pinocchio. It does not implement
translation, full $SE(3)$ rigid-body dynamics, contact, constraints, actuator
allocation, state estimation, or robotics middleware.
