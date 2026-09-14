# Getting started

## Installation

Create a virtual environment and install SympLie from the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install the development dependencies to run tests and regenerate figures:

```bash
python -m pip install -e ".[dev]"
```

The default installation uses CPU-backed JAX. Accelerator packages depend on
the installed driver and CUDA version.
Follow the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
when another backend is required.

### Optional CUDA support

For a compatible CUDA 13 environment:

```bash
python -m pip install --upgrade "jax[cuda13]"
python -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
```

Use `jax[cuda12]` when required by the installed driver or CUDA installation.
The JAX installation guide is authoritative for current compatibility
requirements. SympLie tests use whichever JAX backend is active.

## One forced step

The state consists of a body-to-spatial attitude \(R\) and body angular
momentum \(\boldsymbol{\pi}\):

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

Read [Conventions](conventions.md) for the distinction between endpoint
torques and zero-order-held feedback torque.

## Examples

Run the closed-loop attitude-regulation example:

```bash
python examples/attitude_stabilization.py
```

Run the moving-reference attitude-tracking example:

```bash
python examples/attitude_tracking.py
```

Run the \(SE(3)\) exponential/logarithm round trip:

```bash
python examples/se3_exp_log.py
```

## Development

```bash
pytest
```

Install the documentation dependencies before serving the site locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```
