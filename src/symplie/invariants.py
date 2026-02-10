from __future__ import annotations
import jax.numpy as jnp

def energy(pi: jnp.ndarray, J: jnp.ndarray) -> jnp.ndarray:
    """Kinetic energy: 0.5 * pi^T * J^{-1} * pi"""
    omega = jnp.linalg.solve(J, pi)
    return 0.5 * jnp.dot(pi, omega)

def spatial_momentum(R: jnp.ndarray, pi: jnp.ndarray) -> jnp.ndarray:
    """Spatial angular momentum L = R * pi (if pi is body-frame momentum)."""
    return R @ pi

def ortho_error(R: jnp.ndarray) -> jnp.ndarray:
    """||R^T R - I||_F"""
    I = jnp.eye(3, dtype=R.dtype)
    return jnp.linalg.norm(R.T @ R - I)

