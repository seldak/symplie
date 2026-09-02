"""SE(3) Lie-group operations implemented with JAX.

Twists use the ``xi = [rho, phi]`` convention, where ``rho`` is the
translational component and ``phi`` is the SO(3) rotation vector.
"""

from __future__ import annotations

import jax.numpy as jnp

from .so3 import exp as expSO3
from .so3 import log as logSO3
from .so3 import hat as hatSO3
from .so3 import vee as veeSO3

__all__ = [
    "exp",
    "hat",
    "left_jacobian_SO3",
    "left_jacobian_inverse_SO3",
    "log",
    "vee",
]


def _split_twist(xi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split ``[rho, phi]`` into its translational and rotational parts."""
    return xi[:3], xi[3:]


def _assemble_transform(R: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Assemble an SE(3) matrix from a rotation and translation."""
    upper = jnp.concatenate((R, p[:, None]), axis=1)
    lower = jnp.zeros((1, 4), dtype=R.dtype)
    lower = lower.at[0, 3].set(1.0)
    return jnp.concatenate((upper, lower), axis=0)


def hat(xi: jnp.ndarray) -> jnp.ndarray:
    """Map a twist with shape ``(6,)`` to an se(3) matrix."""
    rho, phi = _split_twist(xi)

    phi_hat = hatSO3(phi)
    upper = jnp.concatenate((phi_hat, rho[:, None]), axis=1)
    lower = jnp.zeros((1, 4), dtype=phi_hat.dtype)
    return jnp.concatenate((upper, lower), axis=0)


def vee(X: jnp.ndarray) -> jnp.ndarray:
    """Map an se(3) matrix with shape ``(4, 4)`` to ``[rho, phi]``."""
    phi = veeSO3(X[:3, :3])
    rho = X[:3, 3]
    return jnp.concatenate((rho, phi))


def left_jacobian_SO3(phi: jnp.ndarray) -> jnp.ndarray:
    """Return the SO(3) left Jacobian used by the SE(3) exponential."""
    raise NotImplementedError("Implement the SO(3) left Jacobian")


def left_jacobian_inverse_SO3(phi: jnp.ndarray) -> jnp.ndarray:
    """Return the inverse SO(3) left Jacobian used by the SE(3) logarithm."""
    raise NotImplementedError("Implement the inverse SO(3) left Jacobian")


def exp(xi: jnp.ndarray) -> jnp.ndarray:
    """Map a twist with shape ``(6,)`` to an SE(3) transformation."""
    rho, phi = _split_twist(xi)
    R = expSO3(phi)
    p = left_jacobian_SO3(phi) @ rho
    return _assemble_transform(R, p)


def log(T: jnp.ndarray) -> jnp.ndarray:
    """Map an SE(3) transformation to its principal twist ``[rho, phi]``."""
    R = T[:3, :3]
    p = T[:3, 3]
    phi = logSO3(R)
    rho = left_jacobian_inverse_SO3(phi) @ p
    return jnp.concatenate((rho, phi))
