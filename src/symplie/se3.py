"""SE(3) Lie-group operations implemented with JAX.

Twists use the ``xi = [rho, phi]`` convention, where ``rho`` is the
translational component and ``phi`` is the SO(3) rotation vector.
"""

from __future__ import annotations

import jax
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
    return xi[..., :3], xi[..., 3:]


def _assemble_transform(R: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Assemble an SE(3) matrix from a rotation and translation."""
    upper = jnp.concatenate((R, p[:, None]), axis=1)
    lower = jnp.zeros((1, 4), dtype=R.dtype)
    lower = lower.at[0, 3].set(1.0)
    return jnp.concatenate((upper, lower), axis=0)


def hat(xi: jnp.ndarray) -> jnp.ndarray:
    """Map twists with shape ``(..., 6)`` to se(3) matrices."""
    if xi.shape[-1:] != (6,):
        raise ValueError("hat expects an array with trailing shape (6,)")

    rho, phi = _split_twist(xi)

    phi_hat = hatSO3(phi)
    upper = jnp.concatenate((phi_hat, rho[..., :, None]), axis=-1)
    lower = jnp.zeros(xi.shape[:-1] + (1, 4), dtype=phi_hat.dtype)
    return jnp.concatenate((upper, lower), axis=-2)


def vee(X: jnp.ndarray) -> jnp.ndarray:
    """Map se(3) matrices with shape ``(..., 4, 4)`` to ``[rho, phi]``."""
    if X.shape[-2:] != (4, 4):
        raise ValueError("vee expects an array with trailing shape (4, 4)")

    phi = veeSO3(X[..., :3, :3])
    rho = X[..., :3, 3]
    return jnp.concatenate((rho, phi), axis=-1)


def _rotation_angle(phi: jnp.ndarray) -> jnp.ndarray:
    """Return the rotation angle, floored below the small-angle threshold."""
    # The series handles smaller angles; the floor keeps unused divisions finite.
    minimum_angle = 0.5 * jnp.cbrt(jnp.finfo(phi.dtype).eps)
    return jnp.sqrt(jnp.maximum(jnp.dot(phi, phi), minimum_angle**2))


def _left_jacobian_SO3_single(phi: jnp.ndarray) -> jnp.ndarray:
    """Return the SO(3) left Jacobian used by the SE(3) exponential."""
    theta = _rotation_angle(phi)
    Phi = hatSO3(phi)
    Phi2 = Phi @ Phi
    I = jnp.eye(3, dtype=phi.dtype)

    def small(_):
        return I + 0.5 * Phi + (1.0 / 6.0) * Phi2

    def general(_):
        theta2 = theta**2
        theta3 = theta2 * theta
        return (
            I
            + ((1.0 - jnp.cos(theta)) / theta2) * Phi
            + ((theta - jnp.sin(theta)) / theta3) * Phi2
        )

    threshold = jnp.cbrt(jnp.finfo(phi.dtype).eps)
    return jax.lax.cond(theta < threshold, small, general, operand=None)


def left_jacobian_SO3(phi: jnp.ndarray) -> jnp.ndarray:
    """SO(3) left Jacobian: (..., 3) -> (..., 3, 3)."""
    if phi.shape[-1:] != (3,):
        raise ValueError("left_jacobian_SO3 expects trailing shape (3,)")
    if phi.ndim == 1:
        return _left_jacobian_SO3_single(phi)

    batch_shape = phi.shape[:-1]
    jacobians = jax.vmap(_left_jacobian_SO3_single)(phi.reshape((-1, 3)))
    return jacobians.reshape(batch_shape + (3, 3))


def _left_jacobian_inverse_SO3_single(phi: jnp.ndarray) -> jnp.ndarray:
    """Return the inverse SO(3) left Jacobian used by the SE(3) logarithm."""
    theta = _rotation_angle(phi)
    Phi = hatSO3(phi)
    Phi2 = Phi @ Phi
    I = jnp.eye(3, dtype=phi.dtype)

    def small(_):
        return I - 0.5 * Phi + (1.0 / 12.0) * Phi2

    def general(_):
        theta2 = theta**2
        coefficient = 1.0 / theta2 - (
            (1.0 + jnp.cos(theta)) / (2.0 * theta * jnp.sin(theta))
        )
        return I - 0.5 * Phi + coefficient * Phi2

    threshold = jnp.cbrt(jnp.finfo(phi.dtype).eps)
    return jax.lax.cond(theta < threshold, small, general, operand=None)


def left_jacobian_inverse_SO3(phi: jnp.ndarray) -> jnp.ndarray:
    """Inverse SO(3) left Jacobian: (..., 3) -> (..., 3, 3)."""
    if phi.shape[-1:] != (3,):
        raise ValueError("left_jacobian_inverse_SO3 expects trailing shape (3,)")
    if phi.ndim == 1:
        return _left_jacobian_inverse_SO3_single(phi)

    batch_shape = phi.shape[:-1]
    jacobians = jax.vmap(_left_jacobian_inverse_SO3_single)(phi.reshape((-1, 3)))
    return jacobians.reshape(batch_shape + (3, 3))


def _exp_single(xi: jnp.ndarray) -> jnp.ndarray:
    """Map a twist with shape ``(6,)`` to an SE(3) transformation."""
    rho, phi = _split_twist(xi)
    R = expSO3(phi)
    p = left_jacobian_SO3(phi) @ rho
    return _assemble_transform(R, p)


def exp(xi: jnp.ndarray) -> jnp.ndarray:
    """SE(3) exponential map: (..., 6) -> (..., 4, 4)."""
    if xi.shape[-1:] != (6,):
        raise ValueError("exp expects an array with trailing shape (6,)")
    if xi.ndim == 1:
        return _exp_single(xi)

    batch_shape = xi.shape[:-1]
    transforms = jax.vmap(_exp_single)(xi.reshape((-1, 6)))
    return transforms.reshape(batch_shape + (4, 4))


def _log_single(T: jnp.ndarray) -> jnp.ndarray:
    """Map an SE(3) transformation to its principal twist ``[rho, phi]``."""
    R = T[:3, :3]
    p = T[:3, 3]
    phi = logSO3(R)
    rho = left_jacobian_inverse_SO3(phi) @ p
    return jnp.concatenate((rho, phi))


def log(T: jnp.ndarray) -> jnp.ndarray:
    """Principal SE(3) logarithm map: (..., 4, 4) -> (..., 6)."""
    if T.shape[-2:] != (4, 4):
        raise ValueError("log expects an array with trailing shape (4, 4)")
    if T.ndim == 2:
        return _log_single(T)

    batch_shape = T.shape[:-2]
    twists = jax.vmap(_log_single)(T.reshape((-1, 4, 4)))
    return twists.reshape(batch_shape + (6,))
