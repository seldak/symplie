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
    r"""Map twists to matrices in \(\mathfrak{se}(3)\).

    SympLie orders a twist as \(\boldsymbol{\xi}=[\boldsymbol{\rho},
    \boldsymbol{\phi}]\), with translation first and rotation second. The hat
    operator is

    \[
        \widehat{\boldsymbol{\xi}} =
        \begin{bmatrix}
            \widehat{\boldsymbol{\phi}} & \boldsymbol{\rho} \\
            \mathbf{0}^{T} & 0
        \end{bmatrix}.
    \]

    Parameters
    ----------
    xi : jax.Array, shape (..., 6)
        One twist or a batch of twists in ``[rho, phi]`` order.

    Returns
    -------
    jax.Array, shape (..., 4, 4)
        Homogeneous Lie-algebra matrices with the same leading batch
        dimensions and dtype as ``xi``.

    Raises
    ------
    ValueError
        If the trailing shape of ``xi`` is not ``(6,)``.
    """
    if xi.shape[-1:] != (6,):
        raise ValueError("hat expects an array with trailing shape (6,)")

    rho, phi = _split_twist(xi)

    phi_hat = hatSO3(phi)
    upper = jnp.concatenate((phi_hat, rho[..., :, None]), axis=-1)
    lower = jnp.zeros(xi.shape[:-1] + (1, 4), dtype=phi_hat.dtype)
    return jnp.concatenate((upper, lower), axis=-2)


def vee(X: jnp.ndarray) -> jnp.ndarray:
    r"""Map matrices in \(\mathfrak{se}(3)\) back to twists.

    For a matrix

    \[
        X = \begin{bmatrix}
            \widehat{\boldsymbol{\phi}} & \boldsymbol{\rho} \\
            \mathbf{0}^{T} & 0
        \end{bmatrix},
    \]

    the result is the translation-first twist
    \([\boldsymbol{\rho},\boldsymbol{\phi}]\).

    Parameters
    ----------
    X : jax.Array, shape (..., 4, 4)
        One Lie-algebra matrix or a batch of matrices. The function assumes the
        expected \(\mathfrak{se}(3)\) structure and does not validate it.

    Returns
    -------
    jax.Array, shape (..., 6)
        Twists in ``[rho, phi]`` order.

    Raises
    ------
    ValueError
        If the trailing shape of ``X`` is not ``(4, 4)``.
    """
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
    r"""Evaluate the left Jacobian of \(SO(3)\).

    For \(\theta=\lVert\boldsymbol{\phi}\rVert\) and
    \(\Phi=\widehat{\boldsymbol{\phi}}\), the Jacobian is

    \[
        J_l(\boldsymbol{\phi})
        = I
        + \frac{1-\cos\theta}{\theta^2}\Phi
        + \frac{\theta-\sin\theta}{\theta^3}\Phi^2.
    \]

    It maps the translational component of an \(\mathfrak{se}(3)\) twist
    to the translation in the corresponding homogeneous transform.

    Parameters
    ----------
    phi : jax.Array, shape (..., 3)
        Rotation vectors in radians.

    Returns
    -------
    jax.Array, shape (..., 3, 3)
        Left Jacobians with the same leading batch dimensions as ``phi``.

    Raises
    ------
    ValueError
        If the trailing shape of ``phi`` is not ``(3,)``.

    Notes
    -----
    A series expansion is used near zero. The implementation is compatible
    with JAX JIT compilation, batching, and automatic differentiation.
    """
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
    r"""Evaluate the inverse left Jacobian of \(SO(3)\).

    For \(\theta=\lVert\boldsymbol{\phi}\rVert\) and
    \(\Phi=\widehat{\boldsymbol{\phi}}\), the inverse is

    \[
        J_l^{-1}(\boldsymbol{\phi})
        = I - \frac{1}{2}\Phi
        + \left[
            \frac{1}{\theta^2}
            - \frac{1+\cos\theta}{2\theta\sin\theta}
          \right]\Phi^2.
    \]

    Parameters
    ----------
    phi : jax.Array, shape (..., 3)
        Rotation vectors in radians.

    Returns
    -------
    jax.Array, shape (..., 3, 3)
        Inverse left Jacobians with the same leading batch dimensions as
        ``phi``.

    Raises
    ------
    ValueError
        If the trailing shape of ``phi`` is not ``(3,)``.

    Notes
    -----
    A series expansion is used near zero. This inverse is used by
    `log` to recover the translational twist coordinate.
    """
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
    r"""Evaluate the exponential map from twists to \(SE(3)\).

    For the translation-first twist
    \(\boldsymbol{\xi}=[\boldsymbol{\rho},\boldsymbol{\phi}]\), SympLie
    constructs

    \[
        \operatorname{Exp}(\boldsymbol{\xi}) =
        \begin{bmatrix}
            \operatorname{Exp}(\boldsymbol{\phi})
            & J_l(\boldsymbol{\phi})\boldsymbol{\rho} \\
            \mathbf{0}^{T} & 1
        \end{bmatrix}.
    \]

    Parameters
    ----------
    xi : jax.Array, shape (..., 6)
        Twists in ``[rho, phi]`` order. Rotational components are in radians.

    Returns
    -------
    jax.Array, shape (..., 4, 4)
        Homogeneous transformation matrices with the same leading batch
        dimensions as ``xi``.

    Raises
    ------
    ValueError
        If the trailing shape of ``xi`` is not ``(6,)``.

    Notes
    -----
    The implementation uses the numerically stable \(SO(3)\) exponential
    and left Jacobian, including their small-angle paths.
    """
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
    r"""Evaluate the principal logarithm of transformations in \(SE(3)\).

    For

    \[
        T = \begin{bmatrix}R & \mathbf{p} \\ \mathbf{0}^{T} & 1\end{bmatrix},
    \]

    the principal twist is

    \[
        \boldsymbol{\phi}=\operatorname{Log}(R),
        \qquad
        \boldsymbol{\rho}=J_l^{-1}(\boldsymbol{\phi})\mathbf{p},
        \qquad
        \boldsymbol{\xi}=[\boldsymbol{\rho},\boldsymbol{\phi}].
    \]

    Parameters
    ----------
    T : jax.Array, shape (..., 4, 4)
        One homogeneous transformation or a batch of transformations.

    Returns
    -------
    jax.Array, shape (..., 6)
        Principal translation-first twists. The rotation-vector norm lies in
        \([0,\pi]\).

    Raises
    ------
    ValueError
        If the trailing shape of ``T`` is not ``(4, 4)``.

    Notes
    -----
    ``log`` assumes that the rotational block is a proper rotation and that
    the final row has homogeneous-transform structure. It does not validate
    either condition. The rotational principal-branch behavior and branch cut
    are inherited from `symplie.so3.log`.
    """
    if T.shape[-2:] != (4, 4):
        raise ValueError("log expects an array with trailing shape (4, 4)")
    if T.ndim == 2:
        return _log_single(T)

    batch_shape = T.shape[:-2]
    twists = jax.vmap(_log_single)(T.reshape((-1, 4, 4)))
    return twists.reshape(batch_shape + (6,))
