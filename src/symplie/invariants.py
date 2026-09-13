from __future__ import annotations
import jax.numpy as jnp

def energy(pi: jnp.ndarray, J: jnp.ndarray) -> jnp.ndarray:
    r"""Compute rotational kinetic energy from body angular momentum.

    \[
        E(\boldsymbol{\pi},J)
        = \frac{1}{2}\boldsymbol{\pi}^{T}J^{-1}\boldsymbol{\pi}.
    \]

    Parameters
    ----------
    pi : jax.Array, shape (3,)
        Body-frame angular momentum.
    J : jax.Array, shape (3, 3)
        Body inertia tensor.

    Returns
    -------
    jax.Array, scalar
        Rotational kinetic energy.

    Notes
    -----
    ``J`` is applied through a linear solve rather than an explicit inverse.
    This function operates on one state; use `jax.vmap` for a trajectory
    or batch.
    """
    omega = jnp.linalg.solve(J, pi)
    return 0.5 * jnp.dot(pi, omega)

def spatial_momentum(R: jnp.ndarray, pi: jnp.ndarray) -> jnp.ndarray:
    r"""Express body angular momentum in spatial coordinates.

    For a body-to-spatial attitude \(R\) and body angular momentum
    \(\boldsymbol{\pi}\), the spatial angular momentum is

    \[
        \mathbf{L} = R\boldsymbol{\pi}.
    \]

    Parameters
    ----------
    R : jax.Array, shape (3, 3)
        Body-to-spatial rotation matrix.
    pi : jax.Array, shape (3,)
        Body-frame angular momentum.

    Returns
    -------
    jax.Array, shape (3,)
        Angular momentum expressed in the spatial frame.

    Notes
    -----
    This function operates on one state; use `jax.vmap` for a trajectory
    or batch.
    """
    return R @ pi

def ortho_error(R: jnp.ndarray) -> jnp.ndarray:
    r"""Measure violation of the rotation-matrix orthogonality condition.

    \[
        e_{\mathrm{orth}}(R) = \lVert R^T R-I\rVert_F.
    \]

    Parameters
    ----------
    R : jax.Array, shape (3, 3)
        Matrix to evaluate.

    Returns
    -------
    jax.Array, scalar
        Frobenius norm of the orthogonality residual. The value is zero for an
        exactly orthogonal matrix.

    Notes
    -----
    Orthogonality alone does not distinguish rotations from reflections. Pair
    this diagnostic with `determinant_error` when testing membership in
    \(SO(3)\). Use `jax.vmap` for a trajectory or batch.
    """
    I = jnp.eye(3, dtype=R.dtype)
    return jnp.linalg.norm(R.T @ R - I)

def determinant_error(R: jnp.ndarray) -> jnp.ndarray:
    r"""Measure violation of the proper-rotation determinant condition.

    \[
        e_{\det}(R) = |\det(R)-1|.
    \]

    Parameters
    ----------
    R : jax.Array, shape (3, 3)
        Matrix to evaluate.

    Returns
    -------
    jax.Array, scalar
        Absolute determinant error. The value is zero when ``det(R)`` is
        exactly ``+1``.

    Notes
    -----
    A unit determinant alone does not imply orthogonality. Pair this diagnostic
    with `ortho_error` when testing membership in \(SO(3)\). Use
    `jax.vmap` for a trajectory or batch.
    """
    return jnp.abs(jnp.linalg.det(R) - 1.0)
