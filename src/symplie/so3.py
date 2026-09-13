from __future__ import annotations

import jax
import jax.numpy as jnp

def hat(w: jnp.ndarray) -> jnp.ndarray:
    r"""Map Euclidean vectors to skew-symmetric matrices in \(\mathfrak{so}(3)\).

    The hat operator is defined so that matrix multiplication reproduces the
    three-dimensional cross product:

    \[
        \widehat{\mathbf{w}} =
        \begin{bmatrix}
            0 & -w_z & w_y \\
            w_z & 0 & -w_x \\
            -w_y & w_x & 0
        \end{bmatrix},
        \qquad
        \widehat{\mathbf{w}}\mathbf{v} = \mathbf{w}\times\mathbf{v}.
    \]

    Parameters
    ----------
    w : jax.Array, shape (..., 3)
        One vector or a batch of vectors.

    Returns
    -------
    jax.Array, shape (..., 3, 3)
        Skew-symmetric matrices with the same leading batch dimensions and
        dtype as ``w``.

    Notes
    -----
    The function is compatible with `jax.jit`, automatic
    differentiation, and arbitrary leading batch dimensions.
    """
    wx, wy, wz = jnp.moveaxis(w, -1, 0)
    zero = jnp.zeros_like(wx)

    row_0 = jnp.stack([zero, -wz, wy], axis=-1)
    row_1 = jnp.stack([wz, zero, -wx], axis=-1)
    row_2 = jnp.stack([-wy, wx, zero], axis=-1)

    return jnp.stack(
        [row_0, row_1, row_2],
        axis=-2,
    )

def vee(W: jnp.ndarray) -> jnp.ndarray:
    r"""Map matrices in \(\mathfrak{so}(3)\) back to Euclidean vectors.

    This is the inverse of `hat` for a skew-symmetric matrix:

    \[
        \left(
        \begin{bmatrix}
            0 & -w_z & w_y \\
            w_z & 0 & -w_x \\
            -w_y & w_x & 0
        \end{bmatrix}
        \right)^\vee
        = \begin{bmatrix}w_x & w_y & w_z\end{bmatrix}^{T}.
    \]

    Parameters
    ----------
    W : jax.Array, shape (..., 3, 3)
        One matrix or a batch of matrices. Inputs are assumed to be
        skew-symmetric; this function does not validate or antisymmetrize them.

    Returns
    -------
    jax.Array, shape (..., 3)
        Vectors formed from entries ``W[..., 2, 1]``, ``W[..., 0, 2]``, and
        ``W[..., 1, 0]``.
    """
    return jnp.stack(
        [W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]],
        axis=-1,
    )

def _exp_single(w: jnp.ndarray) -> jnp.ndarray:
    """SO(3) exponential map using Rodrigues, stable for small angles."""
    theta_squared = jnp.dot(w, w)
    K = hat(w)
    I = jnp.eye(3, dtype=w.dtype)

    def small(_):
        # Series expansions:
        # sin(theta)/theta ≈ 1 - t^2/6 + t^4/120
        # (1-cos(theta))/theta^2 ≈ 1/2 - t^2/24 + t^4/720
        t2 = theta_squared
        a = 1.0 - t2 / 6.0 + (t2 * t2) / 120.0
        b = 0.5 - t2 / 24.0 + (t2 * t2) / 720.0
        return I + a * K + b * (K @ K)

    def general(_):
        # Keep this branch finite when vmap evaluates it at zero.
        theta = jnp.sqrt(jnp.where(theta_squared < 1e-14, 1.0, theta_squared))
        a = jnp.sin(theta) / theta
        b = (1.0 - jnp.cos(theta)) / (theta * theta)
        return I + a * K + b * (K @ K)

    return jax.lax.cond(theta_squared < 1e-14, small, general, operand=None)

def exp(w: jnp.ndarray) -> jnp.ndarray:
    r"""Evaluate the exponential map from rotation vectors to \(SO(3)\).

    For \(\theta=\lVert\mathbf{w}\rVert\) and
    \(W=\widehat{\mathbf{w}}\), Rodrigues' formula gives

    \[
        \operatorname{Exp}(\mathbf{w})
        = I
        + \frac{\sin\theta}{\theta}W
        + \frac{1-\cos\theta}{\theta^2}W^2.
    \]

    The vector direction is the rotation axis and its norm is the rotation
    angle in radians. Near the identity, series expansions replace the two
    removable singularities in Rodrigues' formula.

    Parameters
    ----------
    w : jax.Array, shape (..., 3)
        Rotation vectors in radians.

    Returns
    -------
    jax.Array, shape (..., 3, 3)
        Proper rotation matrices with the same leading batch dimensions as
        ``w``.

    Raises
    ------
    ValueError
        If the trailing shape of ``w`` is not ``(3,)``.

    Notes
    -----
    The small-angle and general branches remain finite during JAX automatic
    differentiation, including exactly at ``w = 0``.
    """
    if w.shape[-1:] != (3,):
        raise ValueError("exp expects an array with trailing shape (3,)")
    if w.ndim == 1:
        return _exp_single(w)

    batch_shape = w.shape[:-1]
    rotations = jax.vmap(_exp_single)(w.reshape((-1, 3)))
    return rotations.reshape(batch_shape + (3, 3))

def _log_single(R: jnp.ndarray) -> jnp.ndarray:
    """Principal rotation vector. Undefined if R is not a proper rotation."""
    # For R in SO(3), trace(R) = 1 + 2*cos(theta). Roundoff can push
    # the inferred cosine just outside arccos's valid interval.
    cosine = jnp.clip(0.5 * (jnp.trace(R) - 1.0), -1.0, 1.0)
    small_angle = cosine >= jnp.cos(jnp.asarray(1e-4, dtype=R.dtype))
    # The small-angle series does not need arccos, whose derivative at 1 is infinite.
    theta = jnp.arccos(jnp.where(small_angle, 0.0, cosine))

    # vee(R - R.T) / 2 = sin(theta) * axis.
    skew_vector = 0.5 * vee(R - R.T)

    def small(_):
        # Near zero, evaluate theta/sin(theta) through its series in
        # sin(theta) to avoid dividing two small quantities.
        sine_squared = jnp.dot(skew_vector, skew_vector)
        scale = 1.0 + sine_squared / 6.0 + 3.0 * sine_squared**2 / 40.0
        return scale * skew_vector

    def near_pi(_):
        # Near pi, the skew part vanishes. Recover axis*axis.T from the
        # symmetric part and use its largest diagonal entry as the pivot.
        symmetric = 0.5 * (R + R.T)
        axis_outer = 0.5 * (
            symmetric + jnp.eye(3, dtype=R.dtype)
        )
        diagonal = jnp.maximum(jnp.diag(axis_outer), 0.0)
        dominant = jnp.argmax(diagonal)

        def from_x(_):
            x = jnp.sqrt(diagonal[0])
            return jnp.array(
                [x, axis_outer[0, 1] / x, axis_outer[0, 2] / x],
                dtype=R.dtype,
            )

        def from_y(_):
            y = jnp.sqrt(diagonal[1])
            return jnp.array(
                [axis_outer[0, 1] / y, y, axis_outer[1, 2] / y],
                dtype=R.dtype,
            )

        def from_z(_):
            z = jnp.sqrt(diagonal[2])
            return jnp.array(
                [axis_outer[0, 2] / z, axis_outer[1, 2] / z, z],
                dtype=R.dtype,
            )

        axis = jax.lax.switch(dominant, (from_x, from_y, from_z), None)

        # The symmetric part determines the axis only up to sign. Away from
        # exactly pi, the residual skew part selects the principal sign.
        axis = jnp.where(
            jnp.dot(axis, skew_vector) < 0.0, -axis, axis
        )
        axis = axis / jnp.linalg.norm(axis)
        return theta * axis

    def general(_):
        # Convert sin(theta)*axis into the rotation vector theta*axis.
        return (theta / jnp.sin(theta)) * skew_vector

    return jax.lax.cond(
        small_angle,
        small,
        lambda _: jax.lax.cond(
            jnp.pi - theta < 1e-4, near_pi, general, operand=None
        ),
        operand=None,
    )

def log(R: jnp.ndarray) -> jnp.ndarray:
    r"""Evaluate the principal logarithm of rotations in \(SO(3)\).

    The result \(\mathbf{w}\) satisfies

    \[
        \operatorname{Exp}(\mathbf{w}) = R,
        \qquad
        \lVert\mathbf{w}\rVert \in [0, \pi].
    \]

    Separate numerical paths are used near the identity and near
    \(\pi\). The near-identity path avoids differentiating ``arccos`` at
    one, while the near-\(\pi\) path recovers the rotation axis from the
    symmetric part of ``R`` when its skew part becomes too small.

    Parameters
    ----------
    R : jax.Array, shape (..., 3, 3)
        One proper rotation matrix or a batch of proper rotation matrices.

    Returns
    -------
    jax.Array, shape (..., 3)
        Principal rotation vectors with the same leading batch dimensions as
        ``R``.

    Raises
    ------
    ValueError
        If the trailing shape of ``R`` is not ``(3, 3)``.

    Notes
    -----
    ``log`` assumes that every input belongs to \(SO(3)\) and does not
    validate orthogonality or determinant. Its value is undefined for an
    improper or otherwise invalid rotation. Use `log_checked` for
    host-side validation or `is_proper_rotation` inside transformed JAX
    code.

    The principal logarithm is discontinuous at rotations of angle
    \(\pi\); derivatives should be interpreted on one side of that branch
    cut.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError("log expects an array with trailing shape (3, 3)")
    if R.ndim == 2:
        return _log_single(R)

    batch_shape = R.shape[:-2]
    vectors = jax.vmap(_log_single)(R.reshape((-1, 3, 3)))
    return vectors.reshape(batch_shape + (3,))

def is_proper_rotation(R: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    r"""Test whether matrices satisfy the defining conditions of \(SO(3)\).

    A matrix is accepted when both

    \[
        \lVert R^T R-I\rVert_F < \mathtt{atol}
        \qquad\text{and}\qquad
        |\det(R)-1| < \mathtt{atol}.
    \]

    Parameters
    ----------
    R : jax.Array, shape (..., 3, 3)
        Matrices to test.
    atol : float, optional
        Absolute tolerance applied independently to the orthogonality and
        determinant errors. The default is ``1e-6``.

    Returns
    -------
    jax.Array, shape (...,), dtype bool
        One boolean result for each matrix. A scalar boolean array is returned
        for a single ``(3, 3)`` matrix.

    Raises
    ------
    ValueError
        If the trailing shape of ``R`` is not ``(3, 3)``.

    Notes
    -----
    Unlike `log_checked`, this function is suitable for use with
    `jax.jit`, `jax.vmap`, and `jax.lax.cond`.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError("is_proper_rotation expects trailing shape (3, 3)")

    identity = jnp.eye(3, dtype=R.dtype)
    transpose = jnp.swapaxes(R, -1, -2)
    ortho_error = jnp.linalg.norm(transpose @ R - identity, axis=(-2, -1))
    determinant_error = jnp.abs(jnp.linalg.det(R) - 1.0)
    return (ortho_error < atol) & (determinant_error < atol)

def log_checked(R: jnp.ndarray, atol: float = 1e-6) -> jnp.ndarray:
    r"""Evaluate the principal logarithm after validating membership in \(SO(3)\).

    Parameters
    ----------
    R : jax.Array, shape (..., 3, 3)
        One rotation matrix or a batch of rotation matrices. Every matrix must
        pass `is_proper_rotation`.
    atol : float, optional
        Absolute tolerance used for both the orthogonality and determinant
        checks. The default is ``1e-6``.

    Returns
    -------
    jax.Array, shape (..., 3)
        Principal rotation vectors returned by `log`.

    Raises
    ------
    ValueError
        If the input shape is invalid or any input matrix fails validation.

    Notes
    -----
    Validation converts the combined JAX boolean to a Python ``bool`` in order
    to raise a normal exception. Consequently, ``log_checked`` is intended for
    input boundaries, tests, and debugging rather than a JIT-compiled hot path.
    Use `is_proper_rotation` when validation must remain inside JAX
    transformations.
    """
    if not bool(jnp.all(is_proper_rotation(R, atol))):
        raise ValueError("log_checked expects a proper rotation")
    return log(R)
