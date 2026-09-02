from __future__ import annotations

import jax
import jax.numpy as jnp

def hat(w: jnp.ndarray) -> jnp.ndarray:
    """so(3) hat operator: (..., 3) -> (..., 3, 3)."""
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
    """so(3) vee operator: (..., 3, 3) -> (..., 3)."""
    return jnp.stack(
        [W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]],
        axis=-1,
    )

def exp(w: jnp.ndarray) -> jnp.ndarray:
    """SO(3) exponential map using Rodrigues, stable for small angles."""
    theta = jnp.linalg.norm(w)
    K = hat(w)
    I = jnp.eye(3, dtype=w.dtype)

    def small(_):
        # Series expansions:
        # sin(theta)/theta ≈ 1 - t^2/6 + t^4/120
        # (1-cos(theta))/theta^2 ≈ 1/2 - t^2/24 + t^4/720
        t2 = theta * theta
        a = 1.0 - t2 / 6.0 + (t2 * t2) / 120.0
        b = 0.5 - t2 / 24.0 + (t2 * t2) / 720.0
        return I + a * K + b * (K @ K)

    def general(_):
        a = jnp.sin(theta) / theta
        b = (1.0 - jnp.cos(theta)) / (theta * theta)
        return I + a * K + b * (K @ K)

    return jax.lax.cond(theta < 1e-7, small, general, operand=None)

def log(R: jnp.ndarray) -> jnp.ndarray:
    """SO(3) logarithm map returning the principal rotation vector."""
    # For R in SO(3), trace(R) = 1 + 2*cos(theta). Roundoff can push
    # the inferred cosine just outside arccos's valid interval.
    cosine = jnp.clip(0.5 * (jnp.trace(R) - 1.0), -1.0, 1.0)
    theta = jnp.arccos(cosine)

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
        theta < 1e-4,
        small,
        lambda _: jax.lax.cond(
            jnp.pi - theta < 1e-4, near_pi, general, operand=None
        ),
        operand=None,
    )
