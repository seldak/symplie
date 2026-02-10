from __future__ import annotations

import jax
import jax.numpy as jnp

def hat(w: jnp.ndarray) -> jnp.ndarray:
    """so(3) hat operator: R^3 -> skew(3)."""
    wx, wy, wz = w
    return jnp.array(
        [[0.0, -wz,  wy],
         [wz,  0.0, -wx],
         [-wy, wx,  0.0]],
        dtype=w.dtype
    )

def vee(W: jnp.ndarray) -> jnp.ndarray:
    """so(3) vee operator: skew(3) -> R^3."""
    return jnp.array([W[2, 1], W[0, 2], W[1, 0]], dtype=W.dtype)

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

