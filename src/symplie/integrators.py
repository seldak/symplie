from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp

from .so3 import exp as expSO3, hat, vee

def discrete_inertia(J: jnp.ndarray) -> jnp.ndarray:
    """
    Discrete inertia matrix used in Moser–Veselov style rigid body VI:
      Jd = 0.5*tr(J)*I - J
    """
    I = jnp.eye(3, dtype=J.dtype)
    return 0.5 * jnp.trace(J) * I - J

def _residual(g: jnp.ndarray, pi: jnp.ndarray, Jd: jnp.ndarray, h: float) -> jnp.ndarray:
    """
    Residual r(g)=vee(F Jd - Jd F^T - h*hat(pi)), where F=Exp(g).
    Solve r(g)=0 for g in R^3.
    """
    F = expSO3(g)
    M = F @ Jd - Jd @ F.T - h * hat(pi)
    return vee(M)

@partial(jax.jit, static_argnames=("newton_iters",))
def solve_F(pi: jnp.ndarray, J: jnp.ndarray, h: float, newton_iters: int = 6) -> jnp.ndarray:
    """
    Solve for relative rotation F ∈ SO(3) for one VI step.
    Uses fixed-iteration Newton for determinism/JIT.
    """
    Jd = discrete_inertia(J)

    # Initial guess: g0 ≈ h * omega  (omega = J^{-1} pi)
    omega = jnp.linalg.solve(J, pi)
    g0 = h * omega

    def newton_body(_, g):
        r = _residual(g, pi, Jd, h)
        # Jacobian dr/dg (3x3)
        Jg = jax.jacfwd(lambda gg: _residual(gg, pi, Jd, h))(g)

        # Tiny Tikhonov regularization for robustness
        Jg = Jg + (1e-12 * jnp.eye(3, dtype=Jg.dtype))

        delta = jnp.linalg.solve(Jg, r)
        return g - delta

    g = jax.lax.fori_loop(0, newton_iters, newton_body, g0)
    return expSO3(g)

@partial(jax.jit, static_argnames=("steps", "newton_iters"))
def simulate_free_rigid_body(
    R0: jnp.ndarray,
    pi0: jnp.ndarray,
    J: jnp.ndarray,
    dt: float,
    steps: int,
    newton_iters: int = 6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate torque-free rigid body on SO(3).
    State is (R, pi) with pi in body coordinates.
    Returns:
      Rs: (steps+1, 3, 3)
      pis: (steps+1, 3)
    """
    def scan_fn(carry, _):
        R, pi = carry
        F = solve_F(pi, J, dt, newton_iters=newton_iters)
        Rn = R @ F
        pin = F.T @ pi
        return (Rn, pin), (Rn, pin)

    (Rf, pif), (Rh, ph) = jax.lax.scan(scan_fn, (R0, pi0), xs=None, length=steps)

    Rs = jnp.concatenate([R0[None, ...], Rh], axis=0)
    pis = jnp.concatenate([pi0[None, ...], ph], axis=0)
    return Rs, pis

