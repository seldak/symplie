from __future__ import annotations

from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp

from .so3 import exp as expSO3, hat, vee

class SolverInfo(NamedTuple):
    """Diagnostics measured after a fixed nonlinear-solver iteration budget."""

    residual_norm: jnp.ndarray
    converged: jnp.ndarray

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
def solve_F_with_info(pi, J, h, newton_iters=8, tolerance=1e-10):
    """
    Solve for relative rotation F ∈ SO(3) for one VI step.
    Runs exactly ``newton_iters`` Newton iterations. ``converged`` is true when
    the final residual is at most ``tolerance``.
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
        # Deterministic backtracking: retain the candidate with least residual.
        scales = jnp.array([0.0, 0.25, 0.5, 1.0], dtype=g.dtype)
        candidates = g[None, :] - scales[:, None] * delta[None, :]
        norms = jax.vmap(lambda candidate: jnp.linalg.norm(_residual(candidate, pi, Jd, h)))(candidates)
        return candidates[jnp.argmin(norms)]

    g = jax.lax.fori_loop(0, newton_iters, newton_body, g0)
    residual_norm = jnp.linalg.norm(_residual(g, pi, Jd, h))
    return expSO3(g), SolverInfo(residual_norm, residual_norm <= tolerance)

@partial(jax.jit, static_argnames=("newton_iters",))
def solve_F(pi, J, h, newton_iters=8):
    """Return the relative rotation; use ``solve_F_with_info`` to inspect convergence."""
    return solve_F_with_info(pi, J, h, newton_iters)[0]

@partial(jax.jit, static_argnames=("steps", "newton_iters"))
def simulate_free_rigid_body(
    R0: jnp.ndarray,
    pi0: jnp.ndarray,
    J: jnp.ndarray,
    dt: float,
    steps: int,
    newton_iters: int = 8,
    tolerance: float = 1e-10,
) -> tuple[jnp.ndarray, jnp.ndarray, SolverInfo]:
    """
    Simulate torque-free rigid body on SO(3).
    State is (R, pi) with pi in body coordinates.
    Returns:
      Rs: (steps+1, 3, 3)
      pis: (steps+1, 3)
      solver_info: per-step residual norms and convergence flags
    """
    def scan_fn(carry, _):
        R, pi = carry
        F, solver_info = solve_F_with_info(
            pi,
            J,
            dt,
            newton_iters=newton_iters,
            tolerance=tolerance,
        )
        Rn = R @ F
        pin = F.T @ pi
        return (Rn, pin), (Rn, pin, solver_info)

    (Rf, pif), (Rh, ph, solver_info) = jax.lax.scan(
        scan_fn,
        (R0, pi0),
        xs=None,
        length=steps,
    )

    Rs = jnp.concatenate([R0[None, ...], Rh], axis=0)
    pis = jnp.concatenate([pi0[None, ...], ph], axis=0)
    return Rs, pis, solver_info
