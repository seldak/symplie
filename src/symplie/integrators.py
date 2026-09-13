from __future__ import annotations

from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp

from .so3 import exp as expSO3, hat, vee

class SolverInfo(NamedTuple):
    """Diagnostics measured after a fixed nonlinear-solver iteration budget.

    Attributes
    ----------
    residual_norm : jax.Array
        Euclidean norm of the discrete Moser–Veselov residual after the final
        Newton iteration. It is scalar for `solve_F_with_info` and has
        shape ``(steps,)`` when returned by `simulate_free_rigid_body`.
    converged : jax.Array, dtype bool
        Whether ``residual_norm`` is no greater than the requested tolerance.
        The flag reports the final residual; it does not imply that the fixed
        iteration loop terminated early.
    """

    residual_norm: jnp.ndarray
    converged: jnp.ndarray

def discrete_inertia(J: jnp.ndarray) -> jnp.ndarray:
    r"""Construct the discrete inertia used by the Moser–Veselov equation.

    The discrete inertia associated with a physical inertia tensor \(J\)
    is

    \[
        J_d = \frac{1}{2}\operatorname{tr}(J)I - J.
    \]

    The definition is coordinate-independent; ``J`` need not be diagonal or
    expressed in principal axes.

    Parameters
    ----------
    J : jax.Array, shape (3, 3)
        Body inertia tensor.

    Returns
    -------
    jax.Array, shape (3, 3)
        Discrete inertia with the same dtype as ``J``.
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
    r"""Solve the discrete Moser–Veselov equation for one relative rotation.

    Given body angular momentum \(\boldsymbol{\pi}_k\), inertia \(J\),
    and timestep \(h\), this function finds \(F_k\in SO(3)\) satisfying

    \[
        F_k J_d - J_d F_k^T
        = h\,\widehat{\boldsymbol{\pi}_k},
    \]

    where \(J_d=\tfrac{1}{2}\operatorname{tr}(J)I-J\). The unknown is
    parameterized as \(F_k=\operatorname{Exp}(\mathbf{g})\) and solved with
    a fixed number of Newton iterations. Each iteration evaluates a small set
    of deterministic damping factors and retains the candidate with the lowest
    residual norm.

    Parameters
    ----------
    pi : jax.Array, shape (3,)
        Body-frame angular momentum at the start of the step.
    J : jax.Array, shape (3, 3)
        Symmetric positive-definite body inertia tensor. It may be non-diagonal.
    h : float or jax.Array
        Timestep.
    newton_iters : int, optional
        Number of Newton iterations. The value is static under JIT compilation.
        The default is ``8``.
    tolerance : float or jax.Array, optional
        Maximum final residual norm accepted as converged. The default is
        ``1e-10``.

    Returns
    -------
    F : jax.Array, shape (3, 3)
        Relative rotation for the step.
    info : SolverInfo
        Final residual norm and convergence flag.

    Notes
    -----
    Every call consumes the complete ``newton_iters`` budget. Callers should
    inspect ``info.converged`` before trusting the returned step; a failed solve
    does not raise an exception or stop a surrounding simulation.
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
    """Solve one Moser–Veselov step and return only its relative rotation.

    Parameters
    ----------
    pi : jax.Array, shape (3,)
        Body-frame angular momentum at the start of the step.
    J : jax.Array, shape (3, 3)
        Symmetric positive-definite body inertia tensor.
    h : float or jax.Array
        Timestep.
    newton_iters : int, optional
        Fixed Newton iteration budget. The default is ``8``.

    Returns
    -------
    jax.Array, shape (3, 3)
        Relative rotation for the step.

    Notes
    -----
    This convenience function discards the nonlinear-solver diagnostics. Use
    `solve_F_with_info` whenever convergence must be checked.
    """
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
    r"""Simulate torque-free rigid-body motion on \(SO(3)\).

    The state is \((R_k,\boldsymbol{\pi}_k)\), where \(R_k\) maps body
    coordinates to spatial coordinates and \(\boldsymbol{\pi}_k\) is body
    angular momentum. For each transition, `solve_F_with_info` computes a
    relative rotation \(F_k\), followed by

    \[
        R_{k+1} = R_k F_k,
        \qquad
        \boldsymbol{\pi}_{k+1} = F_k^T\boldsymbol{\pi}_k.
    \]

    This discrete update preserves spatial angular momentum algebraically:

    \[
        R_{k+1}\boldsymbol{\pi}_{k+1}
        = R_k\boldsymbol{\pi}_k.
    \]

    Parameters
    ----------
    R0 : jax.Array, shape (3, 3)
        Initial body-to-spatial attitude. The caller is responsible for
        providing a proper rotation.
    pi0 : jax.Array, shape (3,)
        Initial body-frame angular momentum.
    J : jax.Array, shape (3, 3)
        Symmetric positive-definite body inertia tensor. It may be non-diagonal.
    dt : float or jax.Array
        Constant simulation timestep.
    steps : int
        Number of discrete transitions. This value is static under JIT
        compilation.
    newton_iters : int, optional
        Newton iterations used for every transition. The default is ``8`` and
        the value is static under JIT compilation.
    tolerance : float or jax.Array, optional
        Residual threshold used to form each convergence flag. The default is
        ``1e-10``.

    Returns
    -------
    Rs : jax.Array, shape (steps + 1, 3, 3)
        Attitude history including ``R0``.
    pis : jax.Array, shape (steps + 1, 3)
        Body-angular-momentum history including ``pi0``.
    solver_info : SolverInfo
        Arrays of residual norms and convergence flags with shape ``(steps,)``;
        there is one diagnostic entry per transition.

    Notes
    -----
    The function returns trajectories even when one or more nonlinear solves
    fail the requested tolerance. Always inspect ``solver_info.converged``
    before using the result. The complete simulation is compatible with
    `jax.jit` and automatic differentiation.
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

@partial(jax.jit, static_argnames=("newton_iters",))
def simulate_rigid_body(
    R0: jnp.ndarray,
    pi0: jnp.ndarray,
    J: jnp.ndarray,
    body_torques: jnp.ndarray,
    dt: float,
    newton_iters: int = 8,
    tolerance: float = 1e-10,
) -> tuple[jnp.ndarray, jnp.ndarray, SolverInfo]:
    r"""Simulate a rigid body driven by prescribed body-frame torques.

    The state at node \(k\) is \((R_k,\boldsymbol{\pi}_k)\), where \(R_k\)
    maps body coordinates to spatial coordinates and
    \(\boldsymbol{\pi}_k\) is body angular momentum. The supplied torque
    \(\boldsymbol{\tau}_k\) is also expressed in body coordinates.

    For a timestep \(h\), the forced Lie-group variational update first finds
    \(F_k\in SO(3)\) from

    \[
        F_k J_d - J_d F_k^T
        = h\,\widehat{
            \boldsymbol{\pi}_k + \frac{h}{2}\boldsymbol{\tau}_k
          },
        \qquad
        J_d = \frac{1}{2}\operatorname{tr}(J)I-J.
    \]

    The state then advances according to

    \[
        R_{k+1} = R_k F_k,
    \]

    \[
        \boldsymbol{\pi}_{k+1}
        = F_k^T\boldsymbol{\pi}_k
        + \frac{h}{2}F_k^T\boldsymbol{\tau}_k
        + \frac{h}{2}\boldsymbol{\tau}_{k+1}.
    \]

    The two half-step torque terms are the discrete forces at the ends of the
    interval. Consequently, a trajectory with ``steps`` transitions requires
    ``steps + 1`` torque samples. Setting every torque to zero recovers
    `simulate_free_rigid_body`.

    Parameters
    ----------
    R0 : jax.Array, shape (3, 3)
        Initial body-to-spatial attitude. The caller is responsible for
        providing a proper rotation.
    pi0 : jax.Array, shape (3,)
        Initial body-frame angular momentum.
    J : jax.Array, shape (3, 3)
        Symmetric positive-definite body inertia tensor. It may be non-diagonal.
    body_torques : jax.Array, shape (steps + 1, 3)
        Prescribed body-frame torque at every state node, including both the
        initial and final nodes. The number of transitions is inferred from
        this leading dimension.
    dt : float or jax.Array
        Constant simulation timestep \(h\).
    newton_iters : int, optional
        Newton iterations used for every transition. The default is ``8`` and
        the value is static under JIT compilation.
    tolerance : float or jax.Array, optional
        Residual threshold used to form each convergence flag. The default is
        ``1e-10``.

    Returns
    -------
    Rs : jax.Array, shape (steps + 1, 3, 3)
        Attitude history including ``R0``.
    pis : jax.Array, shape (steps + 1, 3)
        Body-angular-momentum history including ``pi0``.
    solver_info : SolverInfo
        Arrays of residual norms and convergence flags with shape ``(steps,)``;
        there is one diagnostic entry per transition.

    Notes
    -----
    The nonlinear equation is the same Moser--Veselov solve used by
    `simulate_free_rigid_body`, with the first half of the discrete torque
    impulse included in its momentum argument. As in the free-body simulator,
    the returned trajectory must not be trusted unless every entry of
    ``solver_info.converged`` is true.

    References
    ----------
    T. Lee, N. H. McClamroch, and M. Leok, "A Lie Group Variational
    Integrator for the Attitude Dynamics of a Rigid Body with Applications to
    the 3D Pendulum," *Proceedings of the 2005 IEEE Conference on Control
    Applications*, pp. 962--967, 2005. doi:10.1109/CCA.2005.1507254.
    """
    def scan_fn(carry, torque_pair):
        R, pi = carry
        torque_k, torque_next = torque_pair

        pi_half = pi + 0.5 * dt * torque_k

        F, solver_info = solve_F_with_info(
            pi_half,
            J,
            dt,
            newton_iters=newton_iters,
            tolerance=tolerance,
        )

        R_next = R @ F
        pi_next = F.T @ pi_half + 0.5 * dt * torque_next

        return (R_next, pi_next), (R_next, pi_next, solver_info)

    (_, _), (Rh, ph, solver_info) = jax.lax.scan(
        scan_fn,
        (R0, pi0),
        (body_torques[:-1], body_torques[1:]),
    )

    Rs = jnp.concatenate([R0[None, ...], Rh], axis=0)
    pis = jnp.concatenate([pi0[None, ...], ph], axis=0)
    return Rs, pis, solver_info
