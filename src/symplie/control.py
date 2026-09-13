from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .integrators import SolverInfo, rigid_body_step
from .so3 import vee


def attitude_error_vector(
    R: jnp.ndarray,
    R_target: jnp.ndarray,
) -> jnp.ndarray:
    r"""Compute the intrinsic attitude error used by geometric controllers.

    For the current body-to-spatial attitude \(R\) and target attitude \(R_d\),

    \[
        \mathbf{e}_R
        = \frac{1}{2}
          \left(R_d^T R - R^T R_d\right)^\vee.
    \]

    Parameters
    ----------
    R : jax.Array, shape (3, 3)
        Current body-to-spatial attitude.
    R_target : jax.Array, shape (3, 3)
        Desired body-to-spatial attitude.

    Returns
    -------
    jax.Array, shape (3,)
        Attitude error expressed in the current body frame.

    Notes
    -----
    The error vanishes at both relative angles \(0\) and \(\pi\). The
    180-degree attitudes are unstable critical points of the corresponding
    geometric controller, so this vector is intended for almost-global rather
    than global attitude stabilization.
    """
    relative = jnp.swapaxes(R_target, -1, -2) @ R
    return 0.5 * vee(relative - jnp.swapaxes(relative, -1, -2))


@jax.jit
def geometric_pd_torque(
    R: jnp.ndarray,
    pi: jnp.ndarray,
    J: jnp.ndarray,
    R_target: jnp.ndarray,
    attitude_gain: float,
    rate_gain: float,
) -> jnp.ndarray:
    r"""Compute body torque for fixed-target geometric PD attitude control.

    Let \(\boldsymbol{\omega}=J^{-1}\boldsymbol{\pi}\). For a stationary
    target attitude, the control law is

    \[
        \boldsymbol{\tau}
        = -k_R\mathbf{e}_R
          -k_\omega\boldsymbol{\omega}
          +\boldsymbol{\omega}\times\boldsymbol{\pi}.
    \]

    The final term cancels the gyroscopic term in Euler's rigid-body equation,

    \[
        J\dot{\boldsymbol{\omega}}
        +\boldsymbol{\omega}\times J\boldsymbol{\omega}
        = \boldsymbol{\tau}.
    \]

    Parameters
    ----------
    R : jax.Array, shape (3, 3)
        Current body-to-spatial attitude.
    pi : jax.Array, shape (3,)
        Current body-frame angular momentum.
    J : jax.Array, shape (3, 3)
        Symmetric positive-definite body inertia tensor.
    R_target : jax.Array, shape (3, 3)
        Desired fixed body-to-spatial attitude.
    attitude_gain : float or jax.Array
        Positive proportional gain \(k_R\).
    rate_gain : float or jax.Array
        Positive angular-rate gain \(k_\omega\).

    Returns
    -------
    jax.Array, shape (3,)
        Commanded body-frame torque.
    """
    omega = jnp.linalg.solve(J, pi)
    attitude_error = attitude_error_vector(R, R_target)
    return (
        -attitude_gain * attitude_error
        - rate_gain * omega
        + jnp.cross(omega, pi)
    )


@partial(
    jax.jit,
    static_argnames=("torque_fn", "steps", "newton_iters"),
)
def simulate_controlled_rigid_body(
    R0: jnp.ndarray,
    pi0: jnp.ndarray,
    J: jnp.ndarray,
    torque_fn: Callable,
    torque_params: Any,
    dt: float,
    steps: int,
    newton_iters: int = 8,
    tolerance: float = 1e-10,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, SolverInfo]:
    r"""Simulate state-dependent body torque under zero-order hold.

    At the beginning of interval \(k\), the simulator evaluates

    \[
        \boldsymbol{\tau}_k
        = f_\tau(t_k,R_k,\boldsymbol{\pi}_k,J,p)
    \]

    and holds that value over the complete interval. One call to
    rigid_body_step therefore receives the same torque at both endpoints.
    The controller is evaluated again after the state reaches the next node.

    Parameters
    ----------
    R0 : jax.Array, shape (3, 3)
        Initial body-to-spatial attitude.
    pi0 : jax.Array, shape (3,)
        Initial body-frame angular momentum.
    J : jax.Array, shape (3, 3)
        Symmetric positive-definite body inertia tensor.
    torque_fn : callable
        JAX-compatible function with signature
        torque_fn(t, R, pi, J, torque_params), returning a body-frame vector
        with shape (3,). The callable is static under JIT.
    torque_params : PyTree
        Parameters passed unchanged to torque_fn.
    dt : float or jax.Array
        Constant controller and integration timestep.
    steps : int
        Number of controlled transitions. This value is static under JIT.
    newton_iters : int, optional
        Newton iterations used for every transition. The default is 8.
    tolerance : float or jax.Array, optional
        Residual threshold used to form each convergence flag. The default is
        1e-10.

    Returns
    -------
    Rs : jax.Array, shape (steps + 1, 3, 3)
        Attitude history including R0.
    pis : jax.Array, shape (steps + 1, 3)
        Body-angular-momentum history including pi0.
    body_torques : jax.Array, shape (steps, 3)
        Torque held over each transition.
    solver_info : SolverInfo
        Residual norms and convergence flags with shape (steps,).

    Notes
    -----
    Zero-order hold models a digital controller whose output changes only at
    sample times. It is distinct from evaluating a prescribed torque at both
    nodes, as done by simulate_rigid_body.
    """
    def scan_fn(carry, step_index):
        R, pi = carry
        time = step_index * dt
        torque = torque_fn(time, R, pi, J, torque_params)

        R_next, pi_next, solver_info = rigid_body_step(
            R,
            pi,
            J,
            torque,
            torque,
            dt,
            newton_iters=newton_iters,
            tolerance=tolerance,
        )

        return (R_next, pi_next), (R_next, pi_next, torque, solver_info)

    (_, _), (Rh, ph, body_torques, solver_info) = jax.lax.scan(
        scan_fn,
        (R0, pi0),
        jnp.arange(steps),
    )

    Rs = jnp.concatenate([R0[None, ...], Rh], axis=0)
    pis = jnp.concatenate([pi0[None, ...], ph], axis=0)
    return Rs, pis, body_torques, solver_info
