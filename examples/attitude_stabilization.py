"""Stabilize a rigid body at a fixed attitude with geometric PD control."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from symplie import (
    attitude_error_vector,
    expSO3,
    geometric_pd_torque,
    simulate_controlled_rigid_body,
)


def controller(_, R, pi, J, params):
    R_target, attitude_gain, rate_gain = params
    return geometric_pd_torque(
        R,
        pi,
        J,
        R_target,
        attitude_gain,
        rate_gain,
    )


def main():
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    R_target = expSO3(jnp.array([-0.2, 0.1, 0.3]))
    R0 = R_target @ expSO3(jnp.array([0.6, -0.4, 0.3]))
    pi0 = jnp.array([0.2, -0.15, 0.1])
    params = (R_target, 4.0, 2.5)

    Rs, pis, body_torques, solver_info = simulate_controlled_rigid_body(
        R0,
        pi0,
        J,
        controller,
        params,
        dt=0.01,
        steps=800,
    )
    if not bool(jnp.all(solver_info.converged)):
        raise RuntimeError("A rigid-body step did not converge")

    initial_error = jnp.linalg.norm(attitude_error_vector(Rs[0], R_target))
    final_error = jnp.linalg.norm(attitude_error_vector(Rs[-1], R_target))
    final_rate = jnp.linalg.norm(jnp.linalg.solve(J, pis[-1]))
    peak_torque = jnp.max(jnp.linalg.norm(body_torques, axis=1))

    print(f"initial attitude error: {float(initial_error):.6f}")
    print(f"final attitude error:   {float(final_error):.6f}")
    print(f"final angular rate:     {float(final_rate):.6f}")
    print(f"peak control torque:    {float(peak_torque):.6f}")


if __name__ == "__main__":
    main()
