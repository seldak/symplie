"""Track a smooth attitude trajectory with geometric feedback."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from symplie import (
    expSO3,
    geometric_tracking_torque,
    logSO3,
    simulate_controlled_rigid_body,
)


def reference(time, R_base, axis, amplitude, frequency):
    angle = amplitude * jnp.sin(frequency * time)
    angle_rate = amplitude * frequency * jnp.cos(frequency * time)
    angle_acceleration = (
        -amplitude * frequency**2 * jnp.sin(frequency * time)
    )

    R_target = R_base @ expSO3(axis * angle)
    target_rate = axis * angle_rate
    target_acceleration = axis * angle_acceleration
    return R_target, target_rate, target_acceleration


def controller(time, R, pi, J, params):
    R_base, axis, amplitude, frequency, attitude_gain, rate_gain = params
    target = reference(time, R_base, axis, amplitude, frequency)
    return geometric_tracking_torque(
        R,
        pi,
        J,
        *target,
        attitude_gain,
        rate_gain,
    )


def main():
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    R_base = expSO3(jnp.array([-0.2, 0.1, 0.3]))
    axis = jnp.array([1.0, -0.4, 0.7])
    axis = axis / jnp.linalg.norm(axis)
    amplitude = 0.5
    frequency = 0.7
    attitude_gain = 5.0
    rate_gain = 3.0
    params = (
        R_base,
        axis,
        amplitude,
        frequency,
        attitude_gain,
        rate_gain,
    )
    dt = 0.01
    steps = 1000

    initial_target = reference(0.0, R_base, axis, amplitude, frequency)
    R0 = initial_target[0] @ expSO3(jnp.array([0.4, -0.25, 0.2]))
    target_rate_body = R0.T @ initial_target[0] @ initial_target[1]
    omega0 = target_rate_body + jnp.array([0.15, -0.1, 0.08])
    pi0 = J @ omega0

    Rs, pis, body_torques, solver_info = simulate_controlled_rigid_body(
        R0,
        pi0,
        J,
        controller,
        params,
        dt,
        steps,
    )
    if not bool(jnp.all(solver_info.converged)):
        raise RuntimeError("A rigid-body step did not converge")

    final_target = reference(
        steps * dt,
        R_base,
        axis,
        amplitude,
        frequency,
    )
    final_attitude_error = jnp.linalg.norm(
        logSO3(final_target[0].T @ Rs[-1])
    )
    target_rate_body = Rs[-1].T @ final_target[0] @ final_target[1]
    final_rate_error = jnp.linalg.norm(
        jnp.linalg.solve(J, pis[-1]) - target_rate_body
    )
    peak_torque = jnp.max(jnp.linalg.norm(body_torques, axis=1))
    max_residual = jnp.max(solver_info.residual_norm)

    print(f"final attitude error: {float(final_attitude_error):.6f} rad")
    print(f"final rate error:     {float(final_rate_error):.6f} rad/s")
    print(f"peak control torque:  {float(peak_torque):.6f}")
    print(f"maximum residual:     {float(max_residual):.3e}")


if __name__ == "__main__":
    main()
