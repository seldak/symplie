import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from symplie import (
    expSO3,
    geometric_pd_torque,
    geometric_tracking_torque,
    logSO3,
    simulate_controlled_rigid_body,
)


def sinusoidal_reference(time, R_base, axis, amplitude, frequency):
    angle = amplitude * jnp.sin(frequency * time)
    angle_rate = amplitude * frequency * jnp.cos(frequency * time)
    angle_acceleration = (
        -amplitude * frequency**2 * jnp.sin(frequency * time)
    )

    R_target = R_base @ expSO3(axis * angle)
    target_rate = axis * angle_rate
    target_acceleration = axis * angle_acceleration
    return R_target, target_rate, target_acceleration


def tracking_controller(time, R, pi, J, params):
    R_base, axis, amplitude, frequency, attitude_gain, rate_gain = params
    reference = sinusoidal_reference(
        time,
        R_base,
        axis,
        amplitude,
        frequency,
    )
    return geometric_tracking_torque(
        R,
        pi,
        J,
        *reference,
        attitude_gain,
        rate_gain,
    )


def tracking_errors(R, pi, J, reference):
    R_target, target_rate, _ = reference
    attitude_error = jnp.linalg.norm(logSO3(R_target.T @ R))
    target_rate_body = R.T @ R_target @ target_rate
    rate_error = jnp.linalg.norm(jnp.linalg.solve(J, pi) - target_rate_body)
    return attitude_error, rate_error


def test_tracking_torque_recovers_fixed_target_controller():
    R = expSO3(jnp.array([0.3, -0.2, 0.1]))
    pi = jnp.array([0.2, -0.1, 0.4])
    J = jnp.array(
        [
            [1.2, 0.1, -0.05],
            [0.1, 1.6, 0.08],
            [-0.05, 0.08, 2.0],
        ]
    )
    R_target = expSO3(jnp.array([-0.1, 0.25, 0.15]))

    fixed_torque = geometric_pd_torque(
        R,
        pi,
        J,
        R_target,
        attitude_gain=4.0,
        rate_gain=2.5,
    )
    tracking_torque = geometric_tracking_torque(
        R,
        pi,
        J,
        R_target,
        jnp.zeros(3),
        jnp.zeros(3),
        attitude_gain=4.0,
        rate_gain=2.5,
    )

    np.testing.assert_allclose(tracking_torque, fixed_torque, atol=1e-14)


def test_tracking_feedforward_matches_rigid_body_dynamics():
    R_target = expSO3(jnp.array([0.2, -0.3, 0.1]))
    target_rate = jnp.array([0.4, -0.2, 0.3])
    target_acceleration = jnp.array([-0.1, 0.25, 0.05])
    J = jnp.array(
        [
            [1.2, 0.1, -0.05],
            [0.1, 1.6, 0.08],
            [-0.05, 0.08, 2.0],
        ]
    )
    pi = J @ target_rate

    torque = geometric_tracking_torque(
        R_target,
        pi,
        J,
        R_target,
        target_rate,
        target_acceleration,
        attitude_gain=4.0,
        rate_gain=2.5,
    )
    expected = jnp.cross(target_rate, pi) + J @ target_acceleration

    np.testing.assert_allclose(torque, expected, atol=1e-14)


def test_controller_tracks_a_moving_attitude_reference():
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    R_base = expSO3(jnp.array([-0.2, 0.1, 0.3]))
    axis = jnp.array([1.0, -0.4, 0.7])
    axis = axis / jnp.linalg.norm(axis)
    amplitude = 0.5
    frequency = 0.7
    params = (R_base, axis, amplitude, frequency, 5.0, 3.0)
    dt = 0.01
    steps = 1000

    initial_reference = sinusoidal_reference(
        0.0,
        R_base,
        axis,
        amplitude,
        frequency,
    )
    R0 = initial_reference[0] @ expSO3(jnp.array([0.4, -0.25, 0.2]))
    target_rate_body = R0.T @ initial_reference[0] @ initial_reference[1]
    omega0 = target_rate_body + jnp.array([0.15, -0.1, 0.08])
    pi0 = J @ omega0

    Rs, pis, _, solver_info = simulate_controlled_rigid_body(
        R0,
        pi0,
        J,
        tracking_controller,
        params,
        dt,
        steps,
    )

    final_reference = sinusoidal_reference(
        steps * dt,
        R_base,
        axis,
        amplitude,
        frequency,
    )
    initial_errors = tracking_errors(Rs[0], pis[0], J, initial_reference)
    final_errors = tracking_errors(Rs[-1], pis[-1], J, final_reference)

    assert float(final_errors[0]) < 0.02 * float(initial_errors[0])
    assert float(final_errors[1]) < 0.02 * float(initial_errors[1])
    assert bool(jnp.all(solver_info.converged))


def test_tracking_simulation_is_differentiable_with_respect_to_gains():
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    R_base = expSO3(jnp.array([-0.2, 0.1, 0.3]))
    axis = jnp.array([1.0, -0.4, 0.7])
    axis = axis / jnp.linalg.norm(axis)
    R0 = R_base @ expSO3(jnp.array([0.2, -0.1, 0.15]))
    pi0 = jnp.array([0.05, -0.03, 0.02])

    def final_loss(gains):
        params = (R_base, axis, 0.5, 0.7, gains[0], gains[1])
        Rs, pis, _, _ = simulate_controlled_rigid_body(
            R0,
            pi0,
            J,
            tracking_controller,
            params,
            dt=0.01,
            steps=10,
        )
        reference = sinusoidal_reference(
            0.1,
            R_base,
            axis,
            0.5,
            0.7,
        )
        attitude_error, rate_error = tracking_errors(
            Rs[-1],
            pis[-1],
            J,
            reference,
        )
        return attitude_error**2 + rate_error**2

    gains = jnp.array([5.0, 3.0])
    gradient = jax.jit(jax.grad(final_loss))(gains)

    epsilon = 1e-4
    attitude_gain_direction = jnp.array([1.0, 0.0])
    finite_difference = (
        final_loss(gains + epsilon * attitude_gain_direction)
        - final_loss(gains - epsilon * attitude_gain_direction)
    ) / (2.0 * epsilon)

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
    np.testing.assert_allclose(
        gradient[0],
        finite_difference,
        rtol=1e-6,
        atol=1e-10,
    )
