import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from symplie import (
    attitude_error_vector,
    expSO3,
    geometric_pd_torque,
    simulate_controlled_rigid_body,
    simulate_rigid_body,
)


def constant_torque(_, __, ___, ____, torque):
    return torque


def fixed_attitude_controller(_, R, pi, J, params):
    R_target, attitude_gain, rate_gain = params
    return geometric_pd_torque(
        R,
        pi,
        J,
        R_target,
        attitude_gain,
        rate_gain,
    )


def test_zero_order_hold_matches_prescribed_constant_torque():
    R0 = expSO3(jnp.array([0.2, -0.1, 0.05]))
    pi0 = jnp.array([0.3, -0.2, 0.1])
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    torque = jnp.array([0.02, -0.01, 0.03])
    dt = 0.01
    steps = 12

    Rs, pis, applied_torques, controlled_info = simulate_controlled_rigid_body(
        R0,
        pi0,
        J,
        constant_torque,
        torque,
        dt,
        steps,
    )
    prescribed_Rs, prescribed_pis, prescribed_info = simulate_rigid_body(
        R0,
        pi0,
        J,
        jnp.broadcast_to(torque, (steps + 1, 3)),
        dt,
    )

    np.testing.assert_allclose(Rs, prescribed_Rs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pis, prescribed_pis, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        applied_torques,
        jnp.broadcast_to(torque, (steps, 3)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        controlled_info.residual_norm,
        prescribed_info.residual_norm,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        controlled_info.converged,
        prescribed_info.converged,
    )


def test_geometric_pd_torque_vanishes_at_target_equilibrium():
    R_target = expSO3(jnp.array([0.3, -0.2, 0.1]))
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))

    torque = geometric_pd_torque(
        R_target,
        jnp.zeros(3),
        J,
        R_target,
        attitude_gain=4.0,
        rate_gain=2.0,
    )

    np.testing.assert_allclose(torque, jnp.zeros(3), rtol=0.0, atol=1e-14)


def test_geometric_controller_reduces_attitude_and_rate_error():
    R_target = expSO3(jnp.array([-0.2, 0.1, 0.3]))
    R0 = R_target @ expSO3(jnp.array([0.6, -0.4, 0.3]))
    pi0 = jnp.array([0.2, -0.15, 0.1])
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    params = (R_target, 4.0, 2.5)

    Rs, pis, body_torques, solver_info = simulate_controlled_rigid_body(
        R0,
        pi0,
        J,
        fixed_attitude_controller,
        params,
        dt=0.01,
        steps=800,
    )

    initial_attitude_error = jnp.linalg.norm(
        attitude_error_vector(Rs[0], R_target)
    )
    final_attitude_error = jnp.linalg.norm(
        attitude_error_vector(Rs[-1], R_target)
    )
    initial_rate = jnp.linalg.norm(jnp.linalg.solve(J, pis[0]))
    final_rate = jnp.linalg.norm(jnp.linalg.solve(J, pis[-1]))

    assert float(final_attitude_error) < 0.02 * float(initial_attitude_error)
    assert float(final_rate) < 0.02 * float(initial_rate)
    assert body_torques.shape == (800, 3)
    assert bool(jnp.all(solver_info.converged))


def test_controlled_simulation_is_differentiable_with_respect_to_gains():
    R_target = jnp.eye(3)
    R0 = expSO3(jnp.array([0.2, -0.1, 0.15]))
    pi0 = jnp.array([0.05, -0.03, 0.02])
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))

    def final_loss(gains):
        params = (R_target, gains[0], gains[1])
        Rs, pis, _, _ = simulate_controlled_rigid_body(
            R0,
            pi0,
            J,
            fixed_attitude_controller,
            params,
            dt=0.01,
            steps=10,
        )
        return (
            jnp.sum(attitude_error_vector(Rs[-1], R_target) ** 2)
            + jnp.sum(pis[-1] ** 2)
        )

    gradient = jax.jit(jax.grad(final_loss))(jnp.array([4.0, 2.5]))

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
