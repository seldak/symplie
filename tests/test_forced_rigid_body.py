import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from symplie import (
    expSO3,
    rigid_body_step,
    simulate_free_rigid_body,
    simulate_rigid_body,
)


def test_zero_torque_matches_free_rigid_body():
    R0 = expSO3(jnp.array([0.2, -0.1, 0.3]))
    pi0 = jnp.array([0.4, -0.25, 0.15])
    J = jnp.array(
        [
            [1.8, 0.2, -0.1],
            [0.2, 1.3, 0.15],
            [-0.1, 0.15, 0.9],
        ]
    )
    dt = 0.01
    steps = 20
    body_torques = jnp.zeros((steps + 1, 3))

    Rs, pis, solver_info = simulate_rigid_body(
        R0,
        pi0,
        J,
        body_torques,
        dt,
    )
    free_Rs, free_pis, free_solver_info = simulate_free_rigid_body(
        R0,
        pi0,
        J,
        dt,
        steps,
    )

    np.testing.assert_allclose(Rs, free_Rs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pis, free_pis, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        solver_info.residual_norm,
        free_solver_info.residual_norm,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        solver_info.converged,
        free_solver_info.converged,
    )


def test_rigid_body_step_matches_first_simulation_transition():
    R0 = expSO3(jnp.array([0.1, -0.2, 0.05]))
    pi0 = jnp.array([0.3, -0.1, 0.2])
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    body_torques = jnp.array(
        [
            [0.02, -0.01, 0.03],
            [0.01, 0.00, 0.02],
        ]
    )
    dt = 0.01

    R_next, pi_next, step_info = rigid_body_step(
        R0,
        pi0,
        J,
        body_torques[0],
        body_torques[1],
        dt,
    )
    Rs, pis, simulation_info = simulate_rigid_body(
        R0,
        pi0,
        J,
        body_torques,
        dt,
    )

    np.testing.assert_allclose(R_next, Rs[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pi_next, pis[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        step_info.residual_norm,
        simulation_info.residual_norm[0],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        step_info.converged,
        simulation_info.converged[0],
    )


def test_rigid_body_step_supports_vmap():
    batch_size = 4
    rotations = jnp.broadcast_to(jnp.eye(3), (batch_size, 3, 3))
    momenta = jnp.array(
        [
            [0.2, -0.1, 0.3],
            [0.1, 0.2, -0.2],
            [-0.2, 0.3, 0.1],
            [0.3, 0.1, 0.2],
        ]
    )
    torques_k = jnp.full((batch_size, 3), 0.01)
    torques_next = jnp.full((batch_size, 3), -0.01)
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))

    batched_step = jax.jit(
        jax.vmap(
            rigid_body_step,
            in_axes=(0, 0, None, 0, 0, None),
        )
    )
    Rs_next, pis_next, solver_info = batched_step(
        rotations,
        momenta,
        J,
        torques_k,
        torques_next,
        0.01,
    )

    assert Rs_next.shape == (batch_size, 3, 3)
    assert pis_next.shape == (batch_size, 3)
    assert solver_info.residual_norm.shape == (batch_size,)
    assert solver_info.converged.shape == (batch_size,)
    np.testing.assert_allclose(
        jnp.linalg.det(Rs_next),
        jnp.ones(batch_size),
        rtol=1e-12,
        atol=1e-12,
    )
    assert bool(jnp.all(solver_info.converged))


def test_constant_principal_axis_torque_has_expected_discrete_solution():
    R0 = jnp.eye(3)
    pi0 = jnp.array([0.0, 0.0, 0.4])
    J = jnp.diag(jnp.array([1.0, 1.5, 2.0]))
    torque = jnp.array([0.0, 0.0, 0.3])
    dt = 0.02
    steps = 25
    body_torques = jnp.broadcast_to(torque, (steps + 1, 3))

    Rs, pis, solver_info = simulate_rigid_body(
        R0,
        pi0,
        J,
        body_torques,
        dt,
    )

    expected_pis = pi0 + jnp.arange(steps + 1)[:, None] * dt * torque
    half_step_momenta = expected_pis[:-1, 2] + 0.5 * dt * torque[2]
    angle = jnp.sum(jnp.arcsin(dt * half_step_momenta / J[2, 2]))
    expected_R = expSO3(jnp.array([0.0, 0.0, angle]))

    np.testing.assert_allclose(pis, expected_pis, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(Rs[-1], expected_R, rtol=1e-11, atol=1e-11)
    assert bool(jnp.all(solver_info.converged))
    assert float(jnp.max(solver_info.residual_norm)) < 1e-10


def test_forced_trajectory_and_diagnostics_have_one_entry_per_node_and_step():
    steps = 6
    body_torques = jnp.array(
        [
            [0.01, -0.02, 0.03],
            [0.02, -0.01, 0.02],
            [0.01, 0.00, 0.01],
            [0.00, 0.01, 0.00],
            [-0.01, 0.02, -0.01],
            [-0.02, 0.01, -0.02],
            [-0.01, 0.00, -0.01],
        ]
    )

    Rs, pis, solver_info = simulate_rigid_body(
        jnp.eye(3),
        jnp.array([0.2, 0.1, -0.15]),
        jnp.diag(jnp.array([1.0, 1.4, 1.8])),
        body_torques,
        0.01,
    )

    assert Rs.shape == (steps + 1, 3, 3)
    assert pis.shape == (steps + 1, 3)
    assert solver_info.residual_norm.shape == (steps,)
    assert solver_info.converged.shape == (steps,)
    np.testing.assert_allclose(
        jnp.linalg.det(Rs),
        jnp.ones(steps + 1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert bool(jnp.all(solver_info.converged))


def test_forced_simulation_is_differentiable_with_respect_to_torque():
    R0 = jnp.eye(3)
    pi0 = jnp.array([0.25, -0.1, 0.2])
    J = jnp.diag(jnp.array([1.0, 1.3, 1.7]))
    body_torques = jnp.zeros((5, 3))

    def final_attitude_loss(torques):
        Rs, _, _ = simulate_rigid_body(R0, pi0, J, torques, 0.01)
        return Rs[-1, 0, 1]

    gradient = jax.jit(jax.grad(final_attitude_loss))(body_torques)

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
