import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from symplie import (
    expSO3,
    simulate_free_rigid_body,
    simulate_rigid_body,
    spatial_momentum,
)


def assert_same_trajectory(first, second):
    first_Rs, first_pis, first_info = first
    second_Rs, second_pis, second_info = second

    np.testing.assert_allclose(first_Rs, second_Rs, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(first_pis, second_pis, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        first_info.residual_norm,
        second_info.residual_norm,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_array_equal(first_info.converged, second_info.converged)


def test_free_simulation_accepts_one_timestep_per_transition():
    R0 = expSO3(jnp.array([0.2, -0.1, 0.3]))
    pi0 = jnp.array([0.4, -0.25, 0.15])
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    steps = 12
    dt = 0.01

    scalar_result = simulate_free_rigid_body(R0, pi0, J, dt, steps)
    array_result = simulate_free_rigid_body(
        R0,
        pi0,
        J,
        jnp.full((steps,), dt),
        steps,
    )

    assert_same_trajectory(scalar_result, array_result)


def test_forced_simulation_accepts_one_timestep_per_transition():
    R0 = expSO3(jnp.array([0.2, -0.1, 0.3]))
    pi0 = jnp.array([0.4, -0.25, 0.15])
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    steps = 12
    dt = 0.01
    torques = jnp.linspace(-0.02, 0.03, steps + 1)[:, None] * jnp.array(
        [1.0, -0.5, 0.25]
    )

    scalar_result = simulate_rigid_body(R0, pi0, J, torques, dt)
    array_result = simulate_rigid_body(
        R0,
        pi0,
        J,
        torques,
        jnp.full((steps,), dt),
    )

    assert_same_trajectory(scalar_result, array_result)


def test_variable_timestep_principal_axis_solution():
    R0 = jnp.eye(3)
    pi0 = jnp.array([0.0, 0.0, 0.4])
    J = jnp.diag(jnp.array([1.0, 1.5, 2.0]))
    torque = jnp.array([0.0, 0.0, 0.3])
    timesteps = jnp.array([0.012, 0.018, 0.009, 0.021, 0.015])
    torques = jnp.broadcast_to(torque, (timesteps.size + 1, 3))

    Rs, pis, solver_info = simulate_rigid_body(
        R0,
        pi0,
        J,
        torques,
        timesteps,
    )

    elapsed = jnp.concatenate([jnp.zeros(1), jnp.cumsum(timesteps)])
    expected_pis = pi0 + elapsed[:, None] * torque
    half_step_momenta = expected_pis[:-1, 2] + 0.5 * timesteps * torque[2]
    angle = jnp.sum(
        jnp.arcsin(timesteps * half_step_momenta / J[2, 2])
    )
    expected_R = expSO3(jnp.array([0.0, 0.0, angle]))

    np.testing.assert_allclose(pis, expected_pis, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Rs[-1], expected_R, rtol=1e-12, atol=1e-12)
    assert bool(jnp.all(solver_info.converged))


def test_variable_timestep_free_simulation_preserves_spatial_momentum():
    R0 = expSO3(jnp.array([0.2, -0.1, 0.3]))
    pi0 = jnp.array([0.4, -0.25, 0.15])
    J = jnp.array(
        [
            [1.8, 0.2, -0.1],
            [0.2, 1.3, 0.15],
            [-0.1, 0.15, 0.9],
        ]
    )
    timesteps = jnp.array([0.008, 0.012, 0.009, 0.011, 0.007, 0.013])

    Rs, pis, solver_info = simulate_free_rigid_body(
        R0,
        pi0,
        J,
        timesteps,
        steps=timesteps.size,
    )

    momenta = jax.vmap(spatial_momentum)(Rs, pis)
    expected_momenta = jnp.broadcast_to(momenta[0], momenta.shape)
    np.testing.assert_allclose(
        momenta,
        expected_momenta,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.linalg.det(Rs),
        jnp.ones(timesteps.size + 1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert bool(jnp.all(solver_info.converged))


def test_forced_simulation_is_differentiable_with_respect_to_timesteps():
    R0 = jnp.eye(3)
    pi0 = jnp.array([0.25, -0.1, 0.2])
    J = jnp.diag(jnp.array([1.0, 1.3, 1.7]))
    torques = jnp.zeros((5, 3))

    def final_attitude_loss(timesteps):
        Rs, _, _ = simulate_rigid_body(R0, pi0, J, torques, timesteps)
        return Rs[-1, 0, 1]

    gradient = jax.jit(jax.grad(final_attitude_loss))(jnp.full((4,), 0.01))

    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


@pytest.mark.parametrize("simulator", ["free", "forced"])
def test_simulation_rejects_wrong_timestep_shape(simulator):
    R0 = jnp.eye(3)
    pi0 = jnp.array([0.2, -0.1, 0.3])
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    wrong_shape = jnp.full((4,), 0.01)

    with pytest.raises(ValueError, match="dt must be scalar or have shape"):
        if simulator == "free":
            simulate_free_rigid_body(R0, pi0, J, wrong_shape, steps=3)
        else:
            torques = jnp.zeros((4, 3))
            simulate_rigid_body(R0, pi0, J, torques, wrong_shape)
