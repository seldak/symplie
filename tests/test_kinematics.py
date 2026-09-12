import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from symplie.invariants import determinant_error, ortho_error
from symplie.kinematics import propagate_gyro
from symplie.so3 import exp as expSO3
from symplie.so3 import log as logSO3


def test_propagation_includes_initial_attitude():
    R0 = expSO3(jnp.array([0.1, -0.2, 0.3]))
    angular_velocity = jnp.zeros((4, 3))

    attitudes = propagate_gyro(R0, angular_velocity, dt=0.01)

    assert attitudes.shape == (5, 3, 3)
    assert jnp.allclose(attitudes[0], R0)


def test_empty_measurement_sequence_returns_initial_attitude():
    R0 = expSO3(jnp.array([0.1, -0.2, 0.3]))

    attitudes = propagate_gyro(R0, jnp.zeros((0, 3)), dt=jnp.zeros(0))

    assert attitudes.shape == (1, 3, 3)
    assert jnp.array_equal(attitudes[0], R0)


def test_zero_angular_velocity_keeps_attitude_constant():
    R0 = expSO3(jnp.array([0.1, -0.2, 0.3]))
    angular_velocity = jnp.zeros((10, 3))

    attitudes = propagate_gyro(R0, angular_velocity, dt=0.01)

    assert jnp.allclose(attitudes, jnp.broadcast_to(R0, attitudes.shape))


def test_constant_axis_matches_analytic_rotation():
    R0 = expSO3(jnp.array([0.2, -0.1, 0.3]))
    angular_velocity = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.4]), (20, 3))
    dt = 0.05

    attitudes = propagate_gyro(R0, angular_velocity, dt)
    expected = R0 @ expSO3(jnp.sum(angular_velocity * dt, axis=0))

    assert jnp.allclose(attitudes[-1], expected, atol=1e-12, rtol=1e-12)


def test_per_sample_timesteps_are_supported():
    R0 = jnp.eye(3)
    angular_velocity = jnp.broadcast_to(jnp.array([0.3, 0.0, 0.0]), (4, 3))
    dt = jnp.array([0.01, 0.02, 0.04, 0.08])

    attitudes = propagate_gyro(R0, angular_velocity, dt)
    expected = expSO3(jnp.array([0.3 * jnp.sum(dt), 0.0, 0.0]))

    assert jnp.allclose(attitudes[-1], expected, atol=1e-12, rtol=1e-12)


def test_constant_bias_is_removed_from_measurements():
    R0 = jnp.eye(3)
    true_rate = jnp.array([0.1, -0.2, 0.3])
    bias = jnp.array([0.02, -0.01, 0.03])
    measurements = jnp.broadcast_to(true_rate + bias, (25, 3))
    dt = 0.02

    attitudes = propagate_gyro(R0, measurements, dt, bias=bias)
    expected = expSO3(true_rate * (len(measurements) * dt))

    assert jnp.allclose(attitudes[-1], expected, atol=1e-12, rtol=1e-12)


def test_propagated_attitudes_remain_proper_rotations():
    R0 = expSO3(jnp.array([0.2, 0.1, -0.3]))
    angular_velocity = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [-0.4, 0.1, 0.2],
            [0.2, 0.3, -0.1],
        ]
    )

    attitudes = jax.jit(propagate_gyro)(R0, angular_velocity, 0.01)

    assert jnp.max(jax.vmap(ortho_error)(attitudes)) < 1e-12
    assert jnp.max(jax.vmap(determinant_error)(attitudes)) < 1e-12


def test_final_attitude_loss_has_finite_bias_gradient():
    R0 = jnp.eye(3)
    true_rate = jnp.array([0.1, -0.2, 0.3])
    true_bias = jnp.array([0.02, -0.01, 0.03])
    measurements = jnp.broadcast_to(true_rate + true_bias, (20, 3))
    target = expSO3(true_rate * 0.4)

    def loss(bias):
        final_attitude = propagate_gyro(R0, measurements, 0.02, bias=bias)[-1]
        error = logSO3(target.T @ final_attitude)
        return 0.5 * jnp.dot(error, error)

    gradient = jax.jit(jax.grad(loss))(jnp.zeros(3))

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 1e-4
