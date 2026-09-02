import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from symplie.se3 import exp as expSE3
from symplie.se3 import hat
from symplie.se3 import left_jacobian_SO3
from symplie.se3 import left_jacobian_inverse_SO3
from symplie.se3 import log as logSE3
from symplie.se3 import vee
from symplie.so3 import exp as expSO3


@pytest.mark.parametrize(
    "xi",
    [
        jnp.zeros(6, dtype=jnp.float64),
        jnp.array([1.0, 2.0, 3.0, 0.1, -0.2, 0.3], dtype=jnp.float64),
        jnp.array([-0.4, 0.7, 1.2, -1.1, 0.5, 0.2], dtype=jnp.float64),
    ],
)
def test_vee_hat_recovers_twist(xi):
    recovered = vee(hat(xi))

    assert recovered.shape == (6,)
    assert recovered.dtype == xi.dtype
    assert jnp.allclose(recovered, xi, atol=0.0, rtol=0.0)


def test_hat_has_se3_matrix_structure():
    xi = jnp.array([1.0, 2.0, 3.0, 0.1, -0.2, 0.3], dtype=jnp.float64)

    X = hat(xi)

    expected = jnp.array(
        [
            [0.0, -0.3, -0.2, 1.0],
            [0.3, 0.0, -0.1, 2.0],
            [0.2, 0.1, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    assert X.shape == (4, 4)
    assert X.dtype == xi.dtype
    assert jnp.allclose(X, expected, atol=0.0, rtol=0.0)


def test_hat_vee_recovers_se3_matrix():
    X = jnp.array(
        [
            [0.0, -0.6, -0.2, 1.5],
            [0.6, 0.0, 0.4, -2.0],
            [0.2, -0.4, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    recovered = hat(vee(X))

    assert jnp.allclose(recovered, X, atol=0.0, rtol=0.0)


def test_hat_and_vee_are_jittable():
    xi = jnp.array([0.3, -0.8, 1.1, 0.2, 0.4, -0.5], dtype=jnp.float64)

    X = jax.jit(hat)(xi)
    recovered = jax.jit(vee)(X)

    assert jnp.allclose(recovered, xi, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "phi",
    [
        jnp.zeros(3, dtype=jnp.float64),
        jnp.array([1e-12, -2e-12, 3e-12], dtype=jnp.float64),
        jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64),
        jnp.array([2.0, -1.0, 0.5], dtype=jnp.float64),
    ],
)
def test_left_jacobian_inverse(phi):
    J = left_jacobian_SO3(phi)
    J_inverse = left_jacobian_inverse_SO3(phi)

    assert jnp.all(jnp.isfinite(J))
    assert jnp.all(jnp.isfinite(J_inverse))
    assert jnp.allclose(J @ J_inverse, jnp.eye(3), atol=1e-10, rtol=1e-10)


def test_exp_zero_is_identity():
    T = expSE3(jnp.zeros(6, dtype=jnp.float64))

    assert jnp.array_equal(T, jnp.eye(4, dtype=jnp.float64))


def test_exp_pure_translation():
    rho = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64)
    xi = jnp.concatenate((rho, jnp.zeros(3, dtype=jnp.float64)))

    T = expSE3(xi)

    assert jnp.array_equal(T[:3, :3], jnp.eye(3, dtype=jnp.float64))
    assert jnp.array_equal(T[:3, 3], rho)
    assert jnp.array_equal(T[3], jnp.array([0.0, 0.0, 0.0, 1.0]))


def test_exp_pure_rotation():
    phi = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64)
    xi = jnp.concatenate((jnp.zeros(3, dtype=jnp.float64), phi))

    T = expSE3(xi)

    assert jnp.allclose(T[:3, :3], expSO3(phi), atol=1e-12, rtol=1e-12)
    assert jnp.array_equal(T[:3, 3], jnp.zeros(3, dtype=jnp.float64))
    assert jnp.array_equal(T[3], jnp.array([0.0, 0.0, 0.0, 1.0]))


_NEAR_PI_AXIS = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64)
_NEAR_PI_AXIS = _NEAR_PI_AXIS / jnp.linalg.norm(_NEAR_PI_AXIS)


@pytest.mark.parametrize(
    "xi",
    [
        jnp.zeros(6, dtype=jnp.float64),
        jnp.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.3], dtype=jnp.float64),
        jnp.array([1.0, -2.0, 0.5, 0.1, -0.2, 0.3], dtype=jnp.float64),
        jnp.array(
            [0.4, -0.7, 1.2, 1e-12, -2e-12, 3e-12], dtype=jnp.float64
        ),
        jnp.concatenate(
            (
                jnp.array([0.4, -0.7, 1.2], dtype=jnp.float64),
                (jnp.pi - 1e-6) * _NEAR_PI_AXIS,
            )
        ),
    ],
)
def test_log_exp_recovers_principal_twist(xi):
    recovered = logSE3(expSE3(xi))

    assert jnp.all(jnp.isfinite(recovered))
    assert jnp.allclose(recovered, xi, atol=1e-8, rtol=1e-8)


def test_exp_and_log_are_jittable():
    xi = jnp.array([1.0, -2.0, 0.5, 0.1, -0.2, 0.3], dtype=jnp.float64)

    T = jax.jit(expSE3)(xi)
    recovered = jax.jit(logSE3)(T)

    assert jnp.allclose(recovered, xi, atol=1e-10, rtol=1e-10)
