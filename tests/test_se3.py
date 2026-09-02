import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from symplie.se3 import hat, vee


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
