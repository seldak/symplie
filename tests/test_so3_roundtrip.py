import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from symplie.so3 import exp as expSO3
from symplie.so3 import log as logSO3


def test_log_identity_is_zero():
    R = jnp.eye(3, dtype=jnp.float64)

    result = logSO3(R)

    assert result.shape == (3,)
    assert jnp.allclose(result, jnp.zeros(3), atol=1e-14, rtol=0.0)


@pytest.mark.parametrize(
    "rotation_vector",
    [
        jnp.array([1e-9, -2e-9, 3e-9], dtype=jnp.float64),
        jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64),
        jnp.array([-0.7, 0.4, 1.1], dtype=jnp.float64),
    ],
)
def test_log_exp_recovers_rotation_vector(rotation_vector):
    result = logSO3(expSO3(rotation_vector))

    assert jnp.allclose(result, rotation_vector, atol=1e-10, rtol=1e-10)


def test_exp_log_recovers_rotation_near_pi():
    axis = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64)
    axis = axis / jnp.linalg.norm(axis)
    R = expSO3((jnp.pi - 1e-6) * axis)

    reconstructed = expSO3(logSO3(R))

    assert jnp.allclose(reconstructed, R, atol=1e-8, rtol=1e-8)


def test_log_is_jittable():
    R = expSO3(jnp.array([0.2, 0.1, -0.3], dtype=jnp.float64))

    eager = logSO3(R)
    compiled = jax.jit(logSO3)(R)

    assert jnp.allclose(compiled, eager, atol=1e-12, rtol=1e-12)
