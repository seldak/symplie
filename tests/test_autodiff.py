import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from symplie import se3, so3


def linear_log_SO3(R):
    return 0.5 * so3.vee(R - R.T)


def linear_log_SE3(T):
    return jnp.concatenate((T[:3, 3], linear_log_SO3(T[:3, :3])))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev])
@pytest.mark.parametrize(
    "function, linearization, argument",
    [
        (so3.exp, so3.hat, jnp.zeros(3)),
        (so3.log, linear_log_SO3, jnp.eye(3)),
        (se3.exp, se3.hat, jnp.zeros(6)),
        (se3.log, linear_log_SE3, jnp.eye(4)),
    ],
)
def test_map_jacobian_at_identity(function, linearization, argument, differentiate, dtype):
    argument = argument.astype(dtype)
    expected = differentiate(linearization)(argument)
    jacobian = differentiate(function)

    for evaluate in (jacobian, jax.jit(jacobian)):
        actual = evaluate(argument)
        assert jnp.all(jnp.isfinite(actual))
        assert jnp.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev])
@pytest.mark.parametrize("group, dimension", [(so3, 3), (se3, 6)])
def test_vmapped_roundtrip_jacobian(group, dimension, differentiate, dtype):
    # Mix branches in one batch: identity, a tiny rotation, and a finite one.
    direction = jnp.linspace(0.1, 0.3, dimension, dtype=dtype)
    vectors = jnp.stack((jnp.zeros_like(direction), 1e-9 * direction, direction))

    def roundtrip(vector):
        return group.log(group.exp(vector))

    jacobian = jax.jit(differentiate(jax.vmap(roundtrip)))(vectors)
    expected = jnp.eye(vectors.size, dtype=dtype)

    assert jnp.all(jnp.isfinite(jacobian))
    assert jnp.allclose(jacobian.reshape(expected.shape), expected, atol=2e-6, rtol=2e-6)
