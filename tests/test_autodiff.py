import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

from symplie import se3, simulate_free_rigid_body, so3


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


def test_integrator_gradient_wrt_initial_momentum():
    R0 = jnp.eye(3, dtype=jnp.float64)
    pi0 = jnp.array([0.2, 0.7, 1.0], dtype=jnp.float64)
    J = jnp.diag(jnp.array([0.8, 1.0, 1.2], dtype=jnp.float64))

    def loss(initial_momentum):
        _, pis, solver_info = simulate_free_rigid_body(
            R0, initial_momentum, J, dt=0.01, steps=50
        )
        # Use one body component; the full momentum norm is conserved.
        return 0.5 * (pis[-1, 0] - 0.4)**2, solver_info

    gradient, solver_info = jax.grad(loss, has_aux=True)(pi0)
    assert jnp.all(solver_info.converged)
    assert jnp.all(jnp.isfinite(gradient))
    assert float(jnp.linalg.norm(gradient)) > 1e-3

    epsilon = 1e-5
    finite_difference = []
    for direction in jnp.eye(3, dtype=pi0.dtype):
        plus, plus_info = loss(pi0 + epsilon * direction)
        minus, minus_info = loss(pi0 - epsilon * direction)
        assert jnp.all(plus_info.converged) and jnp.all(minus_info.converged)
        finite_difference.append((plus - minus) / (2 * epsilon))

    assert jnp.allclose(gradient, jnp.array(finite_difference), atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize("differentiate", [jax.jacfwd, jax.jacrev])
@pytest.mark.parametrize("axis", [[-3.0, 1.0, 2.0], [1.0, -3.0, 2.0], [1.0, 2.0, -3.0]])
@pytest.mark.parametrize("gap", [-5e-5, 5e-5])
def test_log_jacobian_near_pi(axis, gap, differentiate):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    R = Rotation.from_rotvec((np.pi - gap) * axis).as_matrix()

    def local_log(delta):
        return so3.log(jnp.asarray(R) @ so3.exp(delta))

    # Stay on one side of pi: the principal log is discontinuous at the cut.
    epsilon = 1e-6
    columns = []
    for direction in np.eye(3):
        plus = R @ Rotation.from_rotvec(epsilon * direction).as_matrix()
        minus = R @ Rotation.from_rotvec(-epsilon * direction).as_matrix()
        columns.append(
            (Rotation.from_matrix(plus).as_rotvec() - Rotation.from_matrix(minus).as_rotvec())
            / (2 * epsilon)
        )
    expected = np.column_stack(columns)

    jacobian = differentiate(local_log)
    for evaluate in (jacobian, jax.jit(jacobian)):
        actual = evaluate(jnp.zeros(3, dtype=jnp.float64))
        assert jnp.all(jnp.isfinite(actual))
        assert np.allclose(actual, expected, atol=1e-4, rtol=1e-4)
