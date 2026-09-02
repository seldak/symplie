import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

from symplie import is_proper_rotation, logSO3_checked
from symplie.so3 import exp as expSO3
from symplie.so3 import hat
from symplie.so3 import log as logSO3
from symplie.so3 import vee


def test_log_identity_is_zero():
    R = jnp.eye(3, dtype=jnp.float64)

    result = logSO3(R)

    assert result.shape == (3,)
    assert jnp.allclose(result, jnp.zeros(3), atol=1e-14, rtol=0.0)


def test_log_on_qr_constructed_rotations():
    rng = np.random.default_rng(0)
    for _ in range(10):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q[:, 0] *= np.linalg.det(Q)
        assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-14)
        assert np.isclose(np.linalg.det(Q), 1.0, atol=1e-14)

        expected = Rotation.from_matrix(Q).as_rotvec()
        actual = jax.jit(logSO3)(jnp.asarray(Q))
        assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)
        assert np.allclose(expSO3(actual), Q, atol=1e-10, rtol=0.0)
        assert np.allclose(logSO3_checked(jnp.asarray(Q)), actual, atol=1e-12)


def test_log_checked_rejects_reflections():
    normal = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float64)
    normal /= jnp.linalg.norm(normal)
    householder = jnp.eye(3) - 2 * jnp.outer(normal, normal)
    for R in (jnp.diag(jnp.array([1.0, 1.0, -1.0])), householder):
        assert jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-14)
        assert jnp.isclose(jnp.linalg.det(R), -1.0, atol=1e-14)
        with pytest.raises(ValueError, match="expects a proper rotation"):
            logSO3_checked(R)


def test_rotation_check_is_jittable_and_vmappable():
    rotations = jnp.stack((
        jnp.eye(3),
        jnp.diag(jnp.array([1.0, 1.0, -1.0])),
        jnp.diag(jnp.array([2.0, 0.5, 1.0])),  # det = 1, but not orthogonal.
    ))
    valid = jax.jit(jax.vmap(is_proper_rotation))(rotations)
    assert jnp.array_equal(valid, jnp.array([True, False, False]))
    with pytest.raises(ValueError, match="expects a proper rotation"):
        logSO3_checked(rotations[-1])


def test_log_checked_respects_tolerance():
    R = jnp.diag(jnp.array([1.0 + 1e-5, 1.0, 1.0], dtype=jnp.float64))
    with pytest.raises(ValueError, match="expects a proper rotation"):
        logSO3_checked(R)
    assert bool(is_proper_rotation(R, atol=1e-4))
    assert jnp.allclose(logSO3_checked(R, atol=1e-4), logSO3(R))


@pytest.mark.parametrize("angle", [np.pi + 0.2, 1.5 * np.pi, 2 * np.pi + 0.3])
def test_log_wraps_to_principal_branch(angle):
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    R = Rotation.from_rotvec(angle * axis).as_matrix()
    principal_angle = (angle + np.pi) % (2 * np.pi) - np.pi

    actual = logSO3(jnp.asarray(R))

    assert np.linalg.norm(actual) <= np.pi
    assert np.allclose(actual, principal_angle * axis, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("axis", [[-3.0, 1.0, 2.0], [1.0, -3.0, 2.0], [1.0, 2.0, -3.0]])
def test_log_sweep_across_pi(axis):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    # Cross the near-pi threshold from both sides of the principal-branch cut.
    gaps = [1e-2, 1e-3, 2e-4, 1.1e-4, 9e-5, 1e-5, 1e-6, 1e-7]
    for gap in gaps:
        for sign in (-1, 1):
            R = Rotation.from_rotvec((np.pi + sign * gap) * axis).as_matrix()
            expected = -sign * (np.pi - gap) * axis
            actual = jax.jit(logSO3)(jnp.asarray(R))
            assert np.allclose(actual, expected, atol=1e-7, rtol=0.0), (gap, sign)

    # At exactly pi either axis sign represents the same rotation.
    R = Rotation.from_rotvec(np.pi * axis).as_matrix()
    actual = logSO3(jnp.asarray(R))
    assert np.all(np.isfinite(actual))
    assert np.isclose(np.linalg.norm(actual), np.pi, atol=3e-8, rtol=0.0)
    assert np.allclose(Rotation.from_rotvec(np.asarray(actual)).as_matrix(), R, atol=3e-8)


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


def test_hat_and_vee_accept_leading_batch_dimensions():
    vectors = jnp.array(
        [
            [0.1, -0.2, 0.3],
            [-0.4, 0.5, -0.6],
        ],
        dtype=jnp.float64,
    )

    matrices = hat(vectors)
    recovered = vee(matrices)

    assert matrices.shape == (2, 3, 3)
    assert recovered.shape == (2, 3)
    assert jnp.array_equal(recovered, vectors)
