import numpy as np
import pytest

from symplie.compat.px4 import (
    matrix_to_px4_quaternion,
    px4_quaternion_to_matrix,
    px4_timestamps_to_timesteps,
)


def test_px4_quaternion_to_matrix_known_rotations():
    root_half = np.sqrt(0.5)
    quaternions = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [root_half, root_half, 0.0, 0.0],
            [root_half, 0.0, root_half, 0.0],
            [root_half, 0.0, 0.0, root_half],
        ]
    )
    expected = np.array(
        [
            np.eye(3),
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )

    np.testing.assert_allclose(
        px4_quaternion_to_matrix(quaternions), expected, atol=1e-15
    )


def test_quaternion_sign_does_not_change_rotation():
    quaternion = np.array([0.5, -0.5, 0.5, -0.5])

    np.testing.assert_allclose(
        px4_quaternion_to_matrix(quaternion),
        px4_quaternion_to_matrix(-quaternion),
    )


def test_matrix_round_trip_uses_canonical_quaternion_sign():
    quaternions = np.array(
        [
            [-0.5, 0.5, -0.5, 0.5],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    rotations = px4_quaternion_to_matrix(quaternions)

    recovered = matrix_to_px4_quaternion(rotations)

    np.testing.assert_allclose(px4_quaternion_to_matrix(recovered), rotations)
    assert np.all(recovered[:, 0] >= 0.0)
    np.testing.assert_allclose(
        recovered[1:],
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        atol=1e-15,
    )


def test_timestamp_filter_returns_input_mask_and_interval_timesteps():
    origin = 1_787_496_889_000_000
    timestamps = origin + np.array([10, 9, 9, 11, 11, 14], dtype=np.int64)

    keep, dt = px4_timestamps_to_timesteps(timestamps)

    np.testing.assert_array_equal(keep, [True, False, False, True, False, True])
    np.testing.assert_allclose(dt, [1e-6, 3e-6])
    assert dt.shape == (np.count_nonzero(keep) - 1,)


@pytest.mark.parametrize("timestamps", [[], [5], [5, 5], [5, 4, 3]])
def test_timestamp_filter_rejects_fewer_than_two_retained_samples(timestamps):
    with pytest.raises(ValueError, match="fewer than two|at least two"):
        px4_timestamps_to_timesteps(np.asarray(timestamps))


def test_codec_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        px4_quaternion_to_matrix(np.ones(3))
    with pytest.raises(ValueError, match="zero quaternion"):
        px4_quaternion_to_matrix(np.zeros(4))
    with pytest.raises(ValueError, match="proper rotation"):
        matrix_to_px4_quaternion(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="shape"):
        px4_timestamps_to_timesteps(np.ones((2, 2)))
