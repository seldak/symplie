"""Convert PX4 attitude data into SympLie conventions.

PX4 attitude quaternions use Hamilton ``[w, x, y, z]`` ordering and map the
body-fixed FRD frame into the local NED frame. The returned rotation matrices
therefore already follow SympLie's body-to-spatial convention when body is
identified with FRD and spatial is identified with NED.

This module operates on NumPy arrays and does not parse ULogs. Timestamp and
signal alignment remain the responsibility of the application reading PX4
data.
"""

from __future__ import annotations

import numpy as np


def px4_quaternion_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert PX4 Hamilton quaternions to body-FRD-to-NED matrices.

    Parameters
    ----------
    q_wxyz : numpy.ndarray, shape (..., 4)
        Quaternion coefficients in Hamilton ``[w, x, y, z]`` order. Each
        quaternion is normalized before conversion.

    Returns
    -------
    numpy.ndarray, shape (..., 3, 3)
        Body-FRD-to-NED rotation matrices.

    Raises
    ------
    ValueError
        If the trailing dimension is not four or a quaternion has zero norm.
    """
    q = np.asarray(q_wxyz)
    if q.ndim == 0 or q.shape[-1] != 4:
        raise ValueError("q_wxyz must have shape (..., 4)")
    if not np.issubdtype(q.dtype, np.floating):
        q = q.astype(np.float64)

    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("q_wxyz must not contain a zero quaternion")

    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    row_0 = np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        axis=-1,
    )
    row_1 = np.stack(
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        axis=-1,
    )
    row_2 = np.stack(
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    )
    return np.stack([row_0, row_1, row_2], axis=-2)


def _canonicalize_quaternion(q_wxyz: np.ndarray) -> np.ndarray:
    """Choose one representative of the quaternion sign pair."""
    q = q_wxyz.copy()
    if q[0] < 0.0:
        return -q

    if q[0] == 0.0:
        nonzero = np.flatnonzero(q[1:])
        if nonzero.size and q[nonzero[0] + 1] < 0.0:
            return -q

    return q


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert one proper rotation matrix to Hamilton quaternion form."""
    trace = np.trace(R)

    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [
                0.25 * scale,
                (R[2, 1] - R[1, 2]) / scale,
                (R[0, 2] - R[2, 0]) / scale,
                (R[1, 0] - R[0, 1]) / scale,
            ]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array(
            [
                (R[2, 1] - R[1, 2]) / scale,
                0.25 * scale,
                (R[0, 1] + R[1, 0]) / scale,
                (R[0, 2] + R[2, 0]) / scale,
            ]
        )
    elif R[1, 1] > R[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array(
            [
                (R[0, 2] - R[2, 0]) / scale,
                (R[0, 1] + R[1, 0]) / scale,
                0.25 * scale,
                (R[1, 2] + R[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array(
            [
                (R[1, 0] - R[0, 1]) / scale,
                (R[0, 2] + R[2, 0]) / scale,
                (R[1, 2] + R[2, 1]) / scale,
                0.25 * scale,
            ]
        )

    q /= np.linalg.norm(q)
    return _canonicalize_quaternion(q)


def matrix_to_px4_quaternion(R: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Convert body-FRD-to-NED matrices to canonical PX4 quaternions.

    The returned Hamilton quaternion has nonnegative scalar component. When
    that component is exactly zero, the first nonzero vector component is
    positive.

    Parameters
    ----------
    R : numpy.ndarray, shape (..., 3, 3)
        Proper body-FRD-to-NED rotation matrices.
    atol : float, optional
        Absolute tolerance used to validate orthogonality and determinant.

    Returns
    -------
    numpy.ndarray, shape (..., 4)
        Canonical quaternions in Hamilton ``[w, x, y, z]`` order.

    Raises
    ------
    ValueError
        If the trailing dimensions are not ``(3, 3)`` or a matrix is not a
        proper rotation within ``atol``.
    """
    matrices = np.asarray(R)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError("R must have shape (..., 3, 3)")

    identity = np.eye(3, dtype=matrices.dtype)
    orthogonality = np.swapaxes(matrices, -1, -2) @ matrices
    if not np.allclose(orthogonality, identity, rtol=0.0, atol=atol):
        raise ValueError("R must contain proper rotation matrices")
    if not np.allclose(np.linalg.det(matrices), 1.0, rtol=0.0, atol=atol):
        raise ValueError("R must contain proper rotation matrices")

    flat_matrices = matrices.reshape((-1, 3, 3))
    flat_quaternions = np.stack(
        [_matrix_to_quaternion(matrix) for matrix in flat_matrices]
    )
    return flat_quaternions.reshape(matrices.shape[:-2] + (4,))


def px4_timestamps_to_timesteps(
    timestamp_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter PX4 timestamps and convert retained intervals to seconds.

    Parameters
    ----------
    timestamp_us : numpy.ndarray, shape (samples,)
        PX4 timestamps in integer microseconds. Non-monotonic and duplicate
        records are removed with a strict record-high filter.

    Returns
    -------
    keep : numpy.ndarray, shape (samples,), dtype bool
        Mask over the input records. Apply the same mask to every signal that
        is aligned with ``timestamp_us``.
    dt : numpy.ndarray, shape (sum(keep) - 1,)
        Strictly positive intervals between retained samples, in seconds.

    Raises
    ------
    ValueError
        If the input is not one-dimensional or fewer than two samples remain.
    """
    timestamps = np.asarray(timestamp_us)
    if timestamps.ndim != 1:
        raise ValueError("timestamp_us must have shape (samples,)")
    if timestamps.size == 0:
        raise ValueError("timestamp_us must contain at least two samples")

    running_max = np.maximum.accumulate(timestamps)
    keep = np.ones(timestamps.shape, dtype=bool)
    keep[1:] = timestamps[1:] > running_max[:-1]

    retained_indices = np.flatnonzero(keep)
    retained_timestamps = timestamps[keep]
    dt = np.diff(retained_timestamps).astype(np.float64) * 1e-6
    positive_interval = dt > 0.0
    if not np.all(positive_interval):
        keep[retained_indices[1:][~positive_interval]] = False
        retained_timestamps = timestamps[keep]
        dt = np.diff(retained_timestamps).astype(np.float64) * 1e-6

    if np.count_nonzero(keep) < 2:
        raise ValueError("fewer than two strictly increasing timestamps remain")

    return keep, dt
