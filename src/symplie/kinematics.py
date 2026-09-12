"""Attitude kinematics on SO(3)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .so3 import exp as expSO3


def propagate_gyro(
    R0: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    dt: float | jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Propagate attitude from body-frame angular-rate measurements.

    This function integrates a sequence of gyroscope measurements directly on
    :math:`SO(3)`. Each angular increment is mapped to a rotation matrix with
    the exponential map and composed with the current attitude. Consequently,
    the propagation does not require Euler angles, quaternion normalization,
    or projection of an additively integrated matrix back onto the rotation
    group.

    Frame convention
    ----------------
    ``R0`` is the initial body-to-world rotation. For a vector
    :math:`\mathbf{v}_B` expressed in the body frame, the corresponding vector
    in the world frame is

    .. math::

        \mathbf{v}_W = R_0\,\mathbf{v}_B.

    Angular velocity is expressed in the body frame. The attitude increment is
    therefore composed on the right. If :math:`R_k` is the attitude at sample
    :math:`k`, the update is

    .. math::

        R_{k+1}
        = R_k\,\operatorname{Exp}\!\left(
            (\boldsymbol{\omega}_{m,k} - \mathbf{b})\,\Delta t_k
          \right),

    where :math:`\boldsymbol{\omega}_{m,k}` is the measured angular velocity,
    :math:`\mathbf{b}` is a constant gyroscope bias, and :math:`\Delta t_k` is
    the duration represented by that measurement. Equivalently, the model
    assumes

    .. math::

        \boldsymbol{\omega}_{m,k}
        = \boldsymbol{\omega}_k + \mathbf{b}.

    Setting ``bias`` to the known sensor bias therefore recovers the corrected
    body rate :math:`\boldsymbol{\omega}_k` before propagation. When ``bias``
    is omitted, it is taken to be zero.

    The angular velocity is treated as constant over each sampling interval.
    Under that assumption, the exponential update is the exact rotation for
    the individual interval. Accuracy over a complete trajectory still depends
    on the sampling rate, measurement quality, timestamp accuracy, and validity
    of the constant-bias assumption.

    Parameters
    ----------
    R0 : jax.Array, shape (3, 3)
        Initial body-to-world rotation matrix. The caller is responsible for
        providing a proper rotation in :math:`SO(3)`.
    angular_velocity : jax.Array, shape (samples, 3)
        Body-frame gyroscope measurements in radians per second. Rows are
        consumed in chronological order.
    dt : float or jax.Array, shape (samples,)
        Sampling interval in seconds. A scalar applies the same interval to
        every measurement. A one-dimensional array supplies one interval per
        measurement.
    bias : jax.Array, shape (3,), optional
        Constant body-frame gyroscope bias in radians per second. The default
        is a zero vector with the same dtype as ``angular_velocity``.

    Returns
    -------
    jax.Array, shape (samples + 1, 3, 3)
        Body-to-world attitude history. Element zero is exactly ``R0`` and
        element :math:`k+1` is the attitude after consuming measurement
        :math:`k`. An empty measurement sequence therefore returns an array
        containing only ``R0``.

    Notes
    -----
    The recurrence is evaluated with :func:`jax.lax.scan`, so the complete
    propagation is compatible with :func:`jax.jit` and can be differentiated
    with respect to the angular velocities, timesteps, initial attitude, or
    bias. The output remains on :math:`SO(3)` up to floating-point roundoff when
    ``R0`` is a proper rotation.

    This is gyroscope-only attitude propagation, not a complete IMU model or
    state estimator. It does not estimate a time-varying bias, fuse
    accelerometer or magnetometer observations, model noise, propagate a
    covariance, or correct accumulated drift. Those responsibilities belong to
    the estimator or application using this kinematic primitive.

    Examples
    --------
    Propagate one second of constant yaw rate, correcting a known sensor bias:

    >>> import jax.numpy as jnp
    >>> from symplie.kinematics import propagate_gyro
    >>> R0 = jnp.eye(3)
    >>> true_rate = jnp.array([0.0, 0.0, 0.5])
    >>> bias = jnp.array([0.01, -0.02, 0.03])
    >>> measurements = jnp.broadcast_to(true_rate + bias, (100, 3))
    >>> attitudes = propagate_gyro(R0, measurements, dt=0.01, bias=bias)
    >>> attitudes.shape
    (101, 3, 3)

    See Also
    --------
    symplie.so3.exp
        Exponential map used for each body-frame angular increment.
    """

    timesteps = jnp.broadcast_to(
        jnp.asarray(dt),
        (angular_velocity.shape[0],),
    )
    if bias is None:
        bias = jnp.zeros(3, dtype=angular_velocity.dtype)

    def step(R, sample):
        omega, dt_i = sample
        R_next = R @ expSO3((omega - bias) * dt_i)
        return R_next, R_next

    _, attitudes = jax.lax.scan(step, R0, (angular_velocity, timesteps))
    return jnp.concatenate((R0[None, ...], attitudes), axis=0)
