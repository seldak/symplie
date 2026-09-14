# PX4 compatibility

The PX4 compatibility module is a frame-and-unit codec. It converts PX4
attitude quaternions and timestamps into the matrix and SI conventions used by
SympLie; it does not parse ULogs or change the core dynamics API.

PX4 attitude quaternions use Hamilton `[w, x, y, z]` ordering and describe the
rotation from body FRD to local NED. This is the same body-to-spatial mapping
used by SympLie when the body frame is identified with FRD and the spatial
frame with NED. PX4 `vehicle_angular_velocity.xyz` is therefore already a body
angular velocity with shape `(..., 3)`.

```python
from symplie import propagate_gyro
from symplie.compat.px4 import (
    px4_quaternion_to_matrix,
    px4_timestamps_to_timesteps,
)

keep, dt = px4_timestamps_to_timesteps(timestamp_us)
R = px4_quaternion_to_matrix(q_wxyz[keep])
omega = omega_at_attitude_timestamps[keep]

Rs = propagate_gyro(R[0], omega[:-1], dt)
```

The timestamp mask refers to the input records, while `dt` contains one
interval fewer than the retained samples. Apply `omega[k]` over the interval
from retained attitude sample `k` to `k + 1`, as shown above.

ULog parsing, topic-clock alignment, interpolation, window selection, and
acceptance thresholds belong to the application consuming the flight log.
In particular, `vehicle_attitude` and `vehicle_angular_velocity` should not be
paired by raw row index because they are sampled on different clocks.

## API

::: symplie.compat.px4.px4_quaternion_to_matrix

::: symplie.compat.px4.matrix_to_px4_quaternion

::: symplie.compat.px4.px4_timestamps_to_timesteps
