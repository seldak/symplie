# API reference

The top-level `symplie` package exports the names documented here. Unless
otherwise stated, vectors and matrices may use any floating dtype supported by
the active JAX backend.

## \(SO(3)\) operations

::: symplie.hatSO3

::: symplie.veeSO3

::: symplie.expSO3

::: symplie.logSO3

::: symplie.is_proper_rotation

::: symplie.logSO3_checked

## \(SE(3)\) operations

::: symplie.hatSE3

::: symplie.veeSE3

::: symplie.expSE3

::: symplie.logSE3

::: symplie.left_jacobian_SO3

::: symplie.left_jacobian_inverse_SO3

## Rigid-body integration

::: symplie.SolverInfo

::: symplie.solve_F

::: symplie.solve_F_with_info

::: symplie.simulate_free_rigid_body

::: symplie.simulate_rigid_body

## Attitude kinematics

::: symplie.propagate_gyro

## Diagnostics and invariants

::: symplie.energy

::: symplie.spatial_momentum

::: symplie.ortho_error

::: symplie.determinant_error

## Additional helpers

::: symplie.integrators.discrete_inertia
