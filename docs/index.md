# SympLie

SympLie is an experimental JAX package for differentiable Lie-group
rigid-body dynamics. It combines batched \(SO(3)\) and \(SE(3)\) operations
with Moser–Veselov rotational integration and geometric attitude control.

## Capabilities

- Numerically stable, batched \(SO(3)\) and \(SE(3)\) exponential and
  logarithm maps.
- Torque-free and externally forced rigid-body integration on \(SO(3)\).
- A composable single-step map for custom JAX scans and control loops.
- State-dependent torque under zero-order hold.
- Fixed-target geometric PD attitude control.
- Gyroscope attitude propagation with optional bias compensation.
- Solver diagnostics, invariant checks, JIT compilation, and automatic
  differentiation.

Start with [installation and examples](getting-started.md), then read the
[frame and sampling conventions](conventions.md) before using the dynamics
APIs.

## Scope

SympLie deliberately remains a rotational-dynamics package. It does not
provide multibody dynamics, translation, contact, constraints, estimation,
actuator allocation, or robotics middleware. It is not a replacement for
jaxlie, Sophus, Drake, or Pinocchio.
