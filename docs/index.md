# SympLie

SympLie is an experimental JAX package for simulating and differentiating
controlled rigid-body attitude dynamics on \(SO(3)\). Its core is a
Moser–Veselov variational integrator for rotational motion.

## Capabilities

- Torque-free and externally forced rigid-body integration on \(SO(3)\).
- A composable single-step map for custom JAX scans and control loops.
- The \(SO(3)\) maps and invariant diagnostics needed to use and inspect the
  integrator.
- State-dependent torque under zero-order hold.
- Fixed-target geometric PD attitude control.
- Geometric attitude trajectory tracking.
- Gyroscope attitude propagation with optional bias compensation.
- Solver diagnostics, batched geometry, JIT compilation, and automatic
  differentiation.

Start with [installation and examples](getting-started.md), then read the
[frame and sampling conventions](conventions.md) before using the dynamics
APIs.

## Scope

SympLie deliberately remains a rotational-dynamics package. It does not
provide multibody dynamics, translation, contact, constraints, estimation,
actuator allocation, or robotics middleware. It is not a replacement for
jaxlie, Sophus, Drake, or Pinocchio. Its \(SE(3)\) exponential and logarithm
utilities are not an \(SE(3)\) dynamics model.
