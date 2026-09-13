# SympLie

SympLie is a small experimental JAX package for Lie-group operations and
structure-preserving rigid-body integration. It provides batched
\(SO(3)\) and \(SE(3)\) maps, a Moser–Veselov free-rigid-body integrator,
and gyroscope attitude propagation.

The package deliberately has a narrow scope. It is not a replacement for a
general geometry library, multibody dynamics engine, state estimator, or
robotics framework.

The [API reference](api.md) documents every public operation and its array
shape conventions.
