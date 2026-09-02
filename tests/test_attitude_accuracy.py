import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from scipy.spatial.transform import Rotation

jax.config.update("jax_enable_x64", True)

from symplie.integrators import simulate_free_rigid_body


# Load the benchmark script without adding it to the installed package.
script = Path(__file__).resolve().parents[1] / "scripts" / "attitude_accuracy.py"
spec = importlib.util.spec_from_file_location("attitude_accuracy", script)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_reference_and_vi_against_exact_symmetric_top():
    transverse_inertia = 0.7
    axial_inertia = 1.2
    J = jnp.diag(jnp.array([transverse_inertia, transverse_inertia, axial_inertia]))
    pi0 = jnp.array([0.2, 0.7, 1.0])
    R0 = jnp.asarray(Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix())
    duration = 2.0

    # R(t) = Exp(t * L / I_perp) R0 Exp(t * spin * e3).
    spatial_momentum = R0 @ pi0
    spin = pi0[2] * (1 / axial_inertia - 1 / transverse_inertia)
    reference = (
        Rotation.from_rotvec(duration * spatial_momentum / transverse_inertia).as_matrix()
        @ R0
        @ Rotation.from_rotvec([0.0, 0.0, duration * spin]).as_matrix()
    )

    scipy_R, scipy_pi, norm_error = benchmark.reference_attitude(R0, pi0, J, duration)
    assert benchmark.attitude_error(scipy_R, reference) < 1e-11
    assert jnp.allclose(scipy_pi, reference.T @ spatial_momentum, atol=1e-11, rtol=0.0)
    assert norm_error < 1e-11

    errors = []
    for dt in (0.2, 0.1, 0.05):
        steps = round(duration / dt)
        Rs, _, info = simulate_free_rigid_body(R0, pi0, J, dt, steps=steps)
        assert bool(jnp.all(info.converged))

        assert jnp.allclose(
            jnp.swapaxes(Rs, -1, -2) @ Rs, jnp.eye(3), atol=1e-12, rtol=0.0
        )
        assert jnp.allclose(jnp.linalg.det(Rs), 1.0, atol=1e-12, rtol=0.0)
        errors.append(benchmark.attitude_error(Rs[-1], reference))

    observed_orders = jnp.log2(jnp.array(errors[:-1]) / jnp.array(errors[1:]))
    assert jnp.all(jnp.abs(observed_orders - 2) < 0.15), observed_orders


def test_reference_zero_momentum_leaves_attitude_unchanged():
    R0 = jnp.asarray(Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix())
    pi0 = jnp.zeros(3)
    J = jnp.diag(jnp.array([0.7, 1.0, 1.2]))

    R, pi, norm_error = benchmark.reference_attitude(R0, pi0, J, duration=1.0)

    assert jnp.allclose(R, R0, atol=1e-14, rtol=0.0)
    assert jnp.array_equal(pi, pi0)
    assert norm_error < 1e-14


def test_asymmetric_benchmark_is_resolved_by_reference():
    rows, metadata = benchmark.run_benchmark()
    reference_gap = metadata["reference_tolerance_difference_rad"]

    assert reference_gap < 1e-10
    errors = jnp.array([row["vi_attitude_error_rad"] for row in rows])
    assert jnp.all(errors > 100 * reference_gap)
    observed_orders = jnp.log2(errors[:-1] / errors[1:])
    assert jnp.all(jnp.abs(observed_orders - 2) < 0.15), observed_orders


def test_reference_reports_solver_failure(monkeypatch):
    def failed_solve(*args, **kwargs):
        return SimpleNamespace(success=False, message="forced failure")

    monkeypatch.setattr(benchmark, "solve_ivp", failed_solve)
    with pytest.raises(RuntimeError, match="SciPy reference failed"):
        benchmark.reference_attitude(jnp.eye(3), jnp.zeros(3), jnp.eye(3), 1.0)


def test_reference_rejects_large_quaternion_drift(monkeypatch):
    def drifted_solve(*args, **kwargs):
        state = jnp.array([0.0, 0.0, 0.0, 2.0, 0.2, 0.7, 1.0])
        return SimpleNamespace(success=True, y=state[:, None])

    monkeypatch.setattr(benchmark, "solve_ivp", drifted_solve)
    with pytest.raises(RuntimeError, match="quaternion norm check failed"):
        benchmark.reference_attitude(jnp.eye(3), jnp.zeros(3), jnp.eye(3), 1.0)
