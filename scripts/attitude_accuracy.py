"""Measure variational-integrator attitude error against a SciPy reference.

Run with ``python scripts/attitude_accuracy.py --out artifacts``.
The DOP853 reference is checked by tightening its integration tolerances.
"""

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from symplie.integrators import simulate_free_rigid_body


def reference_attitude(R0, pi0, J, duration, rtol=1e-13, atol=1e-15):
    """Return final attitude, body momentum, and accepted-step quaternion norm drift."""
    inverse_J = np.linalg.inv(np.asarray(J))
    q0 = Rotation.from_matrix(np.asarray(R0)).as_quat()

    def derivative(t, state):
        # SciPy stores the scalar quaternion component last: q = [v, s].
        v, s, pi = state[:3], state[3], state[4:]
        omega = inverse_J @ pi
        v_dot = 0.5 * (s * omega + np.cross(v, omega))
        s_dot = -0.5 * np.dot(v, omega)
        pi_dot = np.cross(pi, omega)
        return np.concatenate((v_dot, [s_dot], pi_dot))

    solution = solve_ivp(
        derivative,
        (0.0, duration),
        np.concatenate((q0, np.asarray(pi0))),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"SciPy reference failed: {solution.message}")
    norm_error = float(np.max(np.abs(np.linalg.norm(solution.y[:4], axis=0) - 1.0)))
    if not np.all(np.isfinite(solution.y)) or norm_error > 1e-8:
        raise RuntimeError(f"Reference quaternion norm check failed: {norm_error:.3e}")

    # from_quat normalizes the quaternion before constructing the rotation.
    R = Rotation.from_quat(solution.y[:4, -1]).as_matrix()
    return R, solution.y[4:, -1], norm_error


def attitude_error(R, reference):
    """Geodesic attitude error in radians."""
    relative = np.asarray(reference).T @ np.asarray(R)
    return float(Rotation.from_matrix(relative).magnitude())


def run_benchmark():
    J = jnp.diag(jnp.array([0.6, 1.0, 1.8], dtype=jnp.float64))
    R0 = jnp.eye(3, dtype=jnp.float64)
    pi0 = jnp.array([0.2, 0.7, 1.0], dtype=jnp.float64)
    duration = 20.0
    reference_tolerances = {"rtol": 1e-13, "atol": 1e-15}
    check_tolerances = {"rtol": 1e-11, "atol": 1e-13}
    timesteps = [0.2, 0.1, 0.05, 0.025, 0.0125]

    check_R, _, check_norm_error = reference_attitude(
        R0, pi0, J, duration, **check_tolerances
    )
    reference, _, norm_error = reference_attitude(
        R0, pi0, J, duration, **reference_tolerances
    )
    reference_gap = attitude_error(check_R, reference)
    if not jnp.isfinite(reference_gap) or reference_gap > 1e-10:
        raise RuntimeError(f"Reference tolerance check failed: {reference_gap:.3e} rad")
    orthogonality_error = float(jnp.linalg.norm(reference.T @ reference - jnp.eye(3)))
    determinant_error = float(jnp.abs(jnp.linalg.det(reference) - 1.0))
    if not (orthogonality_error < 1e-10 and determinant_error < 1e-10):
        raise RuntimeError("Reference rotation failed the SO(3) validity check")

    rows = []
    for dt in timesteps:
        steps = round(duration / dt)
        vi_Rs, _, info = simulate_free_rigid_body(R0, pi0, J, dt, steps=steps)
        if not bool(jnp.all(info.converged)):
            residual = float(jnp.max(info.residual_norm))
            raise RuntimeError(f"VI solve failed at dt={dt}: residual={residual:.3e}")

        error = attitude_error(vi_Rs[-1], reference)
        if not np.isfinite(error) or error <= 100 * reference_gap:
            raise RuntimeError(f"Reference is not accurate enough at dt={dt}")
        row = {
            "dt_s": dt,
            "vi_attitude_error_rad": error,
            "vi_max_residual": float(jnp.max(info.residual_norm)),
        }
        rows.append(row)

    metadata = {
        "duration_s": duration,
        "inertia_diagonal": jnp.diag(J).tolist(),
        "initial_body_momentum": pi0.tolist(),
        "initial_rotation": R0.tolist(),
        "reference_method": "SciPy DOP853 (quaternion + body momentum)",
        "reference_tolerances": reference_tolerances,
        "reference_check_tolerances": check_tolerances,
        "reference_tolerance_difference_rad": reference_gap,
        "reference_quaternion_norm_error": norm_error,
        "reference_check_quaternion_norm_error": check_norm_error,
        "reference_orthogonality_error": orthogonality_error,
        "reference_determinant_error": determinant_error,
        "dtype": "float64",
        "newton_iters": 8,
        "newton_tolerance": 1e-10,
    }
    return rows, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)

    rows, metadata = run_benchmark()
    args.out.mkdir(parents=True, exist_ok=True)

    with (args.out / "attitude_error_vs_dt.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "attitude_accuracy.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    dt_values = [row["dt_s"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.loglog(dt_values, [row["vi_attitude_error_rad"] for row in rows], "o-",
              label="Moser–Veselov")
    ax.set_xlabel("Timestep [s]")
    ax.set_xticks(dt_values, labels=[f"{dt:g}" for dt in dt_values])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Final attitude error [rad]")
    ax.set_title("Attitude error at t = 20 s\nReference: SciPy DOP853")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.savefig(args.out / "attitude_error_vs_dt.png", dpi=160)
    plt.close(fig)

    print("dt [s]       VI error [rad]")
    for row in rows:
        print(f"{row['dt_s']:<12g} {row['vi_attitude_error_rad']:.6e}")
    print(f"Reference tolerance-check difference: "
          f"{metadata['reference_tolerance_difference_rad']:.3e} rad")


if __name__ == "__main__":
    main()
