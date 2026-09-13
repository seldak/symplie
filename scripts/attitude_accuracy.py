"""Measure variational-integrator attitude error against a SciPy reference.

Run with ``python scripts/attitude_accuracy.py --out artifacts``.
The DOP853 reference is checked by tightening its integration tolerances.
"""

import argparse
import csv
import json
from pathlib import Path
from shutil import copyfile

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from symplie.integrators import simulate_free_rigid_body, simulate_rigid_body


def reference_attitude(
    R0,
    pi0,
    J,
    duration,
    rtol=1e-13,
    atol=1e-15,
    body_torque=None,
):
    """Return final attitude, body momentum, and accepted-step quaternion norm drift."""
    inverse_J = np.linalg.inv(np.asarray(J))
    q0 = Rotation.from_matrix(np.asarray(R0)).as_quat()
    if body_torque is None:
        def body_torque(_):
            return np.zeros(3)

    def derivative(t, state):
        # SciPy stores the scalar quaternion component last: q = [v, s].
        v, s, pi = state[:3], state[3], state[4:]
        omega = inverse_J @ pi
        v_dot = 0.5 * (s * omega + np.cross(v, omega))
        s_dot = -0.5 * np.dot(v, omega)
        pi_dot = np.cross(pi, omega) + np.asarray(body_torque(t))
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


def prescribed_body_torque(time):
    """Smooth three-axis body torque used by the forced benchmark."""
    return np.array(
        [
            0.08 * np.cos(0.7 * time),
            -0.06 * np.sin(0.5 * time),
            0.05 * np.cos(1.3 * time),
        ]
    )


def run_forced_benchmark():
    """Measure forced attitude and momentum convergence against DOP853."""
    J = jnp.diag(jnp.array([0.7, 1.1, 1.8], dtype=jnp.float64))
    R0 = jnp.asarray(
        Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    )
    pi0 = jnp.array([0.3, -0.2, 0.4], dtype=jnp.float64)
    duration = 0.8
    timesteps = [0.08, 0.04, 0.02, 0.01]

    check_R, check_pi, check_norm_error = reference_attitude(
        R0,
        pi0,
        J,
        duration,
        rtol=1e-11,
        atol=1e-13,
        body_torque=prescribed_body_torque,
    )
    reference_R, reference_pi, norm_error = reference_attitude(
        R0,
        pi0,
        J,
        duration,
        body_torque=prescribed_body_torque,
    )
    reference_attitude_gap = attitude_error(check_R, reference_R)
    reference_momentum_gap = float(
        np.linalg.norm(check_pi - reference_pi)
    )

    rows = []
    for dt in timesteps:
        steps = round(duration / dt)
        times = np.arange(steps + 1) * dt
        body_torques = jnp.asarray(
            np.stack([prescribed_body_torque(time) for time in times])
        )
        Rs, pis, solver_info = simulate_rigid_body(
            R0,
            pi0,
            J,
            body_torques,
            dt,
        )
        if not bool(jnp.all(solver_info.converged)):
            max_residual = float(jnp.max(solver_info.residual_norm))
            raise RuntimeError(
                f"Forced solve failed at dt={dt}: residual={max_residual:.3e}"
            )

        rows.append(
            {
                "dt_s": dt,
                "attitude_error_rad": attitude_error(Rs[-1], reference_R),
                "momentum_error": float(
                    jnp.linalg.norm(pis[-1] - reference_pi)
                ),
                "max_residual": float(jnp.max(solver_info.residual_norm)),
            }
        )

    attitude_errors = np.array(
        [row["attitude_error_rad"] for row in rows]
    )
    momentum_errors = np.array(
        [row["momentum_error"] for row in rows]
    )
    metadata = {
        "duration_s": duration,
        "inertia_diagonal": jnp.diag(J).tolist(),
        "initial_body_momentum": pi0.tolist(),
        "initial_rotation": R0.tolist(),
        "reference_method": "SciPy DOP853 (quaternion + body momentum)",
        "reference_quaternion_norm_error": norm_error,
        "reference_check_quaternion_norm_error": check_norm_error,
        "reference_attitude_tolerance_difference_rad": reference_attitude_gap,
        "reference_momentum_tolerance_difference": reference_momentum_gap,
        "attitude_observed_order": float(
            np.polyfit(np.log(timesteps), np.log(attitude_errors), 1)[0]
        ),
        "momentum_observed_order": float(
            np.polyfit(np.log(timesteps), np.log(momentum_errors), 1)[0]
        ),
        "dtype": "float64",
        "newton_iters": 8,
        "newton_tolerance": 1e-10,
    }
    return rows, metadata


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
    parser.add_argument("--docs-out", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)

    rows, metadata = run_benchmark()
    forced_rows, forced_metadata = run_forced_benchmark()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.docs_out is not None:
        args.docs_out.mkdir(parents=True, exist_ok=True)

    with (args.out / "attitude_error_vs_dt.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "attitude_accuracy.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    with (args.out / "forced_error_vs_dt.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=forced_rows[0].keys())
        writer.writeheader()
        writer.writerows(forced_rows)
    with (args.out / "forced_accuracy.json").open("w") as stream:
        json.dump(forced_metadata, stream, indent=2)
        stream.write("\n")

    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    dt_values = [row["dt_s"] for row in rows]
    errors = [row["vi_attitude_error_rad"] for row in rows]
    observed_order = np.polyfit(np.log(dt_values), np.log(errors), 1)[0]
    # Offset the slope guide so it does not cover the measured errors.
    guide = 0.5 * errors[-1] * (np.asarray(dt_values) / dt_values[-1])**2
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.loglog(dt_values, errors, "o-",
              label="Moser–Veselov")
    ax.loglog(dt_values, guide, "--", color="gray", label=r"$C\,dt^2$ guide")
    ax.text(0.97, 0.06, f"Observed order: {observed_order:.2f}",
            transform=ax.transAxes, ha="right")
    ax.set_xlabel("Timestep [s]")
    ax.set_xticks(dt_values, labels=[f"{dt:g}" for dt in dt_values])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Final attitude error [rad]")
    ax.set_title("Attitude error at t = 20 s\nReference: SciPy DOP853")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.savefig(args.out / "attitude_error_vs_dt.png", dpi=160)
    plt.close(fig)

    forced_dt = np.array([row["dt_s"] for row in forced_rows])
    forced_attitude = np.array(
        [row["attitude_error_rad"] for row in forced_rows]
    )
    forced_momentum = np.array(
        [row["momentum_error"] for row in forced_rows]
    )
    attitude_guide = (
        0.55 * forced_attitude[-1] * (forced_dt / forced_dt[-1])**2
    )
    momentum_guide = (
        0.55 * forced_momentum[-1] * (forced_dt / forced_dt[-1])**2
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9, 4),
        layout="constrained",
    )
    axes[0].loglog(
        forced_dt,
        forced_attitude,
        "o-",
        color="#2563a6",
        label="SympLie",
    )
    axes[0].loglog(
        forced_dt,
        attitude_guide,
        "--",
        color="gray",
        label=r"$C\,dt^2$ guide",
    )
    axes[0].set_ylabel("Final attitude error [rad]")
    axes[0].set_title(
        f"Attitude · order {forced_metadata['attitude_observed_order']:.2f}"
    )

    axes[1].loglog(
        forced_dt,
        forced_momentum,
        "o-",
        color="#b45309",
        label="SympLie",
    )
    axes[1].loglog(
        forced_dt,
        momentum_guide,
        "--",
        color="gray",
        label=r"$C\,dt^2$ guide",
    )
    axes[1].set_ylabel("Final momentum error")
    axes[1].set_title(
        f"Momentum · order {forced_metadata['momentum_observed_order']:.2f}"
    )

    for axis in axes:
        axis.set_xlabel("Timestep [s]")
        axis.set_xticks(
            forced_dt,
            labels=[f"{dt:g}" for dt in forced_dt],
        )
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="both", alpha=0.2)
        axis.legend()

    fig.suptitle(
        "Three-axis forced rigid body\nReference: SciPy DOP853",
    )
    forced_figure = args.out / "forced_error_vs_dt.png"
    fig.savefig(forced_figure, dpi=160)
    if args.docs_out is not None:
        copyfile(forced_figure, args.docs_out / forced_figure.name)
    plt.close(fig)

    print("dt [s]       VI error [rad]")
    for row in rows:
        print(f"{row['dt_s']:<12g} {row['vi_attitude_error_rad']:.6e}")
    print(f"Reference tolerance-check difference: "
          f"{metadata['reference_tolerance_difference_rad']:.3e} rad")
    print(
        "Forced observed order: "
        f"attitude={forced_metadata['attitude_observed_order']:.3f}, "
        f"momentum={forced_metadata['momentum_observed_order']:.3f}"
    )


if __name__ == "__main__":
    main()
