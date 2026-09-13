"""Plot a geometric attitude-controller response."""

import argparse
import csv
import json
from pathlib import Path
from shutil import copyfile

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from symplie import (
    expSO3,
    geometric_pd_torque,
    logSO3,
    simulate_controlled_rigid_body,
)


def controller(_, R, pi, J, params):
    R_target, attitude_gain, rate_gain = params
    return geometric_pd_torque(
        R,
        pi,
        J,
        R_target,
        attitude_gain,
        rate_gain,
    )


def run_response():
    """Return one closed-loop attitude-regulation trajectory."""
    J = jnp.diag(jnp.array([1.0, 1.4, 1.8]))
    R_target = expSO3(jnp.array([-0.2, 0.1, 0.3]))
    R0 = R_target @ expSO3(jnp.array([0.6, -0.4, 0.3]))
    pi0 = jnp.array([0.2, -0.15, 0.1])
    attitude_gain = 4.0
    rate_gain = 2.5
    dt = 0.01
    steps = 800

    Rs, pis, body_torques, solver_info = simulate_controlled_rigid_body(
        R0,
        pi0,
        J,
        controller,
        (R_target, attitude_gain, rate_gain),
        dt,
        steps,
    )
    if not bool(jnp.all(solver_info.converged)):
        max_residual = float(jnp.max(solver_info.residual_norm))
        raise RuntimeError(
            f"Controlled solve failed: residual={max_residual:.3e}"
        )

    relative_rotations = jnp.swapaxes(R_target, -1, -2) @ Rs
    attitude_errors = jnp.linalg.norm(logSO3(relative_rotations), axis=-1)
    angular_rates = jax.vmap(lambda pi: jnp.linalg.solve(J, pi))(pis)
    rate_norms = jnp.linalg.norm(angular_rates, axis=-1)
    times = jnp.arange(steps + 1) * dt

    metadata = {
        "duration_s": steps * dt,
        "dt_s": dt,
        "steps": steps,
        "attitude_gain": attitude_gain,
        "rate_gain": rate_gain,
        "inertia_diagonal": jnp.diag(J).tolist(),
        "initial_body_momentum": pi0.tolist(),
        "initial_rotation": R0.tolist(),
        "target_rotation": R_target.tolist(),
        "newton_iters": 8,
        "newton_tolerance": 1e-10,
        "initial_attitude_error_rad": float(attitude_errors[0]),
        "final_attitude_error_rad": float(attitude_errors[-1]),
        "initial_angular_rate_norm": float(rate_norms[0]),
        "final_angular_rate_norm": float(rate_norms[-1]),
        "peak_torque_norm": float(
            jnp.max(jnp.linalg.norm(body_torques, axis=-1))
        ),
        "max_solver_residual": float(
            jnp.max(solver_info.residual_norm)
        ),
    }
    return times, attitude_errors, rate_norms, body_torques, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--docs-out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.docs_out is not None:
        args.docs_out.mkdir(parents=True, exist_ok=True)

    times, attitude_errors, rate_norms, torques, metadata = run_response()
    times = np.asarray(times)
    attitude_errors = np.asarray(attitude_errors)
    rate_norms = np.asarray(rate_norms)
    torques = np.asarray(torques)

    with (args.out / "attitude_control_response.csv").open(
        "w",
        newline="",
    ) as stream:
        fieldnames = [
            "time_s",
            "attitude_error_rad",
            "angular_rate_norm",
            "torque_x",
            "torque_y",
            "torque_z",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, time in enumerate(times):
            torque = (
                torques[index]
                if index < len(torques)
                else [np.nan, np.nan, np.nan]
            )
            writer.writerow(
                {
                    "time_s": time,
                    "attitude_error_rad": attitude_errors[index],
                    "angular_rate_norm": rate_norms[index],
                    "torque_x": torque[0],
                    "torque_y": torque[1],
                    "torque_z": torque[2],
                }
            )

    with (args.out / "attitude_control_response.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 7),
        sharex=True,
        layout="constrained",
    )
    axes[0].semilogy(times, attitude_errors, color="#2563a6")
    axes[0].set_ylabel("Attitude error [rad]")

    axes[1].semilogy(times, rate_norms, color="#0f766e")
    axes[1].set_ylabel(r"$\|\omega\|$ [rad/s]")

    torque_times = times[:-1]
    labels = (r"$\tau_x$", r"$\tau_y$", r"$\tau_z$")
    colors = ("#b45309", "#7c3aed", "#be123c")
    for component, label, color in zip(torques.T, labels, colors):
        axes[2].plot(torque_times, component, label=label, color=color)
    axes[2].axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Body torque")
    axes[2].legend(ncols=3)

    for axis in axes:
        axis.grid(True, which="both", alpha=0.2)

    fig.suptitle("Geometric attitude stabilization on SO(3)")
    control_figure = args.out / "attitude_control_response.png"
    fig.savefig(control_figure, dpi=160)
    if args.docs_out is not None:
        copyfile(control_figure, args.docs_out / control_figure.name)
    plt.close(fig)

    print(
        "Attitude error: "
        f"{metadata['initial_attitude_error_rad']:.6f} -> "
        f"{metadata['final_attitude_error_rad']:.6f} rad"
    )
    print(
        "Angular-rate norm: "
        f"{metadata['initial_angular_rate_norm']:.6f} -> "
        f"{metadata['final_angular_rate_norm']:.6f} rad/s"
    )


if __name__ == "__main__":
    main()
