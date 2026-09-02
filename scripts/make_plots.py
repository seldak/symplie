import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from symplie.integrators import simulate_free_rigid_body
from symplie.invariants import energy, spatial_momentum
from symplie.so3 import exp as expSO3

def rk4_projected(R0, pi0, J, dt, steps):
    """Conventional RK4 baseline for Euler's equations plus SO(3) projection."""
    def derivative(pi):
        return jnp.cross(pi, jnp.linalg.solve(J, pi))
    R, pi = R0, pi0
    rotations, momenta = [R], [pi]
    for _ in range(steps):
        k1 = derivative(pi)
        k2 = derivative(pi + dt * k1 / 2)
        k3 = derivative(pi + dt * k2 / 2)
        k4 = derivative(pi + dt * k3)
        next_pi = pi + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        omega_mid = jnp.linalg.solve(J, (pi + next_pi) / 2)
        R = R @ expSO3(dt * omega_mid)
        pi = next_pi
        rotations.append(R); momenta.append(pi)
    return jnp.stack(rotations), jnp.stack(momenta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="artifacts")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Classic asymmetric inertia case (torque-free precession)
    J = jnp.diag(jnp.array([0.6, 1.0, 1.8], dtype=jnp.float64))
    R0 = jnp.eye(3, dtype=jnp.float64)
    pi0 = jnp.array([0.2, 0.7, 1.0], dtype=jnp.float64)

    dt = 1e-2
    steps = 2000

    Rs, pis, solver_info = simulate_free_rigid_body(
        R0,
        pi0,
        J,
        dt,
        steps=steps,
    )
    if not bool(jnp.all(solver_info.converged)):
        max_residual = float(jnp.max(solver_info.residual_norm))
        raise RuntimeError(
            f"Variational solve failed to converge; maximum residual: {max_residual}"
        )

    Es = jax.vmap(lambda p: energy(p, J))(pis)
    E0 = Es[0]
    relE = (Es - E0) / (E0 + 1e-15)

    Ls = jax.vmap(spatial_momentum)(Rs, pis)
    L0 = Ls[0]
    Lerr = jnp.linalg.norm(Ls - L0, axis=1)
    rk_Rs, rk_pis = rk4_projected(R0, pi0, J, dt, steps)
    rk_Es = jax.vmap(lambda p: energy(p, J))(rk_pis)
    rk_relE = (rk_Es - rk_Es[0]) / rk_Es[0]
    rk_Ls = jax.vmap(spatial_momentum)(rk_Rs, rk_pis)
    rk_Lerr = jnp.linalg.norm(rk_Ls - rk_Ls[0], axis=1)

    t = jnp.arange(steps + 1) * dt

    # Energy drift plot
    plt.figure()
    plt.plot(t, relE, label="variational")
    plt.plot(t, rk_relE, label="RK4 + Lie-group attitude update")
    plt.xlabel("time [s]")
    plt.ylabel("(E - E0) / E0")
    plt.title("Energy drift (torque-free rigid body)")
    plt.legend()
    plt.savefig(out / "energy_drift.png", dpi=160)
    plt.close()

    # Spatial momentum error plot
    plt.figure()
    plt.plot(t, Lerr, label="variational")
    plt.plot(t, rk_Lerr, label="RK4 + Lie-group attitude update")
    plt.xlabel("time [s]")
    plt.ylabel("||L - L0||")
    plt.title("Spatial momentum deviation")
    plt.legend()
    plt.savefig(out / "spatial_momentum_error.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
