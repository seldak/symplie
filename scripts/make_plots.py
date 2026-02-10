import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from symplie.integrators import simulate_free_rigid_body
from symplie.invariants import energy, spatial_momentum

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

    Rs, pis = simulate_free_rigid_body(R0, pi0, J, dt, steps=steps, newton_iters=6)

    Es = jax.vmap(lambda p: energy(p, J))(pis)
    E0 = Es[0]
    relE = (Es - E0) / (E0 + 1e-15)

    Ls = jax.vmap(spatial_momentum)(Rs, pis)
    L0 = Ls[0]
    Lerr = jnp.linalg.norm(Ls - L0, axis=1)

    t = jnp.arange(steps + 1) * dt

    # Energy drift plot
    plt.figure()
    plt.plot(t, relE)
    plt.xlabel("time [s]")
    plt.ylabel("(E - E0) / E0")
    plt.title("Energy drift (torque-free rigid body)")
    plt.savefig(out / "energy_drift.png", dpi=160)
    plt.close()

    # Spatial momentum error plot
    plt.figure()
    plt.plot(t, Lerr)
    plt.xlabel("time [s]")
    plt.ylabel("||L - L0||")
    plt.title("Spatial momentum deviation")
    plt.savefig(out / "spatial_momentum_error.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()

