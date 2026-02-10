import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from symplie.integrators import simulate_free_rigid_body
from symplie.invariants import energy, spatial_momentum, ortho_error

def random_diag_inertia(key):
    d = jax.random.uniform(key, (3,), minval=0.3, maxval=2.0)
    return jnp.diag(d).astype(jnp.float64)

def test_conservation_and_valid_rotation():
    key = jax.random.PRNGKey(0)

    # keep CI fast + stable: few cases, modest step count, dt small enough for Newton to behave.
    for _ in range(6):
        key, kJ, kpi = jax.random.split(key, 3)
        J = random_diag_inertia(kJ)
        pi0 = jax.random.normal(kpi, (3,), dtype=jnp.float64)

        R0 = jnp.eye(3, dtype=jnp.float64)
        dt = 1e-2
        steps = 400

        Rs, pis = simulate_free_rigid_body(R0, pi0, J, dt, steps=steps, newton_iters=6)

        # Spatial momentum should be constant (up to numeric)
        Ls = jax.vmap(spatial_momentum)(Rs, pis)
        L0 = Ls[0]
        max_L_dev = jnp.max(jnp.linalg.norm(Ls - L0, axis=1))

        # Energy should remain bounded (not necessarily exact, but small drift)
        Es = jax.vmap(lambda p: energy(p, J))(pis)
        E0 = Es[0]
        rel_E_dev = jnp.max(jnp.abs((Es - E0) / (E0 + 1e-15)))

        # Rotation validity
        ortho = jax.vmap(ortho_error)(Rs)
        max_ortho = jnp.max(ortho)

        assert float(max_L_dev) < 1e-8
        assert float(rel_E_dev) < 1e-5
        assert float(max_ortho) < 1e-10

