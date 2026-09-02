import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from symplie.integrators import simulate_free_rigid_body
from symplie.invariants import determinant_error, energy, spatial_momentum, ortho_error
from symplie.so3 import hat, log

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

        Rs, pis, solver_info = simulate_free_rigid_body(
            R0,
            pi0,
            J,
            dt,
            steps=steps,
        )

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
        max_det = jnp.max(jax.vmap(determinant_error)(Rs))
        max_residual = jnp.max(solver_info.residual_norm)

        assert float(max_L_dev) < 1e-8
        assert float(rel_E_dev) < 1e-5
        assert float(max_ortho) < 1e-10
        assert float(max_det) < 1e-10
        assert bool(jnp.all(solver_info.converged)), float(max_residual)
        assert float(max_residual) < 1e-10

def test_attitude_against_high_resolution_rk4_reference():
    """Conservation is supplemented by an independent trajectory check."""
    J = jnp.diag(jnp.array([0.6, 1.0, 1.8], dtype=jnp.float64))
    initial = (jnp.eye(3, dtype=jnp.float64), jnp.array([0.2, 0.7, 1.0]))
    h = 1e-4

    def derivative(state):
        R, pi = state
        omega = jnp.linalg.solve(J, pi)
        return R @ hat(omega), jnp.cross(pi, omega)

    def add(state, tangent, scale):
        return state[0] + scale*tangent[0], state[1] + scale*tangent[1]

    def rk4_step(_, state):
        k1 = derivative(state); k2 = derivative(add(state, k1, h/2))
        k3 = derivative(add(state, k2, h/2)); k4 = derivative(add(state, k3, h))
        return tuple(x + h*(a+2*b+2*c+d)/6 for x,a,b,c,d in zip(state,k1,k2,k3,k4))

    R_reference, pi_reference = jax.lax.fori_loop(0, 5000, rk4_step, initial)
    Rs, pis, solver_info = simulate_free_rigid_body(
        *initial,
        J,
        0.005,
        steps=100,
    )
    attitude_error = jnp.linalg.norm(log(R_reference.T @ Rs[-1]))
    assert bool(jnp.all(solver_info.converged))
    assert float(attitude_error) < 2e-6
    assert float(jnp.linalg.norm(pi_reference - pis[-1])) < 1e-6
