"""Demonstrate a JIT-compiled SE(3) exponential/logarithm round trip."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from symplie.se3 import exp as expSE3
from symplie.se3 import log as logSE3


def main() -> None:
    # Twists use [rho, phi]: translation first, rotation vector second.
    rho = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64)
    phi = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float64)
    xi = jnp.concatenate((rho, phi))

    T = jax.jit(expSE3)(xi)
    recovered = jax.jit(logSE3)(T)
    max_error = jnp.max(jnp.abs(recovered - xi))

    jnp.set_printoptions(precision=6, suppress=True)
    print("twist [rho, phi]:")
    print(xi)
    print("\nSE(3) transform:")
    print(T)
    print("\nrecovered twist:")
    print(recovered)
    print(f"\nmaximum round-trip error: {float(max_error):.3e}")

    if not jnp.allclose(recovered, xi, atol=1e-10, rtol=1e-10):
        raise RuntimeError("SE(3) Exp/Log round trip exceeded tolerance")


if __name__ == "__main__":
    main()
