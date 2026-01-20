import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.scipy as jsp
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs, potrf
from functools import partial

N = 20480
dtype = jnp.complex128

# A = L^\dagger L + shift * I
shift = 1e-2
N_row = 10240
L = np.random.random([N_row, N]) + np.random.random([N_row, N]) * 1j
A = L.conj().T @ L
A /= np.linalg.norm(A)
A = jnp.array(A) + shift * jnp.eye(N)
b = jnp.ones((N, 1), dtype=dtype)


@jax.jit
def factor_and_solve(mat: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
    chol, lower = jsp.linalg.cho_factor(mat, lower=True)
    return chol, jsp.linalg.cho_solve((chol, lower), rhs)

expected_L, expected_out = factor_and_solve(A, b)


# multiple devices
# A supports only (str, None) sharding
@partial(jax.jit, static_argnums=(2, 3, 4))
def solve_potrs(A: jax.Array, b: jax.Array, T_A: int, mesh: jax._src.mesh.Mesh, in_specs: P):
    A = jax.device_put(A, NamedSharding(mesh, in_specs[0]))
    # jax.debug.visualize_array_sharding(A)
    b = jax.device_put(b, NamedSharding(mesh, in_specs[1]))
    # jax.debug.visualize_array_sharding(b)
    return potrs(A, b, T_A=T_A, mesh=mesh, in_specs=in_specs)

@partial(jax.jit, static_argnums=(1, 2, 3))
def factor_potrf(A: jax.Array, T_A: int, mesh: jax._src.mesh.Mesh, in_specs: P):
    A = jax.device_put(A, NamedSharding(mesh, in_specs))
    return potrf(A, T_A=T_A, mesh=mesh, in_specs=in_specs)


print(f"Devices: {jax.devices()}")
ndev = len(jax.devices("gpu"))
mesh = jax.make_mesh((ndev,), ("S",))
T_A = N // ndev // 4
in_specs = (P("S", None), P(None, None))

A_potrs = A.copy()
out = solve_potrs(A_potrs, b, T_A, mesh, in_specs)
print(jnp.allclose(out.flatten(), expected_out.flatten()))

out = solve_potrs(A_potrs, b, T_A, mesh, in_specs) # A mutated by potrs
print(jnp.allclose(out.flatten(), expected_out.flatten()))

in_specs_a = P("S", None)
A_potrf = A.copy()
L = factor_potrf(A_potrf, T_A, mesh, in_specs_a)
print(jnp.allclose(jnp.tril(L), jnp.tril(expected_L)))
