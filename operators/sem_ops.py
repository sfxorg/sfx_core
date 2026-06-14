import jax
import jax.numpy as jnp

@jax.jit
def run_pure_sem_2d(u, D_matrix, jacobian, cx, cy, dt):
    du_dx = jnp.dot(D_matrix, u) * jacobian
    du_dy = jnp.dot(u, D_matrix.T) * jacobian
    return u - dt * (cx * du_dx + cy * du_dy)
