import jax
import jax.numpy as jnp

@jax.jit
def fft_rhs(q, ik_x, ik_y, cx, cy):
    """
    Computes the RHS for the 2D advection equation using FFTs.
    Matches the logic from test_master_collision.py.
    """
    q_hat = jnp.fft.fft2(q)
    dqdx = jnp.fft.ifft2(q_hat * ik_x).real
    dqdy = jnp.fft.ifft2(q_hat * ik_y).real
    return -(cx * dqdx + cy * dqdy)

@jax.jit
def run_pure_fft_2d(u, ik_x, ik_y, cx, cy, dt):
    """
    Legacy helper: performs a single forward Euler step.
    (Note: Using fft_rhs with RK4 is recommended for better stability).
    """
    return u + dt * fft_rhs(u, ik_x, ik_y, cx, cy)
