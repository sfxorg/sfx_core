import jax
import jax.numpy as jnp

@jax.jit
def run_pure_fft_2d(u, ik_x, ik_y, cx, cy, dt):
    u_hat = jnp.fft.fft2(u)
    return u - dt * (
        cx * jnp.fft.ifft2(u_hat * ik_x).real +
        cy * jnp.fft.ifft2(u_hat * ik_y).real
    )
