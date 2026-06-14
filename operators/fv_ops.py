import jax
import jax.numpy as jnp

@jax.jit
def run_fv_advection_step(u, flux_x, flux_y, dx, dy, dt):
    """
    Finite Volume Flux-Form Advection (MPAS-style proxy).
    Uses a standard upwind or flux-limiter approach.
    """
    # 1D Flux-form transport operator (PPM-lite proxy)
    # This represents the 'Finite Volume' philosophy
    d_flux_dx = (flux_x[1:, :] - flux_x[:-1, :]) / dx
    d_flux_dy = (flux_y[:, 1:] - flux_y[:, :-1]) / dy
    
    return u - dt * (d_flux_dx + d_flux_dy)
