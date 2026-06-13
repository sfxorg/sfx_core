from jax import jit
import jax.numpy as jnp

class JaxAdvectionSolver:
    """Core JAX advection pipeline isolated from geometric dependencies."""
    
    @staticmethod
    @jit
    def upwind_flux_2d(q, u, v, dx, dy, inv_sqrt_g):
        """
        Pure JAX 2D Upwind Flux operator scaled by the structural inverse Jacobian.
        """
        # Xflux computation
        flux_x_pos = jnp.maximum(u, 0.0) * (q - jnp.roll(q, 1, axis=0)) / dx
        flux_x_neg = jnp.minimum(u, 0.0) * (jnp.roll(q, -1, axis=0) - q) / dx
        
        # Yflux computation
        flux_y_pos = jnp.maximum(v, 0.0) * (q - jnp.roll(q, 1, axis=1)) / dy
        flux_y_neg = jnp.minimum(v, 0.0) * (jnp.roll(q, -1, axis=1) - q) / dy
        
        # Apply metric transformation scaling
        dq_dt = -inv_sqrt_g * (flux_x_pos + flux_x_neg + flux_y_pos + flux_y_neg)
        return dq_dt
