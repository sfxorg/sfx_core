import jax.numpy as jnp
from sfx_core.config import GridConfig
from sfx_core.geometry.flat_grid import FlatCartesianGrid
from sfx_core.solvers.advection import JaxAdvectionSolver

def run_flat_verification():
    # 1. Initialize configuration and geometry engine
    config = GridConfig(nx=128, ny=128, dt=0.002)
    grid = FlatCartesianGrid(config=config)
    
    # 2. Extract spatial metrics
    dx = grid.x[1] - grid.x[0]
    dy = grid.y[1] - grid.y[0]
    
    # 3. Setup verification initialization (e.g., standard Gaussian tracer)
    q_init = jnp.exp(-((grid.X)**2 + (grid.Y)**2) / (2 * 0.1**2))
    u_velocity = jnp.ones_like(grid.X) * 1.0
    v_velocity = jnp.ones_like(grid.Y) * 0.5
    
    # 4. Single step test to verify JAX compilation sequence
    dq_dt = JaxAdvectionSolver.upwind_flux_2d(
        q_init, u_velocity, v_velocity, dx, dy, grid.inv_sqrt_g
    )
    
    print(f"Refactoring verified. Initial state max flux: {jnp.max(jnp.abs(dq_dt)):.6e}")

if __name__ == "__main__":
    run_flat_verification()
