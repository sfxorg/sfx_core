import jax.numpy as jnp
from sfx_core.geometry.base_grid import BaseGrid

class FlatCartesianGrid(BaseGrid):
    """Encapsulates your verified flat 2D sandbox grid logic."""
    
    def setup_grid(self):
        self.x = jnp.linspace(self.config.domain_min, self.config.domain_max, self.config.nx)
        self.y = jnp.linspace(self.config.domain_min, self.config.domain_max, self.config.ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')
        self.compute_metric_tensor()

    def compute_metric_tensor(self):
        # Cartesian grid has a uniform trivial metric tensor (identity matrix)
        self.sqrt_g = jnp.ones_like(self.X)
        self.inv_sqrt_g = jnp.ones_like(self.X)
