from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class GridConfig:
    nx: int = 64
    ny: int = 64
    cfl: float = 0.1
    dt: float = 0.001
    domain_min: float = -jnp.pi / 4
    domain_max: float = jnp.pi / 4
    overlap_cells: int = 2
