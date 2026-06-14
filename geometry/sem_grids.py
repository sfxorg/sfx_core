import jax.numpy as jnp
from .sem_nodes import get_sem_diff_matrix_2d

def generate_tiled_sem_grid_2d(E, P, L):
    """Generates a multi-element continuous grid to eliminate dispersion smearing."""
    nodes_ref, _ = get_sem_diff_matrix_2d(P)
    dx_elem = L / E
    x_coords = []

    for e in range(E):
        x_coords.append((e * dx_elem) + 0.5 * dx_elem * (nodes_ref + 1.0))

    X_elem_1d = jnp.concatenate(x_coords)
    return jnp.meshgrid(X_elem_1d, X_elem_1d)

def generate_sem_1d(E, P, L):
    """Generate 1D SEM element nodes across the domain."""
    nodes_ref, _ = get_sem_diff_matrix_2d(P)
    dx_elem = L / E
    coords = [(e * dx_elem) + 0.5 * dx_elem * (nodes_ref + 1.0)
              for e in range(E)]
    return jnp.concatenate(coords)

def generate_sem_grid_2d(E, P, L):
    """Full 2D SEM grid."""
    X1d = generate_sem_1d(E, P, L)
    return jnp.meshgrid(X1d, X1d)

def generate_ribbon_grid(P_ribbon, L, num_elems):
    """Generate 1D ribbon grid for hybrid SFX frame."""
    nodes_ref, _ = get_sem_diff_matrix_2d(P_ribbon)
    dx_elem = L / num_elems
    coords = [(e * dx_elem) + 0.5 * dx_elem * (nodes_ref + 1.0)
              for e in range(num_elems)]
    return jnp.concatenate(coords)
