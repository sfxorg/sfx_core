import jax.numpy as jnp

def get_sem_diff_matrix_2d(P):
    """Computes exact analytical differentiation matrix for elements."""
    nodes = -jnp.cos(jnp.pi * jnp.arange(P + 1) / P)
    c = jnp.ones(P + 1).at[:].set(2.0).at[-1].set(2.0)
    D = jnp.zeros((P + 1, P + 1))

    for i in range(P + 1):
        for j in range(P + 1):
            if i != j:
                D = D.at[i, j].set((c[i] / c[j]) * ((-1.0)**(i + j)) /
                                   (nodes[i] - nodes[j] + 1e-16))

    for i in range(P + 1):
        D = D.at[i, i].set(-jnp.sum(D[i, :]))

    return nodes, D
