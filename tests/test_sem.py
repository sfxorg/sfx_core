import jax.numpy as jnp
from operators.sem_ops import run_pure_sem_2d

def test_sem_runs():
    u = jnp.ones((5, 5))
    D = jnp.eye(5)
    out = run_pure_sem_2d(u, D, 1.0, 1.0, 1.0, 0.01)
    assert out.shape == u.shape
