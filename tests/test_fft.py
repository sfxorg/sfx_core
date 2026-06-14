import jax.numpy as jnp
from operators.fft_ops import run_pure_fft_2d

def test_fft_runs():
    u = jnp.ones((8, 8))
    ik = 1j * jnp.ones((8, 1))
    out = run_pure_fft_2d(u, ik, ik.T, 1.0, 1.0, 0.01)
    assert out.shape == u.shape
