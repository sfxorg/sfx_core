import time
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

N = 64
L = 10.0

u = jnp.ones((N, N))

ik_x = (
    1j * 2.0 * jnp.pi *
    jnp.fft.fftfreq(N, d=L/N)
)[:, None]

# warmup
uhat = jnp.fft.fft2(u)
jax.block_until_ready(uhat)

# FFT benchmark
t0 = time.time()

for _ in range(1000):
    uhat = jnp.fft.fft2(u)

jax.block_until_ready(uhat)

fft_time = time.time() - t0

# IFFT benchmark
t0 = time.time()

for _ in range(1000):
    tmp = jnp.fft.ifft2(uhat * ik_x)

jax.block_until_ready(tmp)

ifft_time = time.time() - t0

print()
print("FFT 1000 calls :", fft_time)
print("IFFT 1000 calls:", ifft_time)
