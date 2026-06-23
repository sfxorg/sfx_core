import os
import sys
import time

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.jacobians import sem_jacobian

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

N = 64
L = 10.0

dt = 0.05
cx = 1.0
cy = 1.0

u_fft = jnp.ones((N, N))

ik_x = (
    1j
    * 2.0
    * jnp.pi
    * jnp.fft.fftfreq(N, d=L/N)
)[:, None]

ik_y = (
    1j
    * 2.0
    * jnp.pi
    * jnp.fft.fftfreq(N, d=L/N)
)[None, :]

# ----------------------------------------------------
# P=4 STANDARD
# ----------------------------------------------------

P = 4

_, D = get_sem_diff_matrix_2d(P)

jac = sem_jacobian(16, L)

rib_h = jnp.ones((P+1, N))
rib_v = jnp.ones((N, P+1))

# ----------------------------------------------------
# FFT INTERIOR
# ----------------------------------------------------

@jax.jit
def fft_rhs(u):

    uhat = jnp.fft.fft2(u)

    return -(
        cx * jnp.fft.ifft2(
            uhat * ik_x
        ).real
        +
        cy * jnp.fft.ifft2(
            uhat * ik_y
        ).real
    )

fft_rhs(u_fft)
jax.block_until_ready(fft_rhs(u_fft))

t0 = time.time()

for _ in range(1000):
    out = fft_rhs(u_fft)

jax.block_until_ready(out)

fft_time = time.time() - t0

# ----------------------------------------------------
# HORIZONTAL RIBBON
# ----------------------------------------------------

@jax.jit
def horizontal_ribbon(rib):

    du_dx = jnp.dot(D, rib) * jac

    du_dy = jnp.fft.ifft(
        jnp.fft.fft(rib, axis=1)
        * ik_y,
        axis=1
    ).real

    return -(cx * du_dx + cy * du_dy)

horizontal_ribbon(rib_h)
jax.block_until_ready(horizontal_ribbon(rib_h))

t0 = time.time()

for _ in range(1000):
    out = horizontal_ribbon(rib_h)

jax.block_until_ready(out)

horizontal_time = time.time() - t0

# ----------------------------------------------------
# VERTICAL RIBBON
# ----------------------------------------------------

@jax.jit
def vertical_ribbon(rib):

    du_dx = jnp.fft.ifft(
        jnp.fft.fft(rib, axis=0)
        * ik_x,
        axis=0
    ).real

    du_dy = jnp.dot(
        rib,
        D.T
    ) * jac

    return -(cx * du_dx + cy * du_dy)

vertical_ribbon(rib_v)
jax.block_until_ready(vertical_ribbon(rib_v))

t0 = time.time()

for _ in range(1000):
    out = vertical_ribbon(rib_v)

jax.block_until_ready(out)

vertical_time = time.time() - t0

# ----------------------------------------------------
# SINC PROJECTION
# ----------------------------------------------------

sinc_kernel = jnp.sinc(
    jnp.arange(N) - N/2
)

sinc_kernel = sinc_kernel.reshape(1, N)

@jax.jit
def sinc_projection(data):

    projected = sinc_kernel @ data

    return jnp.broadcast_to(
        projected,
        (2, N)
    )

sinc_projection(u_fft)
jax.block_until_ready(
    sinc_projection(u_fft)
)

t0 = time.time()

for _ in range(1000):
    out = sinc_projection(u_fft)

jax.block_until_ready(out)

sinc_time = time.time() - t0

# ----------------------------------------------------
# REPORT
# ----------------------------------------------------

total = (
    fft_time
    + horizontal_time
    + vertical_time
    + sinc_time
)

print()
print("=" * 60)
print("HYBRID BREAKDOWN")
print("=" * 60)

print(f"FFT Interior      : {fft_time:.6f}")
print(f"Ribbon Horizontal : {horizontal_time:.6f}")
print(f"Ribbon Vertical   : {vertical_time:.6f}")
print(f"Sinc Projection   : {sinc_time:.6f}")

print()

print(
    f"FFT %             : "
    f"{100*fft_time/total:.2f}%"
)

print(
    f"Horizontal %      : "
    f"{100*horizontal_time/total:.2f}%"
)

print(
    f"Vertical %        : "
    f"{100*vertical_time/total:.2f}%"
)

print(
    f"Sinc %            : "
    f"{100*sinc_time/total:.2f}%"
)
