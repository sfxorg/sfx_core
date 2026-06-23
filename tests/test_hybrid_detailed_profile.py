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

import time

import jax
import jax.numpy as jnp

from operators.hybrid_ops import (
    run_hybrid_sfx_2d_standard,
    run_hybrid_sfx_2d_sinc,
)

# --------------------------------------------------
# utility
# --------------------------------------------------

def walltime(fn, *args):

    t0 = time.perf_counter()

    result = fn(*args)

    jax.block_until_ready(result)

    return time.perf_counter() - t0

# --------------------------------------------------
# setup
# --------------------------------------------------

N = 64
P = 4

u_fft = jnp.ones((N, N))

u_rib_top = jnp.ones((P, N))
u_rib_bot = jnp.ones((P, N))

u_rib_left = jnp.ones((N, P))
u_rib_right = jnp.ones((N, P))

kx = 2j * jnp.pi * jnp.fft.fftfreq(N)
ky = 2j * jnp.pi * jnp.fft.fftfreq(N)

ik_x = kx.reshape(N, 1)
ik_y = ky.reshape(1, N)

D_matrix = jnp.eye(P)

jac_sem = 1.0

cx = 1.0
cy = 1.0
dt = 0.01

# --------------------------------------------------
# warmup jit
# --------------------------------------------------

_ = run_hybrid_sfx_2d_standard(
    u_fft,
    u_rib_top,
    u_rib_bot,
    u_rib_left,
    u_rib_right,
    ik_x,
    ik_y,
    D_matrix,
    jac_sem,
    cx,
    cy,
    dt,
)

_ = run_hybrid_sfx_2d_sinc(
    u_fft,
    u_rib_top,
    u_rib_bot,
    u_rib_left,
    u_rib_right,
    ik_x,
    ik_y,
    D_matrix,
    jac_sem,
    cx,
    cy,
    dt,
)

jax.block_until_ready(_)

# --------------------------------------------------
# timing
# --------------------------------------------------

nrepeat = 100

std_times = []
sinc_times = []

for _ in range(nrepeat):

    std_times.append(
        walltime(
            run_hybrid_sfx_2d_standard,
            u_fft,
            u_rib_top,
            u_rib_bot,
            u_rib_left,
            u_rib_right,
            ik_x,
            ik_y,
            D_matrix,
            jac_sem,
            cx,
            cy,
            dt,
        )
    )

    sinc_times.append(
        walltime(
            run_hybrid_sfx_2d_sinc,
            u_fft,
            u_rib_top,
            u_rib_bot,
            u_rib_left,
            u_rib_right,
            ik_x,
            ik_y,
            D_matrix,
            jac_sem,
            cx,
            cy,
            dt,
        )
    )

print()
print("=" * 60)
print("HYBRID TIMESTEP PROFILE")
print("=" * 60)

print(f"Standard RK4 : {sum(std_times)/nrepeat:.6e} s")
print(f"Sinc RK4     : {sum(sinc_times)/nrepeat:.6e} s")

print(
    f"Sinc Speedup : "
    f"{(sum(std_times)/sum(sinc_times)):.3f}x"
)
