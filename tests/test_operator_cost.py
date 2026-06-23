import os
import sys

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

jax.config.update("jax_enable_x64", True)

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.jacobians import sem_jacobian
from operators.hybrid_ops import (
    run_hybrid_sfx_2d_standard,
    run_hybrid_sfx_2d_sinc,
)

N = 64
L = 10.0

cx = 1.0
cy = 1.0
dt = 0.05

u = jnp.ones((N, N))

ik_x = (
    1j * 2.0 * jnp.pi *
    jnp.fft.fftfreq(N, d=L/N)
)[:, None]

ik_y = (
    1j * 2.0 * jnp.pi *
    jnp.fft.fftfreq(N, d=L/N)
)[None, :]

# --------------------------------------------------
# STANDARD
# --------------------------------------------------

P_std = 4

_, D_std = get_sem_diff_matrix_2d(P_std)

jac_std = sem_jacobian(16, L)

state_std = (
    u,
    u[:P_std+1,:],
    u[-P_std-1:,:],
    u[:,:P_std+1],
    u[:,-P_std-1:]
)

run_hybrid_sfx_2d_standard(*state_std,
                           ik_x,
                           ik_y,
                           D_std,
                           jac_std,
                           cx,
                           cy,
                           dt)

jax.block_until_ready(state_std[0])

t0 = time.time()

for _ in range(1000):

    result = run_hybrid_sfx_2d_standard(
        *state_std,
        ik_x,
        ik_y,
        D_std,
        jac_std,
        cx,
        cy,
        dt
    )

jax.block_until_ready(result[0])

std_time = time.time() - t0

# --------------------------------------------------
# SINC
# --------------------------------------------------

P_sinc = 1

_, D_sinc = get_sem_diff_matrix_2d(P_sinc)

jac_sinc = sem_jacobian(16, L)

state_sinc = (
    u,
    u[:P_sinc+1,:],
    u[-P_sinc-1:,:],
    u[:,:P_sinc+1],
    u[:,-P_sinc-1:]
)

run_hybrid_sfx_2d_sinc(
    *state_sinc,
    ik_x,
    ik_y,
    D_sinc,
    jac_sinc,
    cx,
    cy,
    dt
)

jax.block_until_ready(state_sinc[0])

t0 = time.time()

for _ in range(1000):

    result = run_hybrid_sfx_2d_sinc(
        *state_sinc,
        ik_x,
        ik_y,
        D_sinc,
        jac_sinc,
        cx,
        cy,
        dt
    )

jax.block_until_ready(result[0])

sinc_time = time.time() - t0

print()
print("Standard Hybrid 1000 calls :", std_time)
print("Sinc Hybrid 1000 calls     :", sinc_time)
print("Speedup                    :", std_time / sinc_time)
