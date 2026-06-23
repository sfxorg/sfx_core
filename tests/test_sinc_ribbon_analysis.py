import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.jacobians import sem_jacobian

from operators.hybrid_ops import (
    run_hybrid_sfx_2d_standard,
    run_hybrid_sfx_2d_sinc,
)

# ============================================================
# CONFIG
# ============================================================

L = 10.0
N_fft = 64

cx = 1.0
cy = 1.0

dt = 0.05
total_time = 12.5
steps = int(total_time / dt)

# ============================================================
# FFT GRID
# ============================================================

X_fft, Y_fft = jnp.meshgrid(
    jnp.linspace(0, L, N_fft, endpoint=False),
    jnp.linspace(0, L, N_fft, endpoint=False),
)

ik_x = (
    1j
    * 2.0
    * jnp.pi
    * jnp.fft.fftfreq(N_fft, d=L / N_fft)
)[:, None]

ik_y = (
    1j
    * 2.0
    * jnp.pi
    * jnp.fft.fftfreq(N_fft, d=L / N_fft)
)[None, :]

u_init = jnp.exp(
    -((X_fft - 5.0) ** 2 + (Y_fft - 5.0) ** 2) / 1.5
)

# ============================================================
# STANDARD RIBBON
# ============================================================

P_rib_std = 4

_, D_rib_std = get_sem_diff_matrix_2d(P_rib_std)

jac_rib_std = sem_jacobian(16, L)

# ============================================================
# SINC RIBBON
# ============================================================

P_rib_sinc = 1

_, D_rib_sinc = get_sem_diff_matrix_2d(P_rib_sinc)

jac_rib_sinc = sem_jacobian(16, L)

# ============================================================
# HYBRID STANDARD
# ============================================================

@jax.jit
def step_std(carry, _):

    return (
        run_hybrid_sfx_2d_standard(
            carry[0],
            carry[1],
            carry[2],
            carry[3],
            carry[4],
            ik_x,
            ik_y,
            D_rib_std,
            jac_rib_std,
            cx,
            cy,
            dt,
        ),
        None,
    )

init_std = (
    jnp.copy(u_init),
    u_init[:P_rib_std+1, :],
    u_init[-P_rib_std-1:, :],
    u_init[:, :P_rib_std+1],
    u_init[:, -P_rib_std-1:],
)

final_std, _ = jax.lax.scan(
    step_std,
    init_std,
    None,
    length=steps,
)

# ============================================================
# HYBRID SINC
# ============================================================

@jax.jit
def step_sinc(carry, _):

    return (
        run_hybrid_sfx_2d_sinc(
            carry[0],
            carry[1],
            carry[2],
            carry[3],
            carry[4],
            ik_x,
            ik_y,
            D_rib_sinc,
            jac_rib_sinc,
            cx,
            cy,
            dt,
        ),
        None,
    )

init_sinc = (
    jnp.copy(u_init),
    u_init[:P_rib_sinc+1, :],
    u_init[-P_rib_sinc-1:, :],
    u_init[:, :P_rib_sinc+1],
    u_init[:, -P_rib_sinc-1:],
)

final_sinc, _ = jax.lax.scan(
    step_sinc,
    init_sinc,
    None,
    length=steps,
)

# ============================================================
# RIBBON ANALYSIS
# ============================================================

std_top = final_std[1]
std_bot = final_std[2]
std_left = final_std[3]
std_right = final_std[4]

sinc_top = final_sinc[1]
sinc_bot = final_sinc[2]
sinc_left = final_sinc[3]
sinc_right = final_sinc[4]

print("\n" + "="*60)
print("RIBBON SIZE ANALYSIS")
print("="*60)

print("Standard Top Shape :", std_top.shape)
print("Sinc Top Shape     :", sinc_top.shape)

std_dof = (
    std_top.size +
    std_bot.size +
    std_left.size +
    std_right.size
)

sinc_dof = (
    sinc_top.size +
    sinc_bot.size +
    sinc_left.size +
    sinc_right.size
)

print("\nStandard Ribbon DoF :", std_dof)
print("Sinc Ribbon DoF     :", sinc_dof)

print(
    "DoF Reduction       : %.2f %%"
    % (100.0 * (1.0 - sinc_dof/std_dof))
)

# ============================================================
# COMPARISON
# ============================================================

def compare(a, b):

    rows = min(a.shape[0], b.shape[0])
    cols = min(a.shape[1], b.shape[1])

    return float(
        jnp.max(
            jnp.abs(
                a[:rows, :cols]
                -
                b[:rows, :cols]
            )
        )
    )

print("\n" + "="*60)
print("RIBBON DIFFERENCES")
print("="*60)

print("Top    :", compare(std_top, sinc_top))
print("Bottom :", compare(std_bot, sinc_bot))
print("Left   :", compare(std_left.T, sinc_left.T))
print("Right  :", compare(std_right.T, sinc_right.T))

print("\nDone.\n")

