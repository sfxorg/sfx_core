import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.sem_grids import generate_tiled_sem_grid_2d
from geometry.jacobians import sem_jacobian

from operators.hybrid_ops import (
    run_hybrid_sfx_2d_standard,
    run_hybrid_sfx_2d_sinc,
)

# ============================================================
# HELPERS
# ============================================================

def rk4_step(rhs_func, u, dt):
    k1 = rhs_func(u)
    k2 = rhs_func(u + 0.5 * dt * k1)
    k3 = rhs_func(u + 0.5 * dt * k2)
    k4 = rhs_func(u + dt * k3)

    return u + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )


@jax.jit
def run_fv_proxy(u, cx, cy, dt, dx, dy):
    flux_x = cx * u
    flux_y = cy * u

    d_flux_dx = (flux_x - jnp.roll(flux_x, 1, axis=0)) / dx
    d_flux_dy = (flux_y - jnp.roll(flux_y, 1, axis=1)) / dy

    return u - dt * (d_flux_dx + d_flux_dy)


def mass(u):
    return float(jnp.sum(u))


# ============================================================
# GLOBAL CONFIG
# ============================================================

L = 10.0

cx = 1.0
cy = 1.0

dt = 0.05

total_time = 12.5
steps = int(total_time / dt)

N_fft = 64
dx = L / N_fft

print("\nRunning Validation Benchmark")
print(f"N_fft = {N_fft}")
print(f"steps = {steps}")

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
# SEM SETUP
# ============================================================

E_sem = 32
P_sem = 4

X_sem, Y_sem = generate_tiled_sem_grid_2d(
    E_sem,
    P_sem,
    L,
)

_, D_ref = get_sem_diff_matrix_2d(P_sem)

D_sem_global = jnp.kron(
    jnp.eye(E_sem),
    D_ref,
)

jac_sem = sem_jacobian(E_sem, L)

u_init_sem = jnp.exp(
    -((X_sem - 5.0) ** 2 + (Y_sem - 5.0) ** 2) / 1.5
)

# ============================================================
# RIBBONS
# ============================================================

P_rib_std = 4

_, D_rib_std = get_sem_diff_matrix_2d(P_rib_std)

jac_rib_std = sem_jacobian(16, L)

P_rib_sinc = 1

_, D_rib_sinc = get_sem_diff_matrix_2d(P_rib_sinc)

jac_rib_sinc = sem_jacobian(16, L)

# ============================================================
# PURE FFT
# ============================================================

@jax.jit
def step_fft(u, _):

    def rhs(v):
        v_hat = jnp.fft.fft2(v)

        dudx = jnp.fft.ifft2(v_hat * ik_x).real
        dudy = jnp.fft.ifft2(v_hat * ik_y).real

        return -(cx * dudx + cy * dudy)

    return rk4_step(rhs, u, dt), None


t0 = time.time()

u_fft, _ = jax.lax.scan(
    step_fft,
    jnp.copy(u_init),
    None,
    length=steps,
)

jax.block_until_ready(u_fft)

t_fft = time.time() - t0

# ============================================================
# HYBRID STANDARD
# ============================================================

@jax.jit
def step_hyb_std(carry, _):

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
    u_init[: P_rib_std + 1, :],
    u_init[-P_rib_std - 1 :, :],
    u_init[:, : P_rib_std + 1],
    u_init[:, -P_rib_std - 1 :],
)

t0 = time.time()

final_std, _ = jax.lax.scan(
    step_hyb_std,
    init_std,
    None,
    length=steps,
)

u_hyb_std = final_std[0]

jax.block_until_ready(u_hyb_std)

t_hyb_std = time.time() - t0

# ============================================================
# HYBRID SINC
# ============================================================

@jax.jit
def step_hyb_sinc(carry, _):

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
    u_init[: P_rib_sinc + 1, :],
    u_init[-P_rib_sinc - 1 :, :],
    u_init[:, : P_rib_sinc + 1],
    u_init[:, -P_rib_sinc - 1 :],
)

t0 = time.time()

final_sinc, _ = jax.lax.scan(
    step_hyb_sinc,
    init_sinc,
    None,
    length=steps,
)

u_hyb_sinc = final_sinc[0]

jax.block_until_ready(u_hyb_sinc)

t_hyb_sinc = time.time() - t0

# ============================================================
# FV PROXY
# ============================================================

@jax.jit
def step_fv(u, _):
    return run_fv_proxy(
        u,
        cx,
        cy,
        dt,
        dx,
        dx,
    ), None


t0 = time.time()

u_fv, _ = jax.lax.scan(
    step_fv,
    jnp.copy(u_init),
    None,
    length=steps,
)

jax.block_until_ready(u_fv)

t_fv = time.time() - t0

# ============================================================
# EXACT SOLUTION
# ============================================================

u_exact = jnp.exp(
    -(
        (((X_fft - cx * total_time) % L) - 5.0) ** 2
        + (((Y_fft - cy * total_time) % L) - 5.0) ** 2
    )
    / 1.5
)

# ============================================================
# ERRORS
# ============================================================

err_fft = jnp.max(jnp.abs(u_fft - u_exact))
err_hyb_std = jnp.max(jnp.abs(u_hyb_std - u_exact))
err_hyb_sinc = jnp.max(jnp.abs(u_hyb_sinc - u_exact))
err_fv = jnp.max(jnp.abs(u_fv - u_exact))

# ============================================================
# COUPLING CHECK
# ============================================================

fft_vs_hyb_std = jnp.max(
    jnp.abs(u_fft - u_hyb_std)
)

fft_vs_hyb_sinc = jnp.max(
    jnp.abs(u_fft - u_hyb_sinc)
)

# ============================================================
# MASS CHECK
# ============================================================

mass_init = mass(u_init)

mass_fft = mass(u_fft)
mass_hyb_std = mass(u_hyb_std)
mass_hyb_sinc = mass(u_hyb_sinc)
mass_fv = mass(u_fv)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 60)
print("ACCURACY")
print("=" * 60)

print(f"FFT Error             : {float(err_fft):.6e}")
print(f"Hybrid Std Error      : {float(err_hyb_std):.6e}")
print(f"Hybrid Sinc Error     : {float(err_hyb_sinc):.6e}")
print(f"FV Error              : {float(err_fv):.6e}")

print("\n" + "=" * 60)
print("RUNTIME")
print("=" * 60)

print(f"FFT Runtime           : {t_fft:.4f} s")
print(f"Hybrid Std Runtime    : {t_hyb_std:.4f} s")
print(f"Hybrid Sinc Runtime   : {t_hyb_sinc:.4f} s")
print(f"FV Runtime            : {t_fv:.4f} s")

print("\n" + "=" * 60)
print("COUPLING VALIDATION")
print("=" * 60)

print(
    f"FFT vs Hybrid Std     : {float(fft_vs_hyb_std):.6e}"
)

print(
    f"FFT vs Hybrid Sinc    : {float(fft_vs_hyb_sinc):.6e}"
)

print("\n" + "=" * 60)
print("MASS CONSERVATION")
print("=" * 60)

print(f"Initial Mass          : {mass_init:.12f}")
print(f"FFT Mass              : {mass_fft:.12f}")
print(f"Hybrid Std Mass       : {mass_hyb_std:.12f}")
print(f"Hybrid Sinc Mass      : {mass_hyb_sinc:.12f}")
print(f"FV Mass               : {mass_fv:.12f}")

print("\nMass Errors")

print(
    f"FFT Mass Error        : "
    f"{abs(mass_fft - mass_init):.6e}"
)

print(
    f"Hybrid Std Error      : "
    f"{abs(mass_hyb_std - mass_init):.6e}"
)

print(
    f"Hybrid Sinc Error     : "
    f"{abs(mass_hyb_sinc - mass_init):.6e}"
)

print(
    f"FV Mass Error         : "
    f"{abs(mass_fv - mass_init):.6e}"
)

print("\nDone.")
