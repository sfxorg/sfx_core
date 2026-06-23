import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.sem_grids import generate_tiled_sem_grid_2d
from geometry.jacobians import sem_jacobian

from operators.hybrid_ops import (
    run_hybrid_sfx_2d_standard,
    run_hybrid_sfx_2d_sinc,
)

from plots.diagnostics import plot_sfx_dashboard


# ============================================================
# TIMING UTILITIES
# ============================================================

def timed_run(name, fn):

    print(f"\nRunning {name}...")

    # compile + execute
    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    compile_exec = time.perf_counter() - t0

    # execute only
    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    exec_only = time.perf_counter() - t0

    print(
        f"{name:<20}"
        f" compile+run={compile_exec:.6f}s "
        f"run={exec_only:.6f}s"
    )

    return result, compile_exec, exec_only


# ============================================================
# RK4
# ============================================================

def rk4_step(rhs_func, u, dt):

    k1 = rhs_func(u)
    k2 = rhs_func(u + 0.5 * dt * k1)
    k3 = rhs_func(u + 0.5 * dt * k2)
    k4 = rhs_func(u + dt * k3)

    return u + (dt / 6.0) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )


# ============================================================
# FV PROXY
# ============================================================

@jax.jit
def run_fv_proxy(
    u,
    cx,
    cy,
    dt,
    dx,
    dy,
):

    flux_x = cx * u
    flux_y = cy * u

    d_flux_dx = (
        flux_x
        - jnp.roll(flux_x, 1, axis=0)
    ) / dx

    d_flux_dy = (
        flux_y
        - jnp.roll(flux_y, 1, axis=1)
    ) / dy

    return u - dt * (d_flux_dx + d_flux_dy)


# ============================================================
# GLOBAL CONFIG
# ============================================================

L = 10.0

cx = 1.0
cy = 1.0

dt = 0.05
total_time = 12.5

steps = int(total_time / dt)

N_fft = 256 #128 #64
dx = L / N_fft


# ============================================================
# FFT GRID
# ============================================================

X_fft, Y_fft = jnp.meshgrid(
    jnp.linspace(0, L, N_fft, endpoint=False),
    jnp.linspace(0, L, N_fft, endpoint=False),
)

ik_x = (
    1j
    * 2
    * jnp.pi
    * jnp.fft.fftfreq(
        N_fft,
        d=L / N_fft,
    )
)[:, None]

ik_y = (
    1j
    * 2
    * jnp.pi
    * jnp.fft.fftfreq(
        N_fft,
        d=L / N_fft,
    )
)[None, :]

u_init = jnp.exp(
    -(
        (X_fft - 5) ** 2
        + (Y_fft - 5) ** 2
    )
    / 1.5
)


# ============================================================
# SEM GRID
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

jac_sem = sem_jacobian(
    E_sem,
    L,
)

u_init_sem = jnp.exp(
    -(
        (X_sem - 5) ** 2
        + (Y_sem - 5) ** 2
    )
    / 1.5
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
# FFT SOLVER
# ============================================================

@jax.jit
def step_fft(u, _):

    def rhs(q):

        return -(
            cx
            * jnp.fft.ifft2(
                jnp.fft.fft2(q) * ik_x
            ).real
            + cy
            * jnp.fft.ifft2(
                jnp.fft.fft2(q) * ik_y
            ).real
        )

    return rk4_step(rhs, u, dt), None


# ============================================================
# SEM SOLVER
# ============================================================

@jax.jit
def step_sem(u, _):

    def rhs(q):

        return -(
            cx
            * jnp.dot(
                D_sem_global,
                q,
            )
            * jac_sem
            + cy
            * jnp.dot(
                q,
                D_sem_global.T,
            )
            * jac_sem
        )

    return rk4_step(rhs, u, dt), None


# ============================================================
# HYBRID STANDARD
# ============================================================

@jax.jit
def step_hyb_std(state, _):

    return (
        run_hybrid_sfx_2d_standard(
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
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


# ============================================================
# HYBRID SINC
# ============================================================

@jax.jit
def step_hyb_sinc(state, _):

    return (
        run_hybrid_sfx_2d_sinc(
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
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


# ============================================================
# RUN BENCHMARKS
# ============================================================

u_fft, _, t_fft = timed_run(
    "PURE FFT",
    lambda: jax.lax.scan(
        step_fft,
        jnp.copy(u_init),
        None,
        length=steps,
    )[0],
)

u_sem, _, t_sem = timed_run(
    "PURE SEM",
    lambda: jax.lax.scan(
        step_sem,
        jnp.copy(u_init_sem),
        None,
        length=steps,
    )[0],
)

init_state_std = (
    jnp.copy(u_init),
    u_init[:P_rib_std + 1, :],
    u_init[-P_rib_std - 1:, :],
    u_init[:, :P_rib_std + 1],
    u_init[:, -P_rib_std - 1:],
)

final_hyb_std, _, t_hyb_std = timed_run(
    "HYBRID STD",
    lambda: jax.lax.scan(
        step_hyb_std,
        init_state_std,
        None,
        length=steps,
    )[0],
)

u_hyb_std = final_hyb_std[0]

init_state_sinc = (
    jnp.copy(u_init),
    u_init[:P_rib_sinc + 1, :],
    u_init[-P_rib_sinc - 1:, :],
    u_init[:, :P_rib_sinc + 1],
    u_init[:, -P_rib_sinc - 1:],
)

final_hyb_sinc, _, t_hyb_sinc = timed_run(
    "HYBRID SINC",
    lambda: jax.lax.scan(
        step_hyb_sinc,
        init_state_sinc,
        None,
        length=steps,
    )[0],
)

u_hyb_sinc = final_hyb_sinc[0]

u_fv, _, t_fv = timed_run(
    "FV PROXY",
    lambda: jax.lax.scan(
        lambda u, _: (
            run_fv_proxy(
                u,
                cx,
                cy,
                dt,
                dx,
                dx,
            ),
            None,
        ),
        jnp.copy(u_init),
        None,
        length=steps,
    )[0],
)

# ============================================================
# ERROR PROFILING
# ============================================================

t0 = time.perf_counter()

u_exact = jnp.exp(
    -(
        (((X_fft - cx * total_time) % L) - 5) ** 2
        + (((Y_fft - cy * total_time) % L) - 5) ** 2
    )
    / 1.5
)

u_exact_sem = jnp.exp(
    -(
        (((X_sem - cx * total_time) % L) - 5) ** 2
        + (((Y_sem - cy * total_time) % L) - 5) ** 2
    )
    / 1.5
)

err_fft = jnp.abs(u_fft - u_exact)
err_sem = jnp.abs(u_sem - u_exact_sem)
err_hyb_std = jnp.abs(u_hyb_std - u_exact)
err_hyb_sinc = jnp.abs(u_hyb_sinc - u_exact)
err_fv = jnp.abs(u_fv - u_exact)

jax.block_until_ready(err_hyb_sinc)

error_time = time.perf_counter() - t0


# ============================================================
# REPORT
# ============================================================

print("\n")
print("=" * 70)
print("EXECUTION PROFILE")
print("=" * 70)

print(f"FFT          : {t_fft:.6f} s")
print(f"SEM          : {t_sem:.6f} s")
print(f"HYB STD      : {t_hyb_std:.6f} s")
print(f"HYB SINC     : {t_hyb_sinc:.6f} s")
print(f"FV           : {t_fv:.6f} s")

print("-" * 70)

print(f"Error Eval   : {error_time:.6f} s")

print("-" * 70)

print(
    f"STD/FFT      : {t_hyb_std/t_fft:.2f}x"
)

print(
    f"SINC/FFT     : {t_hyb_sinc/t_fft:.2f}x"
)

print(
    f"SINC/STD     : {t_hyb_sinc/t_hyb_std:.2f}x"
)

print("-" * 70)

print(
    f"FFT step     : {1e3*t_fft/steps:.4f} ms"
)

print(
    f"STD step     : {1e3*t_hyb_std/steps:.4f} ms"
)

print(
    f"SINC step    : {1e3*t_hyb_sinc/steps:.4f} ms"
)

print("=" * 70)
